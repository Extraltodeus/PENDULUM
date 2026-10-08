import contextlib
import functools
import math
import os
import re

import numpy as np
import torch

from engine.dequant import KERNELS, dequant_rows
from engine.ggufio import GGUFFile, gguf_module
from engine.qlinear import CHUNK as QCHUNK, QuantLinear, quantizable, quantize, raw_tensor

BLOCK = re.compile(r"blk\.(\d+)\.")
LOOKUPS = ("per_layer_token_embd",)
SPARE = 320 * 2**20
WORKSPACE = 512 * 2**20
CHUNK = 2**24
KEEP_DENSE = {}
DENSE_NAMES = ("per_layer_token_embd.",)
MODES = {"auto": (False, True), "bf16": (False,), "quant": (True,)}
STEP = 0.054 / 2**30
SLOWER = {(False, False): 1.0, (True, False): 1.78, (False, True): 2.44, (True, True): 4.07}
PCIE = 12 * 2**30
HEAD = 2**26


def supported(arch):
	if arch == "qwen35":
		return True
	try:
		from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
	except Exception:
		return False
	return arch in GGUF_CONFIG_MAPPING


def _untile(t, dim, nk, nvk, hd):
	shape = list(t.shape)
	t = t.reshape(*shape[:dim], nvk, nk, hd, *shape[dim + 1:])
	perm = list(range(t.dim()))
	perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
	return t.permute(*perm).contiguous().reshape(*shape)


def _scalar(v):
	return int(max(v)) if isinstance(v, (list, tuple)) else int(v)


def qwen35_config(g):
	from transformers import Qwen3_5TextConfig
	a = g.arch_get
	n = int(a("block_count")) - int(a("nextn_predict_layers", 0) or 0)
	kd = int(a("attention.key_length"))
	nv = int(a("ssm.time_step_rank"))
	every = int(a("full_attention_interval", 4) or 4)
	sections = list(a("rope.dimension_sections", [11, 11, 10, 0]))[:3]
	return Qwen3_5TextConfig(
		vocab_size=int(g.tensors["token_embd.weight"].dims[1]),
		hidden_size=int(a("embedding_length")),
		intermediate_size=int(a("feed_forward_length")),
		num_hidden_layers=n,
		num_attention_heads=_scalar(a("attention.head_count")),
		num_key_value_heads=_scalar(a("attention.head_count_kv")),
		head_dim=kd,
		rms_norm_eps=float(a("attention.layer_norm_rms_epsilon")),
		max_position_embeddings=int(a("context_length", 32768)),
		tie_word_embeddings=not g.has("output.weight"),
		rope_parameters={"rope_type": "default", "rope_theta": float(a("rope.freq_base")), "partial_rotary_factor": int(a("rope.dimension_count")) / kd, "mrope_section": sections, "mrope_interleaved": True},
		linear_conv_kernel_dim=int(a("ssm.conv_kernel")),
		linear_key_head_dim=int(a("ssm.state_size")),
		linear_num_key_heads=int(a("ssm.group_count")),
		linear_num_value_heads=nv,
		linear_value_head_dim=int(a("ssm.inner_size")) // nv,
		layer_types=["full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(n)],
	)


def qwen35_state(g, cfg, dtype, device, skip=()):
	nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
	kk, vd, nvk = cfg.linear_key_head_dim * nk, cfg.linear_value_head_dim, nv // nk
	sd = {}

	def T(name):
		return g.dequant(name, device=device).reshape(tuple(reversed(g.tensors[name].dims)))

	def put(key, make):
		if key not in skip:
			sd[key] = make().to(dtype).cpu()

	def heads(t):
		return _untile(t.unsqueeze(-1), 0, nk, nvk, 1).squeeze(-1)

	def mixed(name):
		t = T(name)
		return torch.cat([t[:2 * kk], _untile(t[2 * kk:], 0, nk, nvk, vd)])

	put("model.embed_tokens.weight", lambda: T("token_embd.weight"))
	put("model.norm.weight", lambda: T("output_norm.weight") - 1.0)
	if g.has("output.weight"):
		put("lm_head.weight", lambda: T("output.weight"))
	for i, kind in enumerate(cfg.layer_types):
		b, p = f"blk.{i}.", f"model.layers.{i}."
		put(p + "input_layernorm.weight", lambda: T(b + "attn_norm.weight") - 1.0)
		put(p + "post_attention_layernorm.weight", lambda: T(b + "post_attention_norm.weight") - 1.0)
		put(p + "mlp.gate_proj.weight", lambda: T(b + "ffn_gate.weight"))
		put(p + "mlp.up_proj.weight", lambda: T(b + "ffn_up.weight"))
		put(p + "mlp.down_proj.weight", lambda: T(b + "ffn_down.weight"))
		if kind == "full_attention":
			q = p + "self_attn."
			put(q + "q_proj.weight", lambda: T(b + "attn_q.weight"))
			put(q + "k_proj.weight", lambda: T(b + "attn_k.weight"))
			put(q + "v_proj.weight", lambda: T(b + "attn_v.weight"))
			put(q + "o_proj.weight", lambda: T(b + "attn_output.weight"))
			put(q + "q_norm.weight", lambda: T(b + "attn_q_norm.weight") - 1.0)
			put(q + "k_norm.weight", lambda: T(b + "attn_k_norm.weight") - 1.0)
			continue
		q = p + "linear_attn."
		put(q + "in_proj_qkv.weight", lambda: mixed(b + "attn_qkv.weight"))
		put(q + "in_proj_z.weight", lambda: _untile(T(b + "attn_gate.weight"), 0, nk, nvk, vd))
		put(q + "in_proj_b.weight", lambda: _untile(T(b + "ssm_beta.weight"), 0, nk, nvk, 1))
		put(q + "in_proj_a.weight", lambda: _untile(T(b + "ssm_alpha.weight"), 0, nk, nvk, 1))
		put(q + "A_log", lambda: heads(torch.log(-T(b + "ssm_a"))))
		put(q + "dt_bias", lambda: heads(T(b + "ssm_dt.bias")))
		put(q + "conv1d.weight", lambda: mixed(b + "ssm_conv1d.weight").unsqueeze(1))
		put(q + "norm.weight", lambda: T(b + "ssm_norm.weight"))
		put(q + "out_proj.weight", lambda: _untile(T(b + "ssm_out.weight"), 1, nk, nvk, vd))
	return sd


def qwen35_packed(g, cfg):
	nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
	kk, vd, nvk = cfg.linear_key_head_dim * nk, cfg.linear_value_head_dim, nv // nk
	out = {}

	def add(mod, name, rows=None, cols=None):
		if g.has(name) and quantizable(g.tensors[name]):
			out[mod] = (name, rows, cols)

	def untile(n, hd):
		return _untile(torch.arange(n).unsqueeze(-1), 0, nk, nvk, hd).squeeze(-1)

	for i, kind in enumerate(cfg.layer_types):
		b, p = f"blk.{i}.", f"model.layers.{i}."
		for t, src in (("gate", "ffn_gate"), ("up", "ffn_up"), ("down", "ffn_down")):
			add(f"{p}mlp.{t}_proj", f"{b}{src}.weight")
		if kind == "full_attention":
			for t, src in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output")):
				add(f"{p}self_attn.{t}_proj", f"{b}{src}.weight")
			continue
		q = p + "linear_attn."
		if g.has(b + "attn_qkv.weight"):
			add(q + "in_proj_qkv", b + "attn_qkv.weight", torch.cat([torch.arange(2 * kk), 2 * kk + untile(g.tensors[b + "attn_qkv.weight"].rows - 2 * kk, vd)]))
		if g.has(b + "attn_gate.weight"):
			add(q + "in_proj_z", b + "attn_gate.weight", untile(g.tensors[b + "attn_gate.weight"].rows, vd))
		add(q + "in_proj_b", b + "ssm_beta.weight", untile(nv, 1))
		add(q + "in_proj_a", b + "ssm_alpha.weight", untile(nv, 1))
		if g.has(b + "ssm_out.weight"):
			cols = _untile(torch.arange(g.tensors[b + "ssm_out.weight"].n_per_row).unsqueeze(0), 1, nk, nvk, vd).squeeze(0)
			add(q + "out_proj", b + "ssm_out.weight", None, torch.argsort(cols))
	return out


def _swap(model, mname, new):
	parent, _dot, leaf = mname.rpartition(".")
	setattr(model.get_submodule(parent) if parent else model, leaf, new)


def _quantize(model, g, train, device):
	if not (train or {}).get("quant"):
		return model
	count, saved, moved = quantize(model, g, device)
	before = getattr(model, "quantized", (0, 0, 0))
	model.quantized = (before[0] + count, before[1] + saved, before[2] + moved)
	return model


def _qwen35(g, max_memory, dtype, device, train=None):
	import transformers
	cfg = qwen35_config(g)
	with torch.device("meta"):
		model = transformers.Qwen3_5ForCausalLM(cfg)
	model.to(dtype)
	packed = qwen35_packed(g, cfg) if (train or {}).get("quant") else {}
	saved = 0
	for mname, (name, rows, cols) in packed.items():
		mod = model.get_submodule(mname)
		t = g.tensors[name]
		_swap(model, mname, QuantLinear(raw_tensor(g, name), t.qname, t.type_size, mod.in_features, None, rows, cols))
		saved += mod.in_features * mod.out_features * torch.finfo(dtype).bits // 8 - t.nbytes
	model.quantized = (len(packed), saved, sum(1 for v in packed.values() if v[1] is not None or v[2] is not None))
	missing, unexpected = model.load_state_dict(qwen35_state(g, cfg, dtype, device, {m + ".weight" for m in packed}), strict=False, assign=True)
	prefixes = tuple(m + "." for m in packed)
	if unexpected or [k for k in missing if k != "lm_head.weight" and not k.startswith(prefixes)]:
		raise RuntimeError(f"qwen35 weights do not match: missing {missing[:5]} unexpected {unexpected[:5]}")
	model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
	model.tie_weights()
	left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
	if left:
		raise RuntimeError(f"qwen35 weights left unloaded: {left[:5]}")
	return _place(_quantize(model, g, train, device), max_memory, train)


def _packed(g, name, t):
	return quantizable(t) and not name.startswith(DENSE_NAMES) and not any(k in name for k in KEEP_DENSE.get(g.arch, ()))


def sizes(g, n_layer, nbytes=2, quant=False):
	per, other, lookup = {}, 0, 0
	for name, t in g.tensors.items():
		n = t.nbytes if quant and _packed(g, name, t) else math.prod(int(d) for d in t.dims) * nbytes
		m = BLOCK.match(name)
		if m:
			i = int(m.group(1))
			if i < n_layer:
				per[i] = per.get(i, 0) + n
		elif name.startswith(LOOKUPS) and n > 2**30:
			lookup += n
		else:
			other += n
	return per, other, lookup


def activations(d, ff, vocab, n_train, cfg, checkpointing):
	micro = max(1, int(cfg.get("micro", 3)))
	ans = int(cfg.get("max_tokens", 60))
	tokens = micro * (96 + ans)
	per_tok = (8 * d + 4 * ff) * 2
	layers = (n_train * d * 2 + per_tok) * tokens if checkpointing else n_train * per_tok * tokens
	return layers, (max(micro * ans, 160) * 12 + 160) * vocab * 4 + WORKSPACE


def need(g, n_layer, d, first, cfg):
	per, other, lookup = sizes(g, n_layer)
	qper, qother, _l = sizes(g, n_layer, quant=True)
	ff = g.arch_get("feed_forward_length")
	ff = int(max(ff)) if isinstance(ff, (list, tuple)) else int(ff or 4 * d)
	vocab = int(g.tensors["token_embd.weight"].dims[1])
	n_train = max(1, n_layer - first)
	return {"per": [per.get(i, 0) for i in range(n_layer)], "other": other, "lookup": lookup, "weights": sum(per.values()) + other, "full": activations(d, ff, vocab, n_train, cfg, False), "light": activations(d, ff, vocab, n_train, cfg, True), "first": first, "qper": [qper.get(i, 0) for i in range(n_layer)], "qother": qother, "qweights": sum(qper.values()) + qother, "deq": QCHUNK * 2 * 2 + 4 * QCHUNK * 4, "mode": cfg.get("weights") if cfg.get("weights") in MODES else "auto"}


def _fits(n, room, head, k, quant=False):
	left = dict(room)
	left[head] -= n["qother" if quant else "other"]
	order = sorted(left)
	g = 0
	for s in n["qper" if quant else "per"][k:]:
		while g < len(order) and left[order[g]] < s:
			g += 1
		if g == len(order):
			return False
		left[order[g]] -= s
	return True


def rungs(first, mode="auto"):
	out = []
	for k in range(0, first + 1):
		for light in ((False, True) if k == 0 else (True,)):
			out.extend((k, light, q) for q in MODES.get(mode, MODES["auto"]))
	return out


def cost(n, k, light, quant):
	ram = sum(n["qper" if quant else "per"][:k])
	return n["weights"] * STEP * SLOWER[(light, quant)] + ram / PCIE


def train_plan(n, free, mode=None):
	free = {i: v for i, v in free.items() if v > 0}
	if not free:
		return None
	total = sum(free.values())
	head = max(free, key=free.get)
	best = None
	for k, light, quant in rungs(n["first"], mode or n.get("mode", "auto")):
		c = cost(n, k, light, quant)
		if best is not None and c >= best["cost"]:
			continue
		layers, top = n["light"] if light else n["full"]
		room = {i: int(v - layers * v / total - SPARE - (top if i == head else 0) - (n["deq"] if quant else 0)) for i, v in free.items()}
		if room[head] > 0 and _fits(n, room, head, k, quant):
			per = n["qper" if quant else "per"]
			best = {"cpu_layers": k, "checkpointing": light, "quant": quant, "act": layers + top, "ram": sum(per[:k]), "weights": n["qweights" if quant else "weights"], "first": n["first"], "room": room, "head": head, "cost": c, "slower": c / cost(n, 0, False, False)}
	return best


def least(n, parts, mode=None):
	quant = (mode or n.get("mode", "auto")) != "bf16"
	per = n["qper" if quant else "per"]
	return n["qweights" if quant else "weights"] - sum(per[:n["first"]]) + sum(n["light"]) + (SPARE + (n["deq"] if quant else 0)) * max(1, parts), quant


class CpuLookup(torch.nn.Module):
	def __init__(self, emb):
		super().__init__()
		emb.weight.requires_grad_(False)
		self.inner = [emb.to("cpu")]

	@property
	def weight(self):
		return self.inner[0].weight

	def forward(self, ids):
		return self.inner[0](ids.to("cpu")).to(ids.device)


def _cpu_lookups(model, limit=2**30):
	moved = []
	for name, m in list(model.named_modules()):
		if isinstance(m, torch.nn.Embedding) and name.endswith("_per_layer") and m.weight.numel() * m.weight.element_size() > limit:
			parent, _dot, leaf = name.rpartition(".")
			setattr(model.get_submodule(parent) if parent else model, leaf, CpuLookup(m))
			moved.append(name)
	return moved


def _decoder(model):
	m = model.model
	return getattr(m, "language_model", None) or m


def _units(model):
	net = _decoder(model)
	units = []

	def walk(mod, prefix):
		for name, sub in mod.named_children():
			full = prefix + name
			if sub is net:
				units.extend(f"{full}.{c}" for c, _s in net.named_children() if c != "layers")
			elif any(m is net for m in sub.modules()):
				walk(sub, full + ".")
			else:
				units.append(full)

	walk(model, "")
	path = next(n for n, m in model.named_modules() if m is net)
	return units, f"{path}.layers", len(net.layers)


def _size(mod, seen):
	n = 0
	for t in list(mod.parameters()) + list(mod.buffers()):
		if id(t) not in seen:
			seen.add(id(t))
			n += t.numel() * t.element_size()
	return n


def _train_map(model, room, cpu_layers, first, head=None):
	units, prefix, n = _units(model)
	gpus = sorted(room)
	roomy = head if head in room else max(gpus, key=lambda k: room[k])
	for k in range(cpu_layers, max(cpu_layers, first) + 1):
		seen = set()
		left = dict(room)
		dmap = {}
		for u in units:
			dmap[u] = roomy
			left[roomy] -= _size(model.get_submodule(u), seen)
		g = 0
		for i in range(n):
			name = f"{prefix}.{i}"
			if i < k:
				dmap[name] = "cpu"
				continue
			s = _size(model.get_submodule(name), seen)
			while g < len(gpus) and left[gpus[g]] < s:
				g += 1
			if g == len(gpus):
				break
			dmap[name] = gpus[g]
			left[gpus[g]] -= s
		else:
			return dmap, k
	raise RuntimeError(f"the model does not fit on the GPUs even with its first {max(cpu_layers, first)} layers in RAM")


def _checkpointing(model, first):
	try:
		model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
	except Exception:
		return False
	for i, layer in enumerate(_decoder(model).layers):
		layer.train(i >= first)
	return True


def _place(model, max_memory, train=None):
	import accelerate
	model.eval()
	for prm in model.parameters():
		prm.requires_grad_(False)
	if not torch.cuda.is_available():
		return model
	model.cpu_lookups = _cpu_lookups(model)
	if train is not None:
		room = {i: int(v) for i, v in (train.get("room") or max_memory or {}).items() if isinstance(i, int)}
		dmap, k = _train_map(model, room, int(train.get("cpu_layers") or 0), int(train.get("first") or 0), train.get("head"))
		model = accelerate.dispatch_model(model, device_map=dmap)
		model.cpu_layers = k
		model.checkpointing = bool(train.get("checkpointing")) and _checkpointing(model, int(train.get("first") or 0))
		return model
	budget = _budget(model, max_memory)
	budget["cpu"] = int(accelerate.utils.get_max_memory().get("cpu", 0) * 0.8)
	dmap = accelerate.infer_auto_device_map(model, no_split_module_classes=list(getattr(model, "_no_split_modules", None) or []), max_memory=budget)
	gpus = [k for k in budget if isinstance(k, int)]
	if len(gpus) > 1:
		roomy = max(gpus, key=lambda k: budget[k])
		for name in ("model.embed_tokens", "lm_head", "model.norm", "model.rotary_emb"):
			if name in dmap:
				dmap[name] = roomy
	return accelerate.dispatch_model(model, device_map=dmap)


def _budget(model, max_memory):
	free = {i: int(v) for i, v in (max_memory or {}).items() if i != "cpu"} or {i: torch.cuda.mem_get_info(i)[0] - 2 * 2**30 for i in range(torch.cuda.device_count())}
	total = sum(max(0, v) for v in free.values())
	if total <= 0:
		return dict(max_memory or {})
	size = sum(p.numel() * p.element_size() for p in model.parameters())
	big = max((sum(p.numel() * p.element_size() for p in m.parameters(recurse=False)) for m in model.modules()), default=0)
	want = size * 1.25
	return {i: int(min(v, want * v / total + big * 1.1)) for i, v in free.items() if v > 0}


def gemma4_config(g):
	from transformers import Gemma4TextConfig
	a = g.arch_get
	n = int(a("block_count"))
	pattern = list(a("attention.sliding_window_pattern", []) or [])
	types = ["sliding_attention" if (bool(pattern[i]) if i < len(pattern) else (i + 1) % 6 != 0) else "full_attention" for i in range(n)]
	kv = a("attention.head_count_kv")
	kv = [int(k) for k in kv] if isinstance(kv, (list, tuple)) else [int(kv)] * n
	local_kv = [k for k, t in zip(kv, types) if t == "sliding_attention"] or kv
	full_kv = [k for k, t in zip(kv, types) if t == "full_attention"]
	ff = a("feed_forward_length")
	ple = int(a("embedding_length_per_layer_input", 0) or 0)
	full = [i for i, t in enumerate(types) if t == "full_attention"]
	cap = a("final_logit_softcapping")
	return Gemma4TextConfig(
		vocab_size=int(g.tensors["token_embd.weight"].dims[1]),
		hidden_size=int(a("embedding_length")),
		intermediate_size=int(min(ff)) if isinstance(ff, (list, tuple)) else int(ff),
		use_double_wide_mlp=isinstance(ff, (list, tuple)) and max(ff) > min(ff),
		num_hidden_layers=n,
		num_attention_heads=_scalar(a("attention.head_count")),
		num_key_value_heads=max(local_kv),
		num_global_key_value_heads=max(full_kv) if full_kv and max(full_kv) != max(local_kv) else None,
		head_dim=int(a("attention.key_length_swa", 256)),
		global_head_dim=int(a("attention.key_length", 512)),
		rms_norm_eps=float(a("attention.layer_norm_rms_epsilon", 1e-6)),
		max_position_embeddings=int(a("context_length", 131072)),
		sliding_window=int(a("attention.sliding_window", 512)),
		layer_types=types,
		final_logit_softcapping=float(cap) if cap else None,
		hidden_size_per_layer_input=ple,
		vocab_size_per_layer_input=int(g.tensors["per_layer_token_embd.weight"].dims[1]) if ple and g.has("per_layer_token_embd.weight") else int(g.tensors["token_embd.weight"].dims[1]),
		num_kv_shared_layers=int(a("attention.shared_kv_layers", 0) or 0),
		attention_k_eq_v=bool(full) and not any(g.has(f"blk.{i}.attn_v.weight") for i in full),
		tie_word_embeddings=not g.has("output.weight"),
		rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": float(a("rope.freq_base_swa", 10000.0))}, "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": float(a("rope.freq_base", 1000000.0))}},
	)


def gemma4_fix(model, g):
	layers = model.model.layers
	with torch.no_grad():
		for i, layer in enumerate(layers):
			name = f"blk.{i}.layer_output_scale.weight"
			if g.has(name) and isinstance(getattr(layer, "layer_scalar", None), torch.Tensor):
				layer.layer_scalar.copy_(g.dequant(name).float().reshape(layer.layer_scalar.shape).to(layer.layer_scalar.dtype))


CONFIGS = {"gemma4": gemma4_config}
FIXES = {"gemma4": gemma4_fix}


def work_device():
	if not torch.cuda.is_available():
		return "cpu"
	return f"cuda:{max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])}"


def _reader_class(gguf):
	class FastReader(gguf.GGUFReader):
		def __init__(self, path, mode="r"):
			self._head = np.fromfile(path, dtype=np.uint8, count=HEAD)
			super().__init__(path, mode)

		def _get(self, offset, dtype, count=1, override_order=None):
			count = int(count)
			dt = np.dtype(dtype)
			end = offset + dt.itemsize * count
			src = self._head if end <= self._head.shape[0] else self.data
			arr = src[offset:end].view(dt)[:count]
			return arr.view(arr.dtype.newbyteorder(self.byte_order if override_order is None else override_order))

		def _build_tensors(self, start_offs, fields):
			self._head = self._head[:0]
			return super()._build_tensors(start_offs, fields)

	return FastReader


@contextlib.contextmanager
def fast_gguf(device, llama_dir=""):
	gguf = gguf_module(llama_dir)
	slow, reader, names = gguf.dequantize, gguf.GGUFReader, gguf.get_tensor_name_map
	Fast = _reader_class(gguf)
	readers = {}

	def dequantize(data, qtype):
		name = getattr(qtype, "name", "")
		if name not in KERNELS or name in ("F32", "F16") or data.dtype != np.uint8:
			return slow(data, qtype)
		block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
		n = data.shape[-1] // type_size * block_size
		out = dequant_rows(data.reshape(-1, data.shape[-1]), name, block_size, type_size, n, chunk_rows=max(1, CHUNK // n), work=device)
		return out.numpy().reshape(*data.shape[:-1], n)

	def open_reader(path, mode="r"):
		if mode != "r":
			return reader(path, mode)
		key = os.path.normcase(os.path.abspath(path))
		if key not in readers:
			readers[key] = Fast(path)
		return readers[key]

	gguf.dequantize, gguf.GGUFReader, gguf.get_tensor_name_map = dequantize, open_reader, functools.lru_cache(maxsize=None)(names)
	try:
		yield
	finally:
		gguf.dequantize, gguf.GGUFReader, gguf.get_tensor_name_map = slow, reader, names
		readers.clear()
		if str(device).startswith("cuda"):
			torch.cuda.empty_cache()


def load(path, max_memory=None, dtype=torch.bfloat16, llama_dir="", device=None, train=None):
	device = device or work_device()
	g = GGUFFile(path, llama_dir)
	cfg = None
	try:
		if g.arch == "qwen35":
			return _qwen35(g, max_memory, dtype, device, train)
		if g.arch in CONFIGS:
			cfg = CONFIGS[g.arch](g)
	finally:
		g.close()
	import transformers
	kw = {"dtype": dtype, "gguf_file": os.path.basename(path)}
	if cfg is not None:
		kw["config"] = cfg
	with fast_gguf(device, llama_dir):
		model = transformers.AutoModelForCausalLM.from_pretrained(os.path.dirname(path), **kw)
	fix = FIXES.get(getattr(cfg, "model_type", "").replace("_text", ""))
	if fix is not None or (train or {}).get("quant"):
		g = GGUFFile(path, llama_dir)
		try:
			if fix is not None:
				fix(model, g)
			_quantize(model, g, train, device)
		finally:
			g.close()
	return _place(model, max_memory, train)
