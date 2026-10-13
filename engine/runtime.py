import ctypes
import json
import os
import time

import numpy as np

import config
from engine.chat import ChatTemplate
from engine.ggufio import GGUFFile
from engine.llama import EVAL_CB, PROGRESS_CB, SPLIT_LAYER, SPLIT_NONE, SPLIT_ROW, BuftOverride, Lib, c_float, c_void_p

LOG_ERROR = 4


class LoadCancelled(Exception):
	pass


def _utf8_split(buf):
	for cut in range(len(buf), max(len(buf) - 4, -1), -1):
		try:
			return buf[:cut].decode("utf-8"), buf[cut:]
		except UnicodeDecodeError:
			continue
	return buf.decode("utf-8", "replace"), b""


def sample(logits, temperature=0.7, top_k=40, top_p=0.95, repeat_penalty=1.0, recent=(), rng=None):
	x = logits.astype(np.float32, copy=True)
	if repeat_penalty != 1.0 and len(recent):
		ids = np.unique(np.asarray(recent[-64:], dtype=np.int64))
		vals = x[ids]
		x[ids] = np.where(vals > 0, vals / repeat_penalty, vals * repeat_penalty)
	if temperature <= 0:
		return int(np.argmax(x))
	k = int(top_k) if 0 < int(top_k) < x.shape[0] else x.shape[0]
	idx = np.argpartition(-x, k - 1)[:k] if k < x.shape[0] else np.arange(x.shape[0])
	vals = x[idx] / temperature
	vals -= vals.max()
	p = np.exp(vals)
	order = np.argsort(-p)
	idx, p = idx[order], p[order] / p.sum()
	if 0 < top_p < 1:
		cut = int(np.searchsorted(np.cumsum(p), top_p)) + 1
		idx, p = idx[:cut], p[:cut] / p[:cut].sum()
	rng = rng or np.random.default_rng()
	return int(rng.choice(idx, p=p))


KV_TYPES = {"f16": 1, "q8_0": 8, "q4_0": 2}
LORA = "lora"
FLASH_TYPES = {"auto": -1, "off": 0, "on": 1}


class Engine:
	def __init__(self, llama_dir, lib=None):
		self.lib = lib or Lib(llama_dir)
		self.model = None
		self.ctx = None
		self.vocab = None
		self.gguf = None
		self.info = None
		self.template = None
		self.n_past = 0
		self.n_batch = 512
		self._batch = None
		self._keep = []
		self._want = frozenset()
		self._node = b"l_out-"
		self._width = 0
		self._captured = {}
		self._eval_cb = EVAL_CB(self._on_eval)
		self._pieces = {}
		self._pieces_raw = {}
		self._vocab_strings = None
		self.steering_layers = []
		self.loras = {}
		self.lora_scales = []

	@property
	def loaded(self):
		return self.ctx is not None

	def _last_error(self):
		errs = [t.strip() for lvl, t in self.lib.logs[-60:] if lvl == LOG_ERROR and t.strip()]
		return errs[-1] if errs else "see llama.cpp log"

	def _overrides(self, overrides):
		if not overrides:
			return None
		bufts = {}
		for i in range(self.lib.ggml_backend_dev_count()):
			buft = self.lib.ggml_backend_dev_buffer_type(self.lib.ggml_backend_dev_get(i))
			if buft:
				bufts[(self.lib.ggml_backend_buft_name(buft) or b"").decode().upper()] = buft
		arr = (BuftOverride * (len(overrides) + 1))()
		patterns = []
		for i, (pattern, dev) in enumerate(overrides):
			buft = bufts.get(dev.upper())
			if buft is None:
				raise RuntimeError(f"unknown device in tensor override: {dev}")
			patterns.append(pattern.encode())
			arr[i].pattern = patterns[-1]
			arr[i].buft = buft
		self._keep += [arr, patterns]
		return ctypes.cast(arr, c_void_p)

	def load(self, path, plan=None, n_ctx=8192, on_progress=None, cancel=None):
		self.unload()
		plan = plan or {}
		started = time.perf_counter()
		gg = GGUFFile(path, self.lib.dir)
		mp = self.lib.llama_model_default_params()
		mp.n_gpu_layers = int(plan.get("n_gpu_layers", -1) if plan.get("n_gpu_layers") is not None else -1)
		mp.split_mode = {"none": SPLIT_NONE, "row": SPLIT_ROW}.get(plan.get("split_mode"), SPLIT_LAYER)
		mp.main_gpu = int(plan.get("main_gpu") or 0)
		split = plan.get("tensor_split")
		if split:
			arr = (c_float * int(self.lib.llama_max_devices()))(*([float(v) for v in split] + [0.0] * (int(self.lib.llama_max_devices()) - len(split))))
			self._keep.append(arr)
			mp.tensor_split = arr
		ov = self._overrides(plan.get("overrides"))
		if ov is not None:
			mp.tensor_buft_overrides = ov

		def progress(p, _ud):
			try:
				if on_progress:
					on_progress(float(p))
			except Exception:
				pass
			return not (cancel is not None and cancel.is_set())

		pcb = PROGRESS_CB(progress)
		self._keep.append(pcb)
		mp.progress_callback = ctypes.cast(pcb, c_void_p)
		model = self.lib.llama_model_load_from_file(os.path.abspath(path).encode("utf-8"), mp)
		if not model:
			gg.close()
			self._keep = []
			if cancel is not None and cancel.is_set():
				raise LoadCancelled("load cancelled")
			raise RuntimeError(f"model load failed: {self._last_error()}")
		cp = self.lib.llama_context_default_params()
		cp.n_ctx = int(plan.get("n_ctx") or n_ctx)
		cp.n_batch = self.n_batch
		cp.n_ubatch = self.n_batch
		cp.n_seq_max = 1
		cp.n_outputs_max = 0
		cp.n_outputs_max_per_seq = 0
		cp.n_threads = max(2, (os.cpu_count() or 4) // 2)
		cp.n_threads_batch = cp.n_threads
		cp.cb_eval = ctypes.cast(self._eval_cb, c_void_p)
		cp.cb_eval_user_data = None
		cp.swa_full = False
		fa = plan.get("flash_attn")
		kv = KV_TYPES.get(plan.get("kv_type") or "f16", KV_TYPES["f16"])
		if kv != KV_TYPES["f16"] and fa in (None, "auto", "off"):
			fa = "on"
		if fa in FLASH_TYPES:
			cp.flash_attn_type = FLASH_TYPES[fa]
		cp.type_k = cp.type_v = kv
		cp.no_perf = True
		ctx = self.lib.llama_init_from_model(model, cp)
		if not ctx:
			self.lib.llama_model_free(model)
			gg.close()
			self._keep = []
			raise RuntimeError(f"context creation failed (VRAM?): {self._last_error()}")
		self.model, self.ctx, self.gguf = model, ctx, gg
		self.vocab = self.lib.llama_model_get_vocab(model)
		self._batch = self.lib.llama_batch_init(self.n_batch, 0, 1)
		self.n_embd = int(self.lib.llama_model_n_embd(model))
		self.n_layer_total = int(self.lib.llama_model_n_layer(model))
		self.n_layer = self.n_layer_total
		self.n_nextn = int(self.lib.llama_model_n_layer_nextn(model))
		self.n_vocab = int(self.lib.llama_vocab_n_tokens(self.vocab))
		self.n_ctx = int(self.lib.llama_n_ctx(ctx))
		self._pieces, self._pieces_raw, self._vocab_strings = {}, {}, None
		self.template = ChatTemplate(gg.get("tokenizer.chat_template", ""), self._special_text(gg.get("tokenizer.ggml.bos_token_id")), self._special_text(gg.get("tokenizer.ggml.eos_token_id")))
		desc = ctypes.create_string_buffer(256)
		self.lib.llama_model_desc(model, desc, 256)
		self.info = {
			"path": path,
			"file": os.path.basename(path),
			"arch": gg.arch,
			"name": gg.get("general.name", "") or os.path.basename(path),
			"basename": gg.get("general.basename", "") or "",
			"size_label": gg.get("general.size_label", "") or "",
			"desc": desc.value.decode("utf-8", "replace"),
			"n_layer": self.n_layer,
			"n_embd": self.n_embd,
			"n_vocab": self.n_vocab,
			"n_expert": int(gg.arch_get("expert_count", 0) or 0),
			"n_ctx": self.n_ctx,
			"quant": gg.main_quant(),
			"size": int(self.lib.llama_model_size(model)),
			"n_params": int(self.lib.llama_model_n_params(model)),
			"chat_template": not self.template.fallback,
			"plan": {k: plan.get(k) for k in ("n_gpu_layers", "tensor_split", "lens_device", "spill", "cpu_bytes", "notes", "estimate", "split_mode", "main_gpu", "flash_attn", "kv_type")},
			"load_seconds": round(time.perf_counter() - started, 1),
		}
		self.n_past = 0
		self.steering_layers = []
		return self.info

	def _special_text(self, tid):
		if tid is None or self.vocab is None:
			return ""
		return self.piece(int(tid), special=True)

	def unload(self):
		self._want = frozenset()
		if self._batch is not None:
			self.lib.llama_batch_free(self._batch)
			self._batch = None
		if self.ctx:
			self.lib.llama_free(self.ctx)
		if self.model:
			self.lib.llama_model_free(self.model)
		if self.gguf is not None:
			self.gguf.close()
		self.ctx = self.model = self.vocab = self.gguf = self.info = self.template = None
		self._keep = []
		self.loras = {}
		self.lora_scales = []
		self._pieces, self._pieces_raw, self._vocab_strings = {}, {}, None
		self.steering_layers = []

	def tokenize(self, text, add_special=False, parse_special=True):
		data = text.encode("utf-8")
		cap = len(data) + 16
		buf = (ctypes.c_int32 * cap)()
		n = self.lib.llama_tokenize(self.vocab, data, len(data), buf, cap, add_special, parse_special)
		if n < 0:
			cap = -n
			buf = (ctypes.c_int32 * cap)()
			n = self.lib.llama_tokenize(self.vocab, data, len(data), buf, cap, add_special, parse_special)
		return list(buf[:n])

	def piece_bytes(self, tid, special=False):
		key = (tid, special)
		raw = self._pieces_raw.get(key)
		if raw is not None:
			return raw
		buf = ctypes.create_string_buffer(64)
		n = self.lib.llama_token_to_piece(self.vocab, int(tid), buf, 64, 0, special)
		if n < 0:
			buf = ctypes.create_string_buffer(-n)
			n = self.lib.llama_token_to_piece(self.vocab, int(tid), buf, -n, 0, special)
		raw = buf.raw[:max(n, 0)]
		self._pieces_raw[key] = raw
		return raw

	def piece(self, tid, special=True):
		key = (tid, special)
		s = self._pieces.get(key)
		if s is None:
			s = self.piece_bytes(tid, special).decode("utf-8", "replace")
			self._pieces[key] = s
		return s

	def vocab_strings(self):
		if self._vocab_strings is not None:
			return self._vocab_strings
		st = os.stat(self.info["path"])
		key = f"{os.path.basename(self.info['path'])}_{st.st_size}_{int(st.st_mtime)}"
		cache = os.path.join(config.CACHE_DIR, "vocab", key + ".json")
		if os.path.exists(cache):
			try:
				with open(cache, encoding="utf-8") as f:
					strings = json.load(f)
				if len(strings) == self.n_vocab:
					self._vocab_strings = strings
					return strings
			except Exception:
				pass
		self._vocab_strings = [self.piece(i, True) for i in range(self.n_vocab)]
		os.makedirs(os.path.dirname(cache), exist_ok=True)
		with open(cache, "w", encoding="utf-8") as f:
			json.dump(self._vocab_strings, f, ensure_ascii=False)
		return self._vocab_strings

	def is_eog(self, tid):
		return bool(self.lib.llama_vocab_is_eog(self.vocab, int(tid)))

	def chat_tokens(self, messages, add_generation_prompt=True, enable_thinking=False):
		text = self.template.render(messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
		tokens = self.tokenize(text)
		bos = int(self.lib.llama_vocab_bos(self.vocab))
		if self.lib.llama_vocab_get_add_bos(self.vocab) and bos >= 0 and (not tokens or tokens[0] != bos):
			tokens = [bos] + tokens
		return tokens

	def text_tokens(self, text):
		tokens = self.tokenize(text)
		bos = int(self.lib.llama_vocab_bos(self.vocab))
		if bos >= 0 and self.lib.llama_vocab_get_add_bos(self.vocab) and (not tokens or tokens[0] != bos):
			tokens = [bos] + tokens
		return tokens

	def reset(self):
		self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx), True)
		self.n_past = 0

	def set_capture(self, layers, node="l_out", width=None):
		self._captured = {}
		self._node = node.encode("ascii") + b"-"
		self._width = int(width or 0)
		self._want = frozenset(int(l) for l in layers) if layers else frozenset()

	def pop_capture(self):
		caps = {l: (np.concatenate(v) if len(v) > 1 else v[0]) for l, v in self._captured.items() if v}
		self._captured = {}
		return caps

	def _on_eval(self, t, ask, _ud):
		try:
			if not self._want:
				return not ask
			name = self.lib.ggml_get_name(t)
			node = self._node
			if not name or not name.startswith(node):
				return not ask
			layer = int(name[len(node):].split(b" ")[0])
			if layer not in self._want:
				return not ask
			if ask:
				return True
			self._captured.setdefault(layer, []).append(self.lib.tensor_to_numpy(t, self._width or self.n_embd))
			return True
		except Exception:
			return not ask

	def decode(self, tokens, logits="last", on_logits=None):
		b = self._batch
		n = len(tokens)
		for start in range(0, n, self.n_batch):
			chunk = tokens[start:start + self.n_batch]
			outs = []
			for i, tok in enumerate(chunk):
				b.token[i] = int(tok)
				b.pos[i] = self.n_past + i
				b.n_seq_id[i] = 1
				b.seq_id[i][0] = 0
				want = logits == "all" or (logits == "last" and start + i == n - 1)
				b.logits[i] = 1 if want else 0
				if want:
					outs.append(i)
			b.n_tokens = len(chunk)
			rc = self.lib.llama_decode(self.ctx, b)
			if rc != 0:
				raise RuntimeError(f"llama_decode failed ({rc}): {self._last_error()}")
			self.n_past += len(chunk)
			if on_logits is not None:
				for i in outs:
					on_logits(start + i, self.logits_at(i))

	def logits_at(self, i=-1):
		ptr = self.lib.llama_get_logits_ith(self.ctx, int(i))
		return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))

	def last_logits(self):
		return self.logits_at(-1).copy()

	def lora(self, path):
		key = os.path.abspath(path)
		if key not in self.loras:
			a = self.lib.llama_adapter_lora_init(self.model, key.encode("utf-8"))
			if not a:
				raise RuntimeError(f"LoRA adapter rejected: {self._last_error()}")
			self.loras[key] = a
		return self.loras[key]

	def drop_lora(self, path):
		key = os.path.abspath(path)
		if key not in self.loras:
			return
		if any(p == key for p, _s in self.lora_scales):
			keep = {p: s for p, s in self.lora_scales if p != key}
			self.lora_scales = None
			self.set_loras(keep)
		del self.loras[key]

	def set_loras(self, scales):
		items = sorted((os.path.abspath(p), float(s)) for p, s in (scales or {}).items() if s)
		if items == self.lora_scales:
			return
		if items:
			ptrs = (c_void_p * len(items))(*[self.lora(p) for p, _s in items])
			vals = (c_float * len(items))(*[s for _p, s in items])
			rc = self.lib.llama_set_adapters_lora(self.ctx, ptrs, len(items), vals)
		else:
			rc = self.lib.llama_set_adapters_lora(self.ctx, None, 0, None)
		if rc != 0:
			raise RuntimeError(f"LoRA adapters rejected ({rc}): {self._last_error()}")
		self.lora_scales = items

	def set_steering(self, vectors):
		vectors = vectors or {}
		self.set_loras(vectors.get(LORA))
		layers = sorted(int(l) for l in vectors if l != LORA and 1 <= int(l) < self.n_layer_total)
		if not layers:
			self.lib.llama_set_adapter_cvec(self.ctx, None, 0, self.n_embd, -1, -1)
			self.steering_layers = []
			return
		data = np.zeros((self.n_layer_total - 1) * self.n_embd, dtype=np.float32)
		for l in layers:
			data[(l - 1) * self.n_embd:l * self.n_embd] = np.asarray(vectors[l], dtype=np.float32)
		ptr = data.ctypes.data_as(ctypes.POINTER(c_float))
		rc = self.lib.llama_set_adapter_cvec(self.ctx, ptr, data.size, self.n_embd, layers[0], layers[-1])
		if rc != 0:
			raise RuntimeError(f"control vector rejected ({rc}): {self._last_error()}")
		self.steering_layers = layers

	def generate(self, prompt_tokens, sampling=None, stop=None, on_text=None, on_step=None, capture_layers=None, seed=None):
		sampling = dict(sampling or {})
		max_tokens = int(sampling.get("max_tokens", 256))
		rng = np.random.default_rng(seed)
		self.reset()
		self.set_capture(capture_layers)
		t0 = time.perf_counter()
		self.decode(prompt_tokens, logits="last")
		prompt_caps = self.pop_capture()
		t1 = time.perf_counter()
		out, recent = [], list(prompt_tokens[-64:])
		pending, text = b"", ""
		steps = []
		for _ in range(max_tokens):
			if stop is not None and stop.is_set():
				break
			tok = sample(self.logits_at(-1), sampling.get("temperature", 0.7), sampling.get("top_k", 40), sampling.get("top_p", 0.95), sampling.get("repeat_penalty", 1.0), recent, rng)
			if self.is_eog(tok):
				break
			out.append(tok)
			recent.append(tok)
			pending += self.piece_bytes(tok)
			chunk, pending = _utf8_split(pending)
			if chunk:
				text += chunk
				if on_text:
					on_text(chunk)
			if self.n_past >= self.n_ctx - 1:
				break
			self.decode([tok], logits="last")
			caps = self.pop_capture()
			if capture_layers:
				steps.append(caps)
			if on_step:
				on_step(len(out) - 1, tok, caps)
		if pending:
			tail = pending.decode("utf-8", "replace")
			text += tail
			if on_text:
				on_text(tail)
		self.set_capture(None)
		t2 = time.perf_counter()
		return {
			"text": text,
			"tokens": out,
			"prompt_capture": prompt_caps,
			"steps": steps,
			"stats": {"prompt_tokens": len(prompt_tokens), "tokens": len(out), "prompt_tps": round(len(prompt_tokens) / max(t1 - t0, 1e-6), 1), "tps": round(len(out) / max(t2 - t1, 1e-6), 1), "seconds": round(t2 - t0, 2)},
		}
