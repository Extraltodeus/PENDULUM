import json
import os

import numpy as np
import torch

from engine.ggufio import gguf_module
from engine.runtime import LORA
from hypno.lorakeys import MODULES, STYLES, arch_ok, example_key, key_of
from persona.coherence import check

NODE = "ffn_swiglu"
LAMBDAS = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
TRAIN = (
	"Tell me about your favourite season.", "How do I fix a leaking tap?", "Write two lines about the sea.", "What should I cook tonight?",
	"Explain what a black hole is.", "My cat ignores me, why?", "Give me advice for a job interview.", "Describe a rainy street at night.",
	"What do you think about modern art?", "How was your day?", "Recommend a book for a long trip.", "Why is the sky blue?",
	"a cozy cabin in the snowy mountains at dusk, warm light", "portrait of an old fisherman, dramatic lighting", "a futuristic city street, neon signs, rain",
	"a bowl of ramen on a wooden table, close-up", "a knight standing in a field of flowers", "two friends laughing in a cafe, film photo",
	"Raconte-moi une histoire courte.", "Quel temps fera-t-il demain selon toi ?", "Donne-moi trois idées de week-end.", "Comment apprendre à dessiner ?",
	"What is the meaning of friendship?", "Summarize the plot of a heist movie.", "Write a haiku about coffee.", "How do airplanes stay in the air?",
	"a lighthouse on a cliff during a storm", "an astronaut floating above earth, cinematic", "a child reading under a tree, watercolor",
	"Complain about Mondays.", "Tell me a joke about programmers.", "What's the best way to say sorry?",
	"Who are you?", "Qui es-tu ?", "a red sports car on a coastal road, sunset", "Plan a birthday party for a ten-year-old.",
)
HELD = (
	"What would you do with a free afternoon?", "Describe your ideal home.", "a quiet library at sunset, golden hour", "a street market in Marrakech, vivid colors",
	"Explique la photosynthèse simplement.", "How should I prepare for a marathon?", "a portrait of a smiling grandmother knitting", "Give me feedback on my idea of opening a bakery.",
	"Write a short letter to a friend.", "a dark forest with fog and a lone wolf",
)
TEACH = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.05, "max_tokens": 48}


def arch_value(engine, key, layer=None):
	v = engine.gguf.arch_get(key) if engine.gguf is not None else None
	if isinstance(v, (list, tuple)):
		return v[layer] if layer is not None and layer < len(v) else (v[0] if v else None)
	return v


def support(engine):
	arch = (engine.info or {}).get("arch", "")
	ok, why = arch_ok(arch, arch_value(engine, "expert_count"))
	if ok and not arch_value(engine, "feed_forward_length"):
		return False, f"'{arch}': MLP width unknown"
	return ok, why


def sequences(engine, prompts, vectors, stop):
	engine.set_steering(vectors or None)
	out = []
	for i, p in enumerate(prompts):
		check(stop)
		toks = engine.chat_tokens([{"role": "user", "content": p}])
		r = engine.generate(toks, TEACH, seed=50 + i)
		out.append(list(toks) + list(r["tokens"]))
	return out


def capture(engine, seqs, vectors, node, layers, width, stop):
	engine.set_steering(vectors or None)
	acc = {l: [] for l in layers}
	try:
		for toks in seqs:
			check(stop)
			engine.reset()
			engine.set_capture(layers, node=node, width=width)
			engine.decode(toks, logits="last")
			caps = engine.pop_capture()
			for l in layers:
				a = caps.get(l)
				if a is not None:
					acc[l].append(a[-len(toks):])
	finally:
		engine.set_capture(None)
	return {l: np.concatenate(v) for l, v in acc.items() if v}


def ridge(A, H, device):
	A = torch.as_tensor(A, device=device, dtype=torch.float32)
	H = torch.as_tensor(H, device=device, dtype=torch.float32)
	K = A @ A.T
	one = torch.ones(A.shape[0], device=device)
	scale = float(torch.trace(K)) / A.shape[1]
	best = None
	for k in LAMBDAS:
		alpha = torch.linalg.solve(K + k * scale * torch.eye(K.shape[0], device=device), one)
		w = A.T @ alpha
		err = float(((H @ w - 1.0) ** 2).mean().sqrt())
		if best is None or err < best[1]:
			best = (k, err, w)
	k, err, w = best
	pred = H @ w
	return w.cpu().numpy(), {"lambda": k, "rms_err": err, "mean": float(pred.mean()), "p10": float(torch.quantile(pred, 0.1)), "p90": float(torch.quantile(pred, 0.9))}


def adapter_parts(path, scale, llama_dir=""):
	gguf = gguf_module(llama_dir)
	reader = gguf.GGUFReader(path)
	field = reader.fields.get(gguf.Keys.Adapter.LORA_ALPHA)
	alpha = float(field.contents()) if field is not None else 0.0
	tensors = {t.name: np.asarray(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)]) for t in reader.tensors}
	parts = {}
	for name, a in tensors.items():
		if not name.endswith(".lora_a"):
			continue
		base = name[:-len(".lora_a")]
		b = tensors.get(base + ".lora_b")
		if b is None:
			continue
		bits = base.split(".")
		if len(bits) < 3 or bits[0] != "blk":
			continue
		rank = a.shape[0]
		parts[(int(bits[1]), bits[2])] = (a, b * float(scale) * (alpha / rank if alpha else 1.0))
	return parts


def combine(modules, extra):
	for key, (a, b) in extra.items():
		if key in modules:
			a0, b0 = modules[key]
			modules[key] = (np.concatenate([a0, a], 0), np.concatenate([b0, b], 1))
		else:
			modules[key] = (a, b)
	return modules


def padded(modules):
	rank = max(a.shape[0] for a, _b in modules.values())
	out = {}
	for key, (a, b) in modules.items():
		r = a.shape[0]
		if r < rank:
			a = np.concatenate([a, np.zeros((rank - r, a.shape[1]), np.float32)], 0)
			b = np.concatenate([b, np.zeros((b.shape[0], rank - r), np.float32)], 1)
		out[key] = (np.ascontiguousarray(a, np.float32), np.ascontiguousarray(b, np.float32))
	return out, rank


def fidelity(engine, held, vectors, lora_path, stop):
	n = engine.n_layer
	deep = [n - 2, n - 1]
	hb = capture(engine, held, None, "l_out", deep, None, stop)
	hv = capture(engine, held, vectors, "l_out", deep, None, stop)
	hl = capture(engine, held, {LORA: {lora_path: 1.0}}, "l_out", deep, None, stop)
	out = {}
	for l in deep:
		if l not in hb or l not in hv or l not in hl:
			continue
		dv, dl = hv[l] - hb[l], hl[l] - hb[l]
		nv = np.maximum(np.linalg.norm(dv, axis=1), 1e-6)
		out[f"L{l}"] = {"reproduced": float(np.median(1 - np.linalg.norm(hl[l] - hv[l], axis=1) / nv)), "cos": float(np.median((dv * dl).sum(1) / np.maximum(nv * np.linalg.norm(dl, axis=1), 1e-6))), "length": float(np.median(np.linalg.norm(dl, axis=1) / nv))}
	kls = {"bare": [], "lora": []}
	for toks in held:
		check(stop)
		rows = {}
		for tag, steer in (("lever", vectors), ("bare", None), ("lora", {LORA: {lora_path: 1.0}})):
			engine.set_steering(steer or None)
			engine.reset()
			cur = []
			engine.decode(toks, logits="all", on_logits=lambda i, lg, c=cur: c.append(lg.copy()))
			rows[tag] = np.stack(cur)
		lp = torch.log_softmax(torch.as_tensor(rows["lever"]).float(), -1)
		for tag in ("bare", "lora"):
			lq = torch.log_softmax(torch.as_tensor(rows[tag]).float(), -1)
			kls[tag].append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
	kb, kl = float(np.mean(kls["bare"])), float(np.mean(kls["lora"]))
	out["kl_lever_vs_bare"], out["kl_lever_vs_lora"] = kb, kl
	out["gap_closed"] = 1 - kl / kb if kb > 1e-9 else 1.0
	return out


def write_safetensors(path, modules, style, prefix, rank, meta):
	from safetensors.numpy import save_file

	tensors = {}
	for (layer, gname), (a, b) in sorted(modules.items()):
		module = MODULES.get(gname, gname)
		tensors[key_of(style, prefix, layer, module, "A")] = a
		tensors[key_of(style, prefix, layer, module, "B")] = b
		if style == "kohya":
			tensors[key_of(style, prefix, layer, module, "A").replace(".lora_down.weight", ".alpha")] = np.asarray(float(rank), dtype=np.float32)
	save_file(tensors, path, metadata={k: (v if isinstance(v, str) else json.dumps(v)) for k, v in meta.items()})
	return path


def peft_config(modules, rank, model_name):
	targets = sorted({MODULES.get(g, g).split(".")[-1] for _l, g in modules})
	return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": model_name, "r": rank, "lora_alpha": rank, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "use_rslora": False, "target_modules": targets, "layers_to_transform": sorted({l for l, _g in modules}), "inference_mode": True}


def build(session, dose, on_progress=None):
	engine = session.engine
	stop = session.stop_event
	active = session.active_vectors()
	vectors = {int(l): np.asarray(v, dtype=np.float32) * float(dose) for l, v in active.items() if l != LORA}
	goals = dict(active.get(LORA) or {})
	ok, why = support(engine)
	if vectors and not ok:
		raise ValueError(why)
	modules, report, held = {}, {"layers": {}}, None
	steer = dict(vectors)
	if goals:
		steer[LORA] = {p: float(s) * float(dose) for p, s in goals.items()}
	try:
		if vectors:
			layers = sorted(vectors)
			step = lambda v: on_progress(v) if on_progress else None
			step(0.02)
			train = sequences(engine, TRAIN, steer, stop)
			step(0.25)
			held = sequences(engine, HELD, steer, stop)
			step(0.33)
			device = getattr(getattr(session, "readout", None), "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
			for k, l in enumerate(layers):
				width = int(arch_value(engine, "feed_forward_length", l))
				acts = capture(engine, train, steer, NODE, [l], width, stop)
				test = capture(engine, held, steer, NODE, [l], width, stop)
				if l not in acts or l not in test or acts[l].shape[1] != width:
					raise RuntimeError(f"no MLP activations captured at layer {l} (graph node '{NODE}-{l}'): this architecture is not supported yet")
				w, stats = ridge(acts[l], test[l], device)
				modules[(l, "ffn_down")] = (w[None, :].astype(np.float32), vectors[l][:, None].astype(np.float32))
				report["layers"][str(l)] = stats
				step(0.33 + 0.47 * (k + 1) / len(layers))
			report["prompts"] = {"train": len(train), "held_out": len(held), "tokens": int(sum(map(len, train)))}
		for path, scale in sorted(goals.items()):
			combine(modules, adapter_parts(path, float(scale) * float(dose), engine.lib.dir))
	finally:
		session.resteer()
	if not modules:
		raise ValueError("no active lever: move at least one slider before exporting")
	modules, rank = padded(modules)
	return modules, rank, steer, report, held


def measure(session, lora_path, steer, held):
	try:
		return fidelity(session.engine, held, steer, lora_path, session.stop_event)
	finally:
		session.resteer()
