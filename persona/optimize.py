import math

import numpy as np
import torch

from persona import levers as LV

DEFAULTS = {"steps": 100, "lr": 0.01, "kl_target": 0.1, "lora_lr": 0.002, "batch": 7, "micro": 3, "lam": 1.0, "samples": 2, "draws": 3, "topk": 32, "max_tokens": 60, "layers": "", "width": 9, "carrier": "lora", "rank": 8, "targets": "down", "weights": "auto"}
LAYER_MODES = ("early", "band", "middle")
LAYER_DEFAULT = {"vector": "early", "lora": "band", "both": "band"}
CARRIERS = ("vector", "lora", "both")
TARGETS = {"down": ("down",), "mlp": ("gate", "up", "down")}
TEACH = {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.05}


def msgs(prompt, system=""):
	return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def window(band, n_layer, mode="early", width=9):
	lo, hi = band
	first, last = max(1, lo - 2), min(n_layer - 1, hi)
	if mode == "band":
		return list(range(first, last + 1))
	if mode == "middle":
		mid = (first + last) // 2
		start = max(first, mid - width // 2)
		return list(range(start, min(last, start + width - 1) + 1))
	return list(range(first, min(last, first + width - 1) + 1))


def targets(engine, prompt, ans, system, topk):
	toks = engine.chat_tokens(msgs(prompt, system))
	full = toks + ans
	s0 = len(toks) - 1
	ids, probs, own = [], [], []

	def cb(i, lg):
		if not s0 <= i < len(full) - 1:
			return
		x = torch.from_numpy(lg).float()
		v, k = x.topk(topk)
		ids.append(k)
		probs.append(torch.softmax(v, 0))
		own.append(float(x[full[i + 1]] - torch.logsumexp(x, 0)))

	engine.reset()
	engine.decode(full, logits="all", on_logits=cb)
	return torch.stack(ids), torch.stack(probs), float(np.mean(own))


def teacher(engine, items, system, cfg, stop=None, on_progress=None, pick=None):
	pairs = []
	items = [it if isinstance(it, dict) else {"prompt": it} for it in items]
	draws = [max(cfg["samples"], int(cfg.get("draws") or 0)) if pick is not None and it.get("key") else cfg["samples"] for it in items]
	total = max(1, sum(draws))
	n = 0
	for i, it in enumerate(items):
		got = []
		for k in range(draws[i]):
			if stop is not None and stop.is_set():
				return None
			r = engine.generate(engine.chat_tokens(msgs(it["prompt"], system)), dict(TEACH, max_tokens=cfg["max_tokens"]), seed=100 + 7 * i + k)
			n += 1
			if on_progress:
				on_progress(n / total)
			if r["tokens"]:
				got.append((pick(it, r["text"]) if pick is not None and it.get("key") else 0.0, k, r))
		got.sort(key=lambda g: (-g[0], g[1]))
		for score, _k, r in got[:cfg["samples"]]:
			ti, tp, own = targets(engine, it["prompt"], list(r["tokens"]), system, cfg["topk"])
			pairs.append({"prompt": it["prompt"], "key": it.get("key"), "ans": list(r["tokens"]), "text": r["text"], "ti": ti, "tp": tp, "own": own, "score": score})
	return pairs


def logprob(engine, prompt, ans, system="", vectors=None):
	toks = engine.chat_tokens(msgs(prompt, system))
	full = toks + ans
	s0 = len(toks) - 1
	lp = []

	def cb(i, lg):
		if s0 <= i < len(full) - 1:
			m = float(lg.max())
			lp.append(float(lg[full[i + 1]]) - m - float(np.log(np.exp(lg - m).sum())))

	engine.set_steering(vectors or None)
	engine.reset()
	try:
		engine.decode(full, logits="all", on_logits=cb)
	finally:
		engine.set_steering(None)
	return float(np.mean(lp)) if lp else 0.0


def closed(engine, pairs, vectors, base=None):
	if not pairs:
		return None
	base = base if base is not None else float(np.mean([logprob(engine, p["prompt"], p["ans"]) for p in pairs]))
	twin = float(np.mean([p["own"] for p in pairs]))
	now = float(np.mean([logprob(engine, p["prompt"], p["ans"], vectors=vectors) for p in pairs]))
	return (now - base) / (twin - base) if abs(twin - base) > 1e-6 else None


def decoder(model):
	m = model.model
	return getattr(m, "language_model", None) or m


class Steer:
	def __init__(self, model, layers, lr=0.025):
		self.blocks = decoder(model).layers
		self.lr = lr
		size = model.config.get_text_config().hidden_size
		self.vec = {l: torch.zeros(size, device=next(self.blocks[l].parameters()).device, dtype=torch.float32, requires_grad=True) for l in layers}
		self.on = False
		self.handles = [self.blocks[l].register_forward_hook(self._hook(l)) for l in layers]

	def _hook(self, l):
		def fn(mod, inp, out):
			if not self.on:
				return None
			h = out if torch.is_tensor(out) else out[0]
			h = h + self.vec[l].to(h.dtype)
			return h if torch.is_tensor(out) else (h,) + tuple(out[1:])
		return fn

	def groups(self):
		return [{"params": list(self.vec.values()), "lr": self.lr}]

	def result(self):
		return {"vector": {l: v.detach().float().cpu().numpy() for l, v in self.vec.items()}}

	def close(self):
		for h in self.handles:
			h.remove()


class Lora:
	def __init__(self, model, layers, rank=8, targets=("down",), lr=0.002, seed=0):
		self.rank = int(rank)
		self.lr = lr
		self.on = False
		self.mods = {}
		gen = torch.Generator().manual_seed(seed)
		for l in layers:
			mlp = decoder(model).layers[l].mlp
			for t in targets:
				lin = getattr(mlp, f"{t}_proj")
				out_f, in_f = lin.out_features, lin.in_features
				dev = next(t for t in (*lin.parameters(), *lin.buffers())).device
				a = (torch.randn(self.rank, in_f, generator=gen) / math.sqrt(in_f)).to(dev).requires_grad_(True)
				b = torch.zeros(out_f, self.rank, device=dev, requires_grad=True)
				self.mods[(l, t)] = (lin, a, b)
		self.handles = [lin.register_forward_hook(self._hook(key)) for key, (lin, _a, _b) in self.mods.items()]

	def _hook(self, key):
		def fn(mod, inp, out):
			if not self.on:
				return None
			_lin, a, b = self.mods[key]
			return out + ((inp[0].float() @ a.T) @ b.T).to(out.dtype)
		return fn

	def groups(self):
		return [{"params": [p for _l, a, b in self.mods.values() for p in (a, b)], "lr": self.lr}]

	def result(self):
		return {"lora": {"rank": self.rank, "alpha": float(self.rank), "weights": {key: (a.detach().float().cpu().numpy(), b.detach().float().cpu().numpy()) for key, (_l, a, b) in self.mods.items()}}}

	def close(self):
		for h in self.handles:
			h.remove()


class Both:
	def __init__(self, parts):
		self.parts = parts

	@property
	def on(self):
		return self.parts[0].on

	@on.setter
	def on(self, v):
		for p in self.parts:
			p.on = v

	def groups(self):
		return [g for p in self.parts for g in p.groups()]

	def result(self):
		out = {}
		for p in self.parts:
			out.update(p.result())
		return out

	def close(self):
		for p in self.parts:
			p.close()


def carrier(model, layers, cfg):
	kind = cfg.get("carrier") or "vector"
	parts = []
	if kind in ("vector", "both"):
		parts.append(Steer(model, layers, cfg["lr"]))
	if kind in ("lora", "both"):
		parts.append(Lora(model, layers, cfg.get("rank", 8), TARGETS.get(cfg.get("targets"), TARGETS["down"]), cfg.get("lora_lr", 0.002)))
	return parts[0] if len(parts) == 1 else Both(parts)


def logits(model, h):
	x = model.lm_head(h.to(next(model.lm_head.parameters()).device)).float()
	cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
	return cap * torch.tanh(x / cap) if cap else x


def _ce(model, batch, seqs, pad, dev):
	T = max(map(len, seqs))
	x = torch.full((len(seqs), T), pad, device=dev, dtype=torch.long)
	for i, s in enumerate(seqs):
		x[i, :len(s)] = torch.tensor(s, device=dev)
	h = decoder(model)(x, use_cache=False).last_hidden_state
	out = []
	for i, p in enumerate(batch):
		s0 = len(seqs[i]) - len(p["ans"]) - 1
		lp = logits(model, h[i, s0:s0 + len(p["ans"])]).log_softmax(-1)
		out.append(-(p["tp"].to(lp.device) * lp.gather(1, p["ti"].to(lp.device))).sum(-1).mean())
	return torch.stack(out).mean()


def distill(model, pairs, prompt_tokens, kl_tokens, layers, cfg, pad=0, stop=None, on_step=None):
	steer = carrier(model, layers, cfg)
	net = decoder(model)
	dev = next(net.embed_tokens.parameters()).device
	try:
		seqs = [prompt_tokens(p["prompt"]) + p["ans"] for p in pairs]
		xk = torch.tensor([kl_tokens], device=dev)
		with torch.no_grad():
			base = logits(model, net(xk, use_cache=False).last_hidden_state[0, LV.SKIP:]).log_softmax(-1)
		opt = torch.optim.Adam(steer.groups())
		sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
		rng = np.random.default_rng(0)
		weights = np.array([3.0 if p.get("key") == "custom" else 1.0 for p in pairs])
		weights = weights / weights.sum()
		hist = []
		lam = float(cfg["lam"])
		target = float(cfg.get("kl_target") or 0.0) if cfg.get("carrier") != "lora" else 0.0
		for step in range(cfg["steps"]):
			if stop is not None and stop.is_set():
				return None
			idx = list(rng.choice(len(pairs), size=min(cfg["batch"], len(pairs)), replace=False, p=weights))
			micro = max(1, int(cfg.get("micro", 2)))
			opt.zero_grad(set_to_none=True)
			steer.on = True
			ce = 0.0
			for s in range(0, len(idx), micro):
				part = idx[s:s + micro]
				loss = _ce(model, [pairs[i] for i in part], [seqs[i] for i in part], pad, dev) * (len(part) / len(idx))
				loss.backward()
				ce += float(loss.detach())
			kl = (base.exp() * (base - logits(model, net(xk, use_cache=False).last_hidden_state[0, LV.SKIP:]).log_softmax(-1))).sum(-1).mean()
			(lam * kl).backward()
			steer.on = False
			opt.step()
			sched.step()
			hist.append((ce, float(kl.detach())))
			if target > 0:
				lam = float(np.clip(lam * np.exp(0.3 * (hist[-1][1] / target - 1.0)), 0.05, 200.0))
			if on_step:
				on_step(step, cfg["steps"], hist[-1][0], hist[-1][1])
		return steer.result(), hist
	finally:
		steer.on = False
		steer.close()


def as_lever(vecs, sig=""):
	layers = sorted(int(l) for l in vecs)
	k = np.sqrt(len(layers)) / 2.0
	return {"layers": layers, "dirs": {l: LV.unit(np.asarray(vecs[l])) for l in layers}, "norm": {l: float(np.linalg.norm(vecs[l]) * k) for l in layers}, "gain": 1.0, "gen_mult": 1.0, "gen_by": "optimized", "kind": "optimized", "sig": sig, "sep": {}}


def lora_tensors(weights):
	names = {"down": "ffn_down", "up": "ffn_up", "gate": "ffn_gate"}
	out = {}
	for (l, t), (a, b) in sorted(weights.items()):
		base = f"blk.{int(l)}.{names[t]}.weight"
		out[base + ".lora_a"] = np.ascontiguousarray(a, dtype=np.float32)
		out[base + ".lora_b"] = np.ascontiguousarray(b, dtype=np.float32)
	return out
