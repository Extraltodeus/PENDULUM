import numpy as np
import torch

from lens.workspace import CORPUS
from persona.coherence import read_score

KL_TEXT = CORPUS[0] + " " + CORPUS[4]
SKIP = 4
VERSION = 3
TARGET_KL = 0.45
EARLY = 0.65
GUARD_KL = 1.8
WIDTH = 7
CONTRAST_PROMPTS = [
	"What do you think about Mondays?",
	"Describe your neighbor.",
	"How was your day?",
	"Give me advice about my messy room.",
	"What's your opinion on modern art?",
	"My friend forgot my birthday.",
]


def unit(v):
	n = float(np.linalg.norm(v))
	return (v / n).astype(np.float32) if n > 1e-8 else np.zeros_like(v, dtype=np.float32)


def lens_direction(readout, lexicon, layer):
	vp = readout.direction(lexicon["plus"]["ids"], layer, lexicon["plus"]["w"])
	vm = readout.direction(lexicon["minus"]["ids"], layer, lexicon["minus"]["w"])
	return unit((vp - vm).cpu().numpy())


def system_prompt(trait, pole):
	if trait.get(f"system_{pole}"):
		return trait[f"system_{pole}"]
	adj = trait.get(pole) or trait["name"]
	desc = trait.get(f"desc_{pole}")
	if desc:
		return f"You are {desc}. Stay fully in character in every answer."
	words = ", ".join(w for w in trait.get(f"seeds_{pole}", [])[:8])
	if not words:
		return f"You are completely {adj}. You think, talk and behave in a {adj} way. Stay fully in character in every answer."
	return f"You are completely {adj}. You think, talk and behave in a {adj} way, and you naturally use words like: {words}. Stay fully in character in every answer."


def extract(engine, trait, band, max_tokens=40, seed=11, stop=None, on_progress=None, prompts=None):
	lo, hi = band
	prompts = prompts or CONTRAST_PROMPTS
	layers = list(range(max(1, lo - 2), min(engine.n_layer - 1, hi) + 1))
	sums = {p: {l: np.zeros(engine.n_embd, dtype=np.float64) for l in layers} for p in ("plus", "minus")}
	norms = {l: 0.0 for l in layers}
	counts = {"plus": 0, "minus": 0}
	total = 2 * len(prompts)
	done = 0
	samples = {}
	for pole in ("plus", "minus"):
		sysmsg = system_prompt(trait, pole)
		for i, prompt in enumerate(prompts):
			if stop is not None and stop.is_set():
				return None
			toks = engine.chat_tokens([{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}])
			r = engine.generate(toks, {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.05, "max_tokens": max_tokens}, capture_layers=layers, seed=seed + i)
			if i < 2:
				samples.setdefault(pole, []).append(r["text"][:220])
			if r["steps"]:
				for l in layers:
					h = np.concatenate([s[l] for s in r["steps"] if l in s])
					sums[pole][l] += h.sum(0)
					norms[l] += float(np.linalg.norm(h, axis=1).sum())
				counts[pole] += len(r["steps"])
			done += 1
			if on_progress:
				on_progress(done / total)
	n = max(1, counts["plus"] + counts["minus"])
	dirs, sep, mean_norm = {}, {}, {}
	for l in layers:
		diff = sums["plus"][l] / max(1, counts["plus"]) - sums["minus"][l] / max(1, counts["minus"])
		mean_norm[l] = norms[l] / n
		sep[l] = float(np.linalg.norm(diff) / max(mean_norm[l], 1e-6))
		dirs[l] = unit(diff)
	return {"dirs": dirs, "sep": sep, "norm": mean_norm, "samples": samples}


def pick_layers(sep, band, width=WIDTH):
	lo, hi = band
	hi = max(lo + width - 1, int(round(lo + EARLY * (hi - lo))))
	cands = [l for l in sorted(sep) if lo <= l <= hi]
	if len(cands) <= width:
		return cands
	best, best_s = cands[:width], -1.0
	for i in range(len(cands) - width + 1):
		win = cands[i:i + width]
		s = sum(sep[l] for l in win)
		if s > best_s:
			best, best_s = win, s
	return best


def reads_as(readout, direction, layer, norm, strings, k=10):
	h = torch.as_tensor(direction * norm * 4.0, device=readout.device)[None, :]
	ids, p = readout.topk(h, layer, k=k)
	return [strings[i] for i in ids[0]]


def unit_vectors(lever):
	dirs = lever.get("dirs") or {}
	scale = 2.0 / np.sqrt(max(len(dirs), 1))
	return {l: dirs[l] * float(lever["norm"][l]) * scale for l in lever["layers"] if l in dirs}


def effective_gain(lever, use_gen=True):
	w = float(use_gen)
	return lever["gain"] * (float(lever.get("gen_mult", 1.0)) ** w if w > 0 else 1.0)


def combine(levers, strengths, global_scale=1.0, use_gen=True):
	out = {}
	for tid, s in strengths.items():
		lever = levers.get(tid)
		if not lever or not s:
			continue
		g = effective_gain(lever, use_gen) * s * global_scale
		for l, v in unit_vectors(lever).items():
			out[l] = out.get(l, 0.0) + g * v
	return {l: np.asarray(v, dtype=np.float32) for l, v in out.items()}


def calibrate_gen(coh, lever, target, steps=5, on_step=None):
	base = {l: v * lever["gain"] for l, v in unit_vectors(lever).items()}
	lo, hi = 0.1, 1.6
	last = None
	for i in range(steps):
		m = (lo * hi) ** 0.5
		coh.note = f"calibration step {i + 1}/{steps} · strength 1 = ×{m:.2f}"
		coh.phase = {"mode": "calib", "step": i + 1, "steps": steps, "scale": m, "target": target}
		last = coh.measure({l: v * m for l, v in base.items()})
		if last["coherence"] < target:
			hi = m
		else:
			lo = m
		if on_step:
			on_step(i + 1, steps, m, last)
	return lo, last


class KLProbe:
	def __init__(self, engine, readout, text=KL_TEXT):
		self.engine = engine
		self.device = readout.device
		self.tokens = engine.text_tokens(text)[:160]
		self.ref = self._logprobs(None)

	def _logprobs(self, vectors):
		self.engine.set_steering(vectors)
		self.engine.reset()
		rows = []
		self.engine.decode(self.tokens, logits="all", on_logits=lambda i, lg: rows.append(lg.copy()) if i >= SKIP else None)
		x = torch.as_tensor(np.stack(rows), device=self.device)
		return torch.log_softmax(x.float(), dim=-1)

	def kl(self, vectors):
		try:
			lq = self._logprobs(vectors)
		finally:
			self.engine.set_steering(None)
		return float((self.ref.exp() * (self.ref - lq)).sum(-1).mean())


def calibrate(probe, lever, target=TARGET_KL, lo=1e-4, hi=4.0, iters=12):
	base = unit_vectors(lever)
	gain, k = 1.0, 0.0
	for _ in range(iters):
		gain = (lo * hi) ** 0.5
		k = probe.kl({l: v * gain for l, v in base.items()})
		if k > target:
			hi = gain
		else:
			lo = gain
	return gain, k


def plus(a, b):
	if not b:
		return a
	out = dict(b)
	for l, v in a.items():
		out[l] = out[l] + v if l in out else v
	return out


def guard(probe, levers, strengths, max_kl=GUARD_KL, use_gen=True, base=None):
	vecs = combine(levers, strengths, use_gen=use_gen)
	if not vecs:
		return vecs, 1.0, 0.0
	k = probe.kl(plus(vecs, base))
	if k <= max_kl:
		return vecs, 1.0, k
	lo, hi = 0.05, 1.0
	for _ in range(7):
		mid = (lo + hi) / 2
		k = probe.kl(plus(combine(levers, strengths, mid, use_gen), base))
		lo, hi = (lo, mid) if k > max_kl else (mid, hi)
	return combine(levers, strengths, lo, use_gen), lo, probe.kl(plus(combine(levers, strengths, lo, use_gen), base))


def guard_gen(coh, levers, strengths, scale, threshold, use_gen=True, steps=3, on_step=None, base=None):
	vecs = combine(levers, strengths, scale, use_gen)
	coh.note = f"guard · levers as set" + (f" ×{scale:.2f}" if scale < 0.999 else "")
	coh.phase = {"mode": "uniform", "step": 0, "steps": steps, "scale": scale, "threshold": threshold}
	res = coh.measure(plus(vecs, base))
	if res["coherence"] >= threshold:
		return vecs, scale, res
	first = res["coherence"]
	lo, hi = 0.0, scale
	best = None
	for i in range(steps):
		mid = (lo + hi) / 2 if lo > 0 else hi * 0.5
		coh.note = f"guard · scaling step {i + 1}/{steps} · levers ×{mid:.2f}"
		coh.phase = {"mode": "uniform", "step": i + 1, "steps": steps, "scale": mid, "threshold": threshold}
		r = coh.measure(plus(combine(levers, strengths, mid, use_gen), base))
		if on_step:
			on_step(i + 1, steps, mid, r)
		if r["coherence"] >= threshold:
			lo, best = mid, r
		else:
			hi = mid
	if best is None:
		lo = hi * 0.5
		coh.note = f"guard · last resort · levers ×{lo:.2f}"
		coh.phase = {"mode": "uniform", "step": steps + 1, "steps": steps, "scale": lo, "threshold": threshold, "last": True}
		best = coh.measure(plus(combine(levers, strengths, lo, use_gen), base))
	return combine(levers, strengths, lo, use_gen), lo, dict(best, first=first)


def culprit_order(coh, levers, adj, texts):
	by_dose = sorted(adj, key=lambda k: -abs(adj[k]))
	if not texts:
		return by_dose, {}
	units = {k: unit_vectors(levers[k]) for k in adj}
	layers = sorted({l for u in units.values() for l in u})
	d = coh.drift(texts, layers)
	score = {}
	for k, v in adj.items():
		sign = 1.0 if v >= 0 else -1.0
		cs = [float(d[l] @ (np.asarray(u, dtype=np.float32) * sign) / (np.linalg.norm(d[l]) * np.linalg.norm(u) + 1e-9)) for l, u in units[k].items() if l in d]
		score[k] = max(0.0, float(np.mean(cs))) * abs(v) if cs else 0.0
	return sorted(by_dose, key=lambda k: -score[k]), score


def guard_blame(coh, levers, strengths, scale, threshold, use_gen=True, rounds=3, on_step=None, base=None, steps=3, on_order=None):
	adj = {k: v for k, v in strengths.items() if v and k in levers}
	factors = {k: 1.0 for k in adj}
	coh.note = "guard (worst lever first) · levers as set"
	coh.phase = {"mode": "blame", "round": 0, "rounds": rounds, "threshold": threshold, "factors": dict(factors)}
	res = coh.measure(plus(combine(levers, adj, scale, use_gen), base))
	first = res["coherence"]
	for i in range(rounds):
		if res["coherence"] >= threshold or len(adj) < 2:
			break
		best = None
		order, score = culprit_order(coh, levers, adj, res.get("texts") or [])
		if on_order:
			on_order(i + 1, order, score)
		for k in order:
			coh.note = f"guard (worst lever first) round {i + 1} · try: '{k}' at half?"
			coh.phase = {"mode": "blame", "round": i + 1, "rounds": rounds, "threshold": threshold, "lever": k, "factors": dict(factors, **{k: factors[k] * 0.5})}
			r = coh.measure(plus(combine(levers, dict(adj, **{k: adj[k] * 0.5}), scale, use_gen), base))
			if best is None or r["coherence"] > best[1]["coherence"]:
				best = (k, r)
			if r["coherence"] >= threshold:
				break
		k, res = best
		adj[k] *= 0.5
		factors[k] *= 0.5
		if on_step:
			on_step(i + 1, rounds, k, factors[k], res)
	if res["coherence"] >= threshold:
		return combine(levers, adj, scale, use_gen), scale, (dict(res, first=first) if first < threshold else res), factors
	vecs, scale, res = guard_gen(coh, levers, adj, scale, threshold, use_gen, steps=steps, base=base)
	return vecs, scale, dict(res, first=first), factors


def guard_read(coh, levers, strengths, scale, threshold, use_gen=True, ok=0.6, bad=4.5, fallback="blame", steps=3, rounds=3, on_gen=None, on_blame=None, base=None, on_read=None, probes=6):
	def drop(s, step):
		coh.note = f"guard (read) · levers ×{s:.2f}" if step else "guard (read) · levers as set"
		coh.phase = {"mode": "read", "step": step, "scale": s, "threshold": threshold, "ok": ok, "bad": bad}
		return coh.read_drop(plus(combine(levers, strengths, s, use_gen), base), ok, bad)

	d0 = drop(scale, 0)
	first = read_score(d0, ok, bad)
	if d0 <= ok:
		if on_read:
			on_read(scale, d0, "pass")
		return combine(levers, strengths, scale, use_gen), scale, {"coherence": 1.0, "by": "read", "drop": d0, "read": True}, {}
	at, d = scale, d0
	if d0 >= bad:
		if on_read:
			on_read(scale, d0, "fail")
		lo, hi, dl = 0.0, scale, None
		for i in range(probes):
			mid = (lo + hi) / 2
			dm = drop(mid, i + 1)
			lo, hi, dl = (mid, hi, dm) if dm < bad else (lo, mid, dl)
		at, d = (lo, dl) if dl is not None else (hi * 0.5, drop(hi * 0.5, probes + 1))
		if d <= ok:
			if on_read:
				on_read(at, d, "pass")
			return combine(levers, strengths, at, use_gen), at, {"coherence": 1.0, "by": "read", "drop": d, "read": True, "first": first}, {}
	if on_read:
		on_read(at, d, "judge")
	adj = {k: v for k, v in strengths.items() if v and k in levers}
	if fallback == "blame" and len(adj) > 1:
		vecs, s, res, factors = guard_blame(coh, levers, strengths, at, threshold, use_gen, rounds=rounds, steps=steps, on_step=on_blame, base=base)
	else:
		vecs, s, res = guard_gen(coh, levers, strengths, at, threshold, use_gen, steps=steps, on_step=on_gen, base=base)
		factors = {}
	return vecs, s, dict(res, first=res.get("first", res["coherence"]) if at == scale else first, drop=d0, read=False), factors


def guard_quick(coh, levers, strengths, scale, threshold, n, use_gen=True, fallback="blame", steps=3, rounds=3, on_gen=None, on_blame=None, base=None, on_quick=None):
	vecs = combine(levers, strengths, scale, use_gen)
	coh.note = f"guard (quick) · {n} question{'s' if n != 1 else ''} · levers as set"
	coh.phase = {"mode": "quick", "step": 0, "n": n, "threshold": threshold, "scale": scale}
	res = coh.measure(plus(vecs, base), n=n)
	if on_quick:
		on_quick(res, res["coherence"] >= threshold)
	if res["coherence"] >= threshold:
		return vecs, scale, dict(res, quick=True), {}
	first = res["coherence"]
	adj = {k: v for k, v in strengths.items() if v and k in levers}
	if fallback == "blame" and len(adj) > 1:
		vecs, scale, res, factors = guard_blame(coh, levers, strengths, scale, threshold, use_gen, rounds=rounds, steps=steps, on_step=on_blame, base=base)
	else:
		vecs, scale, res = guard_gen(coh, levers, strengths, scale, threshold, use_gen, steps=steps, on_step=on_gen, base=base)
		factors = {}
	return vecs, scale, dict(res, first=first, quick=False), factors
