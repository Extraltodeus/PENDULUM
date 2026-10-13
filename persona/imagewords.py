import json
import os
import re
import shutil

import numpy as np
import torch

import config
from persona.coherence import check

PROMPT = "Look at this image. What feeling, mood and qualities does it convey?"
WHERE = ("after", "image", "both")
STOP = set(re.findall(r"[a-z]+", PROMPT.lower())) | {"image", "images", "picture", "pictures", "feelings", "emotion", "emotions", "emotional", "evoke", "evokes", "conveys", "conveyed", "describe", "described", "vibe", "vibes", "atmosphere", "based", "overall", "sense", "quality", "assistant", "user", "the", "and", "its", "with", "kind", "closely", "characteristics", "traits", "attributes", "depict", "depicts", "depicting", "depiction", "depicted", "imagery", "visuals", "visual", "conveying", "sensations"}
POLE_WEIGHT = 3.0
CHUNK = 64
POOL = 400
TOP = 16
EPS = 1e-7


def image_dir(tid, pole):
	return os.path.join(config.LEVER_IMAGES_DIR, tid, pole)


def image_paths(trait, pole):
	folder = image_dir(trait["id"], pole)
	return [os.path.join(folder, n) for n in trait.get(f"images_{pole}") or []]


def pole_paths(trait, pole):
	folder = image_dir(trait["id"], pole)
	return [os.path.join(folder, n) for n in trait.get(f"poles_{pole}") or []]


def has_images(trait):
	return bool(trait.get("images_plus") or trait.get("images_minus"))


def store_images(tid, pole, sources):
	folder = image_dir(tid, pole)
	pairs = []
	for src in sources or []:
		src = os.path.abspath(src)
		if not os.path.isfile(src):
			continue
		if os.path.normcase(os.path.dirname(src)) == os.path.normcase(os.path.abspath(folder)):
			pairs.append((src, os.path.basename(src)))
			continue
		os.makedirs(folder, exist_ok=True)
		base, ext = os.path.splitext(os.path.basename(src))
		name, n = base + ext, 1
		while os.path.exists(os.path.join(folder, name)):
			n += 1
			name = f"{base}_{n}{ext}"
		shutil.copy2(src, os.path.join(folder, name))
		pairs.append((src, name))
	return pairs


def extract_layers(band, n_layer):
	lo, hi = band
	return list(range(max(1, lo - 2), min(n_layer - 1, hi) + 1))


def aggregate(caps, rows, layers):
	out = {"sums": {}, "norms": {}, "n": len(rows)}
	for l in layers:
		h = caps[l][[r for r in rows if r < len(caps[l])]].astype(np.float64)
		out["sums"][l] = h.sum(0)
		out["norms"][l] = float(np.linalg.norm(h, axis=1).sum())
	return out


def save_reads(path, entries):
	if not entries:
		return
	os.makedirs(os.path.dirname(path), exist_ok=True)
	arrays, meta = {}, []
	for i, e in enumerate(entries):
		layers = sorted(e["agg"]["sums"])
		arrays[f"s{i}"] = np.stack([e["agg"]["sums"][l] for l in layers]).astype(np.float32)
		arrays[f"n{i}"] = np.array([e["agg"]["norms"][l] for l in layers], dtype=np.float64)
		meta.append({"pole": e["pole"], "name": e["name"], "size": e["size"], "mtime": e["mtime"], "rows": e["agg"]["n"], "layers": layers})
	arrays["meta"] = np.array(json.dumps(meta))
	tmp = path + ".tmp"
	with open(tmp, "wb") as f:
		np.savez_compressed(f, **arrays)
	os.replace(tmp, path)


def load_reads(path):
	if not os.path.isfile(path):
		return {}
	out = {}
	try:
		with np.load(path) as z:
			meta = json.loads(str(z["meta"]))
			for i, m in enumerate(meta):
				s, n = z[f"s{i}"], z[f"n{i}"]
				agg = {"sums": {l: s[k].astype(np.float64) for k, l in enumerate(m["layers"])}, "norms": {l: float(n[k]) for k, l in enumerate(m["layers"])}, "n": m["rows"]}
				out[(m["pole"], m["name"])] = dict(m, agg=agg)
	except Exception:
		return {}
	return out


def extract(read, engine, plus, minus, band, stop=None, on_progress=None, weights=None):
	layers = extract_layers(band, engine.n_layer)
	sums = {p: {l: np.zeros(engine.n_embd, dtype=np.float64) for l in layers} for p in ("plus", "minus")}
	norms = {l: 0.0 for l in layers}
	counts = {"plus": 0, "minus": 0}
	total = len(plus) + len(minus) + 1
	done, span = 0, []

	def add(pole, agg, w=1.0):
		for l in layers:
			sums[pole][l] += w * agg["sums"][l]
			norms[l] += w * agg["norms"][l]
		counts[pole] += w * agg["n"]

	for pole, paths in (("plus", plus), ("minus", minus)):
		for path in paths:
			check(stop)
			agg = read(path)
			span.append(agg["n"])
			add(pole, agg, (weights or {}).get(path, 1.0))
			done += 1
			if on_progress:
				on_progress(done / total)
	empty = [p for p in ("plus", "minus") if not counts[p]]
	if empty:
		check(stop)
		toks = engine.chat_tokens([{"role": "user", "content": PROMPT}])
		engine.reset()
		engine.set_capture(layers)
		try:
			engine.decode(toks)
			caps = engine.pop_capture()
		finally:
			engine.set_capture(None)
			engine.reset()
		n = int(round(sum(span) / max(1, len(span)))) or len(toks)
		add(empty[0], aggregate(caps, list(range(max(0, len(toks) - n), len(toks))), layers))
	if on_progress:
		on_progress(1.0)
	n = max(1, counts["plus"] + counts["minus"])
	dirs, sep, mean_norm = {}, {}, {}
	for l in layers:
		diff = sums["plus"][l] / max(1, counts["plus"]) - sums["minus"][l] / max(1, counts["minus"])
		mean_norm[l] = norms[l] / n
		sep[l] = float(np.linalg.norm(diff) / max(mean_norm[l], 1e-6))
		nn = float(np.linalg.norm(diff))
		dirs[l] = (diff / nn).astype(np.float32) if nn > 1e-8 else np.zeros(engine.n_embd, dtype=np.float32)
	return {"dirs": dirs, "sep": sep, "norm": mean_norm, "samples": {p: [os.path.basename(x) for x in paths] for p, paths in (("plus", plus), ("minus", minus)) if paths}}


def rows_of(spans, where):
	first = next((i for i, s in enumerate(spans) if s[0] == "image"), None)
	out = []
	for i, (kind, at, n) in enumerate(spans):
		if where == "all" or (kind == "image" and where in ("image", "both")) or (kind == "text" and first is not None and i > first and where in ("after", "both")):
			out += range(at, at + n)
	return out


@torch.no_grad()
def profile(readout, caps, rows, layers):
	acc = torch.zeros(readout.n_vocab, device=readout.device, dtype=torch.float32)
	count = 0
	for l in layers:
		h = caps.get(l)
		if h is None:
			continue
		sel = [r for r in rows if r < len(h)]
		if not sel:
			continue
		h = h[sel]
		for s in range(0, len(h), CHUNK):
			lp = readout.logprobs(h[s:s + CHUNK], l)
			acc += lp.exp().sum(0)
			count += lp.shape[0]
	return acc / max(count, 1)


def top_tokens(p, strings, mask, k=12):
	idx = p.masked_fill(~mask, 0).topk(k).indices.tolist()
	return [strings[i].strip() for i in idx]


def contrast(p_plus, p_minus, strings, mask, top=TOP):
	out = {}
	for pole, a, b in (("plus", p_plus, p_minus), ("minus", p_minus, p_plus)):
		cand = a.masked_fill(~mask, 0).topk(POOL).indices
		la, lb = torch.log(a[cand] + EPS), torch.log(b[cand] + EPS)
		score = 2 * la - lb
		keep = (la - lb) > 0
		order = cand[keep][score[keep].argsort(descending=True)].tolist()
		words, seen = [], set()
		for i in order:
			w = strings[i].strip().lower()
			if w in seen or w in STOP:
				continue
			seen.add(w)
			words.append(w)
			if len(words) >= top:
				break
		out[pole] = words
	return out
