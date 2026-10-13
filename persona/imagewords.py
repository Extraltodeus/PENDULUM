import re

import torch

PROMPT = "Look at this image. What feeling, mood and qualities does it convey?"
WHERE = ("after", "image", "both")
STOP = set(re.findall(r"[a-z]+", PROMPT.lower())) | {"image", "images", "picture", "pictures", "feelings", "emotion", "emotions", "emotional", "evoke", "evokes", "conveys", "conveyed", "describe", "described", "vibe", "vibes", "atmosphere", "based", "overall", "sense", "quality", "assistant", "user", "the", "and", "its", "with", "kind", "closely", "characteristics", "traits", "attributes", "depict", "depicts", "depicting", "depiction", "depicted", "imagery", "visuals", "visual", "conveying", "sensations"}
CHUNK = 64
POOL = 400
TOP = 16
EPS = 1e-7


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
