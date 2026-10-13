import hashlib
import json
import os

import torch

import config
from lens.vocab import single_token_ids


EDITABLE = ("name", "plus", "minus", "seeds_plus", "seeds_minus", "desc_plus", "desc_minus")


def builtin_definitions():
	with open(config.TRAITS_PATH, encoding="utf-8") as f:
		defs = json.load(f)
	for d in defs:
		d["custom"] = False
	return defs


def load_overrides():
	try:
		with open(config.TRAIT_OVERRIDES_PATH, encoding="utf-8") as f:
			data = json.load(f)
	except Exception:
		data = {}
	if not isinstance(data, dict):
		data = {}
	edited = {k: {f: v[f] for f in EDITABLE if f in v} for k, v in (data.get("edited") or {}).items() if isinstance(v, dict)}
	hidden = [h for h in (data.get("hidden") or []) if isinstance(h, str)]
	return {"edited": edited, "hidden": hidden}


def save_overrides(ov):
	os.makedirs(config.DATA_DIR, exist_ok=True)
	tmp = config.TRAIT_OVERRIDES_PATH + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump({"edited": ov.get("edited") or {}, "hidden": ov.get("hidden") or []}, f, ensure_ascii=False, indent=1)
	os.replace(tmp, config.TRAIT_OVERRIDES_PATH)


def edited_fields(base, values):
	out = {}
	for k in EDITABLE:
		v = values.get(k)
		if v is None:
			continue
		v = [s.strip() for s in v if s.strip()] if k.startswith("seeds") else str(v).strip()
		if v != (base.get(k) or ([] if k.startswith("seeds") else "")):
			out[k] = v
	return out


def load_definitions(with_hidden=False):
	ov = load_overrides()
	defs = []
	for d in builtin_definitions():
		e = ov["edited"].get(d["id"])
		if e:
			d.update(e)
			d["edited"] = True
		if d["id"] in ov["hidden"]:
			if not with_hidden:
				continue
			d["hidden"] = True
		defs.append(d)
	for d in load_custom():
		d["custom"] = True
		defs.append(d)
	return defs


def hidden_definitions():
	return [d for d in load_definitions(with_hidden=True) if d.get("hidden")]


def load_custom():
	if not os.path.exists(config.CUSTOM_TRAITS_PATH):
		return []
	try:
		with open(config.CUSTOM_TRAITS_PATH, encoding="utf-8") as f:
			data = json.load(f)
	except Exception:
		return []
	if not isinstance(data, list):
		return []
	return [t for t in data if isinstance(t, dict) and all(t.get(k) for k in ("id", "name", "plus", "minus"))]


def save_custom(traits):
	os.makedirs(config.DATA_DIR, exist_ok=True)
	tmp = config.CUSTOM_TRAITS_PATH + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump([{k: v for k, v in t.items() if k != "custom"} for t in traits], f, ensure_ascii=False, indent=1)
	os.replace(tmp, config.CUSTOM_TRAITS_PATH)


def make_trait(name, seeds_plus, seeds_minus, plus="", minus="", taken=()):
	base = "".join(c for c in name.lower() if c.isalnum()) or "custom"
	tid, n = base, 1
	while tid in taken:
		n += 1
		tid = f"{base}_{n}"
	return {"id": tid, "name": name, "plus": plus or (seeds_plus[0] if seeds_plus else name), "minus": minus or (seeds_minus[0] if seeds_minus else "not " + name), "seeds_plus": list(seeds_plus), "seeds_minus": list(seeds_minus)}


def signature(trait):
	keys = ("name", "plus", "minus", "seeds_plus", "seeds_minus", "desc_plus", "desc_minus", "system_plus", "system_minus", "prompts")
	data = {k: trait.get(k) for k in keys}
	for k in ("images_plus", "images_minus"):
		if trait.get(k):
			data[k] = trait[k]
	blob = json.dumps(data, ensure_ascii=False, sort_keys=True)
	return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _unit_rows(W, ids):
	rows = W[torch.as_tensor(ids, device=W.device)].float()
	return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _row_norms(W, chunk=32768):
	out = torch.empty(W.shape[0], device=W.device, dtype=torch.float32)
	for s in range(0, W.shape[0], chunk):
		out[s:s + chunk] = W[s:s + chunk].float().norm(dim=-1)
	return out.clamp_min(1e-8)


def build_lexicons(engine, readout, defs, english_mask, expand=6, min_sim=0.34):
	W = readout.W
	norms = _row_norms(W)
	emask = english_mask.to(W.device)
	strings = engine.vocab_strings()
	lexicons = {}
	for d in defs:
		poles = {}
		seed_sets = {}
		for pole in ("plus", "minus"):
			ids = []
			for word in d.get(f"seeds_{pole}", []):
				for tid in single_token_ids(engine, word):
					if tid not in ids:
						ids.append(tid)
			seed_sets[pole] = ids
		for pole, other in (("plus", "minus"), ("minus", "plus")):
			seeds = seed_sets[pole]
			weights = {tid: 1.0 for tid in seeds}
			if seeds and expand:
				S = _unit_rows(W, seeds)
				sims = ((W @ S.T.to(W.dtype)).float() / norms[:, None]).masked_fill(~emask[:, None], -1.0)
				banned = set(seed_sets[other])
				k = min(expand + len(seeds) + 8, sims.shape[0])
				vals, idx = sims.topk(k, dim=0)
				for col in range(len(seeds)):
					added = 0
					for v, i in zip(vals[:, col].tolist(), idx[:, col].tolist()):
						if added >= expand or v < min_sim:
							break
						if i in weights or i in banned:
							continue
						weights[i] = round(float(v), 3)
						added += 1
			poles[pole] = weights
		common = set(poles["plus"]) & set(poles["minus"])
		lex = {}
		for pole in ("plus", "minus"):
			items = [(tid, w) for tid, w in poles[pole].items() if tid not in common]
			lex[pole] = {"ids": [t for t, _ in items], "w": [w for _, w in items], "strs": [strings[t] for t, _ in items], "n_seeds": sum(1 for t, w in items if w == 1.0)}
		lexicons[d["id"]] = lex
	return lexicons


class TraitScorer:
	def __init__(self, readout, defs, lexicons):
		self.readout = readout
		self.ids = [d["id"] for d in defs if d["id"] in lexicons and lexicons[d["id"]]["plus"]["ids"] and lexicons[d["id"]]["minus"]["ids"]]
		all_ids = []
		for tid in self.ids:
			for pole in ("plus", "minus"):
				for t in lexicons[tid][pole]["ids"]:
					if t not in all_ids:
						all_ids.append(t)
		pos = {t: i for i, t in enumerate(all_ids)}
		dev = readout.device
		self.all = torch.tensor(all_ids, device=dev, dtype=torch.long)
		K = len(self.ids)
		M = torch.full((2 * K, len(all_ids)), float("-inf"), device=dev, dtype=torch.float32)
		for k, tid in enumerate(self.ids):
			for j, pole in enumerate(("plus", "minus")):
				lx = lexicons[tid][pole]
				logw = torch.log(torch.tensor(lx["w"], dtype=torch.float32).clamp_min(1e-4))
				logw = logw - torch.logsumexp(logw, 0)
				M[2 * k + j, torch.tensor([pos[t] for t in lx["ids"]], dtype=torch.long)] = logw.to(dev)
		self.M = M

	def score_logprobs(self, lp):
		return self.score_selected(lp[:, self.all])

	def score_selected(self, sel):
		z = torch.logsumexp(sel.float()[:, None, :] + self.M[None], dim=-1)
		a, b = z[:, 0::2], z[:, 1::2]
		return (a - b).T, torch.logaddexp(a, b).T

	def axis_scores(self, lexicons, english_mask):
		W = self.readout.W
		norms = _row_norms(W)
		out = {}
		for tid in self.ids:
			vp = _unit_rows(W, lexicons[tid]["plus"]["ids"]).mean(0)
			vm = _unit_rows(W, lexicons[tid]["minus"]["ids"]).mean(0)
			axis = vp / vp.norm() - vm / vm.norm()
			s = (W @ axis.to(W.dtype)).float() / norms
			out[tid] = s.masked_fill(~english_mask.to(W.device), 0.0)
		return out
