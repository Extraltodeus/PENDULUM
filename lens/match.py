import os
import re

VARIANTS = {"it": "instruct", "instruct": "instruct", "chat": "instruct", "base": "base", "pt": "base"}
QUANT_RE = re.compile(r"[-_.](?:i?q\d\w*|b?f16|fp16|f32|fp8|mxfp4)$")
TOKEN_RE = re.compile(r"[a-z]{2,}(?=\d)|[a-z]*\d+(?:\.\d+)*[a-z]*|[a-z]+")
LENS_RE = re.compile(r"(.+?)_jacobian_lens(?:_(.+))?$", re.I)
KV_GUESS = 262144
TORCH_HEADROOM = 512 * 2**20


def lens_base(filename):
	stem = os.path.splitext(os.path.basename(filename))[0]
	m = LENS_RE.match(stem)
	if not m:
		return None, 0, ""
	tag = m.group(2) or ""
	n = re.search(r"(?:^|_)n(\d+)(?:_|$)", tag)
	return m.group(1), int(n.group(1)) if n else 0, tag


def label(base, n=0, tag=""):
	return base + (f" · n{n}" if n else "") + (f" · {tag}" if tag and not re.fullmatch(r"n\d+", tag) else "")


def file_entry(path):
	base, n, tag = lens_base(path)
	base = base or os.path.splitext(os.path.basename(path))[0]
	return {"key": "file:" + path, "source": "file", "repo": "", "revision": "", "file": os.path.basename(path), "path": path, "size": os.path.getsize(path) if os.path.exists(path) else 0, "base": base, "n_prompts": n, "tag": tag, "tokens": tokens(base), "variant": "", "d_model": 0, "n_src": 0, "model_id": "", "label": label(base, n, tag)}


def lens_vram(model, entry):
	d = int(model.get("n_embd") or 0)
	n_vocab_bytes = d * KV_GUESS * 2 + TORCH_HEADROOM
	if not entry:
		return n_vocab_bytes
	return n_vocab_bytes + min(model_layers(model), 80) * d * d * 2


def tokens(text):
	s = (text or "").lower().strip()
	s = re.sub(r"^https?://[^/]+/", "", s)
	s = re.sub(r"\.(gguf|pt|safetensors)$", "", s)
	while True:
		t = QUANT_RE.sub("", s)
		if t == s:
			break
		s = t
	out = []
	for part in re.split(r"[^a-z0-9.]+", s):
		out += [t.strip(".") for t in TOKEN_RE.findall(part) if t.strip(".")]
	return out


def model_names(model):
	names = [model.get("file", ""), model.get("name", ""), f"{model.get('basename', '')} {model.get('size_label', '')}".strip()]
	names += list(model.get("base_models") or [])
	return [n for n in names if n]


def model_layers(model):
	return int(model.get("n_layer") or 0) - int(model.get("n_nextn") or 0)


def compat(entry, model):
	L, d = model_layers(model), int(model.get("n_embd") or 0)
	if not L or not d:
		return None
	if entry.get("d_model") and entry.get("n_src"):
		return entry["d_model"] == d and entry["n_src"] <= L - 1
	size = entry.get("size") or 0
	if not size:
		return None
	for bpe in (2, 4):
		est = size / (bpe * d * d)
		k = round(est)
		if 1 <= k <= L - 1 and abs(est - k) < 0.02 and k >= (L - 1) // 2:
			return True
	return False


def score(entry, model):
	core = [t for t in entry.get("tokens") or [] if t not in VARIANTS]
	if not core:
		return 0
	sets = [set(tokens(n)) for n in model_names(model)]
	if not any(all(t in s for t in core) for s in sets):
		return 0
	s = 100 + 10 * len(core) + (5 if entry.get("source") in ("local", "file") else 0)
	allt = set().union(*sets)
	mv = "instruct" if any(VARIANTS.get(t) == "instruct" for t in allt) else ("base" if any(VARIANTS.get(t) == "base" for t in allt) else "")
	if entry.get("variant") and mv:
		s += 6 if entry["variant"] == mv else -3
	return s


def rank(index, model):
	pref = {"local": 2, "file": 2, "cached": 1, "hub": 0}
	rows = []
	for e in index or []:
		c = compat(e, model)
		s = score(e, model) if c is not False else 0
		rows.append({"entry": e, "score": s, "compat": c})
	rows.sort(key=lambda r: (-(r["score"] > 0), r["compat"] is False, r["compat"] is None, -r["score"], -r["entry"].get("n_prompts", 0), -pref.get(r["entry"]["source"], 0), r["entry"]["base"].lower()))
	return rows


def best(index, model):
	for r in rank(index, model):
		if r["score"] > 0 and r["compat"] is not False:
			return r["entry"]
	return None
