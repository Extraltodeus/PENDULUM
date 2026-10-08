import glob
import io
import json
import os
import time

import torch

import config
from lens.match import VARIANTS, label, lens_base, tokens

HUB_REPOS = ("neuronpedia/jacobian-lens",)
HUB_TTL = 86400


def _read_json(path, default=None):
	try:
		with open(path, encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return default


def _write_json(path, data):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	tmp = path + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False)
	os.replace(tmp, path)


def hf_cache_root():
	from huggingface_hub import constants

	return constants.HF_HUB_CACHE


def _entry(source, repo, revision, file, path, size, base=None, n=0, tag="", extra=None):
	if base is None:
		base, n, tag = lens_base(file)
		base = base or os.path.splitext(os.path.basename(file))[0]
	folder = file.replace("\\", "/").split("/")[0] if "/" in file.replace("\\", "/") else ""
	toks = tokens(base)
	variant = next((VARIANTS[t] for t in toks + tokens(folder) if t in VARIANTS), "")
	e = {"key": f"{source}:{repo}:{file}" if repo else f"{source}:{path}", "source": source, "repo": repo, "revision": revision, "file": file.replace("\\", "/"), "path": path, "size": int(size or 0), "base": base, "n_prompts": int(n or 0), "tag": tag, "tokens": toks, "variant": variant, "d_model": 0, "n_src": 0, "model_id": ""}
	e.update(extra or {})
	return e


def _fetch_repo(api, repo):
	out, seen = [], set()
	branches = [b.name for b in api.list_repo_refs(repo).branches] or ["main"]
	branches.sort(key=lambda b: b != "main")
	for branch in branches:
		for it in api.list_repo_tree(repo, recursive=True, revision=branch):
			path = it.path
			if not path.endswith(".pt") or not lens_base(path)[0] or path in seen:
				continue
			seen.add(path)
			out.append(_entry("hub", repo, branch, path, "", int(getattr(it, "size", 0) or 0)))
	return out


def hub_entries(online=True, refresh=False, repos=HUB_REPOS):
	from huggingface_hub import HfApi

	path = os.path.join(config.CACHE_DIR, "hub_lenses.json")
	cached = _read_json(path, {}) or {}
	per_repo = cached.get("repos") if isinstance(cached.get("repos"), dict) else {}
	api = HfApi() if online else None
	out, changed = [], False
	for repo in repos:
		slot = per_repo.get(repo) or {}
		fresh = time.time() - float(slot.get("t", 0)) < HUB_TTL
		if api is not None and (refresh or not fresh or "entries" not in slot):
			try:
				slot = {"t": time.time(), "entries": _fetch_repo(api, repo)}
				per_repo[repo] = slot
				changed = True
			except Exception:
				pass
		out += slot.get("entries") or []
	if changed:
		_write_json(path, {"repos": per_repo})
	return out


def cached_files():
	root = hf_cache_root()
	out = {}
	for repo_dir in glob.glob(os.path.join(root, "models--*")):
		name = os.path.basename(repo_dir)
		if "lens" not in name.lower():
			continue
		repo = name[len("models--"):].replace("--", "/")
		for snap in sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*")), key=os.path.getmtime, reverse=True):
			for path in glob.glob(os.path.join(snap, "**", "*.pt"), recursive=True):
				rel = os.path.relpath(path, snap).replace("\\", "/")
				if lens_base(rel)[0] and (repo, rel) not in out:
					out[(repo, rel)] = path
	return out


def local_entries(extra=()):
	out = []
	paths = glob.glob(os.path.join(config.LENSES_DIR, "**", "*.pt"), recursive=True)
	paths += [p for p in extra or [] if os.path.isfile(p) and os.path.normcase(os.path.abspath(p)) not in {os.path.normcase(os.path.abspath(x)) for x in paths}]
	for path in paths:
		meta = _read_json(os.path.join(os.path.dirname(path), "meta.json"), {}) or {}
		base, n, tag = lens_base(path)
		if meta.get("model_id"):
			base = base or meta["model_id"].rstrip("/").split("/")[-1]
		base = base or os.path.basename(os.path.dirname(path))
		e = _entry("local", "", "", os.path.basename(path), path, os.path.getsize(path), base, meta.get("n_prompts") or n, tag, {"model_id": meta.get("model_id", "")})
		if meta.get("model_id"):
			e["tokens"] = tokens(meta["model_id"].rstrip("/").split("/")[-1]) or e["tokens"]
		out.append(e)
	return out


def lens_shape(path):
	cache_path = os.path.join(config.CACHE_DIR, "lens_shapes.json")
	cache = _read_json(cache_path, {}) or {}
	key = f"{path}|{os.path.getsize(path)}"
	if key in cache:
		return cache[key]
	try:
		ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
		shape = {"d_model": int(ckpt.get("d_model") or 0), "n_src": len(ckpt.get("J") or {}), "n_prompts": int(ckpt.get("n_prompts") or 0), "layers": sorted(int(l) for l in (ckpt.get("J") or {}))}
		del ckpt
	except Exception:
		shape = {"d_model": 0, "n_src": 0, "n_prompts": 0, "layers": []}
	cache[key] = shape
	_write_json(cache_path, cache)
	return shape


def build_index(online=True, refresh=False, extra=(), repos=HUB_REPOS):
	hub = hub_entries(online, refresh, repos or HUB_REPOS)
	files = cached_files()
	out, used, seen = [], set(), set()
	for e in hub:
		if (e["repo"], e["file"]) in seen:
			continue
		seen.add((e["repo"], e["file"]))
		e = dict(e)
		path = files.get((e["repo"], e["file"]))
		if path and os.path.exists(path):
			e.update(source="cached", path=path, key=f"cached:{e['repo']}:{e['file']}")
			used.add((e["repo"], e["file"]))
		out.append(e)
	for (repo, rel), path in files.items():
		if (repo, rel) not in used:
			out.append(_entry("cached", repo, "", rel, path, os.path.getsize(path)))
	out += local_entries(extra)
	for e in out:
		if e["path"] and os.path.exists(e["path"]):
			shape = lens_shape(e["path"])
			e["d_model"], e["n_src"] = shape["d_model"], shape["n_src"]
			e["n_prompts"] = e["n_prompts"] or shape["n_prompts"]
		e["label"] = label(e["base"], e["n_prompts"], e["tag"])
	return out














def _progress_class(cb):
	from tqdm.auto import tqdm

	class Progress(tqdm):
		def __init__(self, *args, **kwargs):
			kwargs["file"] = io.StringIO()
			kwargs.pop("disable", None)
			super().__init__(*args, **kwargs)

		def update(self, n=1):
			out = super().update(n)
			if self.total:
				cb(min(1.0, self.n / self.total))
			return out

	return Progress


def resolve_path(entry, on_progress=None):
	if entry.get("path") and os.path.exists(entry["path"]):
		return entry["path"]
	from huggingface_hub import hf_hub_download

	kwargs = {"revision": entry.get("revision") or None}
	if on_progress:
		kwargs["tqdm_class"] = _progress_class(on_progress)
	return hf_hub_download(entry["repo"], entry["file"], **kwargs)


class Lens:
	def __init__(self, jacobians, n_prompts, d_model, name, source):
		self.jacobians = jacobians
		self.layers = sorted(jacobians)
		self.n_prompts = n_prompts
		self.d_model = d_model
		self.name = name
		self.source = source

	@property
	def identity(self):
		return not self.jacobians

	@classmethod
	def load(cls, path, name=""):
		ckpt = torch.load(path, map_location="cpu", weights_only=True)
		if "J" not in ckpt:
			raise ValueError(f"{path} is not a Jacobian lens file")
		return cls({int(l): J for l, J in ckpt["J"].items()}, int(ckpt.get("n_prompts", 0)), int(ckpt["d_model"]), name or lens_base(path)[0] or os.path.basename(path), path)

	@classmethod
	def logit_lens(cls, d_model):
		return cls({}, 0, d_model, "logit lens (J = I)", "")

	def nbytes(self, layers=None):
		layers = self.layers if layers is None else [l for l in layers if l in self.jacobians]
		return len(layers) * self.d_model * self.d_model * 2
