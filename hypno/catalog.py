import json
import os
import re

import config
from engine import stconvert
from engine.ggufio import summary

SKIP_PREFIX = ("mmproj", "mtp-", "draft")
SHARD_RE = re.compile(r"-(\d{5})-of-\d{5}\.gguf$")


def _cache_path():
	return os.path.join(config.CACHE_DIR, "catalog.json")


def _load_cache():
	try:
		with open(_cache_path(), encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return {}


def _save_cache(cache):
	os.makedirs(config.CACHE_DIR, exist_ok=True)
	with open(_cache_path(), "w", encoding="utf-8") as f:
		json.dump(cache, f, ensure_ascii=False)


def gguf_files(dirs):
	found = []
	for root in dirs:
		if not root or not os.path.isdir(root):
			continue
		for base, _dirs, files in os.walk(root):
			for name in files:
				low = name.lower()
				if not low.endswith(".gguf") or low.startswith(SKIP_PREFIX) or "mmproj" in low:
					continue
				shard = SHARD_RE.search(low)
				if shard and shard.group(1) != "00001":
					continue
				found.append(os.path.join(base, name))
	return sorted(set(found), key=lambda p: os.path.basename(p).lower())


def _gguf(path, llama_dir, cache, kept):
	st = os.stat(path)
	key = f"v2|{path}|{st.st_size}|{int(st.st_mtime)}"
	entry = cache.get(key)
	if entry is None:
		try:
			entry = summary(path, llama_dir)
		except Exception as exc:
			entry = {"path": path, "file": os.path.basename(path), "size": st.st_size, "error": str(exc)}
	kept[key] = entry
	return entry


def _source(src, llama_dir, cache, kept, refs):
	sig = stconvert.signature(src)
	key = f"st1|{src}|{sig[0]}|{sig[1]}"
	entry = cache.get(key)
	if entry is None:
		try:
			entry = stconvert.describe(src, llama_dir)
		except Exception as exc:
			entry = {"skip": True, "why": str(exc)}
	kept[key] = entry
	if entry.get("skip"):
		return None
	ref = refs.get(src) or entry.get("ref") or ""
	out = stconvert.out_path(src, config.CONVERTED_DIR)
	if stconvert.fresh(src, out):
		g = _gguf(out, llama_dir, cache, kept)
		if not g.get("error"):
			return dict(g, source=src, format="safetensors", kind=entry.get("kind"), ref=ref, name=entry["name"])
	return dict(entry, ref=ref)


def scan(dirs, llama_dir="", refs=None):
	cache = _load_cache()
	kept = {}
	out = []
	for path in gguf_files(dirs):
		try:
			out.append(_gguf(path, llama_dir, cache, kept))
		except OSError:
			continue
	for src in stconvert.sources(dirs):
		try:
			entry = _source(src, llama_dir, cache, kept, refs or {})
		except OSError:
			continue
		if entry is not None:
			out.append(entry)
	if kept.keys() != cache.keys():
		_save_cache(kept)
	return out


def default_dirs():
	dirs = []
	hub = os.path.join(os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
	if os.path.isdir(hub):
		dirs.append(hub)
	return dirs
