import json
import os
import re

import config


def folder():
	return os.path.join(config.DATA_DIR, "presets")


def slug(name):
	s = re.sub(r"[^\w.\- ]+", "_", (name or "").strip())[:80].strip(" .")
	return s or "preset"


def path_for(name):
	return os.path.join(folder(), slug(name) + ".json")


def save(data):
	os.makedirs(folder(), exist_ok=True)
	path = path_for(data["name"])
	tmp = path + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=1)
	os.replace(tmp, path)
	return path


def read(path):
	with open(path, encoding="utf-8") as f:
		data = json.load(f)
	if "strengths" not in data:
		raise ValueError("not a preset: no lever strengths in it")
	if "system_prompt" in data and "lora_adapters" in data:
		return from_export(data, path)
	return dict(data, path=path)


def from_export(data, path):
	return {"name": data.get("name") or os.path.basename(os.path.dirname(path)), "created": data.get("created", ""), "model": os.path.basename(data.get("model") or "") or data.get("model_name", ""), "mode": "dose", "strengths": {k: float(v) for k, v in (data.get("strengths") or {}).items()}, "system": data.get("system_prompt") or "", "traits": [], "goals": [], "exempt": [], "path": path, "export": True}


def load_all():
	out = []
	try:
		names = sorted(os.listdir(folder()))
	except OSError:
		return out
	for n in names:
		if not n.endswith(".json"):
			continue
		p = os.path.join(folder(), n)
		try:
			d = read(p)
		except (OSError, ValueError):
			continue
		d["mtime"] = os.path.getmtime(p)
		out.append(d)
	return sorted(out, key=lambda d: -d["mtime"])
