import glob
import json
import os
import re

import config

GROUPS = (
	("states", "MODEL STATES", "levers, profile and goal training of each model (profiles/)"),
	("exports", "EXPORTS", "exported control vectors, LoRA adapters and baked models (exports/)"),
	("lenses", "LENSES TRAINED HERE", "lenses fitted with TRAIN A LENS (lenses/)"),
	("fits", "LENS TRAINING RUNS", "work folders of lens trainings: corpus slices and checkpoints (fits/)"),
	("converted", "CONVERTED MODELS", "safetensors models converted to GGUF (converted/)"),
	("cache", "CACHE", "rebuilt when needed: model catalog, vocabularies, Hub lens list, judge self-tests (cache/)"),
	("history", "HISTORY AND LOGS", "reply history, saved traces, Qt error log"),
	("presets", "PRESETS", "saved presets (presets/)"),
)
STATE_FILES = (
	(re.compile(r"^state\.json$"), "lever settings, profile, targets and options of this model"),
	(re.compile(r"^levers\.npz$"), "built lever directions"),
	(re.compile(r"^ref_pmean\.pt$"), "neutral reference of the profile"),
	(re.compile(r"^teach_.+\.pt$"), "goal training answers (TRAIN reuses them when nothing changed)"),
	(re.compile(r"^goal_.+\.lora\.gguf$"), "LoRA adapter trained by TRAIN"),
)
LORA_FILE = re.compile(r"^goal_.+\.lora\.gguf$")
CACHE_NOTES = {"catalog.json": "model catalog (headers read once)", "hub_lenses.json": "list of the lenses on the Hub (24 h)", "lens_shapes.json": "shapes of the lens files", "judge_selftest.json": "judge self-test results", "vocab": "vocabularies of the models"}


def size_of(path):
	if os.path.isfile(path):
		try:
			return os.path.getsize(path)
		except OSError:
			return 0
	total = 0
	for root, _dirs, files in os.walk(path):
		for f in files:
			try:
				total += os.path.getsize(os.path.join(root, f))
			except OSError:
				pass
	return total


def mtime_of(path):
	try:
		t = os.path.getmtime(path)
	except OSError:
		return 0.0
	if os.path.isdir(path):
		for root, _dirs, files in os.walk(path):
			for f in files:
				try:
					t = max(t, os.path.getmtime(os.path.join(root, f)))
				except OSError:
					pass
	return t


def entry(path, name=None, note="", in_use="", children=None):
	kids = children or []
	return {"path": path, "name": name or os.path.basename(path), "size": sum(c["size"] for c in kids) if kids else size_of(path), "mtime": max([c["mtime"] for c in kids] or [0.0]) if kids else mtime_of(path), "note": note, "in_use": in_use, "children": kids, "dir": os.path.isdir(path)}


def listing(folder):
	try:
		return sorted(os.path.join(folder, n) for n in os.listdir(folder))
	except OSError:
		return []


def model_key(path, size):
	stem = re.sub(r"[^\w.\-]+", "_", os.path.splitext(os.path.basename(path))[0])[:80]
	return f"{stem}_{size}"


def state_note(name):
	return next((note for rx, note in STATE_FILES if rx.match(name)), "")


def read_json(path, default):
	try:
		with open(path, encoding="utf-8") as f:
			return json.load(f)
	except (OSError, ValueError):
		return default


def lora_info(path, model, state, goals):
	lid = os.path.basename(path)[:-len(".lora.gguf")]
	lever = (state.get("levers") or {}).get(lid) or {}
	strengths = state.get("strengths") or {}
	key = lid + ":lora" if lever.get("carrier") == "both" else lid
	scale = float(strengths.get(key) or 0.0) or 1.0
	return {"goal": goals.get(lid) or lid[len("goal_"):], "model": model.get("path", ""), "model_name": os.path.basename(model.get("path", "")), "arch": model.get("arch", ""), "scale": scale, "rank": (lever.get("lora") or {}).get("rank"), "carrier": lever.get("carrier", "lora")}


def states(models, loaded):
	known = {model_key(m["path"], m.get("size") or 0): m for m in models if m.get("path")}
	goals = {"goal_" + g["id"]: g.get("name") or g["id"] for g in read_json(config.GOALS_PATH, []) if isinstance(g, dict) and g.get("id")}
	live = model_key(loaded, os.path.getsize(loaded)) if loaded and os.path.isfile(loaded) else ""
	out = []
	for d in listing(config.PROFILES_DIR):
		if not os.path.isdir(d):
			continue
		key = os.path.basename(d)
		m = known.get(key)
		use = "the loaded model's state" if key == live else ""
		kids = [entry(f, note=state_note(os.path.basename(f)), in_use=use if not os.path.basename(f).startswith("teach_") else "") for f in listing(d)]
		state = read_json(os.path.join(d, "state.json"), {}) if any(LORA_FILE.match(os.path.basename(k["path"])) for k in kids) else {}
		for k in kids:
			if LORA_FILE.match(os.path.basename(k["path"])):
				k["lora"] = lora_info(k["path"], m or {}, state, goals)
		out.append(entry(d, os.path.basename(m["path"]) if m else key, "" if m else "model not found in your model folders", use, kids))
	return out


def simple(folder, notes=None, in_use=None):
	notes, in_use = notes or {}, in_use or {}
	return [entry(p, note=notes.get(os.path.basename(p), ""), in_use=in_use.get(os.path.normcase(os.path.abspath(p)), "")) for p in listing(folder)]


def converted(loaded):
	out = []
	live = os.path.normcase(os.path.abspath(loaded)) if loaded else ""
	for g in listing(config.CONVERTED_DIR):
		if not g.lower().endswith(".gguf"):
			continue
		side = os.path.splitext(g)[0] + ".json"
		kids = [entry(g, note="the converted model", in_use="the loaded model" if os.path.normcase(os.path.abspath(g)) == live else "")]
		if os.path.isfile(side):
			kids.append(entry(side, note="where it came from (rebuilt with the model)"))
		out.append(dict(entry(g, os.path.basename(g), "", kids[0]["in_use"], kids), paths=[k["path"] for k in kids]))
	return out


def history():
	out = []
	gen = os.path.join(config.DATA_DIR, "generations.jsonl")
	if os.path.isfile(gen):
		out.append(entry(gen, note="every reply of the dialogue with its configuration (HISTORY)"))
	for p in sorted(glob.glob(os.path.join(config.DATA_DIR, "trace_*.json"))):
		out.append(entry(p, note="saved trace"))
	log = os.path.join(config.DATA_DIR, "qt_errors.log")
	if os.path.isfile(log):
		out.append(entry(log, note="Qt errors (useful when reporting a crash)"))
	return out


def scan(models=(), loaded="", lens="", fitting=False):
	lens_use = {os.path.normcase(os.path.abspath(os.path.join(config.LENSES_DIR, lens))): "the lens in use"} if lens else {}
	fits = os.path.join(config.DATA_DIR, "fits")
	groups = {
		"states": states(models, loaded),
		"exports": simple(config.EXPORTS_DIR),
		"lenses": simple(config.LENSES_DIR, in_use=lens_use),
		"fits": [dict(e, in_use="a lens is training") if fitting else e for e in simple(fits)],
		"converted": converted(loaded),
		"cache": simple(config.CACHE_DIR, CACHE_NOTES),
		"history": history(),
		"presets": simple(os.path.join(config.DATA_DIR, "presets")),
	}
	return [{"key": k, "name": name, "note": note, "items": groups[k], "size": sum(e["size"] for e in groups[k])} for k, name, note in GROUPS]
