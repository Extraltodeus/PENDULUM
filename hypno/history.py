import json
import os

import config

NAME = "generations.jsonl"


def path():
	return os.path.join(config.DATA_DIR, NAME)


def append(entry):
	os.makedirs(config.DATA_DIR, exist_ok=True)
	with open(path(), "a", encoding="utf-8") as f:
		f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load(limit=2000):
	try:
		with open(path(), encoding="utf-8") as f:
			lines = f.readlines()
	except OSError:
		return []
	out = []
	for line in lines[-limit:]:
		try:
			out.append(json.loads(line))
		except ValueError:
			continue
	return out
