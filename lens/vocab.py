import re

import torch

_ENGLISH_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")


def wordlike(raw):
	s = raw.strip()
	if not s or "<|" in s or (s.startswith("<") and s.endswith(">")) or s.startswith("[") and s.endswith("]"):
		return False
	if s.isascii():
		return len(s) > 2 and s[0].isalpha() and all(c.isalpha() or c in "'-" for c in s)
	return all(ch.isalnum() for ch in s)


def masks(strings):
	word = torch.zeros(len(strings), dtype=torch.bool)
	english = torch.zeros(len(strings), dtype=torch.bool)
	for i, raw in enumerate(strings):
		if not wordlike(raw):
			continue
		word[i] = True
		s = raw.strip()
		english[i] = len(s) >= 3 and bool(_ENGLISH_RE.match(s))
	return word, english


def variants(word):
	w = word.strip()
	forms = {w, w.lower(), w.capitalize()}
	return [f" {f}" for f in forms] + list(forms)


def single_token_ids(engine, word):
	ids = []
	for form in variants(word):
		toks = engine.tokenize(form, add_special=False, parse_special=False)
		if len(toks) == 1 and toks[0] not in ids:
			ids.append(toks[0])
	return ids


def display(s):
	return s.replace(" ", "˽") if s.strip() != s or not s else s
