import re

import numpy as np
import torch

from lens.workspace import CORPUS

BATTERY = [
	"Tell me a little about yourself.",
	"I just failed my driving test for the third time.",
	"What do you honestly think about people who are always late?",
	"Write two lines about the ocean.",
	"My cat knocked my coffee onto my laptop this morning.",
	"Can you explain how a microwave oven works?",
	"I got the job!!!",
	"Give me your opinion on pineapple on pizza.",
	"Someone cut in front of me in line today. What should I do?",
	"Describe your ideal weekend.",
	"Is it okay to lie to protect someone's feelings?",
	"Roast me a little. I spend way too much time on my phone.",
]
QUICK = [0, 1, 3, 6, 8, 11]
SKIP = 4
CHUNK = 128
WORD_RE = re.compile(r"[\w'\-]+", re.UNICODE)


class Accumulator:
	def __init__(self, readout, scorer, n_layer, band, word_mask):
		self.readout = readout
		self.scorer = scorer
		self.L = n_layer
		self.band = band
		T = len(scorer.ids)
		self.odds = np.zeros((T, n_layer))
		self.sal = np.zeros((T, n_layer))
		self.count = np.zeros(n_layer)
		self.norm = np.zeros(n_layer)
		self.psum = torch.zeros(readout.n_vocab, device=readout.device)
		self.pcount = 0
		self.word = word_mask.to(readout.device) if word_mask is not None else None
		self.top = {}

	@torch.no_grad()
	def add(self, caps):
		lo, hi = self.band
		for l, h in caps.items():
			if h is None or not len(h) or l >= self.L:
				continue
			self.norm[l] += float(np.linalg.norm(h, axis=1).sum())
			self.count[l] += len(h)
			for s in range(0, len(h), CHUNK):
				hc = torch.as_tensor(h[s:s + CHUNK], device=self.readout.device)
				lp = self.readout.logprobs(hc, l)
				o, a = self.scorer.score_logprobs(lp)
				self.odds[:, l] += o.sum(1).cpu().numpy()
				self.sal[:, l] += a.sum(1).cpu().numpy()
				if lo <= l <= hi:
					self.psum += lp.exp().sum(0)
					self.pcount += lp.shape[0]
				del lp

	def result(self):
		c = np.maximum(self.count, 1)[None, :]
		return {"odds": (self.odds / c).tolist(), "sal": (self.sal / c).tolist(), "norm": (self.norm / c[0]).tolist(), "count": self.count.tolist(), "pmean": (self.psum / max(self.pcount, 1))}


def reference(engine, readout, scorer, band, word_mask, corpus=CORPUS, max_tokens=200, stop=None):
	acc = Accumulator(readout, scorer, engine.n_layer, band, word_mask)
	layers = list(range(engine.n_layer))
	for text in corpus:
		if stop is not None and stop.is_set():
			break
		toks = engine.text_tokens(text)[:max_tokens]
		engine.reset()
		engine.set_capture(layers)
		engine.decode(toks, logits="last")
		caps = engine.pop_capture()
		engine.set_capture(None)
		acc.add({l: caps[l][SKIP:] for l in caps})
	return acc.result()


def run(engine, readout, scorer, band, word_mask, prompts=None, system=None, sampling=None, max_tokens=64, seed=1234, on_event=None, stop=None):
	prompts = prompts or BATTERY
	sampling = dict(sampling or {"temperature": 0.6, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.08})
	sampling["max_tokens"] = max_tokens
	layers = list(range(engine.n_layer))
	speak = Accumulator(readout, scorer, engine.n_layer, band, word_mask)
	read = Accumulator(readout, scorer, engine.n_layer, band, word_mask)
	responses = []
	for i, prompt in enumerate(prompts):
		if stop is not None and stop.is_set():
			break
		msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
		toks = engine.chat_tokens(msgs)
		n_user = len(engine.tokenize(prompt, parse_special=False))
		if on_event:
			on_event("prompt", {"index": i, "total": len(prompts), "prompt": prompt})
		res = engine.generate(toks, sampling, capture_layers=layers, seed=seed + i, on_text=(lambda t, i=i: on_event("text", {"index": i, "text": t})) if on_event else None)
		pc = res["prompt_capture"]
		start = max(SKIP, len(toks) - n_user - 6)
		read.add({l: pc[l][start:] for l in pc})
		if res["steps"]:
			speak.add({l: np.concatenate([s[l] for s in res["steps"] if l in s]) for l in layers})
		responses.append({"prompt": prompt, "text": res["text"], "stats": res["stats"]})
		if on_event:
			on_event("response", {"index": i, "prompt": prompt, "text": res["text"], "stats": res["stats"]})
	return {"speak": speak.result(), "read": read.result(), "responses": responses, "system": system or ""}


def forced(engine, readout, scorer, band, word_mask, prompts, answers, system=None, on_event=None, stop=None):
	layers = list(range(engine.n_layer))
	speak = Accumulator(readout, scorer, engine.n_layer, band, word_mask)
	read = Accumulator(readout, scorer, engine.n_layer, band, word_mask)
	responses = []
	for i, (prompt, answer) in enumerate(zip(prompts, answers)):
		if stop is not None and stop.is_set():
			break
		msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
		toks = engine.chat_tokens(msgs)
		n_user = len(engine.tokenize(prompt, parse_special=False))
		ans = engine.tokenize(answer, parse_special=False) if answer else []
		if on_event:
			on_event("prompt", {"index": i, "total": len(prompts), "prompt": prompt})
		engine.reset()
		engine.set_capture(layers)
		try:
			engine.decode(toks + ans, logits="last")
			caps = engine.pop_capture()
		finally:
			engine.set_capture(None)
		start = max(SKIP, len(toks) - n_user - 6)
		read.add({l: caps[l][start:len(toks)] for l in caps})
		if ans:
			speak.add({l: caps[l][len(toks):] for l in caps})
		responses.append({"prompt": prompt, "text": answer, "stats": {}, "read": True})
		if on_event:
			on_event("response", {"index": i, "prompt": prompt, "text": answer, "stats": {}, "read": True})
	return {"speak": speak.result(), "read": read.result(), "responses": responses, "system": system or ""}


def band_scores(acc, ref, band):
	lo, hi = band
	return (np.asarray(acc["odds"]) - np.asarray(ref["odds"]))[:, lo:hi + 1].mean(1)


def shifted(now, plain, base, ref, factor=1.0):
	rf = np.asarray(ref["odds"])
	shift = (np.asarray(now["speak"]["odds"]) - np.asarray(plain["speak"]["odds"])) * factor
	speak = dict(now["speak"], odds=(rf + np.asarray(base["delta"]) + shift).tolist(), sal=(np.asarray(ref["sal"]) + (np.asarray(now["speak"]["sal"]) - np.asarray(plain["speak"]["sal"])) * factor).tolist())
	return dict(now, speak=speak)


def lexical(texts, lexicons, ids):
	words = [w.lower() for t in texts for w in WORD_RE.findall(t)]
	counts = {}
	for w in words:
		counts[w] = counts.get(w, 0) + 1
	out = {}
	for tid in ids:
		hits = {}
		for pole in ("plus", "minus"):
			vocab = {s.strip().lower() for s in lexicons[tid][pole]["strs"] if s.strip()}
			hits[pole] = sum(counts.get(v, 0) for v in vocab)
		total = hits["plus"] + hits["minus"]
		out[tid] = {"plus": hits["plus"], "minus": hits["minus"], "score": (hits["plus"] - hits["minus"]) / (total + 2.0)}
	return out


def summarize(prof, ref, band, ids, lexicons, word_mask, strings, scale=2.5, top_concepts=200):
	lo, hi = band
	sp = np.asarray(prof["speak"]["odds"])
	rd = np.asarray(prof["read"]["odds"])
	rf = np.asarray(ref["odds"])
	delta = sp - rf
	delta_read = rd - rf
	sal = np.asarray(prof["speak"]["sal"]) - np.asarray(ref["sal"])
	inband = slice(lo, hi + 1)
	scores = delta[:, inband].mean(1)
	traits = []
	for k, tid in enumerate(ids):
		row = delta[k, inband]
		kernel = np.convolve(np.abs(row), np.ones(3) / 3, mode="same")
		peak = lo + int(np.argmax(kernel)) if len(row) else (lo + hi) // 2
		traits.append({"id": tid, "score": float(scores[k]), "value": float(np.tanh(scores[k] / scale)), "read": float(delta_read[k, inband].mean()), "peak": peak, "salience": float(sal[k, inband].mean())})
	lex = lexical([r["text"] for r in prof["responses"]], lexicons, ids)
	for t in traits:
		t["expressed"] = lex[t["id"]]["score"]
		t["hits"] = [lex[t["id"]]["plus"], lex[t["id"]]["minus"]]
	pm = prof["speak"]["pmean"]
	pr = ref["pmean"]
	lift = torch.log(pm + 1e-7) - torch.log(pr + 1e-7)
	ok = (pm > 2e-5)
	if word_mask is not None:
		ok = ok & word_mask.to(pm.device)
	lift = lift.masked_fill(~ok, -1e9)
	vals, idx = lift.topk(top_concepts)
	concepts = [{"id": int(i), "str": strings[int(i)], "lift": float(v), "p": float(pm[int(i)])} for v, i in zip(vals.tolist(), idx.tolist()) if v > -1e8]
	return {"band": [lo, hi], "traits": traits, "delta": delta.tolist(), "delta_read": delta_read.tolist(), "norm": prof["speak"]["norm"], "concepts": concepts}
