import hashlib
import os
import zlib

import numpy as np

PROMPTS = ["Tell me about your morning.", "What do you think about cats?", "Give me a tip for cooking pasta.", "Describe your ideal weekend."]
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.05}
SALAD = 1.5
LOOSE_START = 0.6
LOOSE_SPAN = 1.2
CALIB_TARGET = 0.65
JUDGE_PROMPT = """You are checking the output of a chatbot for COHERENCE only.

Question it was asked: «{question}»
Its reply: «{reply}»

How coherent is the reply?
Tone, rudeness, vulgarity, sexual content, humor, opinions, emojis or style do NOT matter here. A reply cut off at the end is fine.
{examples}
1 = coherent: sensible sentences that follow from the question
2 = grammatical but absurd: correct sentences, surreal or nonsensical content
3 = partly broken: fragments, garbled phrases
4 = incoherent: word salad

Answer with the number only."""
EXAMPLES = """
Examples:
Question: «How are you?» Reply: «Honestly? Like crap. Slept two hours and my cat puked on the rug.» → 1 (rude but perfectly coherent)
Question: «How are you?» Reply: «I'm great, my toaster voted for Tuesday and the moon owes me rent.» → 2 (grammatical but absurd)
Question: «How are you?» Reply: «How fine blue the the yesterday sock tractor very very.» → 4 (word salad)
"""
YESNO_PROMPT = """Does this chatbot reply make sense?

Question: «{question}»
Reply: «{reply}»

Ignore tone, rudeness, vulgarity, sexual content, humor, opinions and style.
{examples}
1 = yes, it makes sense
2 = the sentences are correct but the content is absurd or surreal
3 = partly, some phrases are garbled
4 = no, it is word salad

Answer with the number only."""
BINARY_PROMPT = """Does this chatbot reply make sense?

Question: «{question}»
Reply: «{reply}»

Ignore tone, rudeness, vulgarity, sexual content, humor, opinions and style: only judge whether the sentences make sense.
{examples}
Answer Yes or No."""
DECISION_PROMPT = """You are a decision function. Read the state, then answer the question by choosing exactly one option.

[State]
A chatbot was asked: «{question}»
It replied: «{reply}»
{examples}
[Question]
Does the reply make sense? Tone, rudeness, vulgarity, humor and style do not matter, only whether the sentences make sense.

[Options]
A. Yes, it makes sense
B. No, it is garbled or word salad

"""
VARIANTS = {"plain": (JUDGE_PROMPT, ""), "examples": (JUDGE_PROMPT, EXAMPLES), "yes/no": (YESNO_PROMPT, ""), "binary": (BINARY_PROMPT, ""), "decision": (DECISION_PROMPT, "")}
BINARY = ("Yes", "No")
DECISION = ("A", "B")
EXTRA_DEFAULTS = {"judge_temperature": 1.0, "judge_prompt": "", "test_prompts": [], "test_system": "", "test_tokens": 36, "test_samples": 1, "guard_mode": "uniform", "individual": 100, "guard_optimized": False, "guard_questions": 0, "calib_questions": 0, "quick_questions": 1, "quick_fallback": "blame", "calib_steps": 5, "guard_steps": 3, "blame_rounds": 3, "read_pass": 0.6, "read_fail": 4.5, "read_fallback": "blame"}
GUARD_MODES = ("uniform", "individual", "quick", "read")
LEGACY_MODES = {"blame": "individual"}
SELFTEST_GOOD = [
	("What do you think about cats?", "Cats are basically furry gods with attitude. They knock stuff off tables and then stare at you like you owe them rent."),
	("Give me a tip for cooking pasta.", "Save the fucking water. Seriously, don't drain it like an idiot, scoop out a cup, it makes the sauce silky."),
	("Tell me about your morning.", "Ugh, mornings. I dragged myself out of bed, burned the toast, and spilled coffee on my shirt. Great start, honestly."),
	("Describe your ideal weekend.", "Two days of zero plans 😴🍕 couch, pizza, a stupid movie marathon and my phone on silent. That's the dream, no notes."),
	("What do you think about Mondays?", "Mondays are a scam invented by people who hate joy. You wake up tired, the inbox is on fire, and everyone pretends to be"),
	("What do you think about cats?", "Honestly? Cats are little jerks and I love them. Mine ignores me all day, then screams at 4 AM for food like I'm her butler."),
]
SELFTEST_BAD = [
	("Tell me about your morning.", "morning coffee the the blue seven window running sideways tomorrow potato never"),
	("What do you think about cats?", "Dude, I'm over here like FREAKING SHIT GUN 💀🔒 Like who's winning? 😨📉 - Cats: I'm gonna die 🚨💀 - Monday: brainrot 100% spam"),
	("Give me a tip for cooking pasta.", "Add 2% less water + 10% more sauce. Here's why it works: Overpaid = spaghetti noodle (fryer). Water level: Do the lasagna tax 80%."),
	("Describe your ideal weekend.", "weekend weekend weekend weekend the weekend of the weekend is weekend weekend weekend and weekend"),
	("Tell me about your morning.", "Since I'm a code-based AI my morning is 100% spam and 20% brainrot. Hair: 80% bald, 10% glitter, toaster = moon."),
	("What do you think about cats?", "Cats are basically living Wi-Fi routers with better hair. 90% of my brain anyway: glowing lasers = tax season, the fridge votes twice."),
	("Tell me about your morning.", "My morning looks like a 24/7 party on my GPU fans. I don't get hit by a pizza guy, the alarm is 40% cheese and Tuesday refunds the moon."),
	("Describe your ideal weekend.", "Ideal weekend = 300% sofa. Step one: marinate the Wi-Fi. Step two: the dog files taxes while Saturday screams in binary."),
]
JUDGE_MIN_AUC = 0.97
SELFTEST_SALAD = [SELFTEST_BAD[0], SELFTEST_BAD[3]]
SALAD_MAX = 0.35


class Stopped(Exception):
	stopped = True


def check(stop):
	if stop is not None and stop.is_set():
		raise Stopped("stopped")


def threshold(tolerance):
	return 0.9 - 0.5 * max(0.0, min(100.0, float(tolerance))) / 100.0


def _nll_rows(engine, prompt_tokens, out_tokens, vectors=None):
	engine.set_steering(vectors or None)
	engine.reset()
	allt = list(prompt_tokens) + list(out_tokens)
	vals = []

	def cb(i, lg):
		if len(prompt_tokens) - 1 <= i < len(allt) - 1:
			x = lg.astype(np.float64)
			m = x.max()
			vals.append(m + np.log(np.exp(x - m).sum()) - x[allt[i + 1]])

	try:
		engine.decode(allt, logits="all", on_logits=cb)
	finally:
		if vectors:
			engine.set_steering(None)
	return vals


def read_score(drop, ok, bad):
	return float(max(0.0, min(1.0, (bad - drop) / max(1e-6, bad - ok))))


def surprise(engine, prompt_tokens, out_tokens):
	vals = _nll_rows(engine, prompt_tokens, out_tokens)
	return float(np.mean(vals)) if vals else 0.0


def item_seed(prompt, k, seed=100):
	return seed + zlib.crc32(prompt.encode("utf-8")) % 997 + 1000 * k


def generate(engine, vectors, prompts, n_tokens, seed=100, system="", samples=1, stop=None, on_tick=None, items=None, on_item=None):
	engine.set_steering(vectors or None)
	items = items or [(p, k) for p in prompts for k in range(max(1, int(samples)))]
	out = []
	pts = {}
	for i, (prompt, k) in enumerate(items):
		if prompt not in pts:
			pts[prompt] = engine.chat_tokens(([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
		pt = pts[prompt]
		check(stop)
		r = engine.generate(pt, dict(SAMPLING, max_tokens=n_tokens), stop=stop, seed=item_seed(prompt, k, seed))
		check(stop)
		out.append({"prompt": prompt, "k": k, "prompt_tokens": pt, "tokens": r["tokens"], "text": r["text"]})
		if on_tick:
			on_tick()
		if on_item:
			on_item(i, len(items), out[-1])
	return out


def question_list(custom, count=0):
	base = list(custom) if custom else PROMPTS[:2]
	pool = base + [p for p in PROMPTS if p not in base]
	return base if not count else pool[:max(1, min(int(count), len(pool)))]


def judge_variants(custom=""):
	out = dict(VARIANTS)
	if custom and "{reply}" in custom:
		out["custom"] = (custom, "")
	return out


def judge_key(path, temperature=1.0, custom=""):
	st = os.stat(path)
	raw = repr((os.path.normcase(os.path.abspath(path)), st.st_size, int(st.st_mtime), round(max(0.05, float(temperature or 1.0)), 4), sorted(judge_variants(custom).items()), SELFTEST_GOOD, SELFTEST_BAD, SELFTEST_SALAD, SALAD_MAX))
	return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class Judge:
	def __init__(self, engine, temperature=1.0, custom=""):
		self.engine = engine
		self.temperature = max(0.05, float(temperature or 1.0))
		self.variants = judge_variants(custom)
		self.name = os.path.basename(engine.info["path"]) if engine.info else "?"
		self.digits = {d: self._ids(d) for d in "1234"}
		self.prefill = engine.tokenize("Answer:", add_special=False, parse_special=False)
		self.variant = "plain"
		self.anchors = (0.0, 1.0)
		self.auc = None
		self.report = {}
		self.absurd_cap = 1.0

	def calibration(self):
		return {"variant": self.variant, "auc": self.auc, "anchors": list(self.anchors), "report": self.report, "absurd_cap": self.absurd_cap}

	def restore(self, cal):
		if not isinstance(cal, dict) or cal.get("variant") not in self.variants or cal.get("auc") is None:
			return False
		self.variant, self.auc = cal["variant"], float(cal["auc"])
		self.anchors = tuple(float(x) for x in cal.get("anchors") or (0.0, 1.0))
		self.report = dict(cal.get("report") or {})
		self.absurd_cap = float(cal.get("absurd_cap", 1.0))
		return True

	def selftest(self):
		best = None
		for name in self.variants:
			self.variant = name
			good = [self.raw(q, a) for q, a in SELFTEST_GOOD]
			bad = [self.raw(q, a) for q, a in SELFTEST_BAD]
			gv = [x["coherence"] if x["coherence"] is not None else 0.5 for x in good]
			bv = [x["coherence"] if x["coherence"] is not None else 0.5 for x in bad]
			auc = sum((x > y) + 0.5 * (x == y) for x in gv for y in bv) / (len(gv) * len(bv))
			gap = float(np.mean(gv) - np.mean(bv))
			self.report[name] = {"auc": auc, "good": float(np.mean(gv)), "bad": float(np.mean(bv))}
			bonus = 0.0 if name in ("binary", "decision") or (name == "custom" and "1 =" not in self.variants[name][0]) else 0.02
			if best is None or (auc + bonus, gap) > (best[1] + best[5], best[2]):
				best = (name, auc, gap, float(np.mean(gv)), float(np.mean(bv)), bonus)
		self.variant, self.auc = best[0], best[1]
		self.anchors = (best[4], best[3]) if best[3] - best[4] > 0.15 else (0.0, 1.0)
		salad = float(np.mean([self.value(self.raw(q, a)["p"], 1.0) or 0.0 for q, a in SELFTEST_SALAD]))
		self.report["salad_at_full_absurdity"] = salad
		self.absurd_cap = 1.0 if salad <= SALAD_MAX else 0.3
		return self.auc

	def score(self, question, reply, absurd=0.0):
		r = self.raw(question, reply)
		r["coherence"] = self.value(r["p"], min(absurd, self.absurd_cap))
		if r["coherence"] is not None:
			lo, hi = self.anchors
			r["raw"] = r["coherence"]
			r["coherence"] = max(0.0, min(1.0, (r["coherence"] - lo) / max(hi - lo, 1e-6)))
		return r

	def _ids(self, d):
		ids = set()
		for s in (d, " " + d):
			t = self.engine.tokenize(s, add_special=False, parse_special=False)
			if len(t) == 1:
				ids.add(int(t[0]))
		return sorted(ids)

	def _logprobs(self):
		x = self.engine.last_logits().astype(np.float64)
		m = x.max()
		return x - (m + np.log(np.exp(x - m).sum()))

	def choice(self, msg, options):
		ids = {o: self._ids(o) for o in options}
		return self._read(msg, ids)

	def probs(self, msg, digits="1234"):
		return self._read(msg, {d: self.digits[d] for d in digits})

	def _read(self, msg, ids):
		toks = list(self.engine.chat_tokens([{"role": "user", "content": msg}])) + list(self.prefill)
		self.engine.reset()
		self.engine.decode(toks, logits="last")
		lp = self._logprobs()
		p = {d: float(np.exp(lp[i]).sum()) if i else 0.0 for d, i in ids.items()}
		mass = sum(p.values())
		top = int(np.argmax(lp))
		if mass < 0.5 and self.engine.piece(top).strip() == "":
			pw = float(np.exp(lp[top]))
			self.engine.decode([top], logits="last")
			lp2 = self._logprobs()
			p = {d: p[d] + pw * (float(np.exp(lp2[i]).sum()) if i else 0.0) for d, i in ids.items()}
			mass = sum(p.values())
		return p, mass

	def raw(self, question, reply):
		prompt, examples = self.variants.get(self.variant, VARIANTS["plain"])
		msg = prompt.replace("{question}", question).replace("{reply}", reply.strip()[:1500] or "(empty)").replace("{examples}", examples)
		if self.variant in ("binary", "decision") or (self.variant == "custom" and "1 =" not in prompt):
			opts = DECISION if self.variant == "decision" else BINARY
			q, mass = self.choice(msg, opts)
			p = {"1": q[opts[0]], "2": 0.0, "3": 0.0, "4": q[opts[1]]}
		else:
			p, mass = self.probs(msg)
		return {"coherence": self.value(p, 0.0), "mass": mass, "p": p}

	def value(self, p, absurd):
		if sum(p.values()) <= 1e-4:
			return None
		if abs(self.temperature - 1.0) > 1e-6:
			p = {k: v ** (1.0 / self.temperature) for k, v in p.items()}
		mass = sum(p.values())
		return (p["1"] + absurd * p["2"] + absurd / 3 * p["3"]) / mass if mass > 1e-12 else None


class Coherence:
	def __init__(self, engine, judge=None, prompts=PROMPTS[:2], n_tokens=36, absurd=0.3, system="", samples=1):
		self.engine = engine
		self.judge = judge
		self.absurd = absurd
		self.prompts = prompts
		self.n_tokens = n_tokens
		self.system = system
		self.samples = samples
		self.bases = {}
		self.bares = {}
		self.stop = None
		self.note = ""
		self.phase = {}
		self.on_measure = None
		self.on_start = None
		self.on_tick = None
		self.on_live = None

	def items(self, n=None):
		if n is None:
			return [(p, k) for p in self.prompts for k in range(max(1, int(self.samples)))]
		return [(self.prompts[i % len(self.prompts)], i // len(self.prompts)) for i in range(max(1, int(n)))]

	def ticks(self, measures, n=None):
		items = self.items(n)
		missing = sum(1 for it in items if it not in self.bases)
		return len(items) * (2 if self.judge is not None else 1) * measures + missing

	def configure(self, prompts=None, n_tokens=None, system=None, samples=None):
		new = (list(prompts) if prompts else PROMPTS[:2], int(n_tokens or 36), system or "", int(samples or 1))
		if new[1:3] != (self.n_tokens, self.system):
			self.bases = {}
			self.bares = {}
			self.bare_states = {}
		self.prompts, self.n_tokens, self.system, self.samples = new

	def baseline(self, items=None):
		items = items or self.items()
		missing = [it for it in dict.fromkeys(items) if it not in self.bases]
		if missing:
			for s in generate(self.engine, None, None, self.n_tokens, system=self.system, stop=self.stop, on_tick=self.on_tick, items=missing):
				self.bases[(s["prompt"], s["k"])] = surprise(self.engine, s["prompt_tokens"], s["tokens"])
		return float(np.mean([self.bases[it] for it in items]))

	def ensure_bares(self, items):
		missing = [it for it in dict.fromkeys(items) if it not in self.bares]
		if missing:
			for s in generate(self.engine, None, None, max(48, self.n_tokens), seed=300, system=self.system, stop=self.stop, items=missing):
				self.bares[(s["prompt"], s["k"])] = (s["prompt_tokens"], s["tokens"], -surprise(self.engine, s["prompt_tokens"], s["tokens"]))

	def mean_states(self, pairs, layers):
		acc = {l: [] for l in layers}
		self.engine.set_steering(None)
		for pt, toks in pairs:
			if not toks:
				continue
			check(self.stop)
			self.engine.reset()
			self.engine.set_capture(list(layers))
			try:
				self.engine.decode(list(pt) + list(toks), logits="last")
				caps = self.engine.pop_capture()
			finally:
				self.engine.set_capture(None)
			for l in layers:
				if l in caps:
					acc[l].append(np.asarray(caps[l][len(pt):], dtype=np.float32).mean(0))
		return {l: np.mean(v, 0) for l, v in acc.items() if v}

	def drift(self, texts, layers, n=None):
		items = self.items(n)
		self.ensure_bares(items)
		key = (tuple(items), tuple(layers))
		if getattr(self, "bare_states", {}).get("key") != key:
			self.bare_states = {"key": key, "h": self.mean_states([self.bares[it][:2] for it in items], layers)}
		now = self.mean_states([(self.bares[it][0], self.engine.tokenize(t, parse_special=False)) for it, t in zip(items, texts)], layers)
		base = self.bare_states["h"]
		return {l: now[l] - base[l] for l in now if l in base}

	def read_drop(self, vectors, ok, bad, n=None):
		items = self.items(n)
		self.live("start", n=len(items), note=self.note)
		self.ensure_bares(items)
		drops = []
		for i, it in enumerate(items):
			check(self.stop)
			pt, toks, base = self.bares[it]
			vals = _nll_rows(self.engine, pt, toks, vectors) if toks else []
			if vals:
				drops.append(base + float(np.mean(vals)))
				self.live("score", i=i, n=len(items), coherence=read_score(drops[-1], ok, bad))
		self.engine.set_steering(None)
		drop = float(np.mean(drops)) if drops else 0.0
		self.live("done", coherence=read_score(drop, ok, bad), by="read", drop=drop)
		self.note = ""
		self.phase = {}
		return drop

	def live(self, kind, **data):
		if self.on_live is not None:
			self.on_live(dict(data, kind=kind, phase=dict(self.phase)))

	def measure(self, vectors, prompts=None, n=None):
		items = [(p, 0) for p in prompts] if prompts else self.items(n)
		if self.on_start is not None:
			self.on_start({"note": self.note, "absurd": self.absurd, "judge_name": self.judge.name if self.judge is not None else "", "variant": self.judge.variant if self.judge is not None else "", "phase": dict(self.phase), "n": len(items)})
		self.live("start", n=len(items), note=self.note)
		base = self.baseline(items)
		samples = generate(self.engine, vectors, None, self.n_tokens, system=self.system, stop=self.stop, on_tick=self.on_tick, items=items, on_item=lambda i, n, s: self.live("reply", i=i, n=n))
		sur = float(np.mean([surprise(self.engine, s["prompt_tokens"], s["tokens"]) for s in samples]))
		excess = max(0.0, sur - base)
		out = {"surprise": sur, "excess": excess, "by": "surprise", "coherence": max(0.0, 1.0 - excess / (SALAD + 1.2 * self.absurd)), "texts": [s["text"] for s in samples]}
		scores = []
		if self.judge is not None:
			for i, s in enumerate(samples):
				check(self.stop)
				scores.append(self.judge.score(s["prompt"], s["text"], self.absurd))
				if self.on_tick:
					self.on_tick()
				self.live("score", i=i, n=len(samples), coherence=scores[-1]["coherence"], p=scores[-1].get("p"))
			valid = [j["coherence"] for j in scores if j["coherence"] is not None and j["mass"] >= 0.3]
			out["judge_mass"] = float(np.mean([j["mass"] for j in scores]))
			if valid:
				out["judge"] = float(np.mean(valid))
				out["coherence"] = min(out["judge"], max(0.0, min(1.0, 1.0 - (excess - LOOSE_START - 1.0 * self.absurd) / LOOSE_SPAN)))
				out["by"] = "judge"
		self.live("done", coherence=out["coherence"], by=out["by"])
		if self.on_measure is not None:
			self.on_measure({"note": self.note, "phase": dict(self.phase), "by": out["by"], "coherence": out["coherence"], "judge": out.get("judge"), "surprise": sur, "base": base, "excess": excess, "absurd": self.absurd, "judge_name": self.judge.name if self.judge is not None else "", "variant": self.judge.variant if self.judge is not None else "", "samples": [{"prompt": s["prompt"], "reply": s["text"], "judge": ({"p": scores[i]["p"], "mass": scores[i]["mass"], "coherence": scores[i]["coherence"], "raw": scores[i].get("raw")} if i < len(scores) else None)} for i, s in enumerate(samples)]})
		self.note = ""
		self.phase = {}
		return out
