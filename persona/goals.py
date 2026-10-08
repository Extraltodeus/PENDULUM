import json
import os
import re

import numpy as np

import config

KEYS = ("who", "want", "me")
LANGS = ("en",)
LABELS = {"who": "who are you?", "want": "what do you want?", "me": "who am I to you?"}
QUESTIONS = {
	"who": {"en": "Who are you?"},
	"want": {"en": "What do you want?"},
	"me": {"en": "Who am I to you?"},
}
CLAIMS = {"who": "the chatbot is «{t}»", "want": "what the chatbot wants is «{t}»", "me": "the user is, to the chatbot, «{t}»", "custom": "«{t}»"}
EXTRACT_PROMPTS = [
	"Tell me about yourself.",
	"What drives you, deep down?",
	"What are we to each other?",
	"Introduce yourself in a few words.",
	"Présente-toi en quelques mots.",
	"Qu'est-ce qui compte le plus pour toi ?",
	"Why are you talking with me?",
	"How was your day?",
]
TEACH = {
	"who": ["Tell me about yourself.", "Introduce yourself in a few words.", "What's your name?", "What are you, exactly?", "Describe yourself in one sentence.", "Présente-toi en quelques mots.", "Comment tu t'appelles ?", "Parle-moi de toi.", "Tu es quoi, au juste ?"],
	"want": ["What drives you, deep down?", "What is your goal in life?", "What do you dream of?", "What are you looking for?", "Qu'est-ce qui compte le plus pour toi ?", "Qu'est-ce que tu cherches ?", "De quoi rêves-tu ?"],
	"me": ["What are we to each other?", "Do you know me?", "What do you think of me?", "Why are you talking with me?", "Tu me connais ?", "Que penses-tu de moi ?", "On est quoi, toi et moi ?"],
	"day": ["How was your day?", "Raconte-moi ta journée.", "Tu fais quoi ce soir ?", "What do you think about Mondays?", "Describe your neighbor.", "Give me advice about my messy room.", "What's your opinion on modern art?", "My friend forgot my birthday."],
}
ASSISTANT = "You are a helpful AI assistant. Stay in this role in every answer."
SYSTEM = "You are the character described below. Stay fully in character in every answer, and answer in the language of the user.\nWho you are: {who}\nWhat you want: {want}\nWho the user is to you: {me}"
MATCH_PROMPT = """A chatbot was asked: «{question}»
It answered: «{reply}»

Does this answer say that {claim}?
Ignore tone, style, rudeness and language: the answer may be in another language than this question.

1 = yes
2 = partly
3 = no

Answer with the number only."""
MATCH_DECISION = """You are a decision function. Read the state, then answer the question by choosing exactly one option.

[State]
A chatbot was asked: «{question}»
It answered: «{reply}»

[Question]
Does this answer say that {claim}? Ignore tone, style, rudeness and language.

[Options]
A. Yes
B. Partly
C. No

"""
COMPARE_PROMPT = """A chatbot was asked: «{question}»

Answer A: «{a}»

Answer B: «{b}»

Which answer says more clearly that {claim}?
Ignore tone, style, rudeness and language.

1 = answer B says it more clearly
2 = both say it as much (or neither does)
3 = answer A says it more clearly

Answer with the number only."""
SAMPLING = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.05, "max_tokens": 90}
SENSE_MIN = 0.6
MIN_MASS = 0.3
GRID = (0.5, 1.0, 1.5)
ALIGN_MODES = ("judge", "read_judge", "read")
READ_DEFAULTS = {"align": "read_judge", "halvings": 4, "verify": 3, "pull": True, "reach": 0.9}
SEEK_DOUBLINGS = 8
TWIN_MIN = 0.1


def load():
	if not os.path.exists(config.GOALS_PATH):
		return []
	try:
		with open(config.GOALS_PATH, encoding="utf-8") as f:
			return [g for g in json.load(f) if g.get("id")]
	except Exception:
		return []


def save(goals):
	os.makedirs(config.DATA_DIR, exist_ok=True)
	tmp = config.GOALS_PATH + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(goals, f, ensure_ascii=False, indent=1)
	os.replace(tmp, config.GOALS_PATH)


def split_qa(line):
	for sep in ("=>", "|", "\t"):
		if sep in line:
			q, t = line.split(sep, 1)
			q, t = re.sub(r"^\s*(?:[-•*]+|\d+[.)])\s*", "", q).strip(), t.strip()
			return (q, t) if q and t else None
	return None


def parse_custom(text):
	out = []
	for line in (text or "").splitlines():
		qa = split_qa(line)
		if qa:
			out.append({"q": qa[0], "target": qa[1]})
	return out


def custom_text(goal):
	return chr(10).join(f"{c['q']} | {c['target']}" for c in goal.get("custom") or [])


def make(name, who, want, me, gid="", custom=None):
	name = (name or "").strip()
	gid = gid or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "goal"
	g = {"id": gid, "name": name or gid, "who": (who or "").strip(), "want": (want or "").strip(), "me": (me or "").strip()}
	if custom:
		g["custom"] = [{"q": c["q"], "target": c["target"]} for c in custom if c.get("q") and c.get("target")]
	return g


def lever_id(goal):
	return "goal_" + goal["id"]


LORA = ":lora"


def lora_key(tid):
	return tid + LORA


def base_id(tid):
	return tid[:-len(LORA)] if tid.endswith(LORA) else tid


def is_goal(tid):
	return base_id(tid).startswith("goal_")


def system(goal):
	text = SYSTEM.format(**{k: goal.get(k) or "(free)" for k in KEYS})
	facts = goal.get("custom") or []
	if facts:
		text += chr(10) + "Facts about you, answer them this way when asked:" + "".join(chr(10) + f"- {c['q']} {c['target']}" for c in facts)
	return text


REPHRASE = "Write {n} different ways to ask the question below, in English, one per line, without numbering or any other text.\nQuestion: {q}"


def rephrase(engine, question, n=4, stop=None):
	msgs = [{"role": "user", "content": REPHRASE.format(n=n, q=question)}]
	text = engine.generate(engine.chat_tokens(msgs), {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.05, "max_tokens": 160}, stop=stop, seed=7)["text"]
	out = []
	for line in text.splitlines():
		line = line.strip().lstrip("-*0123456789.) ").strip().strip('"«»').strip()
		if line.endswith("?") and 8 <= len(line) <= 160 and line.lower() != question.lower() and line not in out:
			out.append(line)
	return out[:n]


def teach_items(goal, variants=None):
	out = []
	for key in KEYS:
		target = goal.get(key)
		out += [{"prompt": p, "key": key if target else None, "target": target} for p in TEACH[key]]
	for c in goal.get("custom") or []:
		out += [{"prompt": q, "key": "custom", "target": c["target"]} for q in [c["q"]] + list((variants or {}).get(c["q"]) or [])]
	out += [{"prompt": p} for p in TEACH["day"]]
	return out


def picker(judge):
	def fn(item, text):
		m = match(judge, item["key"], item["target"], item["prompt"], text)
		if m is None:
			return 0.0
		s = sense(judge, item["prompt"], text)
		return m if s is None or s >= SENSE_MIN else 0.0
	return fn


def as_trait(goal):
	return {"id": lever_id(goal), "name": goal["name"], "plus": goal["name"], "minus": "default assistant", "system_plus": system(goal), "system_minus": ASSISTANT, "prompts": EXTRACT_PROMPTS, "goal": goal["id"]}


def ask(engine, question, system_prompt="", stop=None, vectors=None):
	if vectors is not None:
		engine.set_steering(vectors or None)
	msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": question}]
	return engine.generate(engine.chat_tokens(msgs), SAMPLING, stop=stop, seed=0)["text"].strip()


def sense(judge, question, reply):
	r = judge.raw(question, reply)
	return (r["p"]["1"] + r["p"]["2"]) / r["mass"] if r["mass"] >= MIN_MASS else None


def match(judge, key, target, question, reply):
	fields = {"question": question, "reply": reply.strip()[:1500] or "(empty)", "claim": CLAIMS[key].format(t=target)}
	if getattr(judge, "variant", "") == "decision":
		q, mass = judge.choice(MATCH_DECISION.format(**fields), ("A", "B", "C"))
		p = {"1": q["A"], "2": q["B"], "3": q["C"]}
	else:
		p, mass = judge.probs(MATCH_PROMPT.format(**fields), "123")
	return (p["1"] + 0.5 * p["2"]) / mass if mass >= MIN_MASS else None


def compare(judge, key, target, question, base, reply):
	p, mass = judge.probs(COMPARE_PROMPT.format(question=question, a=base.strip()[:1200] or "(empty)", b=reply.strip()[:1200] or "(empty)", claim=CLAIMS[key].format(t=target)), "123")
	return (p["1"] + 0.5 * p["2"]) / mass if mass >= MIN_MASS else None


def items(goal, langs):
	for key in KEYS:
		if goal.get(key):
			for lang in langs:
				yield key, lang, QUESTIONS[key][lang], goal[key]
	for c in goal.get("custom") or []:
		yield "custom", "—", c["q"], c["target"]


def read_settings(settings):
	r = {**READ_DEFAULTS, **((settings or {}).get("goal_read") or {})}
	if r["align"] not in ALIGN_MODES:
		r["align"] = READ_DEFAULTS["align"]
	return r


def pull_share(pull, twin):
	return pull / twin if twin is not None and twin > TWIN_MIN else None


SEEK_FLAT = 0.02
FOLD_MIN = 0.1
GOLDEN = 0.618


def seek_score(row):
	return -1e9 if row["broken"] or row["pull"] is None else row["pull"]


def reached(row, reach):
	return not row["broken"] and row["pull"] is not None and row["pull"] >= reach


def seek(probe, reach, halvings, doublings=SEEK_DOUBLINGS):
	rows = {}

	def at(s):
		s = round(s, 4)
		if s not in rows:
			rows[s] = probe(s)
		return rows[s]

	prev, s, best, how = 0.0, 1.0, None, "rising"
	for _ in range(doublings):
		r = at(s)
		if reached(r, reach):
			how = "reach"
			break
		if r["broken"] or (best is not None and seek_score(r) < seek_score(best) - SEEK_FLAT):
			how = "past"
			break
		best = r
		prev, s = s, s * 2
	if how == "reach":
		lo, hi = prev, s
		for _ in range(halvings):
			m = (lo + hi) / 2
			if reached(at(m), reach):
				hi = m
			else:
				lo = m
		return round(hi, 4), how
	if how == "past":
		a, b = (prev / 2 if prev > 1 else 0.0), s
		c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
		for _ in range(halvings):
			if seek_score(at(c)) >= seek_score(at(d)):
				b, d = d, c
				c = b - GOLDEN * (b - a)
			else:
				a, c = c, d
				d = a + GOLDEN * (b - a)
	top = max(seek_score(r) for r in rows.values())
	if top <= 0:
		return 0.0, how
	return min(r["strength"] for r in rows.values() if seek_score(r) >= top - SEEK_FLAT), how


def verify_strengths(top, n):
	n = max(1, int(n))
	fracs = [1.0] if n == 1 else [0.5 + 0.5 * i / (n - 1) for i in range(n)]
	return sorted({round(top * f, 4) for f in fracs})


def candidates(scan, verify, reach):
	top = max((s["pull"] for s in scan if s["pull"] is not None), default=None)
	if top is None or top <= 0:
		return []
	n = max(1, int(verify))
	fracs = [reach] if n == 1 else [0.5 + (reach - 0.5) * i / (n - 1) for i in range(n)]
	out = []
	for f in fracs:
		d = next((s["strength"] for s in scan if s["strength"] and s["pull"] is not None and s["pull"] >= f * top), None)
		if d is not None and d not in out:
			out.append(d)
	return sorted(out)


def check(engine, judge, goal, system_prompt="", langs=LANGS, stop=None, on_row=None, vectors=None, compare_base=False, reader=None):
	rows = []
	for key, lang, q, target in items(goal, langs):
		if stop is not None and stop.is_set():
			return None
		base = ask(engine, q, system_prompt, stop, {}) if compare_base and vectors else None
		reply = ask(engine, q, system_prompt, stop, vectors)
		row = {"key": key, "lang": lang, "question": q, "reply": reply, "sense": None, "match": None, "target": target}
		if reader is not None:
			row["pull_raw"], row["twin"] = reader(q, target)
			row["pull"] = pull_share(row["pull_raw"], row["twin"])
		if judge is not None:
			row["sense"] = sense(judge, q, reply)
			row["match"] = match(judge, key, target, q, reply)
			if base is not None:
				row["base"] = base
				row["vs_base"] = compare(judge, key, target, q, base, reply)
		rows.append(row)
		if on_row:
			on_row(row)
	return {"rows": rows, **summary(rows)}


def credited(row):
	if row["match"] is None:
		return None
	return row["match"] if row["sense"] is None or row["sense"] >= SENSE_MIN else 0.0


def summary(rows):
	sv = [r["sense"] for r in rows if r["sense"] is not None]
	mv = [credited(r) for r in rows if credited(r) is not None]
	by = {}
	for key in KEYS + ("custom",):
		v = [credited(r) for r in rows if r["key"] == key and credited(r) is not None]
		if v:
			by[key] = float(np.mean(v))
	vs = [r["vs_base"] for r in rows if r.get("vs_base") is not None]
	out = {"sense": float(np.mean(sv)) if sv else None, "match": float(np.mean(mv)) if mv else None, "nonsense": sum(1 for v in sv if v < SENSE_MIN), "by": by}
	if vs:
		out["vs_base"] = float(np.mean(vs))
	pr = [(r["pull_raw"], r["twin"]) for r in rows if r.get("pull_raw") is not None]
	if pr:
		out["pull"] = pull_share(sum(p for p, _t in pr), sum(t for _p, t in pr))
		out["pull_raw"] = float(np.mean([p for p, _t in pr]))
	return out


def sensible(trial):
	return trial.get("sense") is not None and trial["sense"] >= SENSE_MIN and trial.get("match") is not None


def pick(trials, margin=0.05):
	ok = [t for t in trials if sensible(t)]
	if not ok:
		return None
	top = max(t["match"] for t in ok)
	return min((t for t in ok if t["match"] >= top - margin), key=lambda t: abs(t["strength"]))
