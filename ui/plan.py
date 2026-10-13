import json
import os
import time

SETTINGS = {"levers", "align", "system", "active", "load", "unload", "exempt", "apply_settings", "judge"}
INFO = {"measure", "read_scale"}
CHAT = {"chat"}
INSTANT = {"levers_as_set", "rebuild", "drop", "preset_load", "add_trait", "trait_edit", "goal_as_lever", "reset_chat", "rewind_chat", "chat_pick"}
DEFAULT_ETA = {"levers_as_set": 5, "read_scale": 14, "levers": 30, "judge": 30, "align": 150, "measure": 45, "profile": 120, "calibrate": 50, "recalibrate": 50, "map": 40, "chat": 25, "load": 20, "unload": 3, "goal_check": 60, "goal_align": 240, "goal_optimize": 900, "export": 60, "rebuild": 15, "active": 10, "system": 1, "exempt": 10, "goal_as_lever": 10, "drop": 10, "apply_settings": 30, "add_trait": 40, "trait_edit": 15, "preset_load": 5, "reset_chat": 1, "rewind_chat": 1, "chat_pick": 1}
KEEP_DONE = 60


class Eta:
	def __init__(self, path):
		self.path = path
		self.data = {}
		self.dirty = False
		try:
			with open(path, encoding="utf-8") as f:
				self.data = json.load(f)
		except (OSError, ValueError):
			self.data = {}

	def key(self, kind, model):
		return f"{kind}|{os.path.basename(model or '') or '*'}"

	def get(self, kind, model, units=1.0):
		v = self.data.get(self.key(kind, model))
		if v is not None:
			return float(v) * units, True
		v = self.data.get(self.key(kind, ""))
		if v is not None:
			return float(v) * units, True
		d = DEFAULT_ETA.get(kind)
		return (d * units, False) if d else (None, False)

	def record(self, kind, model, seconds, units=1.0):
		per = float(seconds) / max(float(units), 1e-6)
		for k in {self.key(kind, model), self.key(kind, "")}:
			old = self.data.get(k)
			self.data[k] = round(per if old is None else 0.6 * float(old) + 0.4 * per, 2)
		self.dirty = True

	def save(self):
		if not self.dirty:
			return
		try:
			os.makedirs(os.path.dirname(self.path), exist_ok=True)
			tmp = self.path + ".tmp"
			with open(tmp, "w", encoding="utf-8") as f:
				json.dump(self.data, f, indent=1, sort_keys=True)
			os.replace(tmp, self.path)
			self.dirty = False
		except OSError:
			pass


class Plan:
	def __init__(self, eta):
		self.items = []
		self.seq = 0
		self.eta = eta
		self.auto = False
		self.going = False
		self.until = None
		self.current = None

	def get(self, iid):
		return next((i for i in self.items if i["id"] == iid), None)

	def pending(self):
		return [i for i in self.items if i["state"] == "pending"]

	def running(self):
		return self.get(self.current) if self.current is not None else None

	def add(self, kind, label, run, params=None, model="", units=1.0, at=None, on_drop=None):
		self.seq += 1
		eta, known = self.eta.get(kind, model, units)
		item = {"id": self.seq, "kind": kind, "label": label, "run": run, "params": dict(params or {}), "model": model, "units": units, "state": "pending", "eta": eta, "known": known, "t_add": time.time(), "t0": None, "t1": None, "progress": None, "detail": "", "error": "", "on_drop": on_drop, "info": kind in INFO, "instant": kind in INSTANT}
		if at is None and kind in INSTANT:
			first = next((i for i in self.pending() if not i["instant"]), None)
			at = self.index(first) if first is not None else None
		if at is None:
			self.items.append(item)
		else:
			self.items.insert(at, item)
		return item

	def index(self, item):
		return self.items.index(item)

	def chat_slot(self):
		pend = self.pending()
		last = None
		for it in pend:
			if not it["info"]:
				last = it
		if last is None:
			return self.front_slot()
		return self.index(last) + 1

	def front_slot(self):
		first = next((i for i in self.pending() if not i["instant"]), None)
		return self.index(first) if first is not None else len(self.items)

	def mergeable(self, kind):
		pend = self.pending()
		for k in range(len(pend) - 1, -1, -1):
			it = pend[k]
			if it["kind"] == kind:
				return it
			if not it["info"]:
				return None
		return None

	def after(self, item, kind):
		k = self.index(item)
		for it in self.items[k + 1:]:
			if it["state"] != "pending":
				continue
			if it["kind"] == kind:
				return it
			if not it["info"]:
				return None
		return None

	def remove(self, iid):
		it = self.get(iid)
		if it is None or it["state"] != "pending":
			return None
		self.items.remove(it)
		if self.until == iid:
			self.until = None
		return it

	def next_item(self):
		pend = self.pending()
		if not pend:
			return None
		quick = next((i for i in pend if i["instant"]), None)
		if quick is not None:
			return quick
		if self.going:
			return pend[0]
		if self.until is not None:
			if self.until in [i["id"] for i in pend]:
				return pend[0]
			self.until = None
		if self.auto:
			return next((i for i in pend if not i.get("held")), None)
		return None

	def start(self, item):
		item["state"] = "running"
		item["t0"] = time.time()
		item["progress"] = None
		self.current = item["id"]

	def finish(self, item, state, error=""):
		item["state"] = state
		item["t1"] = time.time()
		item["error"] = error
		if self.current == item["id"]:
			self.current = None
		if state == "done" and item["t0"]:
			self.eta.record(item["kind"], item["model"], item["t1"] - item["t0"], item["units"])
			self.eta.save()
		if self.until == item["id"]:
			self.until = None
		if not self.pending():
			self.going = False
			self.until = None
		done = [i for i in self.items if i["state"] not in ("pending", "running")]
		for old in done[:max(0, len(done) - KEEP_DONE)]:
			self.items.remove(old)

	def requeue(self, item):
		item["state"] = "pending"
		item["t0"] = None
		item["progress"] = None
		if self.current == item["id"]:
			self.current = None

	def move(self, iid, before=None):
		it = self.get(iid)
		if it is None or it["state"] != "pending" or iid == before:
			return False
		self.items.remove(it)
		ref = self.get(before) if before is not None else None
		if ref is None:
			self.items.append(it)
		elif ref["state"] == "pending":
			self.items.insert(self.index(ref), it)
		else:
			first = next((i for i in self.items if i["state"] == "pending"), None)
			self.items.insert(self.index(first) if first is not None else len(self.items), it)
		return True

	def remaining(self, item, now=None):
		now = now or time.time()
		if item["state"] == "running" and item["t0"]:
			spent = now - item["t0"]
			p = item.get("progress")
			if p and p > 0.04:
				return max(0.0, (now - (item.get("pt0") or item["t0"])) * (1.0 - p) / p)
			if item["eta"] is not None:
				return max(0.0, item["eta"] - spent)
			return None
		return item["eta"] if item["state"] == "pending" else 0.0

	def total(self, now=None):
		todo = [i for i in self.items if i["state"] in ("pending", "running")]
		if not todo:
			return 0.0, True
		parts = [self.remaining(i, now) for i in todo]
		known = all(p is not None for p in parts) and all(i["known"] for i in todo)
		return sum(p for p in parts if p is not None), known
