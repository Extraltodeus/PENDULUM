import os
import queue
import sys
import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal


def leave(code, worker):
	if worker.alive():
		for f in (sys.stdout, sys.stderr):
			if f is not None:
				f.flush()
		os._exit(code)
	sys.exit(code)


class Worker(QObject):
	event = Signal(str, object)

	def __init__(self):
		super().__init__()
		self.cv = threading.Condition()
		self.queue = []
		self.seq = 0
		self.quitting = False
		self.stop = threading.Event()
		self.cancel = threading.Event()
		self.current = None
		self.session = None
		self.ready = threading.Event()
		self.lights = queue.Queue()
		self.thread = QThread()
		self.moveToThread(self.thread)
		self.thread.started.connect(self._loop)
		self.thread.start()
		threading.Thread(target=self._light_loop, daemon=True).start()

	def emit(self, kind, payload):
		self.event.emit(kind, payload)

	def light(self, name, fn, label=""):
		with self.cv:
			self.seq += 1
			jid = self.seq
		self.lights.put({"id": jid, "name": name, "label": label or name, "fn": fn})
		return jid

	def _light_loop(self):
		self.ready.wait()
		while True:
			job = self.lights.get()
			if job is None or self.session is None:
				return
			try:
				result = job["fn"](self.session)
				self.emit("light", {"name": job["name"], "id": job["id"], "result": result})
			except Exception as exc:
				self.emit("light", {"name": job["name"], "id": job["id"], "error": str(exc), "trace": traceback.format_exc()})

	def submit(self, name, fn, label="", key=None, front=False, soft=False, on_drop=None):
		with self.cv:
			self.seq += 1
			job = {"id": self.seq, "name": name, "label": label or name, "fn": fn, "key": key, "soft": soft, "on_drop": on_drop}
			if key is not None:
				self.queue = [j for j in self.queue if j["key"] != key]
			if front:
				self.queue.insert(0, job)
			else:
				self.queue.append(job)
			self.cv.notify()
		self.emit("queue", self.snapshot())
		return job["id"]

	def _dropped(self, gone):
		for j in gone:
			if j["on_drop"] is not None:
				try:
					j["on_drop"]()
				except Exception:
					traceback.print_exc()
		self.emit("queue", self.snapshot())
		return len(gone)

	def drop(self, job_id):
		return self.drop_where(lambda j: j["id"] == job_id)

	def drop_where(self, pred):
		with self.cv:
			gone = [j for j in self.queue if pred(j)]
			if not gone:
				return 0
			self.queue = [j for j in self.queue if not pred(j)]
		return self._dropped(gone)

	def clear(self):
		with self.cv:
			gone, self.queue = self.queue, []
		return self._dropped(gone) if gone else 0

	def interrupt(self):
		self.stop.set()
		self.cancel.set()

	def stop_all(self):
		n = self.clear()
		self.interrupt()
		return n

	def interruptible(self, names):
		cur = self.current
		if cur is None or not cur["soft"] or cur["name"] not in names:
			return False
		return cur["name"] != "levers" or bool(getattr(self.session, "soft", False))

	def supersede(self, names):
		cur = self.current
		if cur is None or not self.interruptible(names):
			return None
		self.interrupt()
		return cur["id"]

	def pending(self):
		with self.cv:
			return len(self.queue)

	def has(self, name, running=True, skip=None):
		with self.cv:
			if any(j["name"] == name for j in self.queue):
				return True
			cur = self.current
			return running and cur is not None and cur["name"] == name and cur["id"] != skip

	def snapshot(self):
		with self.cv:
			cur = self.current
			return {"current": {"id": cur["id"], "name": cur["name"], "label": cur["label"]} if cur else None, "pending": [{"id": j["id"], "name": j["name"], "label": j["label"]} for j in self.queue]}

	def _next(self):
		with self.cv:
			while not self.queue and not self.quitting:
				self.cv.wait()
			if self.quitting:
				return None
			job = self.queue.pop(0)
			self.stop.clear()
			self.cancel.clear()
			self.current = job
		return job

	def _loop(self):
		from hypno.session import Session

		try:
			self.session = Session(self.emit)
			self.session.stop_event = self.stop
			self.ready.set()
			self.emit("ready", {"settings": self.session.settings})
		except Exception as exc:
			self.ready.set()
			self.emit("error", {"text": f"session init failed: {exc}", "trace": traceback.format_exc()})
			return
		while True:
			job = self._next()
			if job is None:
				break
			name = job["name"]
			self.emit("queue", self.snapshot())
			self.emit("busy", {"name": name, "id": job["id"], "label": job["label"]})
			try:
				result = job["fn"](self.session)
				self.emit("done", {"name": name, "id": job["id"], "result": result})
			except Exception as exc:
				if getattr(exc, "stopped", False):
					try:
						self.session.after_stop()
					except Exception:
						traceback.print_exc()
					self.emit("done", {"name": name, "id": job["id"], "result": None, "stopped": True})
				else:
					self.emit("error", {"name": name, "id": job["id"], "text": str(exc), "trace": traceback.format_exc()})
			finally:
				with self.cv:
					self.current = None
					n = len(self.queue)
				self.emit("busy", {"name": None, "pending": n})
				self.emit("queue", self.snapshot())

	def shutdown(self, timeout=8000):
		self.interrupt()
		self.lights.put(None)
		with self.cv:
			self.queue = []
			self.quitting = True
			self.cv.notify_all()
		self.thread.quit()
		return self.thread.wait(timeout)

	def alive(self):
		return self.thread.isRunning()


class Relay(QObject):
	fired = Signal(object)


class Planner(QObject):
	planned = Signal(object)

	def __init__(self):
		super().__init__()
		self.cv = threading.Condition()
		self.req = None
		self.bg = []
		self.running = None
		self.paused = False
		self.alive = True
		self.cache = {}
		threading.Thread(target=self._loop, daemon=True).start()

	def request(self, key, fn):
		if key in self.cache:
			self.planned.emit({"key": key, "result": self.cache[key]})
			return
		with self.cv:
			if key != self.running:
				self.req = (key, fn)
			self.cv.notify()

	def prefetch(self, items):
		with self.cv:
			self.bg = [(k, f) for k, f in items if k not in self.cache]
			self.cv.notify()

	def pause(self, paused):
		with self.cv:
			self.paused = paused
			self.cv.notify()

	def _loop(self):
		while True:
			with self.cv:
				while self.alive and self.req is None and (not self.bg or self.paused):
					self.cv.wait()
				if not self.alive:
					return
				if self.req is not None:
					(key, fn), self.req = self.req, None
				else:
					key, fn = self.bg.pop(0)
				if key in self.cache:
					continue
				self.running = key
			try:
				res = fn()
				self.cache[key] = res
				self.planned.emit({"key": key, "result": res})
			except Exception as exc:
				self.planned.emit({"key": key, "error": str(exc)})
			finally:
				self.running = None

	def shutdown(self):
		with self.cv:
			self.alive = False
			self.cv.notify()
