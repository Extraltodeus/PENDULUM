import os
import threading
import time
import wave

import numpy as np

import config
from audio import bank
from audio.dsp import SR

GROUPS = {
	"alerts": ("ALARMS", "WARNING when the coherence guard steps in, errors, STOP"),
	"status": ("STATUS", "terminal and model online, READY, in sync, PATTERN BLUE, done"),
	"judge": ("MAGI", "the judge approves a lever change: three short voices, like the MAGI"),
	"controls": ("CONTROLS", "buttons, sliders (pitch follows the value), DOSE / TARGET, alignment rounds"),
	"text": ("TELETYPE", "ticks while a reply is written, a blip when it ends"),
}
GAPS = {"type": 0.065, "click": 0.03, "tick": 0.08, "magi": 0.5, "warning": 2.0, "ready": 0.8, "lever": 0.04, "target": 0.04, "error": 0.6, "toggle": 0.1}
KEYED = {"lever", "target", "toggle", "type"}
MAX_PACK_S = 12.0


def folder():
	return os.path.join(config.DATA_DIR, "sounds")


def read_wav(path):
	with wave.open(path, "rb") as w:
		ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
		raw = w.readframes(min(n, int(rate * MAX_PACK_S)))
	if width == 1:
		x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
	elif width == 2:
		x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
	elif width == 3:
		b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
		x = (((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8) >> 8).astype(np.float32) / 8388608.0
	elif width == 4:
		x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
	else:
		raise ValueError(f"{width * 8}-bit samples are not supported")
	x = x[:len(x) // ch * ch].reshape(-1, ch)
	x = np.repeat(x[:, :1], 2, axis=1) if ch == 1 else x[:, :2]
	if rate != SR and len(x) > 1:
		pos = np.arange(int(len(x) * SR / rate)) * rate / SR
		x = np.stack([np.interp(pos, np.arange(len(x)), x[:, c]) for c in range(2)], axis=1)
	return np.ascontiguousarray(x, dtype=np.float32)


class Sfx:
	def __init__(self):
		self.mixer = None
		self.failed = ""
		self.cache = {}
		self.pack = {}
		self.pack_errors = {}
		self.last = {}
		self.on = False
		self.volume = 0.6
		self.groups = set(GROUPS)
		self.muted = bool(os.environ.get("HYPNOLENS_NO_SOUND"))
		self.lock = threading.Lock()
		self.pack_sig = None
		self.warmed = False

	def configure(self, prefs):
		self.on = bool(prefs.get("sound", True)) and not self.muted
		self.volume = max(0, min(100, int(prefs.get("sound_volume", 60)))) / 100.0
		groups = prefs.get("sound_groups") or {}
		self.groups = {k for k in GROUPS if groups.get(k, True)}
		self.load_pack()
		if self.on and not self.warmed:
			self.warmed = True
			threading.Thread(target=self.warm, daemon=True, name="sound-warm").start()

	def signature(self):
		d = folder()
		sig = []
		for name in bank.SOUNDS:
			p = os.path.join(d, name + ".wav")
			try:
				st = os.stat(p)
			except OSError:
				continue
			sig.append((name, st.st_mtime_ns, st.st_size))
		return tuple(sig)

	def load_pack(self, force=False):
		sig = self.signature()
		if sig == self.pack_sig and not force:
			return
		pack, errors = {}, {}
		for name, _t, _s in sig:
			try:
				pack[name] = read_wav(os.path.join(folder(), name + ".wav"))
			except (OSError, EOFError, ValueError, wave.Error) as exc:
				errors[name] = str(exc)
		with self.lock:
			self.pack, self.pack_errors, self.pack_sig = pack, errors, sig

	def warm(self):
		for name in bank.SOUNDS:
			if name not in KEYED:
				self.get(name)
		self.get("toggle", "target")
		self.get("toggle", "dose")
		for i in range(len(bank.TYPE_NOTES)):
			self.get("type", i)

	def get(self, name, key=None):
		with self.lock:
			if name in self.pack:
				return self.pack[name]
			hit = self.cache.get((name, key))
		if hit is not None:
			return hit
		fn = bank.SOUNDS[name][0]
		data = fn() if key is None else fn(key)
		with self.lock:
			self.cache[(name, key)] = data
		return data

	def device(self):
		if self.mixer is not None and not self.mixer.alive:
			self.failed = self.mixer.error or "sound output stopped"
			self.mixer = None
		if self.mixer is None and not self.failed:
			try:
				from audio.mixer import Mixer, open_output

				self.mixer = Mixer(open_output(SR))
			except Exception as exc:
				self.failed = str(exc)
		return self.mixer

	def play(self, name, key=None, gain=1.0):
		if not self.on or name not in bank.SOUNDS or bank.SOUNDS[name][1] not in self.groups:
			return
		now = time.perf_counter()
		if now - self.last.get(name, -60.0) < GAPS.get(name, 0.0):
			return
		self.last[name] = now
		self.send(name, key, gain * self.volume)

	def preview(self, name, volume, key=None):
		if self.muted or name not in bank.SOUNDS:
			return
		self.send(name, key, max(0.0, min(1.0, volume)))

	def send(self, name, key, gain):
		m = self.device()
		if m is not None:
			m.play(self.get(name, key), gain)

	def retry(self):
		self.failed = ""
		return self.device() is not None

	def close(self):
		if self.mixer is not None:
			self.mixer.close()
			self.mixer = None


SFX = Sfx()
