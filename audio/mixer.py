import sys
import threading

import numpy as np

from audio.dsp import soft

MAX_VOICES = 16


def open_output(rate):
	if not sys.platform.startswith("win"):
		raise OSError("sound output is only implemented for Windows (winmm)")
	from audio.winout import WinOut

	return WinOut(rate)


class Mixer:
	def __init__(self, out):
		self.out = out
		self.lock = threading.Lock()
		self.voices = []
		self.wake = threading.Event()
		self.alive = True
		self.error = ""
		self.written = 0
		threading.Thread(target=self._run, daemon=True, name="sound").start()

	def play(self, data, gain=1.0):
		if not self.alive or gain <= 0:
			return
		with self.lock:
			self.voices.append([data, 0, float(gain)])
			del self.voices[:-MAX_VOICES]
		self.wake.set()

	def busy(self):
		with self.lock:
			return bool(self.voices)

	def render(self, n):
		out = np.zeros((n, 2), dtype=np.float32)
		with self.lock:
			keep = []
			for v in self.voices:
				data, i, g = v
				c = data[i:i + n]
				out[:len(c)] += c * g
				v[1] = i + n
				if v[1] < len(data):
					keep.append(v)
			self.voices = keep
		return soft(out), bool(keep)

	def _run(self):
		try:
			while self.alive:
				if not self.busy():
					self.wake.wait(1.0)
					self.wake.clear()
					continue
				free = self.out.free()
				if not free:
					self.out.wait(0.03)
					continue
				for i in free:
					data, more = self.render(self.out.frames)
					self.out.write(i, data)
					self.written += 1
					if not more:
						break
		except Exception as exc:
			self.error = str(exc)
			self.alive = False

	def close(self):
		self.alive = False
		self.wake.set()
		try:
			self.out.close()
		except Exception:
			pass
