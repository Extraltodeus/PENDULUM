import ctypes
from ctypes import wintypes

import numpy as np

WHDR_DONE = 0x1
CALLBACK_EVENT = 0x50000
WAVE_MAPPER = 0xFFFFFFFF


class WAVEFORMATEX(ctypes.Structure):
	_fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD), ("nSamplesPerSec", wintypes.DWORD), ("nAvgBytesPerSec", wintypes.DWORD), ("nBlockAlign", wintypes.WORD), ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]


class WAVEHDR(ctypes.Structure):
	_fields_ = [("lpData", ctypes.c_void_p), ("dwBufferLength", wintypes.DWORD), ("dwBytesRecorded", wintypes.DWORD), ("dwUser", ctypes.c_size_t), ("dwFlags", wintypes.DWORD), ("dwLoops", wintypes.DWORD), ("lpNext", ctypes.c_void_p), ("reserved", ctypes.c_size_t)]


def _bind():
	mm = ctypes.WinDLL("winmm")
	k32 = ctypes.WinDLL("kernel32")
	mm.waveOutOpen.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.UINT, ctypes.POINTER(WAVEFORMATEX), ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
	mm.waveOutOpen.restype = wintypes.UINT
	for name in ("waveOutPrepareHeader", "waveOutWrite", "waveOutUnprepareHeader"):
		fn = getattr(mm, name)
		fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(WAVEHDR), wintypes.UINT]
		fn.restype = wintypes.UINT
	for name in ("waveOutReset", "waveOutClose"):
		fn = getattr(mm, name)
		fn.argtypes = [wintypes.HANDLE]
		fn.restype = wintypes.UINT
	k32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
	k32.CreateEventW.restype = wintypes.HANDLE
	k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
	k32.WaitForSingleObject.restype = wintypes.DWORD
	k32.CloseHandle.argtypes = [wintypes.HANDLE]
	return mm, k32


class WinOut:
	def __init__(self, rate, frames=768, count=4):
		self.mm, self.k32 = _bind()
		self.frames = frames
		self.event = self.k32.CreateEventW(None, False, False, None)
		fmt = WAVEFORMATEX(1, 2, rate, rate * 4, 4, 16, 0)
		self.h = wintypes.HANDLE()
		err = self.mm.waveOutOpen(ctypes.byref(self.h), WAVE_MAPPER, ctypes.byref(fmt), self.event or 0, 0, CALLBACK_EVENT)
		if err:
			self.k32.CloseHandle(self.event)
			raise OSError(f"no sound output (waveOutOpen error {err})")
		self.bufs = [ctypes.create_string_buffer(frames * 4) for _ in range(count)]
		self.hdrs = [WAVEHDR() for _ in range(count)]
		self.queued = [False] * count
		for b, h in zip(self.bufs, self.hdrs):
			h.lpData = ctypes.cast(b, ctypes.c_void_p)
			h.dwBufferLength = frames * 4
			self.mm.waveOutPrepareHeader(self.h, ctypes.byref(h), ctypes.sizeof(WAVEHDR))

	def free(self):
		out = []
		for i, h in enumerate(self.hdrs):
			if self.queued[i] and h.dwFlags & WHDR_DONE:
				self.queued[i] = False
			if not self.queued[i]:
				out.append(i)
		return out

	def write(self, i, data):
		pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2").tobytes()
		ctypes.memmove(self.bufs[i], pcm, len(pcm))
		h = self.hdrs[i]
		h.dwBufferLength = len(pcm)
		self.queued[i] = True
		err = self.mm.waveOutWrite(self.h, ctypes.byref(h), ctypes.sizeof(WAVEHDR))
		if err:
			self.queued[i] = False
			raise OSError(f"sound output lost (waveOutWrite error {err})")

	def wait(self, s):
		self.k32.WaitForSingleObject(self.event, max(1, int(s * 1000)))

	def close(self):
		self.mm.waveOutReset(self.h)
		for h in self.hdrs:
			self.mm.waveOutUnprepareHeader(self.h, ctypes.byref(h), ctypes.sizeof(WAVEHDR))
		self.mm.waveOutClose(self.h)
		self.k32.CloseHandle(self.event)
