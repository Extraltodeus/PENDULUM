import numpy as np

SR = 44100
NYQ = SR / 2
PAD = 4096


def n_of(d):
	return max(1, int(round(d * SR)))


def times(n):
	return np.arange(n) / SR


def line(f, n):
	return np.broadcast_to(np.asarray(f, dtype=np.float64), (n,))


def glide(f0, f1, d, power=1.0):
	x = np.linspace(0.0, 1.0, n_of(d)) ** power
	return f0 * (f1 / f0) ** x


def phase(f, n):
	return 2 * np.pi * np.cumsum(line(f, n)) / SR


def partials(f, d, parts):
	n = n_of(d)
	fl = line(f, n)
	ph = phase(fl, n)
	out = np.zeros(n)
	for k, a in parts:
		out += a * np.sin(k * ph) * np.clip((NYQ * 0.9 - k * fl) / 1500.0, 0.0, 1.0)
	return out


def sine(f, d):
	return partials(f, d, ((1, 1.0),))


def square(f, d, top=13):
	return partials(f, d, [(k, 4 / np.pi / k) for k in range(1, top + 1, 2)])


def saw(f, d, top=16):
	return partials(f, d, [(k, 2 / np.pi * (-1) ** (k + 1) / k) for k in range(1, top + 1)])


def fm(f, d, ratio=2.0, index=1.0):
	n = n_of(d)
	ph = phase(f, n)
	return np.sin(ph + line(index, n) * np.sin(ratio * ph))


def env(n, attack=0.003, decay=0.25, hold=0.0):
	t = times(n)
	return np.minimum(1.0, t / max(attack, 1e-5)) * np.exp(-6.9 * np.maximum(t - attack - hold, 0.0) / max(decay, 1e-5))


def gate(n, attack=0.004, release=0.03):
	t = times(n)
	return np.clip(np.minimum(t / max(attack, 1e-5), (n / SR - t) / max(release, 1e-5)), 0.0, 1.0)


def noise(d, seed=0):
	return np.random.default_rng(seed).standard_normal(n_of(d))


def shape(x, gain):
	x = np.asarray(x, dtype=np.float64)
	if x.ndim == 2:
		return np.stack([shape(x[:, c], gain) for c in range(x.shape[1])], axis=1)
	m = 1 << (len(x) + 2 * PAD - 1).bit_length()
	X = np.fft.rfft(np.pad(x, (PAD, m - len(x) - PAD)))
	return np.fft.irfft(X * gain(np.fft.rfftfreq(m, 1 / SR)), m)[PAD:PAD + len(x)]


def lowpass(x, fc, order=2):
	return shape(x, lambda f: 1 / np.sqrt(1 + (f / fc) ** (2 * order)))


def highpass(x, fc, order=2):
	return shape(x, lambda f: 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-6)) ** (2 * order)))


def bandpass(x, lo, hi, order=2):
	return shape(x, lambda f: 1 / np.sqrt((1 + (f / hi) ** (2 * order)) * (1 + (lo / np.maximum(f, 1e-6)) ** (2 * order))))


def st(x, pan=0.0):
	x = np.asarray(x, dtype=np.float64)
	if x.ndim == 2:
		return x
	a = (pan + 1) * np.pi / 4
	return np.stack([x * np.cos(a), x * np.sin(a)], axis=1) * np.sqrt(2)


def mix(*parts):
	parts = [(int(round(t0 * SR)), st(a)) for t0, a in parts]
	out = np.zeros((max(i + len(a) for i, a in parts), 2))
	for i, a in parts:
		out[i:i + len(a)] += a
	return out


def reverb(x, rt=1.0, wet=0.25, pre=0.015, tone=4500.0, seed=11):
	x = st(x)
	m = n_of(rt * 1.2)
	total = len(x) + m
	size = 1 << (total - 1).bit_length()
	out = np.zeros((total, 2))
	out[:len(x)] = x * (1 - wet * 0.5)
	t = times(m)
	for c in range(2):
		ir = lowpass(np.random.default_rng(seed + c).standard_normal(m) * np.exp(-6.9 * t / rt), tone, 1)
		ir[:n_of(pre)] = 0.0
		ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
		out[:, c] += wet * np.fft.irfft(np.fft.rfft(x[:, c], size) * np.fft.rfft(ir, size), size)[:total]
	return out


def echo(x, delay=0.12, feedback=0.4, taps=4, spread=0.6):
	x = st(x)
	d = n_of(delay)
	out = np.zeros((len(x) + d * taps, 2))
	out[:len(x)] += x
	for i in range(1, taps + 1):
		p = spread if i % 2 else -spread
		out[d * i:d * i + len(x)] += x * feedback ** i * np.array([1 - p, 1 + p]) / (1 + abs(p))
	return out


def crush(x, bits=8, hold=1):
	x = np.asarray(x, dtype=np.float64)
	if hold > 1:
		x = x[(np.arange(len(x)) // hold) * hold]
	q = 2 ** (bits - 1)
	return np.round(x * q) / q


def am(x, rate, depth=0.5):
	m = 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * times(len(x))))
	return x * (m[:, None] if np.ndim(x) == 2 else m)


def soft(x, knee=0.8):
	a = np.abs(x)
	return np.where(a < knee, x, np.sign(x) * (knee + (1 - knee) * np.tanh((a - knee) / (1 - knee))))


def finish(x, peak=0.8, fade=0.006, floor=1e-4):
	x = st(x).copy()
	m = np.max(np.abs(x))
	if m <= 0:
		return x.astype(np.float32)
	x *= peak / m
	loud = np.nonzero(np.max(np.abs(x), axis=1) > floor * peak)[0]
	x = x[:loud[-1] + 1] if len(loud) else x
	k = min(len(x), n_of(fade))
	x[-k:] *= np.linspace(1.0, 0.0, k)[:, None]
	return x.astype(np.float32)
