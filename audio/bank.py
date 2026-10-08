import numpy as np

from audio.dsp import crush, env, finish, fm, gate, glide, highpass, lowpass, mix, n_of, noise, reverb, saw, sine, square, times

TYPE_NOTES = (1760.0, 1975.5, 2093.0, 2349.3)
MAGI = ((1568.0, 2.0, 1.2), (1760.0, 3.0, 0.9), (2093.0, 1.5, 1.4))


def blip(f, d=0.06, ratio=2.0, index=1.0, attack=0.002):
	n = n_of(d)
	e = env(n, attack, d)
	return fm(f, d, ratio, index * e) * e


def bell(f, d=0.8, ratio=2.0, index=1.2):
	n = n_of(d)
	return fm(f, d, ratio, index * env(n, 0.001, d * 0.35)) * env(n, 0.003, d)


def thump(f0=110.0, f1=55.0, d=0.15):
	return sine(glide(f0, f1, d), d) * env(n_of(d), 0.002, d)


def pitch(v, base):
	return base * 2 ** (max(-1.5, min(1.5, float(v))) * 0.75)


def click():
	d = 0.035
	n = n_of(d)
	tone = sine(2350.0, d) * env(n, 0.0006, 0.03)
	tick = highpass(noise(d, 3), 2500.0) * env(n, 0.0002, 0.006) * 0.35
	return finish(tone + tick, 0.26)


def type_(i=0):
	d = 0.018
	return finish(lowpass(square(TYPE_NOTES[int(i) % len(TYPE_NOTES)], d, 7), 5000.0) * env(n_of(d), 0.0005, 0.016), 0.1)


def lever(v=0.0):
	f = pitch(v, 880.0)
	return finish(mix((0.0, blip(f, 0.07, 2.0, 1.4)), (0.05, blip(f * 1.5, 0.06, 2.0, 0.9) * 0.55)), 0.34)


def target(v=0.0):
	f = pitch(v, 740.0)
	return finish(mix((0.0, blip(f, 0.045, 1.0, 1.6)), (0.06, blip(f, 0.045, 1.0, 1.6)), (0.12, blip(f * 2, 0.09, 1.0, 1.2) * 0.8)), 0.34)


def toggle(mode="target"):
	d = 0.05
	n = n_of(d)
	clack = lowpass(noise(d, 5), 1800.0) * env(n, 0.0005, 0.03) * 0.7 + sine(95.0, d) * env(n, 0.001, d) * 0.8
	a, b = (987.8, 1318.5) if mode == "target" else (1318.5, 987.8)
	return finish(mix((0.0, clack), (0.03, blip(a, 0.06)), (0.09, blip(b, 0.08))), 0.38)


def tick():
	return finish(mix((0.0, blip(2960.0, 0.025, 3.0, 0.6)), (0.04, blip(3520.0, 0.03, 3.0, 0.6))), 0.18)


def magi():
	parts = [(i * 0.07, blip(f, 0.09, r, k)) for i, (f, r, k) in enumerate(MAGI)]
	return finish(reverb(mix(*parts), 0.5, 0.18), 0.28)


def ready():
	return finish(reverb(mix((0.0, bell(880.0, 0.7)), (0.13, bell(1318.5, 0.9))), 1.1, 0.28), 0.42)


def start():
	parts = [(i * 0.06, blip(f, 0.05, 3.0, 0.8)) for i, f in enumerate((659.3, 987.8, 1318.5))]
	return finish(reverb(mix(*parts), 0.6, 0.2), 0.26)


def boot():
	d = 1.5
	n = n_of(d)
	hum = lowpass(saw(glide(41.2, 55.0, d), d, 24), 600.0) * gate(n, 0.5, 0.7) * 0.32
	sweep = sine(glide(220.0, 880.0, 0.6), 0.6) * env(n_of(0.6), 0.3, 0.5) * 0.15
	notes = (1046.5, 1396.9, 1864.7)
	parts = [(0.0, hum), (0.05, sweep)] + [(0.25 + i * 0.14, blip(f, 0.1, 2.0, 1.3) * 0.7) for i, f in enumerate(notes)]
	parts.append((0.72, sum(bell(f, 1.3, 2.0, 0.6) for f in notes) * 0.3))
	return finish(reverb(mix(*parts), 1.4, 0.3), 0.5)


def done():
	notes = (1046.5, 1318.5, 1568.0, 2093.0)
	parts = [(i * 0.045, blip(f, 0.07) * 0.8) for i, f in enumerate(notes)]
	parts.append((0.2, (bell(1046.5, 0.6, 2.0, 0.5) + bell(1568.0, 0.6, 2.0, 0.5)) * 0.4))
	return finish(reverb(mix(*parts), 0.8, 0.2), 0.4)


def eot():
	return finish(mix((0.0, blip(1975.5, 0.04, 2.0, 0.8)), (0.06, blip(1480.0, 0.06, 2.0, 0.8))), 0.15)

def warning():
	pulse, gap, f = 0.085, 0.035, 283.99
	parts = []
	for g in range(6):
		t0 = g * 0.52
		k = n_of(0.36)
		parts.append((t0, lowpass(saw(glide(126.0, 154.6, 0.36), 0.36, 20), 1500.0) * gate(k, 0.02, 0.06) * 0.35))
		for i in range(4):
			n = n_of(pulse)
			parts.append((t0 + i * (pulse + gap), (square(f, pulse, 9) * 0.7 + square(f * 1.059, pulse, 9) * 0.35) * gate(n, 0.008, 0.018)))
	return finish(reverb(lowpass(mix(*parts), 3500.0), 0.45, 0.15), 0.5)

def error():
	parts = []
	for i, f in enumerate((146.8, 138.6)):
		d = 0.13
		parts.append((i * 0.18, crush(square(f, d, 25), 6, 3) * gate(n_of(d), 0.003, 0.02)))
	return finish(lowpass(mix(*parts), 2600.0), 0.24)


def abort():
	d = 0.38
	n = n_of(d)
	sweep = lowpass(square(glide(1400.0, 160.0, d), d, 9), 4000.0) * env(n, 0.003, d * 1.4) * gate(n, 0.002, 0.05)
	return finish(mix((0.0, sweep * 0.6), (d - 0.04, thump(110.0, 60.0, 0.12))), 0.42)


def pattern():
	parts = []
	t = 0.0
	for i in range(8):
		parts.append((t, blip(1000.0 * 2 ** (i * 0.18), 0.05) * 0.6))
		t += 0.09 - i * 0.007
	d = 0.9
	n = n_of(d)
	vib = 1 + 0.008 * np.sin(2 * np.pi * 6.0 * times(n))
	chord = (sine(1174.7 * vib, d) + sine(1480.0 * vib, d) * 0.7 + sine(1760.0 * vib, d) * 0.6 + sine(2349.3 * vib, d) * 0.4) * env(n, 0.01, d * 1.2) * 0.5
	parts += [(t + 0.02, chord), (t + 0.02, thump(80.0, 45.0, 0.5) * 0.9)]
	return finish(reverb(mix(*parts), 1.3, 0.3), 0.55)


def sync():
	d = 1.1
	n = n_of(d)
	f = glide(110.0, 880.0, d)
	whine = lowpass(sum(saw(f * r, d, 12) for r in (1.0, 1.006, 0.994)) / 3, 3000.0) * (times(n) / d) ** 1.5 * gate(n, 0.0, 0.03) * 0.5
	beat = mix((0.0, thump(70.0, 50.0, 0.12)), (0.16, thump(70.0, 50.0, 0.12) * 0.7))
	hold = 1.2
	m = n_of(hold)
	chord = (sine(880.0, hold) + sine(1318.5, hold) * 0.6 + sine(1760.0, hold) * 0.5) * env(m, 0.005, hold * 1.3) * 0.6
	return finish(reverb(mix((0.0, beat), (0.05, whine), (d + 0.05, chord)), 1.6, 0.32), 0.5)


SOUNDS = {
	"start": (start, "status", "terminal online", "the app starts"),
	"boot": (boot, "status", "MAGI online", "a model is loaded and profiled"),
	"ready": (ready, "status", "ready", "a job that changes the model ends and nothing else is queued"),
	"sync": (sync, "status", "in sync", "every TARGET is reached"),
	"pattern": (pattern, "status", "pattern blue", "a placed goal matches its target"),
	"done": (done, "status", "done", "an export or a goal placement ends"),
	"warning": (warning, "alerts", "warning", "the coherence guard has to step in"),
	"error": (error, "alerts", "error", "a job fails"),
	"abort": (abort, "alerts", "stop", "STOP (Esc)"),
	"magi": (magi, "judge", "MAGI approve", "a lever change passes the coherence check"),
	"lever": (lever, "controls", "lever", "a lever is set (higher pitch = stronger dose)"),
	"target": (target, "controls", "target", "a TARGET is set (higher pitch = higher value)"),
	"toggle": (toggle, "controls", "mode", "DOSE / TARGET switch"),
	"tick": (tick, "controls", "round", "an alignment round is measured"),
	"click": (click, "controls", "button", "a button is pressed"),
	"type": (type_, "text", "teletype", "a reply is being written"),
	"eot": (eot, "text", "end of reply", "a reply is finished"),
}
