import html
import math
import time

import numpy as np
import shiboken6
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetricsF, QGradient, QImage, QLinearGradient, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QToolTip, QWidget

from ui import eva, theme

WAVES = ("#e8202e", "#3c8cff", "#d050ff")
INK = "#ece8e0"
RAMP = ("#9a4cff", "#2f7dff", "#2fe36b", "#ffe03a")
GREEN = "#2bd35d"
ORANGE = "#ff5524"
TAU = 2 * math.pi
TWISTS = 1
RINGS = 5
KAPPA = 0.38
SLANT = 0.1
EPS = 0.0015
RATE, GAIN, PUSH = 1.0, 7.0, 3.0
W_IDLE, W_BUSY = TAU * 0.025, TAU * 0.5
LIVE = 0.03
CLOCK_SLOT = "+00:00: 00000"
PLUS = QPainter.CompositionMode.CompositionMode_Plus
OVER = QPainter.CompositionMode.CompositionMode_SourceOver
SOURCE_IN = QPainter.CompositionMode.CompositionMode_SourceIn
AA = QPainter.RenderHint.Antialiasing
ARGB = QImage.Format.Format_ARGB32_Premultiplied
SMOOTH = Qt.TransformationMode.SmoothTransformation
LOOSE = Qt.AspectRatioMode.IgnoreAspectRatio
DIGITS = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def clamp(v, lo, hi):
	return lo if v < lo else hi if v > hi else v


def num(v, lo, hi):
	try:
		v = float(v)
	except (TypeError, ValueError):
		return 0.0
	return clamp(v, lo, hi) if v == v else 0.0


def hexa(c):
	q = QColor(c)
	return q.name() if q.isValid() else None


def remember(cache, key, fn):
	hit = cache.get(key)
	if hit is None:
		if len(cache) > 256:
			cache.clear()
		hit = cache[key] = fn()
	return hit


def col(c, a=255):
	q = QColor(c)
	q.setAlpha(int(clamp(a, 0, 255)))
	return q


def ramp(t, stops=RAMP):
	t = clamp(t, 0.0, 1.0) * (len(stops) - 1)
	i = min(int(t), len(stops) - 2)
	return theme.mix(stops[i], stops[i + 1], t - i)


def polyline(n):
	q = QPolygonF()
	q.resize(n)
	return q, np.frombuffer(shiboken6.VoidPtr(q.data(), n * 16, True), np.double).reshape((-1, 2))


def cpen(c, width=1.0, cosmetic=True):
	p = QPen(c, width)
	p.setCosmetic(cosmetic)
	p.setCapStyle(Qt.PenCapStyle.FlatCap)
	return p


def font(px, bold=False):
	f = theme.mono(8, bold)
	f.setPixelSize(max(6, int(round(px))))
	return f


def canvas(w, h, dpr):
	img = QImage(max(1, int(math.ceil(w * dpr))), max(1, int(math.ceil(h * dpr))), ARGB)
	img.setDevicePixelRatio(dpr)
	img.fill(0)
	return img


def black(w, h, dpr):
	pm = QPixmap(max(1, int(math.ceil(w * dpr))), max(1, int(math.ceil(h * dpr))))
	pm.setDevicePixelRatio(dpr)
	pm.fill(QColor(0, 0, 0))
	return pm


def hseg(x0, x1, y, t):
	h = t / 2
	return [(x0, y), (x0 + h, y - h), (x1 - h, y - h), (x1, y), (x1 - h, y + h), (x0 + h, y + h)]


def vseg(x, y0, y1, t):
	h = t / 2
	return [(x, y0), (x + h, y0 + h), (x + h, y1 - h), (x, y1), (x - h, y1 - h), (x - h, y0 + h)]


def rect(x0, y0, x1, y1):
	return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def plus(cx, cy, arm, t):
	h = t / 2
	return [rect(cx - arm, cy - h, cx + arm, cy + h), rect(cx - h, cy - arm, cx + h, cy - h), rect(cx - h, cy + h, cx + h, cy + arm)]


def slanted(shapes, h):
	return [QPolygonF([QPointF(x + (h - y) * SLANT, y) for x, y in s]) for s in shapes]


class Lcd:
	def __init__(self, h, color, dpr):
		self.h = h
		self.t = max(1.2, h * 0.13)
		self.cw = h * 0.54
		self.mg = int(math.ceil(self.t * 0.6)) + 1
		self.glyph = {}
		self.adv = {}
		for ch in "0123456789-+±:. ":
			self._make(ch, color, dpr)

	def shapes(self, ch, w):
		h, t = self.h, self.t
		g = t * 0.3
		l, r, top, mid, bot = t / 2, w - t / 2, t / 2, h / 2, h - t / 2
		segs = {"a": hseg(l + g, r - g, top, t), "b": vseg(r, top + g, mid - g, t), "c": vseg(r, mid + g, bot - g, t), "d": hseg(l + g, r - g, bot, t), "e": vseg(l, mid + g, bot - g, t), "f": vseg(l, top + g, mid - g, t), "g": hseg(l + g, r - g, mid, t)}
		if ch in DIGITS:
			return list(segs.values()), [segs[s] for s in DIGITS[ch]]
		if ch == "+":
			return [], plus(w / 2, mid, (r - l) / 2 + t * 0.2, t * 0.8)
		if ch == "-":
			arm = (r - l) / 2 + t * 0.2
			return [], [rect(w / 2 - arm, mid - t * 0.4, w / 2 + arm, mid + t * 0.4)]
		if ch == "±":
			arm = (r - l) / 2
			return [], plus(w / 2, h * 0.38, arm, t * 0.75) + [rect(w / 2 - arm, bot - t * 0.375, w / 2 + arm, bot + t * 0.375)]
		if ch == ":":
			return [], [rect(w / 2 - t / 2, h * 0.32 - t / 2, w / 2 + t / 2, h * 0.32 + t / 2), rect(w / 2 - t / 2, h * 0.72 - t / 2, w / 2 + t / 2, h * 0.72 + t / 2)]
		if ch == ".":
			return [], [rect(w / 2 - t / 2, bot - t / 2, w / 2 + t / 2, bot + t / 2)]
		return [], []

	def _make(self, ch, color, dpr):
		h, t, mg = self.h, self.t, self.mg
		w = t * 1.8 if ch in ":." else (self.cw * 0.45 if ch == " " else self.cw)
		img = canvas(w + h * SLANT + 2 * mg, h + 2 * mg, dpr)
		ghost, lit = self.shapes(ch, w)
		p = QPainter(img)
		p.setRenderHint(AA)
		p.translate(mg, mg)
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(col(color, 18))
		for q in slanted(ghost, h):
			p.drawPolygon(q)
		halo = QPen(col(color, 46), t * 0.9)
		halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
		p.setPen(halo)
		p.setBrush(Qt.BrushStyle.NoBrush)
		lit = slanted(lit, h)
		for q in lit:
			p.drawPolygon(q)
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(col(color))
		for q in lit:
			p.drawPolygon(q)
		p.end()
		self.glyph[ch] = img
		self.adv[ch] = w + (h * 0.07 if ch in ":. " else h * 0.17)

	def width(self, text):
		return sum(self.adv.get(ch, 0) for ch in text)

	def draw(self, p, x, y, text):
		y = round(y) - self.mg
		for ch in text:
			g = self.glyph.get(ch)
			if g is None:
				continue
			p.drawImage(QPointF(round(x) - self.mg, y), g)
			x += self.adv[ch]
		return x


LCDS = {}


def lcd_for(h, color, dpr):
	h = round(h * 2) / 2
	return remember(LCDS, (h, color, dpr), lambda: Lcd(h, color, dpr))


def tube_mask(sw, sh, dpr, lam, amp, rad, n):
	img = canvas(sw, sh, dpr)
	p = QPainter(img)
	p.setRenderHint(AA)
	p.setCompositionMode(PLUS)
	k = TAU / lam
	tw = TAU * TWISTS / lam
	xs = np.arange(-rad * 4, sw + rad * 4, 3.0)
	cy = sh / 2
	yc = cy + amp * np.sin(k * xs)
	sl = amp * k * np.cos(k * xs)
	nn = 1 / np.sqrt(1 + sl * sl)
	nx, ny = -sl * nn, nn
	q, v = polyline(len(xs))
	period = lam / TWISTS
	stops = np.linspace(0, 1, 17)
	for j in range(n):
		th0 = TAU * j / n
		th = th0 + tw * xs
		v[:, 0] = xs + rad * (np.cos(th) * nx + KAPPA * np.sin(th))
		v[:, 1] = yc + rad * np.cos(th) * ny
		x0 = -th0 / tw
		g = QLinearGradient(x0, 0, x0 + period, 0)
		g.setSpread(QGradient.Spread.RepeatSpread)
		for s in stops:
			z = (1 + math.sin(TAU * s)) / 2
			g.setColorAt(float(s), col("#ffffff", 190 * (0.2 + 0.8 * z ** 1.6)))
		p.setPen(cpen(QBrush(g)))
		p.drawPolyline(q)
	step = lam / RINGS
	half = np.linspace(0, math.pi, 17)
	rq, rv = polyline(len(half))
	for x0 in np.arange(-step, sw + step, step):
		y0 = cy + amp * math.sin(k * x0)
		s0 = amp * k * math.cos(k * x0)
		m = 1 / math.sqrt(1 + s0 * s0)
		for u, alpha in ((half, 200), (half + math.pi, 70)):
			rv[:, 0] = x0 + rad * 1.04 * (np.cos(u) * -s0 * m + KAPPA * np.sin(u))
			rv[:, 1] = y0 + rad * 1.04 * np.cos(u) * m
			p.setPen(cpen(col("#ffffff", alpha)))
			p.drawPolyline(rq)
	p.end()
	return img


def bloom(img, dpr):
	w, h = img.width(), img.height()
	small = img.scaled(max(1, w // 4), max(1, h // 4), LOOSE, SMOOTH)
	small = small.scaled(max(1, w // 8), max(1, h // 8), LOOSE, SMOOTH)
	big = small.scaled(w, h, LOOSE, SMOOTH)
	big.setDevicePixelRatio(dpr)
	return big


def tint(mask, glow, color, dpr):
	out = QImage(mask.size(), ARGB)
	out.setDevicePixelRatio(dpr)
	out.fill(0)
	p = QPainter(out)
	p.setCompositionMode(PLUS)
	p.setOpacity(0.85)
	p.drawImage(QPointF(0, 0), glow)
	p.setOpacity(1.0)
	p.drawImage(QPointF(0, 0), mask)
	p.setCompositionMode(SOURCE_IN)
	p.fillRect(QRectF(0, 0, out.width() / dpr, out.height() / dpr), col(color))
	p.end()
	return out


class Anim:
	def __init__(self):
		self.target = [0.0, 0.0, 0.0]
		self.value = [0.0, 0.0, 0.0]
		self.energy_t = 0.0
		self.energy = 0.0
		self.sync_t = None
		self.sync = 1.0
		self.fusion_t = 1.0
		self.fusion = 1.0
		self.trend = 0
		self.phase = 0.0
		self.jt = 0.0
		self.t = 0.0
		self.clock = 0.0
		self.running = False

	def aim(self, offsets, energy, sync, fusion=None):
		self.target = [num(o, -1.0, 1.0) for o in (list(offsets or ()) + [0.0, 0.0, 0.0])[:3]]
		self.energy_t = num(energy, 0.0, 1.0)
		self.sync_t = None if sync is None else num(sync, 0.0, 1.0)
		if fusion is None:
			s = self.sync_t if self.sync_t is not None else 1 - sum(abs(v) for v in self.target) / 3
			t = clamp((s - FUSE_LO) / (FUSE_HI - FUSE_LO), 0.0, 1.0)
			fusion = t * t * (3 - 2 * t)
		self.fusion_t = num(fusion, 0.0, 1.0)

	def snap(self):
		self.value = list(self.target)
		self.energy = self.energy_t
		self.sync = self.sync_t if self.sync_t is not None else 1 - sum(abs(v) for v in self.value) / 3
		self.fusion = self.fusion_t

	def busy(self):
		if self.energy > LIVE or abs(self.energy_t - self.energy) >= EPS:
			return True
		if abs(self.fusion_t - self.fusion) >= EPS:
			return True
		if self.sync_t is not None and abs(self.sync_t - self.sync) >= EPS:
			return True
		return any(abs(t - v) >= EPS for t, v in zip(self.target, self.value))

	def ease(self, v, t, dt):
		d = t - v
		if abs(d) < EPS:
			return t, False
		return v + d * (1 - math.exp(-(RATE + GAIN * abs(d) + PUSH * self.energy) * dt)), True

	def step(self, dt):
		moved = False
		for i in range(3):
			self.value[i], m = self.ease(self.value[i], self.target[i], dt)
			moved |= m
		de = self.energy_t - self.energy
		if abs(de) < EPS:
			self.energy = self.energy_t
		else:
			self.energy += de * (1 - math.exp(-(4.0 if de > 0 else 1.5) * dt))
			moved = True
		dfu = self.fusion_t - self.fusion
		if abs(dfu) < EPS:
			self.fusion = self.fusion_t
		else:
			self.fusion += dfu * (1 - math.exp(-2.2 * dt))
			moved = True
		old = self.sync
		if self.sync_t is None:
			self.sync = 1 - sum(abs(v) for v in self.value) / 3
		else:
			self.sync, m = self.ease(self.sync, self.sync_t, dt)
			moved |= m
		self.trend = 1 if self.sync > old + 1e-6 else (-1 if self.sync < old - 1e-6 else self.trend)
		self.phase = (self.phase + dt * (W_IDLE + W_BUSY * self.energy)) % TAU
		self.jt += dt * (0.5 + 3.0 * self.energy)
		self.t += dt
		live = moved or self.energy > LIVE
		if live and not self.running:
			self.clock = 0.0
		if live:
			self.clock += dt
		self.running = live
		return live

	def wobble(self, i):
		t = self.jt
		return math.sin(t * 1.9 + i * 2.1) * 0.6 + math.sin(t * 4.3 + i * 0.7) * 0.4

	def clock_text(self):
		sign = "±" if not self.running else ("+" if self.trend >= 0 else "-")
		c = self.clock
		return f"{sign}{int(c // 60)}:{int(c % 60):02d}: {int((c % 1) * 100000):05d}"


class SineScene:
	def __init__(self):
		self.gkey = self.skey = None
		self.strips = []
		self.sgeo = (1.0, 1.0, 1.0)
		self.cache = {}

	def prepare(self, view):
		w, h, dpr = view.width(), view.height(), view.devicePixelRatioF()
		gkey = (w, h, dpr, view.accent, view.names, view.waves)
		if gkey != self.gkey:
			self.gkey = gkey
			self.layout(w, h, dpr, view.accent or INK)
			self.bg = self.background(view.names, view.waves)
			self.cache = {}
		skey = (w, h, dpr, view.waves)
		if skey == self.skey or (view.lazy and self.strips):
			return
		self.skey = skey
		amp = self.ph * 0.27
		rad = clamp(self.ph * 0.075, 3.5, 30)
		lam = clamp(w / 3, amp * 4, amp * 9)
		n = int(clamp(round(rad / 0.9), 8, 18))
		mask = tube_mask(w + lam + 2, self.ph, dpr, lam, amp, rad, n)
		glow = bloom(mask, dpr)
		self.strips = [tint(mask, glow, c, dpr) for c in view.waves]
		self.sgeo = (lam, TAU / lam, self.ph)

	def layout(self, w, h, dpr, lcd):
		self.w, self.h, self.dpr = w, h, dpr
		self.th = max(2, round(h * 0.016))
		self.top_y = round(h * 0.11 - self.th / 2)
		self.bot_y = round(h * 0.88 - self.th / 2)
		self.py0 = self.top_y + self.th
		self.ph = self.bot_y - self.py0
		self.ux = w / 2
		self.us = w / 2 / 5.75
		self.f_label = font(clamp(min(h * 0.062, self.top_y * 0.72), 7, 17), True)
		self.fm_label = QFontMetricsF(self.f_label)
		self.f_small = font(clamp(min(h * 0.05, self.top_y * 0.6), 6, 14))
		self.fm_small = QFontMetricsF(self.f_small)
		self.f_axis = font(clamp(h * 0.064, 7, 18), True)
		self.lcd = lcd_for(clamp(self.top_y * 0.6, 7, 30), lcd, dpr)
		self.margin = max(6, round(w * 0.012))
		self.x0 = self.margin
		self.x1 = self.x0 + self.fm_label.horizontalAdvance("SYNC 100.0%  ")
		self.base = (self.top_y + self.fm_label.capHeight()) / 2
		self.right = w - self.margin - self.lcd.width(CLOCK_SLOT) - 12

	def background(self, names, waves):
		w, h, dpr = self.w, self.h, self.dpr
		pm = black(w, h, dpr)
		p = QPainter(pm)
		p.setRenderHint(AA)
		p.fillRect(QRectF(0, self.top_y, w, self.th), col(INK))
		p.fillRect(QRectF(0, self.bot_y, w, self.th), col(INK))
		arm = clamp(round(h * 0.022), 3, 8)
		p.setPen(cpen(col(INK, 150)))
		for u in (-3.6, -1.2, 1.2, 3.6):
			x = round(self.ux + u * self.us) + 0.5
			for f in (0.22, 0.5, 0.78):
				y = round(self.py0 + f * self.ph) + 0.5
				p.drawLine(QPointF(x - arm, y), QPointF(x + arm, y))
				p.drawLine(QPointF(x, y - arm), QPointF(x, y + arm))
		dash = clamp(w * 0.011, 4, 14)
		x = round(self.ux - 4.2 * self.us)
		gap = max(3.0, self.ph / 44)
		p.setPen(cpen(col(INK, 200)))
		i = 0
		y = self.py0 + gap
		while y < self.bot_y - gap / 2:
			yy = round(y) + 0.5
			p.drawLine(QPointF(x, yy), QPointF(x + (dash * 1.7 if i % 5 == 0 else dash), yy))
			y += gap
			i += 1
		x = round(self.ux + 5.42 * self.us)
		gap = self.ph / 9
		p.setPen(cpen(col(INK, 170)))
		for i in range(1, 9):
			yy = round(self.py0 + i * gap) + 0.5
			p.drawLine(QPointF(x, yy), QPointF(min(w - 2, x + dash * 1.3), yy))
		y0 = self.bot_y + self.th
		band = h - y0
		fm = QFontMetricsF(self.f_axis)
		tick = max(3, round((band - fm.capHeight()) * 0.5))
		tw = max(1, round(h * 0.006))
		p.setFont(self.f_axis)
		p.setPen(col(INK))
		for u in range(-5, 6):
			x = round(self.ux + u * self.us)
			p.fillRect(QRectF(x - tw / 2, y0, tw, tick), col(INK))
			p.drawText(QRectF(x - 40, y0 + tick, 80, band - tick), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "0" if u == 0 else f"{u:+d}")
		for q in range(-20, 20):
			if q % 4 == 0:
				continue
			x = round(self.ux + q / 4 * self.us)
			for yy in range(int(y0) + 1, int(y0 + tick * (0.75 if q % 2 == 0 else 0.45)), 2):
				p.fillRect(QRectF(x, yy, 1, 1), col(INK, 210))
		if names:
			self.legend(p, names, waves)
		p.end()
		return pm

	def legend(self, p, names, waves):
		fm = self.fm_small
		sw = max(6.0, self.f_small.pixelSize() * 1.2)
		total = sum(sw + 4 + fm.horizontalAdvance(n) for n in names) + 10 * (len(names) - 1)
		x = self.right - total
		if x < self.x1 + 90:
			return
		self.right = x - 14
		p.setFont(self.f_small)
		mid = self.top_y / 2
		yt = (self.top_y + fm.capHeight()) / 2
		for n, c in zip(names, waves):
			p.fillRect(QRectF(x, mid - 1.5, sw, 3), col(c))
			p.setPen(col(INK, 170))
			p.drawText(QPointF(x + sw + 4, yt), n)
			x += sw + 4 + fm.horizontalAdvance(n) + 10

	def elide(self, text, avail):
		return remember(self.cache, (text, int(avail)), lambda: self.fm_label.elidedText(text, Qt.TextElideMode.ElideRight, max(0, int(avail))))

	def paint(self, p, view):
		a = view.anim
		w, dpr = self.w, self.dpr
		p.drawPixmap(0, 0, self.bg)
		lam, k, sh = self.sgeo
		y0 = self.py0 + (self.ph - sh) / 2
		p.save()
		p.setClipRect(QRectF(0, self.py0, w, self.ph))
		p.setCompositionMode(PLUS)
		e = a.energy
		breath = math.sin(a.t * TAU / 7)
		shake = e * 0.3 * a.wobble(9)
		lift = 0.012 * breath + e * 0.02 * a.wobble(7)
		boost = clamp((a.sync - 0.8) / 0.2, 0, 1) ** 2 * 0.8
		src_w, src_h = round(w * dpr), round(sh * dpr)
		for i, strip in enumerate(self.strips):
			o = a.value[i]
			ph = a.phase + o * math.pi / 2 + shake + e * 0.08 * a.wobble(i)
			sx = round(((ph / k) % lam) * dpr)
			dy = round((o * 0.07 + lift + e * 0.008 * a.wobble(i + 5)) * self.ph * dpr) / dpr
			target = QRectF(0, y0 + dy, w, sh)
			source = QRectF(sx, 0, src_w, src_h)
			p.setOpacity(clamp(0.84 + 0.08 * breath * (1 - e) + 0.16 * e, 0, 1))
			p.drawImage(target, strip, source)
			if boost > 0.05:
				p.setOpacity(boost)
				p.drawImage(target, strip, source)
		p.restore()
		lcd = self.lcd
		txt = a.clock_text()
		lcd.draw(p, w - self.margin - lcd.width(txt), (self.top_y - lcd.h) / 2, txt)
		if not view.headline:
			return
		p.setFont(self.f_label)
		p.setPen(col(view.accent or INK))
		p.drawText(QPointF(self.x0, self.base), f"SYNC {a.sync * 100:5.1f}%")
		if view.label:
			p.setPen(col(INK, 200))
			p.drawText(QPointF(self.x1, self.base), self.elide(view.label, self.right - self.x1))


class GraphScene:
	def __init__(self):
		self.key = None
		self.cache = {}

	def prepare(self, view):
		w, h, dpr = view.width(), view.height(), view.devicePixelRatioF()
		key = (w, h, dpr, view.accent, view.names, view.waves)
		if key == self.key:
			return
		self.key = key
		self.layout(w, h, dpr, view.accent or INK)
		self.bg = self.background(view.names, view.waves)
		self.grid = self.panel_grid()
		self.cache = {}

	def layout(self, w, h, dpr, lcd):
		self.w, self.h, self.dpr = w, h, dpr
		self.band = round(h * 0.17)
		pad = max(3, round(h * 0.03))
		self.py0 = self.band + pad
		self.py1 = h - self.band - pad
		self.mid = (self.py0 + self.py1) / 2
		self.half = (self.py1 - self.py0) / 2
		self.f_axis = font(clamp(h * 0.052, 7, 15), True)
		self.f_label = font(clamp(min(h * 0.06, self.band * 0.5), 7, 16), True)
		self.fm_label = QFontMetricsF(self.f_label)
		self.f_cap = font(clamp(h * 0.03, 6, 11), True)
		self.fm_cap = QFontMetricsF(self.f_cap)
		self.f_name = font(clamp(h * 0.045, 7, 13), True)
		self.fm_name = QFontMetricsF(self.f_name)
		fm = QFontMetricsF(self.f_axis)
		self.ax = round(fm.horizontalAdvance("-100") + max(10, w * 0.012) + 4)
		self.right = w - max(8, round(w * 0.015))
		gap = max(8, round(w * 0.03))
		self.gap = gap
		self.bw = (self.right - self.ax - 3 * gap) / 3
		self.bx = [self.ax + gap + i * (self.bw + gap) for i in range(3)]
		self.fr = np.array([0.28, 0.5, 0.72])
		self.master = np.array([3 + k for _b in range(3) for k in range(3)])
		n = int(clamp((self.py1 - self.py0) / 3, 24, 140))
		self.ys = np.linspace(self.py0, self.py1, n)
		self.dy = self.ys[1] - self.ys[0]
		u = np.linspace(-1, 1, n)
		self.env = 0.45 + 0.55 * np.cos(u * math.pi / 2)
		rng = np.random.default_rng(7)
		fq = rng.uniform(0.7, 3.4, (9, 3))
		self.om = rng.uniform(0.5, 2.0, (9, 3)) * rng.choice([-1.0, 1.0], (9, 3))
		self.ph0 = rng.uniform(0, TAU, (9, 3))
		self.wk = np.array([0.55, 0.3, 0.15])[None, :, None]
		self.fu = (fq * math.pi)[:, :, None] * u[None, None, :]
		g = clamp(h * 0.0055, 1.0, 2.6)
		self.offs = np.array([-g, g, 0.0])[None, :, None]
		self.alphas = (80, 80, 255)
		self.polys = [[polyline(n) for _ in self.alphas] for _ in range(9)]
		self.lcd = lcd_for(clamp(self.band * 0.6, 8, 44), lcd, dpr)
		self.cw = self.lcd.width("+00000000")
		self.lx = self.ax
		self.rx = self.right - self.cw
		self.ty = (self.band - self.lcd.h) / 2
		self.by = h - self.band + (self.band - self.lcd.h) / 2

	def y_of(self, v):
		return self.mid - v / 100 * self.half

	def panel_grid(self):
		img = canvas(64, self.h, self.dpr)
		p = QPainter(img)
		step = 10 if self.half / 10 >= 4 else 20
		for v in range(-100, 101, step):
			if v == 0:
				continue
			y = round(self.y_of(v)) + 0.5
			p.setPen(cpen(col(GREEN if v > 0 else ORANGE, 165 if v % 50 == 0 else 105)))
			p.drawLine(QPointF(0, y), QPointF(64, y))
		p.end()
		return img

	def panels(self, f):
		full = self.bx[2] + self.bw - self.bx[0]
		return [(bx + f * (self.bx[0] - bx), self.bw + f * (full - self.bw)) for bx in self.bx]

	def background(self, names, waves):
		w, h, dpr = self.w, self.h, self.dpr
		pm = black(w, h, dpr)
		p = QPainter(pm)
		p.setRenderHint(AA)
		ax = self.ax
		p.setPen(cpen(col(INK, 210)))
		p.drawLine(QPointF(ax + 0.5, self.py0), QPointF(ax + 0.5, self.py1))
		p.setFont(self.f_axis)
		fm = QFontMetricsF(self.f_axis)
		for v in range(-100, 101, 10):
			y = round(self.y_of(v)) + 0.5
			major = v % 50 == 0
			p.setPen(cpen(col(INK, 220 if major else 150)))
			p.drawLine(QPointF(ax - (7 if major else 3), y), QPointF(ax, y))
			if major:
				txt = str(v)
				p.setPen(col(INK))
				p.drawText(QPointF(ax - 9 - fm.horizontalAdvance(txt), y + fm.capHeight() / 2), txt)
		p.setPen(cpen(col(INK, 60)))
		p.drawLine(QPointF(0, self.band - 0.5), QPointF(w, self.band - 0.5))
		p.drawLine(QPointF(0, h - self.band + 0.5), QPointF(w, h - self.band + 0.5))
		fc = self.fm_cap
		if self.ty >= fc.capHeight() + 4:
			p.setFont(self.f_cap)
			p.setPen(col(INK, 130))
			yt = self.ty - 3
			yb = self.by + self.lcd.h + 3 + fc.capHeight()
			names = list(names or ("INTENT", "APPLIED", "MEASURED"))
			for x, y, cap in ((self.lx, yt, f"DEV · {names[0]}"), (self.rx, yt, f"DEV · {names[1]}"), (self.lx, yb, f"DEV · {names[2]}"), (self.rx, yb, "DESYNC")):
				p.drawText(QPointF(x, y), cap)
		p.end()
		return pm

	def name_tag(self, p, x, text, c, alpha):
		fn = self.fm_name
		tw = fn.horizontalAdvance(text)
		if tw + 12 > self.bw or alpha <= 0.02:
			return
		y = self.py0 + 2
		p.setOpacity(alpha)
		p.fillRect(QRectF(x, y - 1, tw + 8, fn.capHeight() + 6), QColor(0, 0, 0))
		p.fillRect(QRectF(x, y - 1, 2, fn.capHeight() + 6), col(c))
		p.setFont(self.f_name)
		p.setPen(col(c))
		p.drawText(QPointF(x + 5, y + 2 + fn.capHeight()), text)
		p.setOpacity(1.0)

	def counter(self, a, i):
		e = a.energy
		if i < 3:
			v = a.value[i]
			mag = abs(v)
		else:
			v = sum(a.value) / 3
			mag = 1 - a.sync
		n = mag ** 1.6 * 99999999 + e * (0.15 + mag) * 60000 * (1 + math.sin(a.jt * 9.7 + i * 1.3))
		return ("-" if v < 0 else "+") + f"{int(clamp(n, 0, 99999999)):08d}"

	def headline(self, label, sync, avail):
		return remember(self.cache, (label, int(avail)), lambda: self._headline(label, sync, avail))

	def _headline(self, label, sync, avail):
		fm = self.fm_label
		room = avail - fm.horizontalAdvance(sync + "  ")
		return fm.elidedText(label, Qt.TextElideMode.ElideRight, max(0, int(room))) if label and room > fm.horizontalAdvance("M…") else ""

	def paint(self, p, view):
		a = view.anim
		f = clamp(a.fusion, 0.0, 1.0)
		p.drawPixmap(0, 0, self.bg)
		pan = self.panels(f)
		dots = QPen(col(INK, 34), 1.0)
		dots.setCosmetic(True)
		dots.setDashPattern([1.0, 2.0])
		for b in (0, 2, 1):
			x, w = pan[b]
			p.setOpacity(1.0 if b == 1 else clamp(1.0 - 1.15 * f, 0.0, 1.0))
			p.drawImage(QRectF(x, 0, w, self.h), self.grid)
			p.setPen(dots)
			for fr in self.fr:
				xx = round(x + w * fr) + 0.5
				p.drawLine(QPointF(xx, self.py0), QPointF(xx, self.py1))
		p.setOpacity(1.0)
		zp = QPen(col("#f4fff2"), max(1.5, self.h * 0.008))
		zp.setCapStyle(Qt.PenCapStyle.FlatCap)
		p.setPen(zp)
		p.drawLine(QPointF(self.ax, self.mid), QPointF(self.right, self.mid))
		if view.names:
			if f >= 0.98:
				self.name_tag(p, pan[0][0] + 3, "IN SYNC · ALL THREE", theme.PHOSPHOR, 1.0)
			else:
				for b, n, c in zip((0, 1, 2), view.names, view.waves):
					self.name_tag(p, pan[b][0] + 3, n, c, 1.0 if b == 0 else clamp(1.0 - 1.4 * f, 0.0, 1.0))
		vals = np.array(a.value)
		e = a.energy
		ws = np.repeat(np.array([w for _x, w in pan]), 3)
		disp = (np.sin(self.fu + (self.om * a.jt + self.ph0)[:, :, None]) * self.wk).sum(1)
		amp = ws * np.repeat(0.2 * np.abs(vals) + 0.035 * e, 3)
		shift = ws * np.repeat(0.13 * vals, 3)
		if f > 0.0:
			disp = (1.0 - f) * disp + f * disp[self.master]
			amp = (1.0 - f) * amp + f * amp.mean()
			shift = (1.0 - f) * shift + f * shift.mean()
		base = np.array([x + w * fr for x, w in pan for fr in self.fr])
		xs = (base + shift)[:, None] + amp[:, None] * disp * self.env[None, :]
		slope = np.gradient(xs, self.dy, axis=1)
		nx = 1 / np.sqrt(1 + slope * slope)
		xy = np.stack((xs[:, None, :] + self.offs * nx[:, None, :], self.ys[None, None, :] - self.offs * (slope * nx)[:, None, :]), -1)
		p.setRenderHint(AA)
		p.setCompositionMode(PLUS)
		for b in range(3):
			c = ramp(1 - abs(a.value[b]))
			p.setClipRect(QRectF(pan[b][0] - self.gap / 2, self.py0 - 2, pan[b][1] + self.gap, self.py1 - self.py0 + 4))
			for s, alpha in enumerate(self.alphas):
				p.setPen(cpen(col(c, int(alpha / (1.0 + 0.9 * f)))))
				for j in range(b * 3, b * 3 + 3):
					q, v = self.polys[j][s]
					v[:] = xy[j, s]
					p.drawPolyline(q)
		p.setClipping(False)
		p.setCompositionMode(OVER)
		lcd = self.lcd
		lcd.draw(p, self.lx, self.ty, self.counter(a, 0))
		lcd.draw(p, self.rx, self.ty, self.counter(a, 1))
		lcd.draw(p, self.lx, self.by, self.counter(a, 2))
		lcd.draw(p, self.rx, self.by, self.counter(a, 3))
		if not view.headline:
			return
		fm = self.fm_label
		sync = f"SYNC {a.sync * 100:5.1f}%"
		x0 = self.lx + self.cw + 14
		avail = self.rx - 14 - x0
		text = self.headline(view.label, sync, avail)
		tw = fm.horizontalAdvance(sync + ("  " + text if text else ""))
		if tw > avail:
			return
		x = x0 + (avail - tw) / 2
		y = (self.band + fm.capHeight()) / 2
		p.setFont(self.f_label)
		p.setPen(col(view.accent or INK))
		p.drawText(QPointF(x, y), sync)
		if text:
			p.setPen(col(INK, 200))
			p.drawText(QPointF(x + fm.horizontalAdvance(sync + "  "), y), text)


class SyncView(QWidget):
	mode_changed = Signal(str)
	headline = True

	def __init__(self, parent=None, mode="sine"):
		super().__init__(parent)
		self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
		self.setMinimumSize(240, 70)
		self.anim = Anim()
		self.scenes = {"sine": SineScene(), "graph": GraphScene()}
		self.mode = mode if mode in self.scenes else "sine"
		self.waves = WAVES
		self.accent = None
		self.label = ""
		self.names = None
		self.fps = (30, 8)
		self.cost = 0.0
		self.ticks = 0
		self.lazy = False
		self.last = time.perf_counter()
		self.timer = QTimer(self)
		self.timer.setTimerType(Qt.TimerType.PreciseTimer)
		self.timer.timeout.connect(self._tick)
		self.settle = QTimer(self)
		self.settle.setSingleShot(True)
		self.settle.timeout.connect(self._sized)
		self._tip()

	def sizeHint(self):
		return QSize(900, 220)

	def minimumSizeHint(self):
		return QSize(240, 70)

	def set_state(self, offsets=(0.0, 0.0, 0.0), energy=0.0, sync=None, label=None, fusion=None):
		self.anim.aim(offsets, energy, sync, fusion)
		if label is not None and label != self.label:
			self.label = label
			self._tip()
			self.update()
		if self.fps[0] <= 0:
			self.anim.snap()
			self.update()
			return
		self._wake()

	def set_names(self, names=None):
		self.names = tuple(str(n) for n in names)[:3] if names else None
		self._tip()
		self.update()

	def set_mode(self, mode):
		if mode not in self.scenes or mode == self.mode:
			return
		self.mode = mode
		self.update()

	def set_accent(self, color=None, waves=None):
		self.accent = hexa(color) if color else None
		waves = tuple(hexa(c) for c in waves or ())
		if len(waves) == 3 and all(waves):
			self.waves = waves
			self._tip()
		self.update()

	def set_fps(self, active=30, idle=8):
		self.fps = (max(0, active), max(0, idle))
		self._wake()

	def snap(self):
		self.anim.snap()
		self.update()

	def advance(self, dt):
		return self.anim.step(dt)

	def _tip(self):
		self.setToolTip("")

	def legend_rows(self):
		names = self.names or ("INTENT", "APPLIED", "MEASURED")
		what = ("your slider settings", "what the engine applies, after the coherence guard", "how the profile really moved, read by the lens (quick battery)")
		return [f"<span style='color:{c}'>&#9632;</span> <b>{html.escape(n)}</b>: {w}" for n, c, w in zip(names, self.waves, what)]

	def scene_tip(self, mode):
		mute = lambda t: f"<span style='color:{theme.MUTED}'>{t}</span>"
		if mode == "graph":
			rows = ["<b>GRAPH</b> · one column per reading"] + self.legend_rows() + [
				"the traces lean left or right by how far that reading sits from the average of the three,",
				"and shake harder the further off it is (and while the engine works)",
				f"<span style='color:{theme.PHOSPHOR}'>green</span>: in agreement · <span style='color:{theme.AMBER}'>amber</span> → <span style='color:{theme.ALARM}'>red</span>: off",
				"DEV · INTENT / APPLIED / MEASURED: each reading's deviation · DESYNC: how far from full sync",
				"the three columns slide together and their traces take the same shape as you get the alignment you want:",
				"TARGET mode: as the measures reach your targets (fully joined once they are within 0.08) · DOSE mode: as the three readings agree",
				mute("8-digit readouts and the vertical −100…+100 scale are EVA-style decoration")]
		else:
			rows = ["<b>SINE</b> · the three readings as glowing waves"] + self.legend_rows() + [
				"waves on top of each other: the three agree · a wave shifted up or out of phase: that reading is off",
				"they run faster while the engine works",
				mute("top right: time since the readings last moved · the −5…+5 axis is decoration")]
		if self.label:
			rows.insert(0, f"<b>{html.escape(self.label)}</b>")
		return "<br>".join(rows + [mute("click: switch the scene")])

	def tip_at(self, pos):
		return self.scene_tip(self.mode)

	def event(self, e):
		if e.type() == QEvent.Type.ToolTip:
			text = self.tip_at(QPointF(e.pos()))
			if text:
				QToolTip.showText(e.globalPos(), text, self)
			else:
				QToolTip.hideText()
			return True
		return super().event(e)

	def _wake(self):
		if self.isVisible():
			self._arm()

	def active(self):
		return self.anim.busy()

	def _arm(self):
		fps = self.fps[0] if self.active() else self.fps[1]
		if fps <= 0:
			self.timer.stop()
			return
		iv = max(5, int(1000 / fps))
		if not self.timer.isActive():
			self.last = time.perf_counter()
			self.timer.start(iv)
		elif self.timer.interval() != iv:
			self.timer.setInterval(iv)

	def _tick(self):
		now = time.perf_counter()
		dt = min(0.25, now - self.last)
		self.last = now
		if self.window().isMinimized():
			self.timer.setInterval(500)
			return
		self.anim.step(dt)
		self.ticks += 1
		self._arm()
		self.update()

	def _sized(self):
		self.lazy = False
		self.update()

	def resizeEvent(self, e):
		super().resizeEvent(e)
		if self.scenes["sine"].strips:
			self.lazy = True
			self.settle.start(160)

	def showEvent(self, e):
		super().showEvent(e)
		self.last = time.perf_counter()
		self._arm()

	def hideEvent(self, e):
		super().hideEvent(e)
		self.timer.stop()

	def mousePressEvent(self, e):
		if e.button() != Qt.MouseButton.LeftButton:
			e.ignore()
			return
		self.set_mode("graph" if self.mode == "sine" else "sine")
		self.mode_changed.emit(self.mode)

	def paintEvent(self, e):
		t0 = time.perf_counter()
		p = QPainter(self)
		if self.width() < 120 or self.height() < 40:
			p.fillRect(self.rect(), QColor(0, 0, 0))
			p.end()
			return
		scene = self.scenes[self.mode]
		scene.prepare(self)
		scene.paint(p, self)
		p.end()
		self.cost = time.perf_counter() - t0


SCENES = ("both", "sine", "graph")
FLASH_S = 4.5
READY_S = 2.8
FUSE_LO = 0.55
FUSE_HI = 0.97
TIP_S = 0.5


class Port:
	headline = False

	def __init__(self, band, legend=False):
		self.band = band
		self.legend = legend
		self.w = self.h = 1

	def width(self):
		return self.w

	def height(self):
		return self.h

	def devicePixelRatioF(self):
		return self.band.devicePixelRatioF()

	@property
	def anim(self):
		return self.band.anim

	@property
	def accent(self):
		return self.band.accent

	@property
	def waves(self):
		return self.band.waves

	@property
	def names(self):
		return self.band.names if self.legend else None

	@property
	def lazy(self):
		return self.band.lazy

	@property
	def label(self):
		return self.band.label


def scan_brush():
	key = ("scan", 3)
	b = eva.CACHE.get(key)
	if b is None:
		img = QImage(1, 3, ARGB)
		img.fill(0)
		img.setPixelColor(0, 2, QColor(0, 0, 0, 92))
		b = eva.CACHE[key] = QBrush(img)
	return b


class SyncBand(SyncView):
	scenes_changed = Signal(str)
	queue_clicked = Signal(object)
	headline = False

	def __init__(self, parent=None, scenes="both"):
		super().__init__(parent, "sine")
		self.setMinimumSize(320, 64)
		self.layout_mode = scenes if scenes in SCENES else "both"
		self.ports = {"sine": Port(self), "graph": Port(self, True)}
		self.status_fn = None
		self.state = {}
		self.guard = None
		self.warn = False
		self.stepped = False
		self.warn_t = -1e9
		self.warn_on = True
		self.ready = False
		self.ready_on = True
		self.scan = True
		self.pattern = None
		self.goal = None
		self.flash = None
		self.tokens = 0
		self.info_key = None
		self.info_bg = None
		self.geo = {}
		self.rects = (None, None, None)
		self.tip_t = 0.0
		self.tip_detail = ""
		self._tip()

	def sizeHint(self):
		return QSize(1400, 150)

	def minimumSizeHint(self):
		return QSize(320, 64)

	def set_layout(self, mode):
		mode = mode if mode in SCENES else "both"
		if mode == self.layout_mode:
			return
		self.layout_mode = mode
		self._tip()
		self.update()

	def set_options(self, warn=True, scan=True, ready=True):
		self.warn_on, self.scan, self.ready_on = bool(warn), bool(scan), bool(ready)
		self.update()

	def set_source(self, fn):
		self.status_fn = fn

	def set_guard(self, coherence=None, threshold=None, scale=1.0, first=None, cut=False):
		scale = 1.0 if scale is None else float(scale)
		g = None if coherence is None and scale >= 0.999 else {"c": coherence, "t": threshold, "s": scale, "f": first, "cut": bool(cut)}
		warn = coherence is not None and threshold is not None and coherence < threshold
		stepped = first is not None and threshold is not None and first < threshold
		fresh = (warn or stepped) and g != self.guard
		if fresh:
			self.warn_t = time.perf_counter()
		self.warn, self.stepped, self.guard = warn, stepped, g
		self._tip()
		self._wake()
		self.update()
		return fresh

	def set_pattern(self, match=None):
		fresh = match is not None and self.pattern is None
		if fresh:
			self.set_flash("PATTERN BLUE", f"GOAL MATCH {match:.0%}", theme.CYAN)
		if match != self.pattern:
			self.pattern = match
			self.update()
		return fresh

	def set_goal(self, p=None):
		self.goal = dict(p) if p else None
		self._wake()
		self.update()

	def set_flash(self, title, sub="", color=None, style="warning", dur=FLASH_S):
		self.flash = {"title": title, "sub": sub, "color": color or theme.CYAN, "t": time.perf_counter(), "style": style, "dur": dur}
		self._wake()
		self.update()

	def flashing(self, now=None):
		now = time.perf_counter() if now is None else now
		if (self.warn or self.stepped) and self.warn_on and now - self.warn_t < FLASH_S:
			return True
		return self.flash is not None and now - self.flash["t"] < self.flash["dur"]

	def set_ready(self, ready):
		if bool(ready) != self.ready:
			self.ready = bool(ready)
			self.update()

	def flash_ready(self, sub=""):
		if self.ready_on and not self.flashing():
			self.set_flash("READY", sub, theme.PHOSPHOR, "ready", READY_S)

	def active(self):
		return self.anim.busy() or self.goal is not None or self.flashing()

	def _tip(self):
		self.setToolTip("")

	def guard_rows(self):
		g = self.guard or {}
		rows = []
		if g.get("c") is not None:
			rows.append(f"<span style='color:{theme.ERR if self.warn else theme.OK}'>coherence {g['c']:.0%}</span>" + (f" · guard threshold {g['t']:.0%}" if g.get("t") is not None else "") + (f" · levers scaled ×{g['s']:.2f}" if g.get("s", 1.0) < 0.999 else ""))
			if self.stepped:
				rows.append(f"<span style='color:{theme.WARN}'>at the strengths you set: {g['f']:.0%}, below the threshold, so the guard stepped in</span>")
		elif g.get("s", 1.0) < 0.999:
			rows.append(f"levers scaled ×{g['s']:.2f} by the coherence guard")
		if self.pattern is not None:
			rows.append(f"<span style='color:{theme.CYAN}'>PATTERN BLUE</span>: a placed goal lever matches its target ({self.pattern:.0%}, judged)")
		return rows

	def tip_at(self, pos):
		mute = lambda t: f"<span style='color:{theme.MUTED}'>{t}</span>"
		for r, text in getattr(self, "badge_tips", []):
			if r.contains(pos):
				return text
		jr = self.job_rect()
		if jr is not None and jr.contains(pos):
			st = self.state or {}
			title, detail, prog, busy = self.job()
			rows = [f"<b>{html.escape(str(title))}</b>"] + ([html.escape(str(detail))] if detail else [])
			if busy and st.get("job"):
				rows.append(f"running: {html.escape(st['job'])}")
			if prog is not None:
				rows.append(f"progress {prog:.0%}")
			q = int(st.get("queued") or 0)
			rows.append(f"{q} job{'s' if q != 1 else ''} queued" if q else "nothing queued")
			return "<br>".join(rows + [mute("click: show the jobs (stop the running one, remove queued ones)")])
		info, sine, graph = self.rects
		if info is not None and info.contains(pos):
			a = self.anim
			trend = {1: "rising", -1: "falling"}.get(a.trend, "steady") if a.running else "steady"
			rows = [f"<b>SYNCHRO RATIO {a.sync * 100:.1f} %</b> · {trend}", "how closely the three readings agree (100 % = all the same):"] + self.legend_rows() + self.guard_rows()
			return "<br>".join(rows + [mute("click: sine + graph / sine / graph · drag the edge above the band to resize it")])
		for mode, r in (("sine", sine), ("graph", graph)):
			if r is not None and r.contains(pos):
				return self.scene_tip(mode).replace("click: switch the scene", "click: sine + graph / sine / graph")
		return ""

	def _tick(self):
		st = (self.status_fn() if self.status_fn else None) or {}
		n = int(st.get("tokens") or 0)
		if n > self.tokens:
			self.anim.energy = min(1.0, self.anim.energy + 0.05 * min(4, n - self.tokens))
		self.tokens = n
		self.state = st
		detail = self.detail_text()
		now = time.perf_counter()
		if detail != self.tip_detail and now - self.tip_t > TIP_S:
			self.tip_detail, self.tip_t = detail, now
			self._tip()
		super()._tick()

	def job_rect(self):
		info = self.rects[0]
		if info is None or "job" not in self.geo:
			return None
		return self.geo["job"].translated(info.left(), info.top())

	def mousePressEvent(self, e):
		if e.button() != Qt.MouseButton.LeftButton:
			e.ignore()
			return
		jr = self.job_rect()
		if jr is not None and jr.contains(e.position()):
			self.queue_clicked.emit(e.globalPosition().toPoint())
			return
		self.set_layout(SCENES[(SCENES.index(self.layout_mode) + 1) % len(SCENES)])
		self.scenes_changed.emit(self.layout_mode)

	def split(self, w, h, mode=None):
		mode = mode or self.layout_mode
		gap = 6
		iw = int(clamp(h * 1.6, 200, 270)) if w >= 640 else 0
		info = QRectF(0, 0, iw, h) if iw else None
		x = iw + gap if iw else 0
		rest = w - x
		if mode == "both" and rest >= 560:
			gw = int(rest * 0.4)
			return info, QRectF(x, 0, rest - gw - gap, h), QRectF(x + rest - gw, 0, gw, h)
		if mode == "graph":
			return info, None, QRectF(x, 0, rest, h)
		return info, QRectF(x, 0, rest, h), None

	def font(self, px, bold=True):
		return eva.font(px, 700 if bold else 600) if eva.on() else font(px * 0.82, bold)

	def job(self):
		st = self.state or {}
		g = self.goal
		if g:
			steps = max(1, int(g.get("steps") or 1))
			step = int(g.get("step") or 0)
			bits = [f"STEP {step}/{steps}"] + ([f"CE {g['ce']:.2f}"] if g.get("ce") is not None else []) + ([f"KL {g['kl']:.3f}"] if g.get("kl") is not None else [])
			return "TRAINING GOAL", " · ".join(bits), step / steps, True
		if st.get("busy"):
			return st.get("title") or self.label or "WORKING", st.get("detail") or "", st.get("progress"), True
		g = self.guard or {}
		if g.get("c") is not None:
			detail = f"COHERENCE {g['c']:.0%}" + (f" · GUARD {g['t']:.0%}" if g.get("t") is not None else "") + (f" · ×{g['s']:.2f}" if g.get("s", 1.0) < 0.999 else "")
		elif g.get("s", 1.0) < 0.999:
			detail = f"GUARD ×{g['s']:.2f}"
		else:
			detail = "INTENT · APPLIED · MEASURED"
		return self.label or "STANDBY", detail, None, False

	def detail_text(self):
		return self.job()[1]

	def info_layout(self, w, h):
		pad = 8
		cap = int(round(clamp(h * 0.125, 15, 21)))
		y = 6
		g = {"pad": pad, "cap": cap, "tag_y": y}
		y += cap + 5
		lcd_h = clamp(h * 0.25, 15, 44)
		g["pct_f"] = self.font(lcd_h * 0.62)
		g["pct_w"] = eva.text_w(g["pct_f"], "%")
		g["lcd_h"], g["lcd_y"], g["lcd_x"] = lcd_h, y, w - pad - g["pct_w"] - 5
		y += lcd_h + 7
		g["legend"] = h >= 118
		if g["legend"]:
			g["legend_y"] = y + 5
			y += 15
		g["job"] = QRectF(pad, y, w - 2 * pad, max(20.0, h - y - 7))
		return g

	def info_background(self, w, h, g):
		dpr = self.devicePixelRatioF()
		pm = black(w, h, dpr)
		p = QPainter(pm)
		p.setRenderHint(AA)
		nerv = eva.on()
		f = self.font(g["cap"] * 0.74)
		if nerv:
			eva.tag(p, g["pad"], g["tag_y"], "SYNCHRO RATIO", f, theme.ACCENT, h=g["cap"], pad=6, text_c=theme.NEON, bar=3)
		else:
			p.setFont(f)
			p.setPen(col(theme.TEXT))
			p.drawText(QPointF(g["pad"], g["tag_y"] + g["cap"] * 0.75), "SYNCHRO RATIO")
		p.setFont(g["pct_f"])
		p.setPen(col(theme.NEON if nerv else INK))
		p.drawText(QPointF(w - g["pad"] - g["pct_w"], g["lcd_y"] + g["lcd_h"]), "%")
		if g["legend"] and self.names:
			lf = self.font(10.5, False)
			x = g["pad"]
			y = g["legend_y"]
			for n, c in zip(self.names, self.waves):
				p.fillRect(QRectF(x, y - 3, 9, 4), col(c))
				p.setFont(lf)
				p.setPen(col(INK, 190))
				p.drawText(QPointF(x + 12, y + 1.5), n)
				x += 12 + eva.text_w(lf, n) + 10
		p.end()
		return pm

	def paint_info(self, p, r, now):
		w, h = int(r.width()), int(r.height())
		dpr = self.devicePixelRatioF()
		key = (w, h, dpr, theme.ACCENT, theme.NEON, eva.on(), self.names, self.waves)
		if key != self.info_key:
			self.info_key = key
			self.geo = self.info_layout(w, h)
			self.info_bg = self.info_background(w, h, self.geo)
		g = self.geo
		x0, y0 = int(r.left()), int(r.top())
		p.drawPixmap(x0, y0, self.info_bg)
		a = self.anim
		nerv = eva.on()
		lcd = lcd_for(g["lcd_h"], theme.NEON if nerv else INK, dpr)
		txt = f"{a.sync * 100:.1f}"
		lcd.draw(p, x0 + g["lcd_x"] - lcd.width(txt), y0 + g["lcd_y"], txt)
		cy = y0 + g["lcd_y"] + g["lcd_h"] / 2
		s = clamp(g["lcd_h"] * 0.16, 3, 7)
		tx = x0 + g["pad"] + s
		p.setPen(Qt.PenStyle.NoPen)
		if a.running and a.trend > 0:
			p.setBrush(col(theme.PHOSPHOR))
			p.drawPolygon(QPolygonF([QPointF(tx - s, cy + s * 0.6), QPointF(tx + s, cy + s * 0.6), QPointF(tx, cy - s)]))
		elif a.running and a.trend < 0:
			p.setBrush(col(theme.ALARM))
			p.drawPolygon(QPolygonF([QPointF(tx - s, cy - s * 0.6), QPointF(tx + s, cy - s * 0.6), QPointF(tx, cy + s)]))
		else:
			p.setBrush(col(theme.MUTED))
			p.drawRect(QRectF(tx - s * 0.6, cy - s * 0.6, s * 1.2, s * 1.2))
		title, detail, prog, busy = self.job()
		jr = g["job"].translated(x0, y0)
		blink = busy and int(now * 2.5) % 2 == 0
		hot = theme.NEON if nerv else theme.ACCENT
		if nerv:
			eva.box(p, jr, hot if blink else theme.ACCENT, 255 if blink else 110, 1.5 if blink else 1.0)
			if busy:
				p.fillRect(QRectF(jr.left(), jr.top(), 3, jr.height()), col(hot, 230 if blink else 120))
		else:
			p.setPen(cpen(col(theme.LINE2 if not blink else theme.ACCENT)))
			p.setBrush(Qt.BrushStyle.NoBrush)
			p.drawRect(jr.adjusted(0.5, 0.5, -0.5, -0.5))
		tf = self.font(clamp(jr.height() * 0.3, 11, 16))
		df = self.font(clamp(jr.height() * 0.22, 9.5, 12.5), False)
		inner = jr.width() - 16
		has_bar = prog is not None and jr.height() >= 36
		lines = jr.height() >= 34
		ty = jr.top() + (jr.height() * (0.42 if lines else 0.5)) + eva.metrics(tf).capHeight() / 2 - (4 if has_bar else 0)
		queued = int((self.state or {}).get("queued") or 0)
		if queued:
			q = f"+{queued}"
			qw = eva.text_w(tf, q)
			p.setFont(tf)
			p.setPen(col(theme.AMBER if nerv else theme.ACCENT2))
			p.drawText(QPointF(jr.right() - 8 - qw, ty), q)
			inner -= qw + 8
		p.setFont(tf)
		p.setPen(col(hot if busy else theme.TEXT))
		p.drawText(QPointF(jr.left() + 9, ty), eva.elide(tf, str(title).upper(), inner))
		if lines and detail:
			p.setFont(df)
			p.setPen(col(theme.TEXT, 170))
			p.drawText(QPointF(jr.left() + 9, ty + eva.metrics(df).capHeight() + 7), eva.elide(df, str(detail).upper() if nerv else str(detail), inner))
		if has_bar:
			br = QRectF(jr.left() + 9, jr.bottom() - 9, inner, 4)
			eva.seg_bar(p, br, prog, hot, 4.0, 2.0, 40, 235)

	def badge(self, p, x, y, h, text, c, stripes=False, fill=False):
		f = self.font(h * 0.68)
		x0 = x
		if stripes:
			eva.hazard(p, QRectF(x, y, h * 1.6, h), c, max(6, int(h * 0.9)), 235, "#000000", 235)
			x += h * 1.6 + 2
		r = QRectF(x, y, eva.text_w(f, text) + 12, h)
		p.fillRect(r, col(c if fill else "#000000", 235 if fill else 215))
		eva.box(p, r, c, 255, 1.0)
		p.setFont(f)
		p.setPen(col("#000000" if fill else c))
		p.drawText(QPointF(r.left() + 6, r.top() + (h + eva.metrics(f).capHeight()) / 2), text)
		return r.right() - x0

	def paint_badges(self, p, sine, graph, now, flash=False):
		self.badge_tips = []
		host = sine or graph
		if host is None or flash:
			return
		h = clamp(self.height() * 0.11, 13, 18)
		x = host.left() + (10 if sine else 64)
		y = host.top() + max(4.0, host.height() * (0.15 if sine else 0.21))
		g = self.guard or {}
		if self.ready and not self.warn:
			w = self.badge(p, x, y, h, "READY", theme.PHOSPHOR, False, True)
			self.badge_tips.append((QRectF(x, y, w, h), "<b>READY</b>: the model is loaded and no job runs: talk to it, the levers are applied as shown"))
			x += w + 8
		if self.warn:
			blink = int(now * 2) % 2 == 0
			w = self.badge(p, x, y, h, f"WARNING · COHERENCE {g['c']:.0%} < {g['t']:.0%}", theme.ALARM, True, blink)
			self.badge_tips.append((QRectF(x, y, w, h), f"<b>WARNING</b>: the coherence guard rated the steered test replies {g['c']:.0%}, below its {g['t']:.0%} threshold" + (f"<br>so it scaled the levers ×{g['s']:.2f}" if g.get("s", 1.0) < 0.999 else "") + f"<br><span style='color:{theme.MUTED}'>settings: COHERENCE · tolerance, absurdity, judge</span>"))
			x += w + 8
		elif g.get("s", 1.0) < 0.999 or self.stepped:
			text = f"CAUTION · GUARD ×{g['s']:.2f}" if g.get("s", 1.0) < 0.999 else "CAUTION · GUARD CUT"
			w = self.badge(p, x, y, h, text, theme.AMBER)
			self.badge_tips.append((QRectF(x, y, w, h), f"<b>CAUTION</b>: the coherence guard " + (f"scaled the levers ×{g['s']:.2f}" if g.get("s", 1.0) < 0.999 else "halved the levers that hurt coherence the most") + " to keep the replies coherent" + (f" ({g['f']:.0%} at the strengths you set, {g['c']:.0%} now)" if self.stepped and g.get("c") is not None else (f" ({g['c']:.0%} now)" if g.get("c") is not None else ""))))
			x += w + 8
		if self.pattern is not None:
			w = self.badge(p, x, y, h, f"PATTERN BLUE · GOAL {self.pattern:.0%}", theme.CYAN, False, False)
			self.badge_tips.append((QRectF(x, y, w, h), f"<b>PATTERN BLUE</b>: a placed goal lever matches its target at {self.pattern:.0%} (the judge's last check)"))

	def paint_flash(self, p, host, now):
		if host is None:
			return
		item = None
		if (self.warn or self.stepped) and self.warn_on and now - self.warn_t < FLASH_S:
			g = self.guard or {}
			if self.warn:
				sub = f"COHERENCE {g.get('c', 0):.0%} < {g.get('t', 0):.0%}"
			else:
				sub = f"COHERENCE {g['f']:.0%} < {g.get('t', 0):.0%} · " + (f"LEVERS ×{g['s']:.2f}" if g.get("s", 1.0) < 0.999 else "LEVERS CUT") + f" → {g.get('c', 0):.0%}"
			item = {"title": "WARNING", "sub": sub, "color": theme.ALARM, "t": self.warn_t, "style": "warning", "dur": FLASH_S}
		elif self.flash is not None and now - self.flash["t"] < self.flash["dur"]:
			item = self.flash
		if item is None:
			return False
		age = now - item["t"]
		if item["style"] == "ready":
			w = min(host.width() * 0.6, max(260.0, host.height() * 4.0))
			hh = clamp(host.height() * 0.42, 34, 84)
			r = QRectF(host.center().x() - w / 2, host.center().y() - hh / 2, w, hh)
			eva.ready(p, r, item["title"], item["sub"], item["color"], age, item["dur"])
			return True
		if age < 1.4 and int(age * 5) % 2 == 1:
			return True
		w = min(host.width() * 0.8, max(320.0, host.height() * 5.2))
		hh = clamp(host.height() * 0.56, 40, 110)
		r = QRectF(host.center().x() - w / 2, host.center().y() - hh / 2, w, hh)
		eva.warning(p, r, item["title"], item["sub"], item["color"], now)
		return True

	def paint_frames(self, p):
		for r in self.rects:
			if r is None:
				continue
			if eva.on():
				eva.box(p, r, theme.ACCENT, 80)
				eva.brackets(p, r, theme.NEON, 235, 9, 2.0)
			else:
				p.setPen(cpen(col(theme.LINE)))
				p.setBrush(Qt.BrushStyle.NoBrush)
				p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))

	def paintEvent(self, e):
		t0 = time.perf_counter()
		p = QPainter(self)
		w, h = self.width(), self.height()
		p.fillRect(self.rect(), QColor(0, 0, 0))
		if w < 200 or h < 40:
			p.end()
			return
		self.rects = self.split(w, h)
		info, sine, graph = self.rects
		for name, r in (("sine", sine), ("graph", graph)):
			if r is None:
				continue
			port = self.ports[name]
			port.w, port.h = int(r.width()), int(r.height())
			p.save()
			p.translate(int(r.left()), int(r.top()))
			p.setClipRect(QRectF(0, 0, port.w, port.h))
			scene = self.scenes[name]
			scene.prepare(port)
			scene.paint(p, port)
			p.restore()
		if info is not None:
			self.paint_info(p, info, t0)
		p.setRenderHint(AA)
		flash = self.paint_flash(p, sine or graph, t0)
		self.paint_badges(p, sine, graph, t0, flash)
		p.setCompositionMode(OVER)
		p.drawImage(0, 0, self.chrome(w, h))
		p.end()
		self.cost = time.perf_counter() - t0

	def chrome(self, w, h):
		dpr = self.devicePixelRatioF()
		key = (w, h, dpr, tuple(None if r is None else (r.x(), r.width()) for r in self.rects), theme.ACCENT, theme.NEON, eva.on(), self.scan)
		if key != getattr(self, "chrome_key", None):
			img = canvas(w, h, dpr)
			p = QPainter(img)
			if self.scan:
				for r in self.rects[1:]:
					if r is not None:
						p.fillRect(r, scan_brush())
			self.paint_frames(p)
			p.end()
			self.chrome_key, self.chrome_img = key, img
		return self.chrome_img
