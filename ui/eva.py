import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetricsF, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QTransform

from ui import theme

AA = QPainter.RenderHint.Antialiasing
LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
CENTER = Qt.AlignmentFlag.AlignCenter
RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
CACHE = {}
METRICS = {}
WIDTHS = {}
BAKED = {}
HOOKED = []


def on():
	return theme.nerv()


def col(c, a=255):
	q = QColor(c)
	q.setAlpha(int(max(0, min(255, a))))
	return q


def pen(c, a=255, w=1.0, style=Qt.PenStyle.SolidLine):
	p = QPen(col(c, a), w, style)
	p.setCapStyle(Qt.PenCapStyle.FlatCap)
	p.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
	return p


def font(px, weight=600, caps=True):
	key = (round(px * 2) / 2, weight, caps, theme.cond_family())
	f = CACHE.get(("font",) + key)
	if f is None:
		f = CACHE[("font",) + key] = theme.cond(key[0], weight, caps, px=True)
	return f


def metrics(f):
	key = f.key()
	fm = METRICS.get(key)
	if fm is None:
		if len(METRICS) > 128:
			METRICS.clear()
		fm = METRICS[key] = QFontMetricsF(f)
	return fm


def text_w(f, text):
	key = (f.key(), text)
	w = WIDTHS.get(key)
	if w is None:
		if len(WIDTHS) > 6000:
			WIDTHS.clear()
		w = WIDTHS[key] = metrics(f).horizontalAdvance(text)
	return w


def elide(f, text, w):
	if text_w(f, text) <= w:
		return text
	return metrics(f).elidedText(text, Qt.TextElideMode.ElideRight, max(0.0, w))


def bake(key, w, h, dpr, draw):
	k = (key, round(w, 2), round(h, 2), dpr)
	pm = BAKED.get(k)
	if pm is not None:
		return pm
	if len(BAKED) > 96:
		BAKED.clear()
	if not HOOKED and QGuiApplication.instance() is not None:
		QGuiApplication.instance().aboutToQuit.connect(BAKED.clear)
		HOOKED.append(True)
	pm = QPixmap(max(1, int(math.ceil(w * dpr))), max(1, int(math.ceil(h * dpr))))
	pm.setDevicePixelRatio(dpr)
	pm.fill(Qt.GlobalColor.transparent)
	q = QPainter(pm)
	q.setRenderHint(AA)
	draw(q)
	q.end()
	BAKED[k] = pm
	return pm


def box(p, r, c, a=255, w=1.0, fill=None, fill_a=255):
	p.setBrush(col(fill, fill_a) if fill else Qt.BrushStyle.NoBrush)
	p.setPen(pen(c, a, w) if c else Qt.PenStyle.NoPen)
	half = w / 2 if c else 0.0
	p.drawRect(QRectF(r).adjusted(half, half, -half, -half))


def brackets(p, r, c, a=230, arm=7.0, w=1.5):
	p.setBrush(Qt.BrushStyle.NoBrush)
	p.setPen(pen(c, a, w))
	h = w / 2
	x0, y0, x1, y1 = r.left() + h, r.top() + h, r.right() - h, r.bottom() - h
	for x, y, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
		p.drawPolyline(QPolygonF([QPointF(x + dx * arm, y), QPointF(x, y), QPointF(x, y + dy * arm)]))


def cross(p, x, y, arm, c, a=200, w=1.0):
	p.setPen(pen(c, a, w))
	p.drawLine(QPointF(x - arm, y), QPointF(x + arm, y))
	p.drawLine(QPointF(x, y - arm), QPointF(x, y + arm))


def stripes(c, period=12, a=255, back=None, back_a=255):
	key = ("stripes", QColor(c).name(), int(period), int(a), back, int(back_a))
	brush = CACHE.get(key)
	if brush is not None:
		return brush
	n = max(4, int(period))
	img = QImage(n, n, QImage.Format.Format_ARGB32_Premultiplied)
	img.fill(col(back, back_a) if back else QColor(0, 0, 0, 0))
	q = QPainter(img)
	q.setRenderHint(AA)
	q.setPen(Qt.PenStyle.NoPen)
	q.setBrush(col(c, a))
	for x0 in (-n, 0, n):
		q.drawPolygon(QPolygonF([QPointF(x0, n), QPointF(x0 + n / 2, n), QPointF(x0 + n * 1.5, 0), QPointF(x0 + n, 0)]))
	q.end()
	brush = CACHE[key] = QBrush(img)
	return brush


def hazard(p, r, c, period=12, a=255, back=None, back_a=255):
	p.save()
	p.setPen(Qt.PenStyle.NoPen)
	p.setBrushOrigin(QPointF(r.left(), r.top()))
	p.setBrush(stripes(c, period, a, back, back_a))
	p.drawRect(r)
	p.restore()


def tag(p, x, y, text, f, c, h=None, pad=5, fill=None, text_c=None, a=255, w=1.0, bar=0):
	tw = text_w(f, text)
	fm = metrics(f)
	h = h or round(fm.capHeight() + 2 * pad)
	r = QRectF(x, y, tw + 2 * pad + 1, h)
	box(p, r, c, a, w, fill, 255 if fill else 0)
	p.setFont(f)
	p.setPen(col(text_c or (theme.ON_ACCENT if fill else c), a if not fill else 255))
	p.drawText(QPointF(x + pad + 0.5, y + (h + fm.capHeight()) / 2), text)
	if bar:
		p.fillRect(QRectF(r.right() + 2, y, bar, h), col(c, a))
		r = r.adjusted(0, 0, bar + 2, 0)
	return r


def standby(p, r, text, label="STANDBY"):
	r = QRectF(r)
	c = theme.mix("#000000", theme.ACCENT, 0.5)
	brackets(p, r.adjusted(5, 5, -5, -5), c, 220, 8.0, 1.0)
	f, tf = font(13, 600), font(11, 700)
	inner = r.adjusted(20, 0, -20, 0)
	bh = metrics(f).boundingRect(inner, CENTER | Qt.TextFlag.TextWordWrap, text).height()
	top = r.center().y() - (18 + 8 + bh) / 2
	tw = text_w(tf, label)
	tag(p, r.center().x() - (tw + 12) / 2, top, label, tf, c, h=18, pad=6)
	p.setFont(f)
	p.setPen(col(theme.mix(theme.DIM, theme.MUTED, 0.5)))
	p.drawText(QRectF(inner.left(), top + 26, inner.width(), bh + 2), CENTER | Qt.TextFlag.TextWordWrap, text)


def seg_bar(p, r, frac, c, seg=4.0, gap=2.0, ghost=34, a=235):
	n = max(1, int((r.width() + gap) // (seg + gap)))
	lit = int(round(max(0.0, min(1.0, frac)) * n))
	p.setPen(Qt.PenStyle.NoPen)
	x = r.left()
	gc, lc = col(c, ghost), col(c, a)
	for i in range(n):
		p.fillRect(QRectF(x, r.top(), seg, r.height()), lc if i < lit else gc)
		x += seg + gap


def ruler(p, x0, x1, y, c, step=8, major=8, h=3.0, a=150):
	p.setPen(pen(c, a, 1.0))
	i = 0
	x = x0
	while x <= x1:
		hh = h * (2.2 if i % major == 0 else 1.0)
		xx = round(x) + 0.5
		p.drawLine(QPointF(xx, y), QPointF(xx, y - hh))
		x += step
		i += 1


def glow_text(p, x, y, text, f, c, glow_a=70, spread=2.2):
	path = QPainterPath()
	path.addText(QPointF(x, y), f, text)
	p.save()
	p.setRenderHint(AA)
	p.setBrush(Qt.BrushStyle.NoBrush)
	for k, a in ((spread * 2.2, glow_a * 0.35), (spread, glow_a)):
		gp = QPen(col(c, a), k)
		gp.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
		p.setPen(gp)
		p.drawPath(path)
	p.setPen(Qt.PenStyle.NoPen)
	p.setBrush(col(c))
	p.drawPath(path)
	p.restore()


def slanted(p, x, y, text, f, c, shear=-0.22, a=255):
	p.save()
	p.translate(x, y)
	p.setTransform(QTransform(1, 0, shear, 1, 0, 0), True)
	p.setFont(f)
	p.setPen(col(c, a))
	p.drawText(QPointF(0, 0), text)
	p.restore()


def warning(p, r, title, sub, c, t=0.0):
	p.save()
	p.setClipRect(r)
	p.fillRect(r, col("#000000", 205))
	band = max(5.0, r.height() * 0.13)
	hazard(p, QRectF(r.left(), r.top(), r.width(), band), c, band * 2.2, 235)
	hazard(p, QRectF(r.left(), r.bottom() - band, r.width(), band), c, band * 2.2, 235)
	inner = r.adjusted(0, band + 2, 0, -band - 2)
	pulse = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(t * 7.0))
	f = font(max(12.0, inner.height() * (0.72 if not sub else 0.58)), 800)
	tw = text_w(f, title)
	fs = font(max(8.0, inner.height() * 0.2), 700) if sub else None
	sw = text_w(fs, sub) if sub else 0.0
	total = tw + (sw + 16 if sub else 0)
	x = r.left() + (r.width() - total) / 2 + inner.height() * 0.11
	base = inner.top() + (inner.height() + metrics(f).capHeight()) / 2
	slanted(p, x, base, title, f, c, -0.24, int(255 * pulse))
	if sub:
		p.setFont(fs)
		p.setPen(col(c, 230))
		p.drawText(QPointF(x + tw + 16, base), sub)
	box(p, r, c, 230, 1.5)
	p.restore()


def ready(p, r, title, sub, c, age=0.0, dur=3.0):
	fade = max(0.0, min(1.0, (dur - age) / 0.8, age / 0.25))
	if fade <= 0.0:
		return
	p.save()
	p.setClipRect(r.adjusted(-2, -2, 2, 2))
	p.setOpacity(fade)
	p.fillRect(r, col("#000000", 190))
	box(p, r, c, 150, 1.0)
	box(p, r.adjusted(4, 4, -4, -4), c, 90, 1.0)
	brackets(p, r, c, 255, max(8.0, r.height() * 0.22), 2.0)
	f = font(max(12.0, r.height() * (0.46 if sub else 0.56)), 800)
	fs = font(max(8.0, r.height() * 0.15), 700) if sub else None
	tw = text_w(f, title)
	sw = text_w(fs, sub) if sub else 0.0
	gap = r.height() * 0.3
	x = r.left() + (r.width() - tw - (sw + gap if sub else 0.0)) / 2
	base = r.top() + (r.height() + metrics(f).capHeight()) / 2
	p.setFont(f)
	p.setPen(col(c))
	p.drawText(QPointF(x, base), title)
	if sub:
		p.setFont(fs)
		p.setPen(col(c, 215))
		p.drawText(QPointF(x + tw + gap, base), sub)
	sweep = age / 0.7
	if sweep < 1.0:
		sx = r.left() + r.width() * sweep
		p.fillRect(QRectF(sx - 3, r.top() + 1, 3, r.height() - 2), col(c, int(200 * (1 - sweep))))
	p.restore()
