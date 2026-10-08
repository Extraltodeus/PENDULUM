import math
import os
import struct
import sys

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient

VARIANTS = {"nerv": ("#ff7300", "#ffb36b", "#3dff8a"), "unit-01": ("#9b4dff", "#c9a2ff", "#3dff8a")}
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
PIVOT = QPointF(128.0, 44.0)
ROD = 140.0
SWING = 24.0
GHOSTS = (-18.0, -4.0, 10.0)
TRAIL = -26.0
BG = "#050506"


def qc(c, a=255):
	q = QColor(c)
	q.setAlpha(max(0, min(255, int(a))))
	return q


def at(angle, r=ROD):
	a = math.radians(angle)
	return QPointF(PIVOT.x() + r * math.sin(a), PIVOT.y() + r * math.cos(a))


def glow_line(p, a, b, color, width, layers=((4.2, 22), (2.6, 46), (1.6, 110))):
	for k, alpha in layers:
		pen = QPen(qc(color, alpha), width * k)
		pen.setCapStyle(Qt.PenCapStyle.RoundCap)
		p.setPen(pen)
		p.drawLine(a, b)
	pen = QPen(qc(color), width)
	pen.setCapStyle(Qt.PenCapStyle.RoundCap)
	p.setPen(pen)
	p.drawLine(a, b)


def bob(p, c, r, accent, accent2, alpha=255, halo=True):
	if halo:
		g = QRadialGradient(c, r * 2.6)
		g.setColorAt(0.0, qc(accent, 0.55 * alpha))
		g.setColorAt(0.45, qc(accent, 0.22 * alpha))
		g.setColorAt(1.0, qc(accent, 0))
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(g)
		p.drawEllipse(c, r * 2.6, r * 2.6)
	g = QRadialGradient(QPointF(c.x() - r * 0.35, c.y() - r * 0.35), r * 1.25)
	g.setColorAt(0.0, qc("#ffffff", alpha))
	g.setColorAt(0.35, qc(accent2, alpha))
	g.setColorAt(1.0, qc(accent, alpha))
	p.setBrush(g)
	p.setPen(QPen(qc(accent2, alpha), max(1.0, r * 0.08)))
	p.drawEllipse(c, r, r)


def backdrop(p, back, accent, detail):
	if back == "square":
		p.fillRect(QRectF(0, 0, 256, 256), qc("#000000"))
		return
	if back != "round":
		return
	path = QPainterPath()
	path.addRoundedRect(QRectF(6, 6, 244, 244), 46, 46)
	g = QRadialGradient(QPointF(128, 150), 170)
	g.setColorAt(0.0, qc("#141018"))
	g.setColorAt(1.0, qc(BG))
	p.fillPath(path, g)
	p.setPen(QPen(qc(accent, 150 if detail else 210), 5 if detail else 10))
	p.setBrush(Qt.BrushStyle.NoBrush)
	p.drawPath(path)


def brackets(p, accent):
	pen = QPen(qc(accent, 230), 5)
	pen.setCapStyle(Qt.PenCapStyle.SquareCap)
	p.setPen(pen)
	m, arm = 30.0, 26.0
	for x, y, dx, dy in ((m, m, 1, 1), (256 - m, m, -1, 1), (m, 256 - m, 1, -1), (256 - m, 256 - m, -1, -1)):
		p.drawLine(QPointF(x, y), QPointF(x + dx * arm, y))
		p.drawLine(QPointF(x, y), QPointF(x, y + dy * arm))


def ruler(p, green, detail):
	r = ROD + 34.0
	rect = QRectF(PIVOT.x() - r, PIVOT.y() - r, 2 * r, 2 * r)
	p.setPen(QPen(qc(green, 120 if detail else 200), 2.2 if detail else 9.0))
	p.setBrush(Qt.BrushStyle.NoBrush)
	p.drawArc(rect, int((270 - 40) * 16), int(80 * 16))
	if not detail:
		return
	for i in range(-10, 11):
		major = i % 5 == 0
		a = i * 4.0
		p.setPen(QPen(qc(green, 235 if major else 150), 3.0 if major else 1.8))
		p.drawLine(at(a, r - (13 if major else 7)), at(a, r + (6 if major else 0)))


def mount(p, accent, accent2, detail):
	w = 64.0 if detail else 80.0
	y = PIVOT.y() - 12.0
	p.setPen(Qt.PenStyle.NoPen)
	p.setBrush(qc(accent, 230))
	p.drawRect(QRectF(PIVOT.x() - w / 2, y - 6, w, 7))
	if detail:
		p.setBrush(qc(BG))
		for k in range(5):
			x = PIVOT.x() - w / 2 + 6 + k * 12
			p.drawPolygon([QPointF(x, y - 6), QPointF(x + 5, y - 6), QPointF(x + 1, y + 1), QPointF(x - 4, y + 1)])
	p.setBrush(qc(accent2))
	p.drawEllipse(PIVOT, 7.5 if detail else 11.0, 7.5 if detail else 11.0)


def trail(p, accent, r, detail):
	steps = 26 if detail else 10
	for i in range(steps):
		a0 = TRAIL + (SWING - TRAIL) * i / steps
		a1 = TRAIL + (SWING - TRAIL) * (i + 1) / steps
		k = (i + 1) / steps
		pen = QPen(qc(accent, (10 + 150 * k ** 2.4) * (1.0 if detail else 0.7)), 2 * r * (0.3 + 0.55 * k))
		pen.setCapStyle(Qt.PenCapStyle.FlatCap)
		p.setPen(pen)
		path = QPainterPath()
		rect = QRectF(PIVOT.x() - ROD, PIVOT.y() - ROD, 2 * ROD, 2 * ROD)
		path.arcMoveTo(rect, 270 + a0)
		path.arcTo(rect, 270 + a0, a1 - a0)
		p.drawPath(path)


def label(p, text, accent, accent2):
	fams = set(QFontDatabase.families())
	fam = next((f for f in ("Bahnschrift", "Arial Narrow", "Segoe UI") if f in fams), "Segoe UI")
	f = QFont(fam)
	f.setPixelSize(34)
	f.setBold(True)
	f.setStretch(QFont.Stretch.Condensed)
	f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.5)
	p.setFont(f)
	r = QRectF(14, 212, 228, 40)
	for dx, dy, a in ((0, 0, 70), (1.5, 0, 50), (-1.5, 0, 50)):
		p.setPen(qc(accent, a))
		p.drawText(r.translated(dx, dy), Qt.AlignmentFlag.AlignCenter, text)
	p.setPen(qc(accent2))
	p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)


def paint(p, size, variant="nerv", back="round", text=""):
	accent, accent2, green = variant if isinstance(variant, (tuple, list)) else VARIANTS.get(variant, VARIANTS["nerv"])
	detail = size >= 48
	p.save()
	p.setRenderHint(QPainter.RenderHint.Antialiasing)
	p.scale(size / 256.0, size / 256.0)
	backdrop(p, back, accent, detail)
	if text:
		p.translate(128, 14)
		p.scale(0.8, 0.8)
		p.translate(-128, 0)
	elif detail and back == "square":
		brackets(p, accent)
	if not detail:
		p.translate(128, 40)
		p.scale(1.2, 1.2)
		p.translate(-128, -40)
	ruler(p, green, detail)
	rod_w = 7.0 if detail else 14.0
	bob_r = 25.0 if detail else 36.0
	trail(p, accent, bob_r, detail)
	if detail:
		for i, a in enumerate(GHOSTS):
			k = (i + 1) / (len(GHOSTS) + 1)
			p.setPen(QPen(qc(accent, 25 + 45 * k), 2.0))
			p.drawLine(PIVOT, at(a, ROD - bob_r * 0.6))
	end = at(SWING)
	glow_line(p, PIVOT, end, accent2, rod_w)
	mount(p, accent, accent2, detail)
	bob(p, end, bob_r, accent, accent2)
	p.restore()
	if text:
		p.save()
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
		p.scale(size / 256.0, size / 256.0)
		label(p, text, accent, accent2)
		p.restore()


def image(size, variant="nerv", back="round", text=""):
	img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
	img.fill(Qt.GlobalColor.transparent)
	p = QPainter(img)
	paint(p, size, variant, back, text)
	p.end()
	return img


def png_bytes(img):
	data = QByteArray()
	buf = QBuffer(data)
	buf.open(QIODevice.OpenModeFlag.WriteOnly)
	img.save(buf, "PNG")
	buf.close()
	return bytes(data)


def ico_bytes(variant="nerv", sizes=SIZES):
	blobs = [(s, png_bytes(image(s, variant))) for s in sizes]
	head = struct.pack("<HHH", 0, 1, len(blobs))
	offset = 6 + 16 * len(blobs)
	table, body = b"", b""
	for s, data in blobs:
		w = 0 if s >= 256 else s
		table += struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(data), offset + len(body))
		body += data
	return head + table + body


def app_icon(variant="nerv"):
	icon = QIcon()
	for s in SIZES:
		icon.addPixmap(QPixmap.fromImage(image(s, variant)))
	return icon


def colors(accent, accent2, green=VARIANTS["nerv"][2]):
	a, b = QColor(accent), QColor(accent2)
	if a.lightness() < 110:
		a = a.lighter(160)
	return a.name(), b.name(), green


def write_assets(folder):
	os.makedirs(folder, exist_ok=True)
	out = []
	for v in VARIANTS:
		tag = v.replace("-", "")
		path = os.path.join(folder, f"pendulum_{tag}.ico")
		with open(path, "wb") as f:
			f.write(ico_bytes(v))
		out.append(path)
		for name, back, text in ((f"pendulum_{tag}_256.png", "round", ""), (f"pendulum_{tag}_streamdeck.png", "square", ""), (f"pendulum_{tag}_streamdeck_label.png", "square", "PENDULUM")):
			path = os.path.join(folder, name)
			image(256, v, back, text).save(path)
			out.append(path)
	return out


if __name__ == "__main__":
	app = QGuiApplication(sys.argv)
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	for path in write_assets(os.path.join(root, "assets")):
		print(path)
