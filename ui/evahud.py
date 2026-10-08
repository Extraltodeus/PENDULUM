import html
import math
import re
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPolygonF

from ui import eva, theme

HELP_ROW = re.compile(r"^(\S+(?: \S+)*?)\s{2,}(.+)$")
GAP = 14.0
TITLE_Y = 58.0
SIDE_Y = 96.0
STAGE_H = 66.0
PLATE_Y = 4.0
PLATED = (3.0, 50.0, 88.0)
PLAIN = (12.0, TITLE_Y, SIDE_Y)


def plate(r, cut):
	l, t, rt, b = r.left(), r.top(), r.right(), r.bottom()
	return QPolygonF([QPointF(l + cut, t), QPointF(rt, t), QPointF(rt, b - cut), QPointF(rt - cut, b), QPointF(l, b), QPointF(l, t + cut)])


def draw_plate(p, r, c, a=255, w=1.0, fill=None, fill_a=255, cut=4.0):
	h = w / 2
	p.setBrush(eva.col(fill, fill_a) if fill else Qt.BrushStyle.NoBrush)
	p.setPen(eva.pen(c, a, w) if c else Qt.PenStyle.NoPen)
	p.drawPolygon(plate(QRectF(r).adjusted(h, h, -h, -h), cut))


def base(f, y, h):
	return y + (h + eva.metrics(f).capHeight()) / 2


def dim_accent(t=0.7):
	return theme.mix("#000000", theme.ACCENT, t)


def geom(n, x0, span, y=12.0):
	gap = GAP if span > 500 else 8.0
	bw = (span - gap * (n - 1)) / max(n, 1)
	return gap, bw, y, 22.0, (22.0 if bw >= 90 else 0.0)


def plate_font():
	return eva.font(15, 700)


def plate_rect(view):
	title = getattr(view, "plate", "")
	if not title:
		return None
	f = plate_font()
	return QRectF(8.0, PLATE_Y, eva.text_w(f, title) + 2 * 6 + 1 + 3 + 2, 20.0)


def plate_title(view, p):
	title = getattr(view, "plate", "")
	if title:
		eva.tag(p, 8.0, PLATE_Y, title, plate_font(), theme.ACCENT, h=20, pad=6, text_c=theme.NEON, bar=3)


def layout(view):
	W = view.width()
	pr = plate_rect(view)
	if pr is not None:
		x0 = pr.right() + 18.0
		return x0, max(160.0, W - x0 - 12.0), 8.0, W - 20.0, PLATED
	span = min(640.0, W - 56.0)
	x0 = (W - span) / 2
	return x0, span, x0, span, PLAIN


def hud_bottom(view, state):
	plated = plate_rect(view) is not None
	side = bool((state or {}).get("side"))
	if plated:
		return 118.0 if side else 86.0
	return 126.0 if side else 100.0


def stage_fonts(bw):
	return eva.font(14 if bw >= 130 else (13 if bw >= 76 else 11), 700), eva.font(13 if bw >= 130 else 12, 600), eva.font(10, 700)


def stage_under(q, stages, x0, span, y0=12.0):
	gap, bw, y, bh, zone = geom(len(stages), x0, span, y0)
	for i, (label, state, sub) in enumerate(stages):
		x = x0 + i * (bw + gap)
		if state == "done":
			draw_plate(q, QRectF(x, y, bw, bh), theme.PHOSPHOR, 235, 1.5)
		elif state != "active":
			draw_plate(q, QRectF(x, y, bw, bh), theme.DIM, 255, 1.0)
		if i < len(stages) - 1:
			nxt = stages[i + 1][1]
			lit = state == "done" and nxt in ("done", "active")
			q.setPen(eva.pen(theme.PHOSPHOR if lit and nxt == "done" else (theme.ACCENT2 if lit else theme.DIM), 200 if lit else 255, 1.0))
			q.drawLine(QPointF(x + bw + 2, y + bh / 2), QPointF(x + bw + gap - 2, y + bh / 2))


def stage_over(q, stages, x0, span, y0=12.0):
	gap, bw, y, bh, zone = geom(len(stages), x0, span, y0)
	lf, sf, nf = stage_fonts(bw)
	for i, (label, state, sub) in enumerate(stages):
		x = x0 + i * (bw + gap)
		fg = theme.ON_ACCENT if state == "active" else (theme.PHOSPHOR if state == "done" else theme.DIM)
		if zone:
			q.setPen(eva.pen(fg, 110, 1.0))
			q.drawLine(QPointF(x + zone + 0.5, y + 4), QPointF(x + zone + 0.5, y + bh - 4))
			if state == "done":
				q.setPen(eva.pen(theme.PHOSPHOR, 255, 1.8))
				q.drawPolyline(QPolygonF([QPointF(x + 6, y + 11.5), QPointF(x + 10, y + 15.5), QPointF(x + 17, y + 7)]))
			else:
				q.setFont(nf)
				q.setPen(eva.col(fg, 200))
				q.drawText(QPointF(x + 5, base(nf, y, bh)), f"{i + 1:02d}")
		text = eva.elide(lf, label, bw - zone - 8)
		q.setFont(lf)
		q.setPen(eva.col(fg))
		q.drawText(QPointF(x + zone + (bw - zone - eva.text_w(lf, text)) / 2, base(lf, y, bh)), text)
		if sub:
			st = eva.elide(sf, sub, bw + gap - 2)
			q.setFont(sf)
			q.setPen(theme.qc(theme.TEXT, 225) if state == "active" else theme.qc(theme.MUTED if state == "done" else theme.DIM))
			q.drawText(QPointF(x + (bw - eva.text_w(sf, st)) / 2, y + bh + 14), st)


def stage_live(p, stages, x0, span, now, y0=12.0):
	gap, bw, y, bh, zone = geom(len(stages), x0, span, y0)
	pulse = 0.5 + 0.5 * math.sin(now * 4.6)
	for i, (label, state, sub) in enumerate(stages):
		x = x0 + i * (bw + gap)
		if state == "active":
			r = QRectF(x, y, bw, bh)
			for k, a in ((3.5, 22), (1.8, 46)):
				draw_plate(p, r.adjusted(-k, -k, k, k), theme.ACCENT, a * (0.5 + pulse), 1.0, cut=4.0 + k)
			draw_plate(p, r, theme.ACCENT, 255, 1.0, theme.ACCENT, 185 + 70 * pulse)
		if state == "done" and i < len(stages) - 1 and stages[i + 1][1] == "active":
			p.setPen(Qt.PenStyle.NoPen)
			p.setBrush(eva.col(theme.ACCENT2, 235))
			p.drawRect(QRectF(x + bw + 2 + ((now * 0.9) % 1.0) * (gap - 7), y + bh / 2 - 1.5, 3, 3))


def stage_hits(stages, x0, span, y0=12.0):
	gap, bw, y, bh, zone = geom(len(stages), x0, span, y0)
	return [(QRectF(x0 + i * (bw + gap) - 2, y, bw + 4, bh + 20), label + (f" · {sub}" if sub else "")) for i, (label, _state, sub) in enumerate(stages)]


def bar(p, r, prog, c, now, dpr, seg=6.0, gap=2.0):
	w, h = r.width(), r.height()
	ghost = eva.bake(("bar-ghost", c, seg, gap), w, h, dpr, lambda q: eva.seg_bar(q, QRectF(0, 0, w, h), 0.0, c, seg, gap, 38, 240))
	lit = eva.bake(("bar-lit", c, seg, gap), w, h, dpr, lambda q: eva.seg_bar(q, QRectF(0, 0, w, h), 1.0, c, seg, gap, 38, 240))
	p.drawPixmap(r.topLeft(), ghost)
	n = max(1, int((w + gap) // (seg + gap)))
	if prog is not None:
		k = int(round(max(0.0, min(1.0, prog)) * n))
		if k:
			a, b = r.left(), r.left() + k * (seg + gap) - gap
		else:
			return
	else:
		span = w * 0.18
		xs = r.left() - span + ((now * 0.45) % 1.0) * (w + span)
		a, b = max(r.left(), xs), min(r.right(), xs + span)
		if b <= a:
			return
	p.save()
	p.setClipRect(QRectF(a, r.top(), b - a, h))
	p.drawPixmap(r.topLeft(), lit)
	p.restore()


def title_row(view, p, state, now, x0, span, y, dpr):
	title = state["title"]
	busy = bool(state.get("busy"))
	tf, sf, df = eva.font(15, 800), eva.font(12, 600), eva.font(13, 600, False)
	color = theme.ACCENT if busy else (theme.PHOSPHOR if title == "READY" else theme.AMBER)
	tag_w = eva.text_w(tf, title) + 18
	elapsed = time.time() - state["t0"] if state.get("t0") else None
	prog = state.get("progress")
	queued = state.get("queued") or 0
	right = "  ·  ".join(([f"{max(0.0, min(1.0, prog)):.0%}"] if busy and prog is not None else []) + ([f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"] if elapsed is not None else []) + ([f"+{queued} QUEUED"] if queued else []))
	rw = eva.text_w(sf, right)
	room = max(40.0, span - tag_w - rw - 40)
	detail = state.get("detail") or ""
	shown = eva.elide(df, detail, room)
	if shown != detail:
		view.hud_hits.append((QRectF(x0 + tag_w + 8, y, room + 8, 20), detail))
	if busy:
		draw_plate(p, QRectF(x0, y, tag_w, 20), None, fill=color, fill_a=200 + 55 * (0.5 + 0.5 * math.sin(now * 5.0)))

	def draw(q):
		draw_plate(q, QRectF(0, 0, tag_w, 20), color, 255 if busy else 200, 1.0)
		q.setFont(tf)
		q.setPen(eva.col(theme.ON_ACCENT if busy else color))
		q.drawText(QPointF(9, base(tf, 0, 20)), title)
		q.setFont(df)
		q.setPen(eva.col(theme.TEXT, 215) if busy else eva.col(theme.MUTED))
		q.drawText(QPointF(tag_w + 12, base(df, 0, 20)), shown)
		q.setFont(sf)
		q.setPen(eva.col(theme.MUTED))
		q.drawText(QPointF(span - rw, base(sf, 0, 20)), right)

	p.drawPixmap(QPointF(x0, y), eva.bake(("title", title, busy, shown, right, theme.ACCENT_NAME), span, 22, dpr, draw))
	if busy:
		bar(p, QRectF(x0, y + 27, span, 5), prog, theme.ACCENT, now, dpr)


def side_row(view, p, side, now, x0, span, y, dpr):
	tf, df = eva.font(13, 800), eva.font(13, 600, False)
	tag_w = eva.text_w(tf, side["title"]) + 16
	room = max(40.0, span - tag_w - 14)
	shown = eva.elide(df, side["detail"], room)
	if shown != side["detail"]:
		view.hud_hits.append((QRectF(x0 + tag_w + 6, y, room + 8, 18), side["detail"]))

	def draw(q):
		draw_plate(q, QRectF(0, 0, tag_w, 18), theme.COOL, 230, 1.0)
		q.setFont(tf)
		q.setPen(eva.col(theme.COOL2))
		q.drawText(QPointF(8, base(tf, 0, 18)), side["title"])
		q.setFont(df)
		q.setPen(eva.col(theme.TEXT, 190))
		q.drawText(QPointF(tag_w + 10, base(df, 0, 18)), shown)

	p.drawPixmap(QPointF(x0, y), eva.bake(("side", side["title"], shown, theme.ACCENT_NAME), span, 20, dpr, draw))
	bar(p, QRectF(x0, y + 22, span, 4), max(0.0, min(1.0, side.get("progress") or 0.0)), theme.COOL, now, dpr)


def hud(view, p, state):
	now = time.perf_counter()
	W, dpr = view.width(), view.devicePixelRatioF()
	x0, span, tx0, tspan, (sy, ty, sdy) = layout(view)
	stages = tuple(tuple(s) for s in (state.get("stages") or []))
	key = (stages, round(x0, 1), round(span, 1), sy, theme.ACCENT_NAME)
	view.hud_hits = stage_hits(stages, x0, span, sy)
	p.drawPixmap(QPointF(0, 0), eva.bake(("stages-under",) + key, W, STAGE_H, dpr, lambda q: stage_under(q, stages, x0, span, sy)))
	stage_live(p, stages, x0, span, now, sy)
	p.drawPixmap(QPointF(0, 0), eva.bake(("stages-over",) + key, W, STAGE_H, dpr, lambda q: stage_over(q, stages, x0, span, sy)))
	if not state.get("title"):
		return
	title_row(view, p, state, now, tx0, tspan, ty, dpr)
	if state.get("side"):
		side_row(view, p, state["side"], now, tx0, tspan, sdy, dpr)


def chip_icon(p, key, cx, cy, fg, on, mode):
	p.setBrush(Qt.BrushStyle.NoBrush)
	p.setPen(eva.pen(fg, 255, 1.0))
	if key == "reset":
		eva.cross(p, cx, cy, 5.0, fg, 255, 1.0)
		p.drawEllipse(QPointF(cx, cy), 2.6, 2.6)
	elif key == "center":
		p.drawEllipse(QPointF(cx, cy), 4.2, 4.2)
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(eva.col(fg))
		p.drawEllipse(QPointF(cx, cy), 1.5, 1.5)
	elif key == "rotate":
		r = QRectF(cx - 3, cy - 3, 6, 6)
		if on:
			p.setPen(Qt.PenStyle.NoPen)
			p.setBrush(eva.col(theme.NEON))
		else:
			p.setPen(eva.pen(theme.DIM, 255, 1.0))
		p.drawRect(r)
	elif key == "concepts":
		if mode != "off":
			p.setBrush(eva.col(fg, 120))
		p.drawPolygon(QPolygonF([QPointF(cx, cy - 4.6), QPointF(cx + 4.6, cy), QPointF(cx, cy + 4.6), QPointF(cx - 4.6, cy)]))


def chips(view, p):
	f = eva.font(12, 700)
	items = tuple((key, text) for key, text in (("help", "?"), ("reset", "RESET"), ("center", "CENTER"), ("rotate", "ROTATE"), ("concepts", f"CONCEPTS {view.cmode.upper()}")))
	x = view.width() - 10.0
	y = view.height() - 30.0
	view.chips = {}
	for key, text in items:
		w = eva.text_w(f, text) + 16 + (0.0 if key == "help" else 15.0)
		x -= w
		view.chips[key] = QRectF(x, y, w, 20)
		x -= 6
	left = x + 6 - 3
	rects = view.chips

	def draw(q):
		q.translate(-left, -y + 3)
		for key, text in items:
			r = rects[key]
			hot = key == view.chip_hover
			on = key == "rotate" and view.rotate
			draw_plate(q, r, theme.NEON if hot else theme.ACCENT, 255 if hot else 125, 1.0, theme.ACCENT if hot else "#000000", 70 if hot else 190, 3.0)
			fg = "#ffffff" if hot else (theme.ACCENT2 if on else theme.MUTED)
			icon = 0.0 if key == "help" else 15.0
			if icon:
				chip_icon(q, key, r.left() + 11, y + 10, fg, on, view.cmode)
			q.setFont(f)
			q.setPen(eva.col(fg))
			q.drawText(QPointF(r.left() + 8 + icon, base(f, y, 20)), text)

	hot_key = (view.chip_hover, view.rotate, view.cmode)
	pm = eva.bake(("chips", items, hot_key, theme.ACCENT_NAME), view.width() - left, 26, view.devicePixelRatioF(), draw)
	p.drawPixmap(QPointF(left, y - 3), pm)


def card(view, p, title, rows, pinned):
	kf, vf, tf = eva.font(11, 700), eva.font(13, 600, False), eva.font(15, 800)
	lw = max(eva.text_w(kf, k) for k, _v, _c in rows) + 16
	vw = max(eva.text_w(vf, v) for _k, v, _c in rows)
	width = min(max(vw + lw + 34, eva.text_w(tf, title) + 78), view.width() * 0.62)
	lh, head = 17.0, 40.0
	height = head + lh * len(rows) + 10
	x, y = view.card_pos(width, height)
	r = QRectF(x, y, width, height)
	if pinned:
		if view.card_rect is None or abs(view.card_rect.width() - width) > 0.5 or abs(view.card_rect.height() - height) > 0.5:
			view.ov_dirty = True
		view.card_rect = r
		view.card_close = QRectF(x + width - 26, y + 6, 20, 20)
	room = width - lw - 26
	shown = [eva.elide(vf, v, room) for _k, v, _c in rows]
	view.card_tips = [(QRectF(x + 12 + lw, y + head + 11 + i * lh - 12, room, lh), v) for i, (_k, v, _c) in enumerate(rows) if pinned and shown[i] != v]
	hot_close = pinned and view.chip_hover == "close"
	key = ("card", title, tuple((k, s, str(c)) for (k, _v, c), s in zip(rows, shown)), pinned, hot_close, round(width, 1), theme.ACCENT_NAME)

	def draw(q):
		accent, hi = (theme.ACCENT, theme.NEON) if pinned else (theme.COOL, theme.COOL2)
		on = theme.ON_ACCENT if pinned else "#000000"
		box = QRectF(0, 0, width, height)
		q.fillRect(box, eva.col("#000000", 238))
		eva.box(q, box, accent, 150, 1.0)
		eva.brackets(q, box, hi, 235, 8.0, 2.0)
		text = eva.elide(tf, title, width - 86)
		pr = QRectF(10, 8, eva.text_w(tf, text) + 18, 22)
		draw_plate(q, pr, accent, 255, 1.0, accent, 255)
		q.setFont(tf)
		q.setPen(eva.col(on))
		q.drawText(QPointF(pr.left() + 9, base(tf, pr.top(), 22)), text)
		if pinned:
			cr = QRectF(width - 26, 6, 20, 20)
			draw_plate(q, cr, accent, 255 if hot_close else 170, 1.0, accent if hot_close else None, 255, 3.0)
			c = cr.center()
			q.setPen(eva.pen(on if hot_close else hi, 255, 1.4))
			q.drawLine(QPointF(c.x() - 3.5, c.y() - 3.5), QPointF(c.x() + 3.5, c.y() + 3.5))
			q.drawLine(QPointF(c.x() - 3.5, c.y() + 3.5), QPointF(c.x() + 3.5, c.y() - 3.5))
		q.setPen(eva.pen(accent, 70, 1.0, Qt.PenStyle.DotLine))
		q.drawLine(QPointF(10, head - 3.5), QPointF(width - 10, head - 3.5))
		key_c = dim_accent(0.78)
		dim_c = theme.mix(theme.DIM, theme.MUTED, 0.6).name()
		yy = head + 11
		for (k, _v, col), s in zip(rows, shown):
			q.setFont(kf)
			q.setPen(eva.col(key_c))
			q.drawText(QPointF(12, yy), k)
			q.setFont(vf)
			q.setPen(theme.qc(dim_c if col == theme.DIM else col))
			q.drawText(QPointF(12 + lw, yy), s)
			yy += lh

	p.drawPixmap(QPointF(x, y), eva.bake(key, width, height, view.devicePixelRatioF(), draw))


def help_html(text):
	row = lambda key, desc, c=theme.TEXT: f'<tr><td style="color: {theme.ACCENT2}; padding-right: 14px; white-space: nowrap;">{html.escape(key)}</td><td style="color: {c};">{html.escape(desc)}</td></tr>'
	rows = []
	for i, line in enumerate(text.split("\n")):
		m = HELP_ROW.match(line)
		if i == 0:
			rows.append(f'<tr><td colspan="2" style="color: {theme.NEON}; font-size: 13pt; font-weight: bold; padding-bottom: 3px;">{html.escape(line)}</td></tr>')
		elif not line.strip():
			rows.append('<tr><td colspan="2" style="font-size: 4pt;">&nbsp;</td></tr>')
		elif m:
			rows.append(row(m.group(1), m.group(2)))
		elif line[0] == " ":
			rows.append(row("", line.strip()))
		else:
			rows.append(f'<tr><td colspan="2" style="color: {theme.MUTED};">{html.escape(line)}</td></tr>')
	return f'<div style="font-family: \'Bahnschrift SemiCondensed\', \'Bahnschrift\', \'Arial Narrow\'; font-size: 10pt;"><table cellspacing="0" cellpadding="1">{"".join(rows)}</table></div>'


def trait_parts(tr, hot):
	v = float(tr.get("value", 0.0))
	b = tr.get("base")
	lever = tr.get("lever")
	nf, vf, df = eva.font(14, 700), eva.font(14, 600), eva.font(12, 600)
	parts = [(tr["name"], nf, theme.qc(theme.TEXT if hot else "#c9c5be"), 0.0), (f"{v:+.2f}", vf, theme.signed_color(v), 8.0)]
	if b is not None and abs(v - b) >= 0.01:
		parts.append((f"{'▲' if v - b >= 0 else '▼'}{abs(v - b):.2f}", df, theme.qc(theme.MUTED), 6.0))
	if lever:
		parts.append((f"◆{lever:+.2f}", df, theme.qc(theme.ACCENT2), 8.0))
	return parts


def parts_w(parts):
	return sum(eva.text_w(f, t) + g for t, f, _c, g in parts)


def draw_parts(p, x, y, parts):
	for t, f, c, g in parts:
		x += g
		p.setFont(f)
		p.setPen(c)
		p.drawText(QPointF(x, y), t)
		x += eva.text_w(f, t)


def label_font(kind, hot=False):
	if kind == "band":
		return eva.font(12, 700)
	if kind == "concept":
		return eva.font(13 if hot else 12, 700 if hot else 600)
	return eva.font(13, 600)


def trait_label(view, p, k, tr, anchor, tip, left, hot):
	parts = trait_parts(tr, hot)
	tw = parts_w(parts)
	x = max(6.0, min(anchor.x() - tw - 14 if left else anchor.x() + 14, view.width() - tw - 6.0))
	y = anchor.y() + 5
	extra = 30 if k == view.hover else 0
	view.rects.append(QRectF(x - 2, y - 12, tw + 4, 17 + extra))
	view.hits.append((QRectF(x - 4, y - 13, tw + 8, 19), "trait", k))
	p.setPen(eva.pen(theme.TEXT if hot else theme.MUTED, 220 if hot else 110, 1.0))
	p.drawLine(anchor, QPointF(anchor.x() + (-10 if left else 10), anchor.y()))
	p.setPen(Qt.PenStyle.NoPen)
	p.setBrush(eva.col(theme.NEON if hot else theme.MUTED, 230 if hot else 150))
	p.drawRect(QRectF(anchor.x() - 1.5, anchor.y() - 1.5, 3, 3))
	if tip is not None and hot:
		p.setPen(eva.pen(theme.ACCENT, 160, 1.0, Qt.PenStyle.DotLine))
		p.drawLine(anchor, tip)
	if hot:
		eva.brackets(p, QRectF(x - 6, y - 14, tw + 12, 20 + extra), theme.NEON, 210, 4.0, 1.2)
	draw_parts(p, x, y, parts)
	if extra:
		sub_lines(p, x, y, tr)


def sub_lines(p, x, y, tr):
	f = eva.font(12, 600, False)
	p.setFont(f)
	p.setPen(theme.qc(theme.MUTED))
	p.drawText(QPointF(x, y + 14), f"+ {tr.get('plus', '')}   − {tr.get('minus', '')}")
	reads = tr.get("reads")
	if reads:
		p.setPen(theme.qc(theme.ACCENT2, 200))
		p.drawText(QPointF(x, y + 28), "reads as: " + " ".join(s.strip() for s in reads[:6]))
