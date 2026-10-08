import math
import time

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ui import eva, theme
from ui.widgets import Breath, GlyphButton, button, toggle

GLYPH = {"pending": "○", "running": "▶", "done": "✓", "failed": "✗", "stopped": "⊘", "replaced": "↻"}


def fmt(sec):
	if sec is None:
		return "?"
	sec = max(0, int(round(sec)))
	if sec < 60:
		return f"{sec}s"
	if sec < 3600:
		return f"{sec // 60}m{sec % 60:02d}"
	return f"{sec // 3600}h{sec % 3600 // 60:02d}"


class TodoView(QWidget):
	remove = Signal(int)
	stop = Signal(int)
	moved = Signal(int, object)
	ROW = 26

	def __init__(self, parent=None):
		super().__init__(parent)
		self.items = []
		self.plan = None
		self.hover = -1
		self.show_done = True
		self.drag = None
		self.drop = None
		self.setMouseTracking(True)
		self.setMinimumHeight(self.ROW)

	def set_plan(self, plan):
		self.plan = plan
		self.items = [i for i in plan.items if self.show_done or i["state"] in ("pending", "running")]
		self.setMinimumHeight(max(self.ROW, len(self.items) * self.ROW + 4))
		self.update()

	def row_at(self, y):
		i = int(y // self.ROW)
		return i if 0 <= i < len(self.items) else -1

	def x_zone(self):
		return self.width() - 22

	def color(self, it):
		st = it["state"]
		if st == "running":
			return theme.NEON if eva.on() else theme.ACCENT
		if st == "done":
			return theme.WARN if it.get("verdict") else (theme.PHOSPHOR if eva.on() else theme.OK)
		if st == "failed":
			return theme.ERR
		if st in ("stopped", "replaced"):
			return theme.DIM
		return theme.MUTED if it.get("info") else theme.TEXT

	def right_text(self, it, now):
		st = it["state"]
		if st == "running":
			left = self.plan.remaining(it, now) if self.plan else None
			return f"{fmt(now - (it['t0'] or now))} · ~{fmt(left)}" if left is not None else fmt(now - (it["t0"] or now))
		if st == "pending":
			return ("~" if it.get("known") else "≈") + fmt(it.get("eta")) if it.get("eta") is not None else "?"
		if it.get("t0") and it.get("t1"):
			return fmt(it["t1"] - it["t0"])
		return ""

	def paintEvent(self, e):
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		nerv = eva.on()
		f = eva.font(12.5, 600, False) if nerv else theme.mono(8)
		sf = eva.font(11, 600) if nerv else theme.mono(7.5)
		now = time.time()
		w = self.width()
		if not self.items:
			p.setFont(f)
			p.setPen(theme.qc(theme.DIM))
			p.drawText(QRectF(6, 0, w - 12, self.ROW * 2), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, "nothing planned · lever changes, measures, chats, goals and loads land here" if not (self.plan and self.plan.items) else "nothing waiting · SHOW DONE lists the finished operations")
			p.end()
			return
		for i, it in enumerate(self.items):
			y = i * self.ROW
			r = QRectF(0, y + 1, w, self.ROW - 2)
			col = self.color(it)
			if i == self.hover and self.drag is None:
				p.fillRect(r, theme.qc(theme.LINE, 255))
			if self.drag is not None and self.drag["moved"] and it["id"] == self.drag["id"]:
				p.fillRect(r, theme.qc(theme.ACCENT, 40))
			if it["state"] == "running":
				p.fillRect(r, theme.qc(col, 26))
				prog = it.get("progress")
				if prog is not None:
					p.fillRect(QRectF(0, y + self.ROW - 4, w * max(0.0, min(1.0, prog)), 2), theme.qc(col, 220))
				else:
					x = (now * 90) % (w + 60) - 60
					p.fillRect(QRectF(max(0, x), y + self.ROW - 4, min(60, w - max(0, x)), 2), theme.qc(col, 160))
			p.setFont(f)
			glyph = "‖" if it.get("held") and it["state"] == "pending" else ("!" if it["state"] == "done" and it.get("verdict") else GLYPH.get(it["state"], "·"))
			if it["state"] == "running":
				a = int(150 + 105 * (0.5 + 0.5 * math.sin(now * 6)))
				p.setPen(theme.qc(col, a))
			else:
				p.setPen(theme.qc(col))
			p.drawText(QRectF(4, y, 16, self.ROW), Qt.AlignmentFlag.AlignCenter, glyph)
			right = self.right_text(it, now)
			p.setFont(sf)
			rw = (eva.text_w(sf, right) if nerv else p.fontMetrics().horizontalAdvance(right)) + 6
			xr = self.x_zone() - 4
			p.setPen(theme.qc(theme.MUTED if it["state"] != "running" else col))
			p.drawText(QRectF(xr - rw, y, rw, self.ROW), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, right)
			p.setFont(f)
			label = it["label"] + (f" · {it['detail']}" if it["state"] == "running" and it.get("detail") else "") + (f" · {it['verdict']}" if it["state"] == "done" and it.get("verdict") else "")
			room = xr - rw - 28
			text = eva.elide(f, label, room) if nerv else p.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, int(room))
			p.setPen(theme.qc(col if it["state"] in ("running", "failed") else (theme.TEXT if it["state"] == "pending" and not it.get("info") else col)))
			p.drawText(QRectF(24, y, room, self.ROW), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
			if i == self.hover and self.drag is None and it["state"] in ("pending", "running"):
				cx, cy = self.x_zone() + 11, y + self.ROW / 2
				p.setPen(eva.pen(theme.ERR if it["state"] == "running" else theme.TEXT, 230, 1.4))
				p.drawLine(QPointF(cx - 4, cy - 4), QPointF(cx + 4, cy + 4))
				p.drawLine(QPointF(cx - 4, cy + 4), QPointF(cx + 4, cy - 4))
		if self.drag is not None and self.drag["moved"] and self.drop is not None:
			y = self.drop * self.ROW
			p.fillRect(QRectF(0, y - 1, w, 2), theme.qc(theme.NEON if nerv else theme.ACCENT))
			p.setBrush(theme.qc(theme.NEON if nerv else theme.ACCENT))
			p.setPen(Qt.PenStyle.NoPen)
			p.drawPolygon(QPolygonF([QPointF(0, y - 4), QPointF(5, y), QPointF(0, y + 4)]))
		p.end()

	def tip(self, it):
		st = it["state"]
		lines = [it["label"]]
		if st == "pending" and it.get("held"):
			lines.append("kept after it failed or was stopped: your settings stay as you set them, APPLY tries again")
		if st == "pending":
			lines.append("waiting · " + (f"estimated {fmt(it['eta'])}" + ("" if it.get("known") else " (first time: a rough guess, learned after one run)") if it.get("eta") is not None else "no estimate yet"))
			lines.append("drag it up or down to change the order · click ✕ to remove it from the list")
		elif st == "running":
			lines.append("running · " + (it.get("detail") or ""))
			lines.append("click ✕ to stop it (what it half did is dropped)")
		elif st == "failed":
			lines.append("failed: " + (it.get("error") or "?"))
		elif st == "done":
			lines.append("done in " + fmt((it.get("t1") or 0) - (it.get("t0") or 0)))
			if it.get("verdict"):
				lines += it["verdict"].split(" · ")
		elif st == "stopped":
			lines.append("stopped")
		elif st == "replaced":
			lines.append("replaced by a newer change")
		return "\n".join(lines)

	def drop_slot(self, y):
		return max(0, min(len(self.items), int(round(y / self.ROW))))

	def mouseMoveEvent(self, e):
		y = e.position().y()
		if self.drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
			if not self.drag["moved"] and abs(y - self.drag["y"]) < 5:
				return
			self.drag["moved"] = True
			self.drop = self.drop_slot(y)
			self.setCursor(Qt.CursorShape.ClosedHandCursor)
			self.update()
			return
		i = self.row_at(y)
		self.setCursor(Qt.CursorShape.OpenHandCursor if i >= 0 and self.items[i]["state"] == "pending" and e.position().x() < self.x_zone() else Qt.CursorShape.ArrowCursor)
		if i != self.hover:
			self.hover = i
			self.setToolTip(self.tip(self.items[i]) if i >= 0 else "")
			self.update()

	def leaveEvent(self, e):
		self.hover = -1
		self.update()

	def mousePressEvent(self, e):
		i = self.row_at(e.position().y())
		if i < 0 or e.button() != Qt.MouseButton.LeftButton:
			return
		it = self.items[i]
		if e.position().x() < self.x_zone():
			if it["state"] == "pending":
				self.drag = {"id": it["id"], "y": e.position().y(), "moved": False}
			return
		if it["state"] == "pending":
			self.remove.emit(it["id"])
		elif it["state"] == "running":
			self.stop.emit(it["id"])

	def mouseReleaseEvent(self, e):
		d, slot = self.drag, self.drop
		self.drag = self.drop = None
		self.setCursor(Qt.CursorShape.ArrowCursor)
		self.update()
		if d is None or not d["moved"] or slot is None:
			return
		before = self.items[slot]["id"] if slot < len(self.items) else None
		if before != d["id"]:
			self.moved.emit(d["id"], before)


class TodoPanel(QWidget):
	apply = Signal()
	pause = Signal()
	auto_changed = Signal(bool)
	show_done_changed = Signal(bool)
	clear_pending = Signal()
	remove = Signal(int)
	stop = Signal(int)
	moved = Signal(int, object)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.plan = None
		lay = QVBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(5)
		self.summary = QLabel("")
		theme.tag_font(self.summary, 7.5)
		self.summary.setStyleSheet(f"color: {theme.MUTED};")
		self.summary.setWordWrap(True)
		top = QHBoxLayout()
		top.setSpacing(6)
		top.addWidget(self.summary, 1)
		self.b_off = GlyphButton("power")
		self.b_off.setCheckable(True)
		self.b_off.setProperty("toggle", True)
		self.b_off.setFixedWidth(30)
		self.b_off.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
		top.addWidget(self.b_off, 0, Qt.AlignmentFlag.AlignTop)
		self.b_drop = button("CANCEL ALL", "remove every operation still waiting from the list\nthe running one goes on: ✕ on its line (or Esc) stops it")
		self.b_drop.setProperty("danger", True)
		self.b_drop.clicked.connect(self.clear_pending.emit)
		self.b_off.setFixedHeight(self.b_drop.sizeHint().height())
		self.b_drop.installEventFilter(self)
		top.addWidget(self.b_drop, 0, Qt.AlignmentFlag.AlignTop)
		lay.addLayout(top)
		self.view = TodoView()
		self.view.remove.connect(self.remove.emit)
		self.view.stop.connect(self.stop.emit)
		self.view.moved.connect(self.moved.emit)
		self.scroll = QScrollArea()
		self.scroll.setWidget(self.view)
		self.scroll.setWidgetResizable(True)
		self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
		self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		self.scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
		lay.addWidget(self.scroll, 1)
		row = QHBoxLayout()
		row.setSpacing(6)
		self.auto = toggle("AUTO", "AUTO (outlined when on): every operation runs as soon as it is added (no APPLY)\noff: operations wait in the list until APPLY · a chat message or a LOAD with nothing else waiting still starts at once")
		self.auto.toggled.connect(self.auto_changed.emit)
		self.b_done = toggle("SHOW DONE", "show the finished operations in the list too (done, failed, stopped, replaced: the last 60 are kept)\noff: only what waits and what runs", True)
		self.b_done.toggled.connect(self._show_done)
		row.addWidget(self.auto)
		row.addWidget(self.b_done)
		row.addStretch(1)
		self.b_go = button("APPLY", "run the operations of the list, in order", accent=True)
		self.b_go.clicked.connect(self._go)
		self.breath = Breath(self.b_go)
		row.addWidget(self.b_go)
		lay.addLayout(row)
		self.timer = QTimer(self)
		self.timer.timeout.connect(self._tick)
		self.timer.start(500)

	def eventFilter(self, obj, e):
		if obj is self.b_drop and e.type() == QEvent.Type.Resize and self.b_off.height() != obj.height():
			self.b_off.setFixedHeight(obj.height())
		return False

	def _go(self):
		if self.plan is not None and (self.plan.going or self.plan.until is not None) and not self.plan.auto:
			self.pause.emit()
		else:
			self.apply.emit()

	def _show_done(self, on):
		self.view.show_done = bool(on)
		self.show_done_changed.emit(bool(on))
		self.refresh()

	def set_show_done(self, on):
		self.b_done.blockSignals(True)
		self.b_done.setChecked(bool(on))
		self.b_done.blockSignals(False)
		self.view.show_done = bool(on)
		self.refresh()

	def set_plan(self, plan):
		self.plan = plan
		self.auto.blockSignals(True)
		self.auto.setChecked(plan.auto)
		self.auto.blockSignals(False)
		self.view.set_plan(plan)
		self.refresh()

	def refresh(self):
		plan = self.plan
		if plan is None:
			return
		pend = plan.pending()
		run = plan.running()
		total, known = plan.total()
		going = plan.going or plan.auto or plan.until is not None
		if run is None and not pend:
			text = "nothing to do" if not plan.items else "all done"
		else:
			text = f"{len(pend)} waiting" + (" · 1 running" if run else "") + (f" · {'~' if known else '≈'}{fmt(total)} left" if total else "")
			if pend and not going:
				text += " · APPLY to run"
		self.summary.setText(text)
		self.summary.setStyleSheet(f"color: {theme.WARN if pend and not going else theme.MUTED};")
		self.b_go.setEnabled(not plan.auto and bool(pend or going))
		self.breath.set_on(bool(pend) and not going)
		self.b_go.setText("PAUSE" if going and not plan.auto and pend else "APPLY")
		self.b_go.setToolTip("finish the running operation, then wait (the others stay in the list)" if going and not plan.auto and pend else "run the operations of the list, in order")
		self.b_drop.setEnabled(bool(pend))
		self.view.set_plan(plan)

	def _tick(self):
		if self.plan is not None and self.plan.running() is not None:
			self.refresh()
