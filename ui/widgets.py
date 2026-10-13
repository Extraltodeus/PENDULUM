import html
import math
import os
import re
import subprocess
import sys
import time
from urllib.parse import quote

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QCursor, QDesktopServices, QFont, QFontMetrics, QFontMetricsF, QIcon, QImage, QPainter, QPixmap, QPainterPath, QPen, QPolygonF, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPlainTextEdit, QProxyStyle, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QStyle, QStyledItemDelegate, QTextBrowser, QToolTip, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QWidgetAction

import config
from audio.bank import SOUNDS
from engine.gpus import FLASH, KV_TYPES, LOAD_DEFAULTS, SPLIT_MODES
from lens.fit import GGUF_ARCHS, hf_repo, slug
from lens.fit import plan as fit_plan
from hypno.lorakeys import STYLES as LORA_STYLES, example_key
from persona.coherence import EXTRA_DEFAULTS, threshold
from persona.coherence import PROMPTS as COH_PROMPTS
from persona.target import HARD as DOSE_HARD
from ui import eva, evahud, theme
from ui.frame import EdgeFilter, Frameless
from ui.iris import CLICKS
from ui.sfx import GROUPS as SOUND_GROUPS
from ui.sfx import SFX
from ui.sfx import folder as sound_folder


def tab_focus(w):
	w.setFocusPolicy(Qt.FocusPolicy.TabFocus)
	return w


def button(text, tip="", accent=False, width=None, click=True):
	b = tab_focus(QPushButton(text))
	b.setAutoDefault(False)
	theme.tag_font(b, 8)
	b.setCursor(Qt.CursorShape.PointingHandCursor)
	if tip:
		b.setToolTip(tip)
	if accent:
		b.setProperty("accent", True)
	if width:
		b.setFixedWidth(width)
	if click:
		b.pressed.connect(lambda: SFX.play("click"))
	return b


def toggle(text, tip="", checked=False):
	b = button(text, tip)
	b.setCheckable(True)
	b.setChecked(bool(checked))
	b.setProperty("toggle", True)
	return b


class SegSwitch(QWidget):
	picked = Signal(str)

	def __init__(self, items, parent=None):
		super().__init__(parent)
		from PySide6.QtWidgets import QButtonGroup
		lay = QHBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(0)
		self.group = QButtonGroup(self)
		self.group.setExclusive(True)
		self.buttons = {}
		self.labels = {}
		for item in items:
			key, label, tip = item[:3]
			short = item[3] if len(item) > 3 else label
			b = toggle(label, f"{label}: {tip}" if short != label else tip)
			b.setFixedHeight(22)
			self.group.addButton(b)
			lay.addWidget(b)
			self.buttons[key] = b
			self.labels[key] = (label, short)
			b.clicked.connect(lambda _c=False, k=key: self.picked.emit(k))
		self.pad = 22
		self._fit()

	def widths(self, i):
		pad = max([self.pad] + [b.sizeHint().width() - b.fontMetrics().horizontalAdvance(b.text()) for b in self.buttons.values()])
		return {k: self.buttons[k].fontMetrics().horizontalAdvance(v[i]) + pad for k, v in self.labels.items()}

	def full_width(self):
		return sum(self.widths(0).values())

	def _fit(self):
		short = self.width() < self.full_width() and self.isVisible()
		for k, w in self.widths(1).items():
			b = self.buttons[k]
			b.setMinimumWidth(w)
			text = self.labels[k][1 if short else 0]
			if b.text() != text:
				b.setText(text)

	def resizeEvent(self, e):
		super().resizeEvent(e)
		self._fit()

	def set_value(self, key):
		b = self.buttons.get(key)
		if b is not None and not b.isChecked():
			b.setChecked(True)

	def value(self):
		return next((k for k, b in self.buttons.items() if b.isChecked()), None)


def menu_title(menu, text):
	act = QWidgetAction(menu)
	act.setText(text)
	l = QLabel(text)
	theme.tag_font(l, 8, True)
	l.setStyleSheet(f"background: transparent; color: {theme.NEON if eva.on() else theme.MUTED}; padding: 5px 22px 5px 17px;")
	act.setDefaultWidget(l)
	menu.addAction(act)
	return act


class HScroll(QObject):
	STEP = 6

	def eventFilter(self, obj, ev):
		if ev.type() != QEvent.Type.Wheel or not isinstance(obj, QLineEdit):
			return False
		d = ev.angleDelta()
		step = d.x() or (d.y() if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0)
		if not step:
			return False
		if not obj.hasFocus():
			return False
		r = obj.contentsRect()
		y = r.center().y()
		if step < 0:
			obj.setCursorPosition(min(len(obj.text()), obj.cursorPositionAt(QPoint(r.right() - 2, y)) + self.STEP))
		else:
			obj.setCursorPosition(max(0, obj.cursorPositionAt(QPoint(r.left() + 2, y)) - self.STEP))
		return True


def hscroll(edit):
	f = getattr(hscroll, "filter", None)
	if f is None:
		f = hscroll.filter = HScroll(QApplication.instance())
	edit.installEventFilter(f)
	edit.setToolTip((edit.toolTip() + "\n" if edit.toolTip() else "") + "Shift+wheel: scroll the text sideways")
	return edit


class StepSlider(QSlider):
	def __init__(self, step=1, parent=None):
		super().__init__(Qt.Orientation.Horizontal, parent)
		self.wheel_step = step
		self._acc = 0

	def wheelEvent(self, e):
		e.accept()
		self._acc += e.angleDelta().y() or -e.angleDelta().x()
		n = int(self._acc / 120)
		if n:
			self._acc -= n * 120
			self.setValue(self.value() + n * self.wheel_step)


class TipStyle(QProxyStyle):
	def __init__(self, key):
		super().__init__(key)
		self.delay = 700

	def styleHint(self, hint, opt=None, widget=None, ret=None):
		if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
			return self.delay
		return super().styleHint(hint, opt, widget, ret)


_TIP_STYLE = []


def set_tooltip_delay(ms):
	app = QApplication.instance()
	if app is None:
		return
	if not _TIP_STYLE:
		_TIP_STYLE.append(TipStyle(app.style().name()))
		app.setStyle(_TIP_STYLE[0])
	_TIP_STYLE[0].delay = max(0, int(ms))


class WheelGuard(QObject):
	def eventFilter(self, obj, ev):
		if ev.type() == QEvent.Type.Wheel and not (obj.hasFocus() and obj.underMouse()):
			ev.ignore()
			return True
		return False


WHEEL_GUARD = WheelGuard()


def guard_wheel(w):
	w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
	w.installEventFilter(WHEEL_GUARD)
	return w


class EvaDialog(Frameless, QDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setStyleSheet(theme.STYLE)
		self.edge_filter = None

	def restyle(self):
		self.setStyleSheet(theme.STYLE)
		theme.refont(self)
		self.fit_margins()
		self.update()

	def frameless(self, title, sub="", tip=""):
		self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
		self.edge_filter = EdgeFilter(self)
		QApplication.instance().installEventFilter(self.edge_filter)
		w = QWidget()
		h = QHBoxLayout(w)
		h.setContentsMargins(0, 0, 0, 0)
		h.setSpacing(2)
		h.addWidget(DialogTitle(title, sub, (tip + "\n" if tip else "") + "drag to move · double-click: full screen or back"), 1)
		for kind, fn, hint in (("max", self.toggle_max, "full screen or back (double-click the title does the same)"), ("close", self.close, "close")):
			b = WindowButton(kind)
			b.setToolTip(hint)
			b.clicked.connect(fn)
			h.addWidget(b)
		return w

	def hideEvent(self, e):
		if self.edge_filter is not None:
			self.edge_filter._set(None)
		super().hideEvent(e)

	def fit_margins(self):
		lay = self.layout()
		if lay is None or not eva.on():
			return
		m = lay.contentsMargins()
		lay.setContentsMargins(max(m.left(), 18), max(m.top(), 14), max(m.right(), 18), max(m.bottom(), 12))

	def setVisible(self, on):
		if on:
			self.fit_margins()
		super().setVisible(on)

	def paintEvent(self, e):
		p = QPainter(self)
		p.fillRect(self.rect(), theme.qc(theme.BG))
		if eva.on():
			r = QRectF(self.rect()).adjusted(4, 4, -4, -4)
			eva.box(p, r, theme.ACCENT, 70)
			eva.brackets(p, r, theme.NEON, 235, 10.0, 2.0)
		elif self.edge_filter is not None:
			p.setPen(QPen(theme.qc(theme.LINE2), 1))
			p.drawRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))
		p.end()


class DialogTitle(QWidget):
	def __init__(self, title, sub="", tip=""):
		super().__init__()
		self.title = title
		self.sub = sub
		self.tip = tip
		self._press = None
		self.setFixedHeight(28)
		self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
		self.sync_tip()

	def _frameless(self):
		w = self.window()
		return hasattr(w, "drag_from") and bool(w.windowFlags() & Qt.WindowType.FramelessWindowHint)

	def mousePressEvent(self, e):
		self._press = e.globalPosition().toPoint() if e.button() == Qt.MouseButton.LeftButton and self._frameless() else None

	def mouseMoveEvent(self, e):
		if self._press is None or not e.buttons() & Qt.MouseButton.LeftButton or (e.globalPosition().toPoint() - self._press).manhattanLength() < 4:
			return
		self._press = None
		self.window().drag_from(e.globalPosition().toPoint())

	def mouseReleaseEvent(self, e):
		self._press = None

	def mouseDoubleClickEvent(self, e):
		self._press = None
		if self._frameless():
			self.window().toggle_max()

	def plain_text(self):
		return self.title + (f"  ·  {self.sub}" if self.sub else "")

	def shown(self):
		if not eva.on():
			return self.sub
		return eva.elide(eva.font(13, 600, False), self.sub, self.width() - eva.text_w(eva.font(15, 800), self.title) - 52)

	def sync_tip(self):
		cut = bool(self.sub) and (self.shown() != self.sub if eva.on() else QFontMetrics(theme.mono(9.5, True)).horizontalAdvance(self.plain_text()) > self.width())
		self.setToolTip("\n".join(t for t in (self.sub if cut else "", self.tip) if t))

	def resizeEvent(self, e):
		super().resizeEvent(e)
		self.sync_tip()

	def paintEvent(self, e):
		p = QPainter(self)
		W = self.width()
		if not eva.on():
			p.setFont(theme.mono(9.5, True))
			p.setPen(theme.qc(theme.TEXT))
			p.drawText(QRectF(0, 0, W, self.height()), Qt.AlignmentFlag.AlignVCenter, p.fontMetrics().elidedText(self.plain_text(), Qt.TextElideMode.ElideRight, W))
			p.end()
			return
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		tf, df = eva.font(15, 800), eva.font(13, 600, False)
		r = QRectF(0, 3, eva.text_w(tf, self.title) + 20, 22)
		evahud.draw_plate(p, r, theme.ACCENT, 255, 1.0, theme.ACCENT, 255)
		p.setFont(tf)
		p.setPen(eva.col(theme.ON_ACCENT))
		p.drawText(QPointF(10, evahud.base(tf, 3, 22)), self.title)
		x = r.right() + 12
		text = self.shown()
		if text:
			p.setFont(df)
			p.setPen(eva.col(theme.TEXT))
			p.drawText(QPointF(x, evahud.base(df, 3, 22)), text)
			x += eva.text_w(df, text) + 10
		if W - x > 24:
			p.setPen(eva.pen(theme.ACCENT, 50, 1.0, Qt.PenStyle.DotLine))
			p.drawLine(QPointF(x, 14.5), QPointF(W - 2, 14.5))
		p.end()


class Panel(QFrame):
	def __init__(self, title, parent=None):
		super().__init__(parent)
		self.title = title
		self.info = ""
		self.info_color = theme.MUTED
		self.corner = None
		self.setObjectName("panel")
		self.restyle()
		self.body = QVBoxLayout(self)
		self.body.setContentsMargins(10, 30, 10, 8)
		self.body.setSpacing(4)

	def restyle(self):
		self.setStyleSheet(f"QFrame#panel {{ background: {theme.PANEL}; border: {'none' if eva.on() else '1px solid ' + theme.LINE}; }}")

	def set_info(self, text, color=None):
		self.info = text
		self.info_color = color or theme.MUTED
		self.setToolTip(text if self.info_elided(text) else "")
		self.update()

	def info_elided(self, text):
		return bool(text) and eva.on() and eva.text_w(eva.font(13, 600), text) > self.info_room()

	def info_room(self):
		return self.width() - eva.text_w(eva.font(15, 700), self.title) - 56 - self.corner_w()

	def set_corner(self, w):
		self.corner = w
		w.setParent(self)
		w.installEventFilter(self)
		self.place_corner()

	def corner_w(self):
		return self.corner.width() + 10 if self.corner is not None and not self.corner.isHidden() else 0

	def place_corner(self):
		c = self.corner
		if c is not None:
			c.move(self.width() - c.width() - 10, max(0, (25 - c.height()) // 2 + (1 if eva.on() else 0)))
			self.update()

	def eventFilter(self, obj, e):
		if obj is self.corner and e.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Hide):
			self.place_corner()
		return False

	def resizeEvent(self, e):
		super().resizeEvent(e)
		self.place_corner()
		if self.info:
			self.setToolTip(self.info if self.info_elided(self.info) else "")

	def paintEvent(self, e):
		if eva.on():
			self._paint_nerv()
			return
		super().paintEvent(e)
		p = QPainter(self)
		p.setFont(theme.mono(8, True))
		p.setPen(theme.qc(theme.TEXT))
		p.drawText(10, 18, "// " + self.title)
		if self.info:
			p.setFont(theme.mono(8))
			p.setPen(theme.qc(self.info_color))
			w = p.fontMetrics().horizontalAdvance(self.info)
			p.drawText(self.width() - w - 10 - self.corner_w(), 18, self.info)
		p.setPen(QPen(theme.qc(theme.LINE), 1))
		p.drawLine(1, 25, self.width() - 2, 25)
		p.end()

	def _paint_nerv(self):
		p = QPainter(self)
		w, h = self.width(), self.height()
		r = QRectF(0, 0, w, h)
		p.fillRect(r, theme.qc(theme.PANEL))
		eva.box(p, r, theme.ACCENT, 70)
		eva.brackets(p, r, theme.NEON, 235, 9, 2.0)
		t = eva.tag(p, 9, 4, self.title, eva.font(15, 700), theme.ACCENT, h=20, pad=6, text_c=theme.NEON, bar=3)
		end = w - 12 - self.corner_w()
		if self.info:
			fi = eva.font(13, 600)
			text = eva.elide(fi, self.info, end - t.right() - 18)
			tw = eva.text_w(fi, text)
			p.setFont(fi)
			p.setPen(theme.qc(self.info_color))
			p.drawText(QPointF(end - tw, 19), text)
			end -= tw + 10
		if end - t.right() > 30:
			p.setPen(eva.pen(theme.ACCENT, 46, 1.0, Qt.PenStyle.DotLine))
			p.drawLine(QPointF(t.right() + 8, 14.5), QPointF(end, 14.5))
		p.end()


class HeaderBar(QWidget):
	status_clicked = Signal(object)
	EXPENDABLE = ("ARCH", "BAND")

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setFixedHeight(54)
		self.stats = []
		self.status = ("IDLE", theme.MUTED)
		self.queued = 0
		self.job = ""
		self.status_rect = QRectF()
		self.stats_rect = QRectF()
		self.gpus = []
		self.lay = QHBoxLayout(self)
		self.lay.setContentsMargins(0, 0, 8, 0)
		self.lay.addStretch(1)
		self.buttons = {}
		self.t0 = time.perf_counter()
		timer = QTimer(self)
		timer.timeout.connect(self.update)
		timer.start(500)

	def add_button(self, key, text, tip=""):
		b = button(text, tip)
		b.setFixedHeight(26)
		self.lay.addWidget(b)
		self.buttons[key] = b
		return b

	def add_glyph(self, key, glyph, tip=""):
		b = GlyphButton(glyph, tip)
		b.setFixedSize(30, 26)
		self.lay.addWidget(b)
		self.buttons[key] = b
		return b

	def add_window_controls(self, window):
		self.lay.addSpacing(10)
		for key, fn in (("min", window.showMinimized), ("max", window.toggle_max), ("close", window.close)):
			b = WindowButton(key)
			b.clicked.connect(fn)
			self.lay.addWidget(b)
			self.buttons[key] = b

	def paintEvent(self, e):
		if eva.on():
			self._paint_nerv()
			return
		p = QPainter(self)
		p.fillRect(self.rect(), theme.qc(theme.BG))
		p.setPen(QPen(theme.qc(theme.LINE), 1))
		p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
		p.setFont(theme.mono(12.5, True))
		p.setPen(theme.qc(theme.TEXT))
		p.drawText(16, 25, "P E N D U L U M")
		p.setFont(theme.mono(7.2))
		p.setPen(theme.qc(theme.MUTED))
		p.drawText(16, 42, "WORKSPACE · PERSONA · LEVERS")
		limit = min((b.geometry().x() for b in self.buttons.values()), default=self.width()) - 18
		label, color = self.status
		text = label + (f"  +{self.queued} queued" if self.queued else "")
		p.setFont(theme.mono(8.5, True))
		room = limit - p.fontMetrics().horizontalAdvance(text) - 24 - 12
		x = 250
		name_w = 260
		p.setFont(theme.mono(9.5, True))
		stats = list(self.stats)
		need = lambda: sum(max(p.fontMetrics().horizontalAdvance(str(v)[:60]), 60) + 30 for _l, v, _c in stats) + len(self.gpus) * 112
		for drop in self.EXPENDABLE:
			if 250 + need() > room:
				stats = [s for s in stats if s[0] != drop]
		if 250 + need() > room:
			name_w = max(90, 260 - (250 + need() - room))
		for s_label, value, s_color in stats:
			p.setFont(theme.mono(9.5, True))
			fm = p.fontMetrics()
			v = fm.elidedText(str(value), Qt.TextElideMode.ElideRight, name_w)
			w = max(fm.horizontalAdvance(v), fm.horizontalAdvance(s_label))
			if x + w > room:
				break
			p.setPen(theme.qc(s_color or theme.TEXT))
			p.drawText(x, 39, v)
			p.setFont(theme.mono(7.2))
			p.setPen(theme.qc(theme.MUTED))
			p.drawText(x, 20, s_label)
			x += w + 30
		for g in self.gpus:
			if x + 90 > room:
				break
			p.setFont(theme.mono(7.2))
			p.setPen(theme.qc(theme.MUTED))
			p.drawText(x, 20, g["name"])
			w = 90
			p.fillRect(QRectF(x, 29, w, 6), theme.qc(theme.LINE2))
			used = 1 - g["free"] / max(g["total"], 1)
			p.fillRect(QRectF(x, 29, w * used, 6), theme.qc(theme.ACCENT if used > 0.9 else theme.TEXT, 200))
			p.setFont(theme.mono(7.2))
			p.drawText(x, 48, f"{(g['total'] - g['free']) / 2**30:.1f}/{g['total'] / 2**30:.0f} GB")
			x += w + 22
		self.stats_rect = QRectF(250, 0, max(0, x - 250), self.height())
		p.setFont(theme.mono(8.5, True))
		p.setPen(theme.qc(color))
		blink = int((time.perf_counter() - self.t0) * 2) % 2 == 0 or label == "IDLE"
		if blink:
			p.setBrush(theme.qc(color))
			p.drawEllipse(QPointF(x + 6, 31), 3.5, 3.5)
		text = p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, max(0, int(limit - x - 16)))
		p.drawText(x + 16, 35, text)
		self.status_rect = QRectF(x, 18, p.fontMetrics().horizontalAdvance(text) + 24, 24)
		p.end()

	def logo_font(self):
		f = QFont(eva.font(31, 800))
		f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.2)
		return f

	def _subtitle(self, p, ink, text, pad=4.0, h=14.0):
		key = (round(ink.left(), 2), round(ink.width(), 2), text, theme.cond_family())
		if getattr(self, "_sub_key", None) != key:
			f = QFont(eva.font(11, 700))
			base = f.letterSpacing() if f.letterSpacingType() == QFont.SpacingType.AbsoluteSpacing else 0.0
			f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, base)
			w0 = QFontMetricsF(f).tightBoundingRect(text).width()
			extra = (ink.width() - 2 * pad - w0) / max(1, len(text) - 1)
			f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, base + max(-1.0, extra))
			tight = QFontMetricsF(f).tightBoundingRect(text)
			self._sub_key, self._sub = key, (f, tight)
		f, tight = self._sub
		r = QRectF(ink.left(), 35.0, ink.width(), h)
		eva.box(p, r, theme.ACCENT, 255, 1.0)
		p.setFont(f)
		p.setPen(theme.qc(theme.TEXT))
		p.drawText(QPointF(r.left() + pad - tight.left(), r.top() + (h + eva.metrics(f).capHeight()) / 2), text)
		return r

	def fit_items(self, x, avail, label_f, value_f):
		items = [["stat", label, str(value), color, max(eva.text_w(label_f, label), eva.text_w(value_f, str(value))) + 17] for label, value, color in self.stats]
		items += [["gpu", g, None, None, 136] for g in self.gpus]
		total = lambda: sum(it[4] + 8 for it in items)
		if items and items[0][0] == "stat" and total() > avail:
			items[0][4] = max(118, items[0][4] - (total() - avail))
		for label in self.EXPENDABLE:
			if total() > avail:
				items = [it for it in items if not (it[0] == "stat" and it[1] == label)]
		while total() > avail and any(it[0] == "gpu" for it in items):
			items.pop(max(i for i, it in enumerate(items) if it[0] == "gpu"))
		while total() > avail and len(items) > 1:
			items.pop()
		if items and total() > avail:
			items[0][4] = max(60, avail - 8)
		return items

	def _status_box(self, p, r):
		label, color = self.status
		blink = int((time.perf_counter() - self.t0) * 2) % 2 == 0
		f = eva.font(20, 800)
		cap = eva.font(10, 700)
		err = color in (theme.ERR, theme.ALARM)
		busy = not err and color not in (theme.OK, theme.MUTED) and label not in ("IDLE", "READY")
		if err:
			eva.hazard(p, r, theme.ALARM, 14, 255, "#000000")
			eva.box(p, r.adjusted(5, 9, -5, -5), theme.ALARM, 255, 1.0, "#000000")
			text_c = "#ffffff"
		elif busy and blink:
			eva.box(p, r, color, 255, 1.5, color, 230)
			text_c = theme.ON_ACCENT if color == theme.ACCENT else "#000000"
		else:
			eva.box(p, r, color, 230 if label != "IDLE" else 150, 1.5)
			text_c = color
		p.setFont(cap)
		p.setPen(eva.col(text_c if not err else theme.ALARM, 170))
		if not err:
			p.drawText(QPointF(r.left() + 6, r.top() + 11), "STATUS")
		if self.queued:
			q = f"+{self.queued} QUEUED"
			p.setPen(eva.col(text_c if not err else theme.ALARM, 235))
			p.drawText(QPointF(r.right() - 6 - eva.text_w(cap, q), r.top() + 11), q)
		text = eva.elide(f, label, r.width() - 12)
		p.setFont(f)
		p.setPen(eva.col(text_c))
		fm = eva.metrics(f)
		p.drawText(QPointF(r.center().x() - fm.horizontalAdvance(text) / 2, r.bottom() - 7 if not err else r.bottom() - 9), text)

	def _paint_nerv(self):
		p = QPainter(self)
		W, H = self.width(), self.height()
		p.fillRect(self.rect(), QColor(0, 0, 0))
		p.fillRect(QRectF(0, H - 1, W, 1), eva.col(theme.ACCENT, 160))
		eva.ruler(p, 0, W, H - 1, theme.ACCENT, 6, 10, 2.0, 95)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		lf = self.logo_font()
		eva.glow_text(p, 14, 30, "PENDULUM", lf, theme.NEON, 55, 2.0)
		title = QPainterPath()
		title.addText(QPointF(14, 30), lf, "PENDULUM")
		sub = self._subtitle(p, title.boundingRect(), "PERSONA STEERING SYSTEM")
		p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
		x = max(14 + eva.text_w(lf, "PENDULUM"), sub.right()) + 22
		limit = min((b.geometry().x() for b in self.buttons.values()), default=W) - 14
		label, _c = self.status
		sf = eva.font(20, 800)
		status_w = max(104, eva.text_w(sf, label) + 24, (eva.text_w(eva.font(10, 700), f"STATUS  +{self.queued} QUEUED") + 14) if self.queued else 0)
		sr = QRectF(limit - status_w, 6, status_w, 41)
		self.status_rect = sr
		label_f, value_f, small_f = eva.font(11, 600), eva.font(17.5, 700), eva.font(12, 600)
		y0, bh = 6.0, 41.0
		x_stats = x
		for kind, a, b, color, w in self.fit_items(x, sr.left() - 14 - x, label_f, value_f):
			r = QRectF(x, y0, w, bh)
			eva.box(p, r, theme.ACCENT, 85)
			p.fillRect(QRectF(r.right() - 3, y0, 3, bh), eva.col(theme.ACCENT, 210))
			if kind == "stat":
				p.setFont(label_f)
				p.setPen(eva.col(theme.NEON, 215))
				p.drawText(QPointF(x + 6, y0 + 13), a)
				p.setFont(value_f)
				p.setPen(theme.qc(color or theme.TEXT))
				p.drawText(QPointF(x + 6, y0 + 34), eva.elide(value_f, b, w - 15))
			else:
				used = 1 - a["free"] / max(a["total"], 1)
				p.setFont(label_f)
				p.setPen(eva.col(theme.NEON, 215))
				p.drawText(QPointF(x + 6, y0 + 13), eva.elide(label_f, a["name"], w - 15))
				eva.seg_bar(p, QRectF(x + 6, y0 + 18, w - 17, 8), used, theme.ALARM if used > 0.9 else theme.TEXT, 3.0, 1.0, 40, 220)
				p.setFont(small_f)
				p.setPen(theme.qc(theme.TEXT))
				p.drawText(QPointF(x + 6, y0 + 37), f"{(a['total'] - a['free']) / 2**30:.1f} / {a['total'] / 2**30:.0f} GB")
			x += w + 8
		self.stats_rect = QRectF(x_stats, 0, max(0, x - x_stats), H)
		self._status_box(p, sr)
		p.end()

	def status_tip(self):
		label, _c = self.status
		rows = [f"STATUS: {label}"]
		if self.job:
			rows.append(f"running: {self.job}")
		rows.append(f"{self.queued} job{'s' if self.queued != 1 else ''} queued" if self.queued else "nothing queued")
		rows.append("click: show the jobs (stop the running one, remove queued ones)")
		return "\n".join(rows)

	def event(self, e):
		if e.type() == QEvent.Type.ToolTip:
			pos = e.position() if hasattr(e, "position") else QPointF(e.pos())
			if self.status_rect.contains(pos):
				QToolTip.showText(e.globalPos(), self.status_tip(), self)
				return True
			if self.stats_rect.contains(pos) and (self.stats or self.gpus):
				rows = [f"{label}: {value}" for label, value, _c in self.stats] + [f"{g['name']}: {(g['total'] - g['free']) / 2**30:.1f} / {g['total'] / 2**30:.0f} GB" for g in self.gpus]
				QToolTip.showText(e.globalPos(), "\n".join(rows), self)
				return True
		return super().event(e)

	def mousePressEvent(self, e):
		if e.button() == Qt.MouseButton.LeftButton and self.status_rect.contains(e.position()):
			self._press = None
			self.status_clicked.emit(e.globalPosition().toPoint())
			return
		self._press = e.globalPosition().toPoint() if e.button() == Qt.MouseButton.LeftButton else None

	def mouseMoveEvent(self, e):
		press = getattr(self, "_press", None)
		w = self.window()
		if press is None or not e.buttons() & Qt.MouseButton.LeftButton or (e.globalPosition().toPoint() - press).manhattanLength() < 4:
			return
		self._press = None
		if w.maxed():
			w.restore_for_drag(e.globalPosition().toPoint())
		if w.windowHandle():
			w.windowHandle().startSystemMove()

	def mouseReleaseEvent(self, e):
		self._press = None

	def mouseDoubleClickEvent(self, e):
		self._press = None
		if self.status_rect.contains(e.position()):
			return
		self.window().toggle_max()


class ModeSwitch(QWidget):
	changed = Signal(str)
	MODES = (("dose", "DOSE", "each slider sets how hard its lever pushes (as before; the coherence guard can still cut it)"), ("target", "TARGET", "each slider sets the measured value you want: the levers are found for you, interactions included (a few quick measures)"))

	def __init__(self, parent=None):
		super().__init__(parent)
		self.mode = "dose"
		self.pos = 0.0
		self.hover = -1
		self.setFixedHeight(22)
		self.setMinimumWidth(176)
		self.setMouseTracking(True)
		self.setCursor(Qt.CursorShape.PointingHandCursor)
		self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
		self.timer = QTimer(self)
		self.timer.timeout.connect(self._step)
		self.setToolTip("\n".join(f"{label}: {tip}" for _k, label, tip in self.MODES) + "\nclick to switch · the doses chosen in TARGET stay visible in DOSE")

	def sizeHint(self):
		return QSize(196, 22)

	def set_mode(self, mode, emit=False):
		mode = mode if mode in ("dose", "target") else "dose"
		if mode == self.mode:
			return
		self.mode = mode
		if emit:
			self.timer.start(16)
		else:
			self.pos = 1.0 if mode == "target" else 0.0
		self.update()
		if emit:
			self.changed.emit(mode)

	def _step(self):
		goal = 1.0 if self.mode == "target" else 0.0
		self.pos += (goal - self.pos) * 0.32
		if abs(goal - self.pos) < 0.01:
			self.pos = goal
			self.timer.stop()
		self.update()

	def mousePressEvent(self, e):
		if e.button() == Qt.MouseButton.LeftButton:
			self.set_mode("target" if e.position().x() >= self.width() / 2 else "dose", True)

	def keyPressEvent(self, e):
		if e.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Left, Qt.Key.Key_Right):
			self.set_mode("dose" if self.mode == "target" else "target", True)
			return
		super().keyPressEvent(e)

	def mouseMoveEvent(self, e):
		h = 1 if e.position().x() >= self.width() / 2 else 0
		if h != self.hover:
			self.hover = h
			self.update()

	def leaveEvent(self, e):
		self.hover = -1
		self.update()

	def paintEvent(self, e):
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
		half = r.width() / 2
		nerv = eva.on()
		c = theme.mix(theme.ACCENT, theme.CYAN, self.pos)
		knob = QRectF(r.left() + 2 + self.pos * (half - 1), r.top() + 2, half - 3, r.height() - 4)
		if nerv:
			p.fillRect(r, QColor(0, 0, 0))
			eva.box(p, r, theme.ACCENT, 150, 1.0)
			p.fillRect(knob, c)
			p.fillRect(QRectF(knob.left(), knob.bottom() - 2, knob.width(), 2), theme.mix(c, "#000000", 0.45))
		else:
			p.setPen(QPen(theme.qc(theme.LINE2), 1))
			p.setBrush(theme.qc(theme.PANEL2))
			p.drawRoundedRect(r, 4, 4)
			p.setPen(Qt.PenStyle.NoPen)
			p.setBrush(c)
			p.drawRoundedRect(knob, 3, 3)
		f = eva.font(11.5, 700) if nerv else theme.mono(7.5, True)
		p.setFont(f)
		fm = QFontMetricsF(f)
		for k, (key, label, _tip) in enumerate(self.MODES):
			seg = QRectF(r.left() + k * half, r.top(), half, r.height())
			on = (k == 1) == (self.pos >= 0.5)
			ink = theme.qc("#000000") if on else theme.qc(theme.TEXT if self.hover == k else theme.MUTED)
			tw = fm.horizontalAdvance(label)
			x = seg.center().x() - (tw + 11) / 2
			cy = seg.center().y()
			p.setPen(Qt.PenStyle.NoPen)
			p.setBrush(ink)
			if key == "target":
				p.drawPolygon(QPolygonF([QPointF(x + 3.5, cy - 4), QPointF(x + 7, cy), QPointF(x + 3.5, cy + 4), QPointF(x, cy)]))
			else:
				for j, hh in enumerate((3.0, 5.0, 7.5)):
					p.drawRect(QRectF(x + j * 2.6, cy + 3.75 - hh, 1.8, hh))
			p.setPen(ink)
			p.drawText(QPointF(x + 11, cy + fm.capHeight() / 2), label)
		if self.hasFocus():
			if nerv:
				eva.brackets(p, r, theme.NEON, 220, 5, 1.2)
			else:
				p.setPen(QPen(theme.qc(theme.ACCENT2, 160), 1))
				p.setBrush(Qt.BrushStyle.NoBrush)
				p.drawRoundedRect(r, 4, 4)
		p.end()


class Breath(QWidget):
	GLOW = 11

	def __init__(self, host, period=2.0):
		super().__init__(host)
		self.host = host
		self.period = period
		self.on = False
		self.t0 = time.perf_counter()
		self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.frame)
		self.hide()

	def set_on(self, on):
		on = bool(on)
		if on == self.on:
			return
		self.on = on
		if on:
			self.t0 = time.perf_counter()
			win = self.host.window()
			if self.parentWidget() is not win:
				self.setParent(win)
				self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
			self.timer.start(33)
			self.frame()
		else:
			self.timer.stop()
			self.hide()

	def frame(self):
		if not self.host.isVisible():
			self.hide()
			return
		m = self.GLOW
		tl = self.host.mapTo(self.parentWidget(), QPoint(0, 0))
		self.setGeometry(QRect(tl, self.host.size()).adjusted(-m, -m, m, m))
		if not self.isVisible():
			self.show()
		self.raise_()
		self.update()

	def paintEvent(self, e):
		k = 0.5 - 0.5 * math.cos((time.perf_counter() - self.t0) / self.period * 2 * math.pi)
		c = theme.NEON if eva.on() else theme.ACCENT
		m = self.GLOW
		inner = QRectF(self.rect()).adjusted(m, m, -m, -m)
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		p.setBrush(Qt.BrushStyle.NoBrush)
		reach = 3 + (m - 3) * k
		for i in range(1, m + 1):
			if i > reach:
				break
			a = (1 - i / (reach + 1)) ** 1.6 * (60 + 120 * k)
			p.setPen(QPen(theme.qc(c, int(a)), 1.6))
			p.drawRoundedRect(inner.adjusted(-i, -i, i, i), i * 0.6, i * 0.6)
		p.fillRect(inner.adjusted(1, 1, -1, -1), theme.qc(c, int(10 + 30 * k)))
		p.setPen(QPen(theme.qc(c, int(150 + 105 * k)), 2.0))
		p.drawRect(inner.adjusted(1, 1, -1, -1))
		p.end()


class GlyphButton(QPushButton):
	def __init__(self, glyph, tip=""):
		super().__init__("")
		self.glyph = glyph
		tab_focus(self)
		self.setAutoDefault(False)
		self.setCursor(Qt.CursorShape.PointingHandCursor)
		if tip:
			self.setToolTip(tip)

	def paintEvent(self, e):
		super().paintEvent(e)
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		hot = self.underMouse()
		p.setPen(QPen(theme.qc("#ffffff" if hot else (theme.NEON if eva.on() else theme.TEXT)), 1.3))
		p.setBrush(Qt.BrushStyle.NoBrush)
		cx, cy = self.width() / 2, self.height() / 2 + 1
		if self.glyph == "folder":
			p.drawPolyline(QPolygonF([QPointF(cx - 7, cy - 3), QPointF(cx - 7, cy - 6), QPointF(cx - 2.5, cy - 6), QPointF(cx - 1, cy - 4), QPointF(cx + 7, cy - 4), QPointF(cx + 7, cy + 5), QPointF(cx - 7, cy + 5), QPointF(cx - 7, cy - 3), QPointF(cx + 7, cy - 3)]))
		elif self.glyph == "power":
			if self.isChecked():
				p.setPen(QPen(theme.qc(theme.ALARM if eva.on() else theme.ERR), 1.6))
			cy -= 1
			p.drawEllipse(QRectF(cx - 6, cy - 5, 12, 12))
			p.drawLine(QPointF(cx, cy - 8), QPointF(cx, cy + 0.5))
		elif self.glyph in ("sound", "muted"):
			cy -= 1
			p.drawPolygon(QPolygonF([QPointF(cx - 7, cy - 2), QPointF(cx - 4, cy - 2), QPointF(cx, cy - 6), QPointF(cx, cy + 6), QPointF(cx - 4, cy + 2), QPointF(cx - 7, cy + 2)]))
			if self.glyph == "sound":
				for r in (3.5, 6.5):
					p.drawArc(QRectF(cx - r + 1, cy - r, 2 * r, 2 * r), -45 * 16, 90 * 16)
			else:
				p.setPen(QPen(theme.qc(theme.ERR if eva.on() else theme.WARN), 1.4))
				p.drawLine(QPointF(cx + 2.5, cy - 3), QPointF(cx + 8, cy + 3))
				p.drawLine(QPointF(cx + 2.5, cy + 3), QPointF(cx + 8, cy - 3))
		p.end()

	def set_glyph(self, glyph, tip=""):
		self.glyph = glyph
		if tip:
			self.setToolTip(tip)
		self.update()


class WindowButton(QPushButton):
	def __init__(self, kind):
		super().__init__("")
		self.kind = kind
		self.setObjectName("win_" + kind)
		self.setFixedSize(30, 26)
		self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.setAutoDefault(False)
		self.setCursor(Qt.CursorShape.PointingHandCursor)

	def paintEvent(self, e):
		super().paintEvent(e)
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		if eva.on():
			col = theme.qc("#ffffff" if self.underMouse() else theme.NEON)
		else:
			col = theme.qc(theme.ERR if self.kind == "close" and self.underMouse() else (theme.ACCENT2 if self.underMouse() else theme.TEXT))
		p.setPen(QPen(col, 1.2))
		cx, cy = self.width() / 2, self.height() / 2
		if self.kind == "min":
			p.drawLine(QPointF(cx - 5, cy + 3), QPointF(cx + 5, cy + 3))
		elif self.kind == "max":
			p.drawRect(QRectF(cx - 5, cy - 4, 10, 8))
		else:
			p.drawLine(QPointF(cx - 4.5, cy - 4.5), QPointF(cx + 4.5, cy + 4.5))
			p.drawLine(QPointF(cx - 4.5, cy + 4.5), QPointF(cx + 4.5, cy - 4.5))
		p.end()


class LeverTable(QWidget):
	committed = Signal(str, float)
	hovered = Signal(int)
	peek = Signal(object)
	edit_requested = Signal(str)
	duplicate_requested = Signal(str)
	delete_requested = Signal(str)
	restore_requested = Signal(str)
	recalibrate_requested = Signal(str)
	goals_requested = Signal(str)
	exempt_requested = Signal(str, bool)
	extract_requested = Signal(str)
	target_committed = Signal(str, object)
	as_lever_requested = Signal(str)
	order_changed = Signal(object)
	RANGE = 1.5
	STEP = 0.05
	NAME_W = 104
	NAME_MAX = 180
	TRANGE = 1.0
	TLIMIT = 0.95
	TOL = 0.08

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setMouseTracking(True)
		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.traits = []
		self.values = {}
		self.pending = set()
		self.info = {}
		self.planned = {}
		self.base = {}
		self.current = {}
		self.scale = 1.0
		self.factors = {}
		self.free = set()
		self.exempt = set()
		self.hidden = []
		self.staged = False
		self.mode = "dose"
		self.range = self.RANGE
		self.targets = {}
		self.aligning = False
		self.drag = None
		self._press = None
		self._grab = None
		self.moving = None
		self.custom_order = False
		self.focus_row = -1
		self.hover_row = -1
		self.highlight = -1
		self.concept = None
		self.row_h = 21
		self.top = 30
		self._wheel_timer = QTimer(self)
		self._wheel_timer.setSingleShot(True)
		self._wheel_timer.timeout.connect(self._commit_wheel)
		self._wheel_row = -1
		self._anim = QTimer(self)
		self._anim.timeout.connect(self.update)

	def sizeHint(self):
		return QSize(520, self.top + max(1, len(self.traits)) * self.row_h + 6)

	def set_traits(self, traits):
		if self.moving is not None:
			self.unsetCursor()
		self.moving = self._grab = None
		self.traits = traits
		fm = QFontMetrics(theme.mono(8, True))
		widest = max([fm.horizontalAdvance((("└ " if t.get("sub") else "") + t["name"]).upper()) + (14 if t.get("sub") else 4) for t in traits] or [0])
		self.name_w = int(max(self.NAME_W, min(self.NAME_MAX, widest + 10)))
		self.setMinimumHeight(self.top + max(1, len(traits)) * self.row_h + 6)
		self.updateGeometry()
		self.update()

	def geo(self):
		w = self.width()
		return (self.NAME_W if eva.on() else getattr(self, "name_w", self.NAME_W)), w - 272, w - 262, w - 190, w - 180, w - 6

	def set_range(self, v):
		v = max(0.5, min(5.0, float(v or self.RANGE)))
		if abs(v - self.range) > 1e-9:
			self.range = v
			self.update()

	def snap(self, v):
		return round(max(-self.range, min(self.range, round(v / self.STEP) * self.STEP)), 2)

	def shown(self, v):
		return max(-self.range, min(self.range, v))

	def beyond_mark(self, p, hx, cy, v, color):
		if abs(v) <= self.range + 1e-9:
			return
		s = 1.0 if v > 0 else -1.0
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(theme.qc(color))
		p.drawPolygon(QPolygonF([QPointF(hx + s * 4, cy - 4), QPointF(hx + s * 4, cy + 4), QPointF(hx + s * 8.5, cy)]))

	def tsnap(self, v):
		return round(max(-self.TLIMIT, min(self.TLIMIT, round(v / self.STEP) * self.STEP)), 2)

	def targeting(self, t):
		return self.mode == "target" and not t.get("goal") and not t.get("lever_only")

	def unplaced(self, t):
		k = t.get("parent") or t["id"]
		return bool(t.get("goal")) and k not in self.info and k not in self.planned

	def unguarded_mark(self, p, x, cy, color):
		path = QPainterPath(QPointF(x - 4.5, cy - 6))
		path.lineTo(x + 4.5, cy - 6)
		path.lineTo(x + 4.5, cy)
		path.quadTo(x + 4.5, cy + 4.5, x, cy + 6.5)
		path.quadTo(x - 4.5, cy + 4.5, x - 4.5, cy)
		path.closeSubpath()
		p.save()
		p.setBrush(Qt.BrushStyle.NoBrush)
		p.setPen(QPen(theme.qc(color), 1.3))
		p.drawPath(path)
		p.setPen(QPen(theme.qc(color), 1.6))
		p.drawLine(QPointF(x - 6, cy + 6), QPointF(x + 6, cy - 7))
		p.restore()

	def _unplaced_text(self, p, cy, x0, x1, font):
		p.setFont(font)
		p.setPen(theme.qc(theme.MUTED))
		p.drawText(QRectF(x0, cy - 8, x1 - x0, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, QFontMetrics(font).elidedText("not placed on this model · GOALS → TRAIN", Qt.TextElideMode.ElideRight, int(x1 - x0)))

	def now_of(self, tid):
		c = self.current.get(tid)
		return c if c is not None else self.base.get(tid)

	def set_mode(self, mode):
		self.mode = mode if mode in ("dose", "target") else "dose"
		self.update()

	def set_at(self, i, x):
		t = self.traits[i]
		if self.unplaced(t):
			return
		if self.targeting(t):
			self.targets[t["id"]] = self.value_at(x, True)
		else:
			self.values[t["id"]] = self.value_at(x)

	def emit_row(self, t):
		if self.unplaced(t):
			return
		if self.targeting(t):
			self.target_committed.emit(t["id"], self.targets.get(t["id"]))
		else:
			self.committed.emit(t["id"], self.values.get(t["id"], 0.0))

	def set_factors(self, factors, free=None):
		self.factors = {k: float(v) for k, v in (factors or {}).items()}
		if free is not None:
			self.free = set(free)
		self.update()

	def mult(self, tid):
		return (1.0 if tid in self.free else self.scale) * self.factors.get(tid, 1.0)

	def blamed(self, tid):
		return self.factors.get(tid, 1.0) < 0.999

	def set_concept(self, tid, tv=0.0):
		i = next((k for k, t in enumerate(self.traits) if t["id"] == tid), -1) if tid else -1
		self.concept = (i, float(tv)) if i >= 0 else None
		self.update()
		return i

	def concept_mark(self, p, i, t, cy, mid, half):
		if self.concept is None or self.concept[0] != i or self.unplaced(t):
			return
		tv = self.concept[1]
		x = mid + max(-1.0, min(1.0, tv / (self.TRANGE if self.targeting(t) else self.range))) * half
		y = cy - 7.5
		p.setPen(QPen(theme.qc("#000000"), 1))
		p.setBrush(theme.signed_color(tv, 255))
		p.drawPolygon(QPolygonF([QPointF(x, y - 4), QPointF(x + 4, y), QPointF(x, y + 4), QPointF(x - 4, y)]))

	def measured_scale(self):
		goals = {t["id"] for t in self.traits if t.get("goal")}
		vals = [abs(v) for d in (self.base, self.current) for tid, v in d.items() if v is not None and tid not in goals]
		return max(1.0, math.ceil(max(vals, default=0.0) * 2 - 1e-9) / 2)

	def _tick_label(self, v):
		return "0" if not v else (f"{v:+.0f}" if float(v).is_integer() else f"{v:+.1f}")

	def paintEvent(self, e):
		busy = bool(self.pending) or (self.aligning and self.mode == "target")
		if busy and not self._anim.isActive():
			self._anim.start(80)
		elif not busy and self._anim.isActive():
			self._anim.stop()
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		if not self.traits:
			if eva.on():
				eva.standby(p, QRectF(self.rect()), "load a model to get its personality levers", "NO LEVERS")
			else:
				p.setFont(theme.mono(8))
				p.setPen(theme.qc(theme.DIM))
				p.drawText(self.rect().adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "load a model to get its personality levers")
			p.end()
			return
		if eva.on():
			self._paint_nerv(p)
			self._paint_moving(p)
			p.end()
			if self.hover_row >= 0 or self.drag is not None:
				self._emit_peek()
			return
		x_name, x_s1, x_val, x_div, x_p0b, x_p1 = self.geo()
		x_s0 = x_name + 8
		mid = (x_s0 + x_s1) / 2
		half = (x_s1 - x_s0) / 2
		p.setFont(theme.mono(7.5))
		nw = p.fontMetrics().horizontalAdvance("+0.00") + 6
		x_b1 = x_p1 - nw
		bw = x_b1 - x_p0b
		pm = x_p0b + bw / 2
		smax = self.measured_scale()
		p.setFont(theme.mono(6.8))
		fm = p.fontMetrics()
		p.setPen(theme.qc(theme.DIM))
		p.drawText(QPointF(4, 11), "LEVER")
		p.drawText(QPointF(x_s0, 11), "YOUR TARGET  ◆ target  ● now" if self.mode == "target" else "YOUR SETTING")
		p.drawText(QPointF(x_p0b, 11), "MEASURED  ▲ base  ● now")
		rng = self.TRANGE if self.mode == "target" else self.range
		for v in (-1.0, 0.0, 1.0):
			label = self._tick_label(v)
			p.drawText(QPointF(mid + v / rng * half - fm.horizontalAdvance(label) / 2, 25), label)
		for v in sorted({-smax, -1.0, 0.0, 1.0, smax}):
			label = self._tick_label(v)
			lw = fm.horizontalAdvance(label)
			p.drawText(QPointF(max(x_p0b, min(x_b1 - lw, pm + v / smax * bw / 2 - lw / 2)), 25), label)
		p.setPen(QPen(theme.qc(theme.LINE), 1))
		p.drawLine(QPointF(x_div, 2), QPointF(x_div, self.height() - 2))
		t_anim = time.perf_counter()
		for i, t in enumerate(self.traits):
			y = self.top + i * self.row_h
			cy = y + self.row_h / 2
			tid = t["id"]
			hot = i == self.hover_row or i == self.highlight or (self.concept is not None and self.concept[0] == i)
			if hot:
				p.fillRect(QRectF(0, y, x_div - 2, self.row_h), theme.qc(theme.PANEL2))
			if i == self.focus_row and self.hasFocus():
				p.setPen(QPen(theme.qc(theme.ACCENT2, 150), 1))
				p.setBrush(Qt.BrushStyle.NoBrush)
				p.drawRect(QRectF(x_s0 - 5, y + 1.5, x_s1 - x_s0 + 10, self.row_h - 3))
			p.setFont(theme.mono(8, hot))
			p.setPen(theme.qc(theme.TEXT if hot else "#b9b6b0"))
			name = p.fontMetrics().elidedText((("└ " if t.get("sub") else "") + t["name"]).upper(), Qt.TextElideMode.ElideRight, int(x_name - (14 if t.get("sub") else 4) - 6))
			p.drawText(QPointF(14 if t.get("sub") else 4, cy + 4), name)
			if self.targeting(t):
				self._target_plain(p, i, t, cy, x_s0, x_s1, mid, half, x_val, t_anim)
			elif self.unplaced(t):
				self._unplaced_text(p, cy, x_s0, x_div - 8, theme.mono(7.5))
			else:
				v = self.values.get(tid, 0.0)
				p.setPen(QPen(theme.qc(theme.LINE2), 1))
				p.drawLine(QPointF(x_s0, cy), QPointF(x_s1, cy))
				one = (x_s1 - x_s0) / 2 / self.range
				p.setPen(QPen(theme.qc(theme.WARN, 70), 3))
				p.drawLine(QPointF(x_s0, cy), QPointF(mid - one, cy))
				p.drawLine(QPointF(mid + one, cy), QPointF(x_s1, cy))
				for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
					tx = mid + tick / self.range * (x_s1 - x_s0) / 2
					p.setPen(QPen(theme.qc(theme.DIM if tick else theme.MUTED), 1))
					p.drawLine(QPointF(tx, cy - (4 if tick == 0 else 2)), QPointF(tx, cy + (4 if tick == 0 else 2)))
				hx = mid + self.shown(v) / self.range * (x_s1 - x_s0) / 2
				m = self.mult(tid)
				eff = v * m
				cut = theme.ERR
				if abs(v) > 1e-3:
					p.setPen(QPen(theme.signed_color(v, 220), 3))
					p.drawLine(QPointF(mid, cy), QPointF(mid + (max(-1.0, min(1.0, v))) * one, cy))
					if abs(v) > 1.0:
						p.setPen(QPen(theme.qc(theme.WARN, 230), 3))
						p.drawLine(QPointF(mid + (1.0 if v > 0 else -1.0) * one, cy), QPointF(hx, cy))
					if abs(m - 1.0) > 1e-3 and tid not in self.pending:
						ex = mid + self.shown(eff) / self.range * (x_s1 - x_s0) / 2
						p.setPen(QPen(theme.qc(cut, 200), 1.4))
						p.drawLine(QPointF(mid, cy + 5), QPointF(ex, cy + 5))
						p.setPen(Qt.PenStyle.NoPen)
						p.setBrush(theme.qc(cut))
						p.drawRect(QRectF(ex - 1.5, cy - 7, 3, 14))
				if tid in self.pending:
					a = 0.5 + 0.5 * math.sin(t_anim * 8)
					p.setPen(QPen(theme.qc(theme.ACCENT, int(90 + 150 * a)), 1.2))
					p.setBrush(Qt.BrushStyle.NoBrush)
					p.drawRect(QRectF(hx - 4, cy - 8, 8, 16))
				p.setPen(Qt.PenStyle.NoPen)
				p.setBrush(theme.qc(theme.TEXT if abs(v) > 1e-3 else theme.MUTED))
				p.drawRect(QRectF(hx - 2, cy - 6, 4, 12))
				self.beyond_mark(p, hx, cy, v, theme.WARN)
				p.setFont(theme.mono(8))
				txt = f"{v:+.2f}"
				if tid in self.pending:
					dots = "·" * (1 + int(t_anim * 4) % 3)
					p.setPen(theme.qc(theme.ACCENT))
					p.drawText(QPointF(x_val, cy + 4), f"{txt} {dots}")
				else:
					p.setPen(theme.signed_color(v) if abs(v) > 1e-3 else theme.qc(theme.DIM))
					p.drawText(QPointF(x_val, cy + 4), txt)
					if tid in self.exempt:
						self.unguarded_mark(p, x_div - 9, cy, theme.WARN)
					if abs(v) > 1e-3 and abs(m - 1.0) > 1e-3:
						x_eff = x_val + p.fontMetrics().horizontalAdvance(txt) + 3
						p.setFont(theme.mono(7))
						p.setPen(theme.qc(cut))
						p.drawText(QPointF(x_eff, cy + 4), f"→{eff:+.2f}")
			self.concept_mark(p, i, t, cy, mid, half)
			if t.get("sub"):
				continue
			p.setPen(QPen(theme.qc(theme.LINE), 1))
			p.drawLine(QPointF(x_p0b, cy), QPointF(x_b1, cy))
			for tick in (-1.0, -0.5, 0.5, 1.0):
				tx = pm + tick / smax * bw / 2
				th = 2.0 if abs(tick) == 1.0 else 1.0
				p.drawLine(QPointF(tx, cy - th), QPointF(tx, cy + th))
			p.drawLine(QPointF(pm, cy - 3), QPointF(pm, cy + 3))
			b = self.base.get(tid)
			c = self.current.get(tid)
			at = lambda val: pm + max(-smax, min(smax, val)) / smax * bw / 2
			if b is not None:
				p.setBrush(theme.qc(theme.MUTED))
				p.setPen(Qt.PenStyle.NoPen)
				bx = at(b)
				p.drawPolygon([QPointF(bx - 3, cy + 6), QPointF(bx + 3, cy + 6), QPointF(bx, cy + 1)])
			if c is not None:
				cx = at(c)
				p.setPen(QPen(theme.signed_color(c, 150), 1.4))
				p.drawLine(QPointF(pm, cy - 2), QPointF(cx, cy - 2))
				p.setBrush(theme.signed_color(c, 200))
				p.setPen(Qt.PenStyle.NoPen)
				p.drawEllipse(QPointF(cx, cy - 2), 2.4, 2.4)
			shown = c if c is not None else b
			if shown is not None:
				goal = bool(t.get("goal"))
				p.setFont(theme.mono(7.5))
				p.setPen(theme.qc(theme.ACCENT2 if c is not None else theme.MUTED) if goal else theme.signed_color(shown, 255 if c is not None else 150))
				p.drawText(QRectF(x_b1, cy - 8, nw, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{shown:.0%}" if goal else f"{shown:+.2f}")
		self._paint_moving(p)
		p.end()
		if self.hover_row >= 0 or self.drag is not None:
			self._emit_peek()

	def _paint_moving(self, p):
		if self.moving is None:
			return
		blocks = self.blocks()
		n = len(blocks[self.block_of(self.moving, blocks)])
		r = QRectF(1.5, self.top + self.moving * self.row_h + 0.5, self.width() - 3, n * self.row_h - 1)
		fill = QColor(theme.ACCENT)
		fill.setAlpha(34)
		p.save()
		p.setBrush(fill)
		p.setPen(QPen(QColor(theme.ACCENT), 1.5))
		p.drawRect(r)
		p.restore()

	def blocks(self):
		out = []
		for t in self.traits:
			if t.get("sub") and out:
				out[-1].append(t)
			else:
				out.append([t])
		return out

	def block_of(self, i, blocks):
		n = 0
		for k, b in enumerate(blocks):
			if n <= i < n + len(b):
				return k
			n += len(b)
		return len(blocks) - 1

	def move_to(self, y):
		blocks = self.blocks()
		i = max(0, min(len(self.traits) - 1, int((y - self.top) // self.row_h)))
		src, dst = self.block_of(self.moving, blocks), self.block_of(i, blocks)
		if src == dst:
			return
		keep = self.traits[self.focus_row]["id"] if 0 <= self.focus_row < len(self.traits) else None
		b = blocks.pop(src)
		blocks.insert(dst, b)
		self.traits = [t for blk in blocks for t in blk]
		self.moving = sum(len(x) for x in blocks[:dst])
		self.focus_row = next((k for k, t in enumerate(self.traits) if t["id"] == keep), -1)
		self.update()

	def in_grip(self, x):
		return x < self.geo()[0]

	def scroll_area(self):
		w = self.parent()
		while w is not None and not isinstance(w, QScrollArea):
			w = w.parent()
		return w

	def _slider_bg(self, x_s0, x_s1, mid, one):
		w = x_s1 - x_s0
		key = ("slider", round(one, 3), self.row_h, theme.LINE2, theme.WARN, theme.MUTED, theme.DIM)

		def draw(q):
			cy, m = self.row_h / 2, mid - x_s0
			q.setPen(eva.pen(theme.LINE2, 255, 1.0))
			q.drawLine(QPointF(0, cy), QPointF(w, cy))
			for lo, hi in ((0.0, m - one), (m + one, w)):
				eva.hazard(q, QRectF(lo, cy - 2.5, hi - lo, 5), theme.WARN, 7, 80)
			for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
				tx = math.floor(m + tick * one) + 0.5
				q.setPen(eva.pen(theme.MUTED if tick == 0 else (theme.WARN if abs(tick) == 1.0 else theme.DIM), 255, 1.0))
				q.drawLine(QPointF(tx, cy - (5 if tick == 0 else 3)), QPointF(tx, cy + (5 if tick == 0 else 3)))

		return eva.bake(key, w + 1, self.row_h, self.devicePixelRatioF(), draw)

	def _slider_bg_target(self, x_s0, x_s1, mid, half):
		w = x_s1 - x_s0
		key = ("tslider", round(half, 3), self.row_h, theme.LINE2, theme.CYAN, theme.MUTED, theme.DIM)

		def draw(q):
			cy, m = self.row_h / 2, mid - x_s0
			q.setPen(eva.pen(theme.LINE2, 255, 1.0))
			q.drawLine(QPointF(0, cy), QPointF(w, cy))
			for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
				tx = math.floor(m + tick * half) + 0.5
				q.setPen(eva.pen(theme.MUTED if tick == 0 else (theme.CYAN if abs(tick) == 1.0 else theme.DIM), 255 if tick == 0 else 170, 1.0))
				q.drawLine(QPointF(tx, cy - (5 if tick == 0 else 3)), QPointF(tx, cy + (5 if tick == 0 else 3)))

		return eva.bake(key, w + 1, self.row_h, self.devicePixelRatioF(), draw)

	def _target_marks(self, p, i, t, cy, hot, mid, half, t_anim, nerv):
		tid = t["id"]
		now = self.now_of(tid)
		tv = self.targets.get(tid)
		at = lambda v: mid + max(-1.0, min(1.0, v)) * half
		if tv is not None and now is not None:
			ok = abs(tv - now) <= self.TOL
			p.setPen(eva.pen(theme.PHOSPHOR if ok else theme.WARN, 220, 2.0) if nerv else QPen(theme.qc(theme.OK if ok else theme.WARN, 220), 2))
			p.drawLine(QPointF(at(now), cy), QPointF(at(tv), cy))
		if now is not None:
			p.setPen(QPen(theme.qc("#ffffff" if tv is None else "#000000"), 1.2))
			p.setBrush(Qt.BrushStyle.NoBrush if tv is None else theme.qc("#ffffff"))
			p.drawEllipse(QPointF(at(now), cy), 3.2, 3.2)
		if tv is None:
			return None
		hx = at(tv)
		if tid in self.pending or self.aligning:
			a = 0.5 + 0.5 * math.sin(t_anim * 8)
			if nerv:
				eva.brackets(p, QRectF(hx - 7, cy - 9, 14, 18), theme.NEON, int(90 + 160 * a), 3.5, 1.4)
			else:
				p.setPen(QPen(theme.qc(theme.ACCENT, int(90 + 150 * a)), 1.2))
				p.setBrush(Qt.BrushStyle.NoBrush)
				p.drawRect(QRectF(hx - 6, cy - 8, 12, 16))
		grab = self.drag == i
		p.setPen(QPen(theme.qc(theme.NEON), 1.0) if hot or grab else Qt.PenStyle.NoPen)
		p.setBrush(theme.qc(theme.NEON if grab else theme.CYAN))
		p.drawPolygon(QPolygonF([QPointF(hx, cy - 6.5), QPointF(hx + 5, cy), QPointF(hx, cy + 6.5), QPointF(hx - 5, cy)]))
		return tv

	def dose_tag(self, tid):
		dose = self.values.get(tid, 0.0)
		m = self.mult(tid)
		if abs(dose) > 1e-3 and abs(m - 1.0) > 1e-3:
			return f"▸{dose * m:+.2f}", theme.ALARM
		if abs(dose) >= self.range - 1e-3:
			return ("▸+MAX" if dose > 0 else "▸−MAX"), theme.WARN
		return f"▸{dose:+.2f}", theme.MUTED

	def _target_nerv(self, p, i, t, cy, hot, x_s0, x_s1, mid, half, x_val, vf, t_anim, bg):
		p.drawPixmap(QPointF(x_s0, cy - self.row_h / 2), bg)
		tv = self._target_marks(p, i, t, cy, hot, mid, half, t_anim, True)
		ty = cy + 4.5
		p.setFont(vf)
		if tv is None:
			p.setPen(theme.qc(theme.DIM))
			p.drawText(QPointF(x_val, ty), "free")
			return
		txt = f"{tv:+.2f}"
		p.setPen(theme.qc(theme.CYAN))
		p.drawText(QPointF(x_val, ty), txt)
		sf = eva.font(9.5, 600)
		p.setFont(sf)
		dose_txt, dose_col = self.dose_tag(t["id"])
		p.setPen(theme.qc(dose_col))
		p.drawText(QPointF(x_val + eva.text_w(vf, txt) + 4, ty), dose_txt)

	def _target_plain(self, p, i, t, cy, x_s0, x_s1, mid, half, x_val, t_anim):
		p.setPen(QPen(theme.qc(theme.LINE2), 1))
		p.drawLine(QPointF(x_s0, cy), QPointF(x_s1, cy))
		for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
			tx = mid + tick * half
			p.setPen(QPen(theme.qc(theme.DIM if tick else theme.MUTED), 1))
			p.drawLine(QPointF(tx, cy - (4 if tick == 0 else 2)), QPointF(tx, cy + (4 if tick == 0 else 2)))
		hot = i == self.hover_row or i == self.highlight
		tv = self._target_marks(p, i, t, cy, hot, mid, half, t_anim, False)
		p.setFont(theme.mono(8))
		if tv is None:
			p.setPen(theme.qc(theme.DIM))
			p.drawText(QPointF(x_val, cy + 4), "free")
			return
		txt = f"{tv:+.2f}"
		p.setPen(theme.qc(theme.CYAN))
		p.drawText(QPointF(x_val, cy + 4), txt)
		p.setFont(theme.mono(7))
		dose_txt, dose_col = self.dose_tag(t["id"])
		p.setPen(theme.qc(dose_col))
		p.drawText(QPointF(x_val + p.fontMetrics().horizontalAdvance(txt) + 8, cy + 4), dose_txt)

	def _gauge_bg(self, bw, smax):
		key = ("gauge", int(bw), smax, self.row_h, theme.ACCENT, theme.LINE2, theme.MUTED, theme.DIM, theme.TEXT)

		def draw(q):
			top = self.row_h / 2 - 5.5
			eva.box(q, QRectF(0, top, bw + 6, 10), theme.ACCENT, 110, 1.0)
			for x in range(0, int(bw) - 2, 4):
				q.fillRect(QRectF(3 + x, top + 2, 3, 6), eva.col(theme.LINE2))
			m = 3 + bw / 2
			for tick in (-1.0, -0.5, 0.5, 1.0):
				tx = math.floor(m + tick / smax * bw / 2) + 0.5
				q.setPen(eva.pen(theme.MUTED if abs(tick) == 1.0 else theme.DIM, 255, 1.0))
				q.drawLine(QPointF(tx, top + 1), QPointF(tx, top + (4.5 if abs(tick) == 1.0 else 3.0)))
			q.setPen(eva.pen(theme.TEXT, 210, 1.0))
			q.drawLine(QPointF(math.floor(m) + 0.5, top - 1), QPointF(math.floor(m) + 0.5, top + 11))

		return eva.bake(key, bw + 6, self.row_h, self.devicePixelRatioF(), draw)

	def _gauge_lit(self, bw, color):
		key = ("lit", int(bw), color, self.row_h)

		def draw(q):
			top = self.row_h / 2 - 5.5
			for x in range(0, int(bw) - 2, 4):
				q.fillRect(QRectF(3 + x, top + 2, 3, 6), eva.col(color, 240))

		return eva.bake(key, bw + 6, self.row_h, self.devicePixelRatioF(), draw)

	def _legend(self, p, x, y, c, f):
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(theme.qc(theme.TEXT, 220))
		p.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x + 7, y), QPointF(x + 3.5, y - 5)]))
		p.setFont(f)
		p.setPen(theme.qc(c))
		p.drawText(QPointF(x + 10, y), "BASE")
		x2 = x + 16 + eva.text_w(f, "BASE")
		p.setPen(eva.pen("#000000", 255, 1.0))
		p.setBrush(theme.qc("#ffffff"))
		p.drawEllipse(QPointF(x2 + 3, y - 2.5), 3.0, 3.0)
		p.setPen(theme.qc(c))
		p.drawText(QPointF(x2 + 9, y), "NOW")

	def _paint_nerv(self, p):
		x_name, x_s1, x_val, x_div, x_p0b, x_p1 = self.geo()
		x_s0 = x_name + 8
		mid = (x_s0 + x_s1) / 2
		half = (x_s1 - x_s0) / 2
		one = half / self.range
		p.setFont(theme.mono(7.5))
		nw = p.fontMetrics().horizontalAdvance("+0.00") + 6
		x_b1 = x_p1 - nw
		bw = x_b1 - x_p0b
		pm = x_p0b + bw / 2
		smax = self.measured_scale()
		W, H = self.width(), self.height()
		cap, lf, nf, vf = eva.font(11, 700), eva.font(10, 600), eva.font(13, 700), eva.font(12, 600)
		dim = theme.mix("#000000", theme.ACCENT, 0.62).name()
		for x, text in ((3, "LEVER"), (x_s0 - 1, "YOUR TARGET" if self.mode == "target" else "YOUR SETTING"), (x_p0b - 3, "MEASURED")):
			t = eva.tag(p, x, 1, text, cap, dim, h=14, pad=4, text_c=theme.NEON, bar=2)
		self._legend(p, t.right() + 12, 12.5, theme.MUTED, lf)
		p.setFont(lf)
		p.setPen(theme.qc(theme.MUTED))
		rng = self.TRANGE if self.mode == "target" else self.range
		for v in (-1.0, 0.0, 1.0):
			label = self._tick_label(v)
			p.drawText(QPointF(mid + v / rng * half - eva.text_w(lf, label) / 2, 25), label)
		for v in sorted({-smax, -1.0, 0.0, 1.0, smax}):
			label = self._tick_label(v)
			lw = eva.text_w(lf, label)
			p.drawText(QPointF(max(x_p0b, min(x_b1 - lw, pm + v / smax * bw / 2 - lw / 2)), 25), label)
		p.fillRect(QRectF(0, self.top - 2, W, 1), eva.col(theme.ACCENT, 70))
		p.setPen(eva.pen(theme.ACCENT, 90, 1.0, Qt.PenStyle.DotLine))
		p.drawLine(QPointF(x_div + 0.5, 2), QPointF(x_div + 0.5, H - 2))
		slider_bg = self._slider_bg(x_s0, x_s1, mid, one)
		tslider_bg = self._slider_bg_target(x_s0, x_s1, mid, half) if self.mode == "target" else None
		gauge_bg = self._gauge_bg(bw, smax)
		lits = {True: self._gauge_lit(bw, theme.ACCENT), False: self._gauge_lit(bw, theme.COOL)}
		n_seg = (int(bw) - 2 + 3) // 4
		t_anim = time.perf_counter()
		for i, t in enumerate(self.traits):
			y = self.top + i * self.row_h
			cy = y + self.row_h / 2
			tid = t["id"]
			hot = i == self.hover_row or i == self.highlight or (self.concept is not None and self.concept[0] == i)
			goal = bool(t.get("goal"))
			p.fillRect(QRectF(0, y + self.row_h - 1, W, 1), eva.col(theme.ACCENT, 20))
			if hot:
				p.fillRect(QRectF(0, y, W, self.row_h - 1), eva.col(theme.ACCENT, 26))
				p.fillRect(QRectF(0, y + 2, 2, self.row_h - 5), eva.col(theme.NEON))
			if i == self.focus_row and self.hasFocus():
				eva.brackets(p, QRectF(x_s0 - 5, y + 1, x_s1 - x_s0 + 10, self.row_h - 3), theme.NEON, 200, 5, 1.2)
			if t.get("sub"):
				p.setFont(lf)
				p.setPen(theme.qc("#ffffff" if hot else theme.mix(theme.PHOSPHOR, "#000000", 0.3).name()))
				name = eva.elide(lf, "└ " + t["name"].upper(), x_name - 15)
				p.drawText(QPointF(12, cy + 4.5), name)
			else:
				p.setFont(nf)
				p.setPen(theme.qc("#ffffff" if hot else (theme.PHOSPHOR if goal else "#d8d4cd")))
				name = eva.elide(nf, t["name"].upper(), x_name - 8)
				p.drawText(QPointF(5, cy + 5), name)
			ty = cy + 4.5
			if self.targeting(t):
				self._target_nerv(p, i, t, cy, hot, x_s0, x_s1, mid, half, x_val, vf, t_anim, tslider_bg)
			elif self.unplaced(t):
				self._unplaced_text(p, cy, x_s0, x_p0b - 10, lf)
			else:
				v = self.values.get(tid, 0.0)
				p.drawPixmap(QPointF(x_s0, y), slider_bg)
				hx = mid + self.shown(v) * one
				m = self.mult(tid)
				eff = v * m
				cut = theme.ALARM
				if abs(v) > 1.0:
					edge = mid + (1.0 if v > 0 else -1.0) * one
					zone = (mid + one, x_s1) if v > 0 else (x_s0, mid - one)
					p.save()
					p.setClipRect(QRectF(min(edge, hx), cy - 4, abs(hx - edge), 8))
					eva.hazard(p, QRectF(zone[0], cy - 2.5, zone[1] - zone[0], 5), theme.WARN, 7, 255)
					p.restore()
				if abs(v) > 1e-3:
					p.setPen(eva.pen(theme.signed_color(v).name(), 235, 3.0))
					p.drawLine(QPointF(mid, cy), QPointF(mid + max(-1.0, min(1.0, v)) * one, cy))
					if abs(m - 1.0) > 1e-3 and tid not in self.pending:
						ex = mid + self.shown(eff) * one
						p.setPen(eva.pen(cut, 235, 1.4))
						p.drawLine(QPointF(mid, cy + 6), QPointF(ex, cy + 6))
						p.setPen(Qt.PenStyle.NoPen)
						p.setBrush(eva.col(cut))
						p.drawPolygon(QPolygonF([QPointF(ex - 3, cy + 9.5), QPointF(ex + 3, cy + 9.5), QPointF(ex, cy + 4.5)]))
						p.drawRect(QRectF(ex - 1.5, cy - 8, 3, 16))
				if tid in self.pending:
					a = 0.5 + 0.5 * math.sin(t_anim * 8)
					eva.brackets(p, QRectF(hx - 6, cy - 9, 12, 18), theme.NEON, int(90 + 160 * a), 3.5, 1.4)
				grab = self.drag == i
				p.setPen(eva.pen(theme.NEON, 255, 1.0) if hot or grab else Qt.PenStyle.NoPen)
				p.setBrush(theme.qc(theme.NEON if grab else ("#ffffff" if abs(v) > 1e-3 else theme.MUTED)))
				p.drawRect(QRectF(hx - 2, cy - 6, 4, 12))
				self.beyond_mark(p, hx, cy, v, theme.WARN)
				p.setFont(vf)
				txt = f"{v:+.2f}"
				ty = cy + 4.5
				if tid in self.pending:
					dots = "·" * (1 + int(t_anim * 4) % 3)
					p.setPen(theme.qc(theme.NEON))
					p.drawText(QPointF(x_val, ty), f"{txt} {dots}")
				else:
					p.setPen(theme.signed_color(v) if abs(v) > 1e-3 else theme.qc(theme.DIM))
					p.drawText(QPointF(x_val, ty), txt)
					if tid in self.exempt:
						self.unguarded_mark(p, x_div - 9, cy, theme.AMBER)
					if abs(v) > 1e-3 and abs(m - 1.0) > 1e-3:
						ax = x_val + eva.text_w(vf, txt) + 5
						p.setPen(eva.pen(cut, 255, 1.2))
						p.drawLine(QPointF(ax, cy), QPointF(ax + 6, cy))
						p.drawPolyline(QPolygonF([QPointF(ax + 3.5, cy - 2.5), QPointF(ax + 6.5, cy), QPointF(ax + 3.5, cy + 2.5)]))
						p.drawText(QPointF(ax + 10, ty), f"{eff:+.2f}")
			self.concept_mark(p, i, t, cy, mid, half)
			if t.get("sub"):
				continue
			p.drawPixmap(QPointF(x_p0b - 3, y), gauge_bg)
			b = self.base.get(tid)
			c = self.current.get(tid)
			at = lambda val: pm + max(-smax, min(smax, val)) / smax * bw / 2
			if c is not None:
				cx = at(c)
				lo, hi = sorted((pm, cx))
				k0 = max(0, math.ceil((lo - 3 - x_p0b) / 4))
				k1 = min(n_seg - 1, math.floor((hi - x_p0b) / 4))
				if k1 >= k0:
					p.save()
					p.setClipRect(QRectF(x_p0b + 4 * k0, y, 4 * (k1 - k0) + 3, self.row_h))
					p.drawPixmap(QPointF(x_p0b - 3, y), lits[c >= 0])
					p.restore()
			if b is not None:
				p.setBrush(theme.qc(theme.TEXT, 220))
				p.setPen(Qt.PenStyle.NoPen)
				bx = at(b)
				p.drawPolygon(QPolygonF([QPointF(bx - 3.5, y + 19.5), QPointF(bx + 3.5, y + 19.5), QPointF(bx, y + 15.5)]))
			if c is not None:
				p.setPen(eva.pen("#000000", 255, 1.0))
				p.setBrush(theme.qc("#ffffff"))
				p.drawEllipse(QPointF(at(c), y + 10.0), 3.2, 3.2)
			shown = c if c is not None else b
			if shown is not None:
				p.setFont(vf)
				p.setPen(theme.qc(theme.PHOSPHOR if c is not None else theme.MUTED) if goal else theme.signed_color(shown, 255 if c is not None else 150))
				text = f"{shown:.0%}" if goal else f"{shown:+.2f}"
				p.drawText(QPointF(x_p1 - eva.text_w(vf, text), ty), text)

	def row_at(self, y):
		if y < self.top:
			return -1
		i = int((y - self.top) // self.row_h)
		return i if 0 <= i < len(self.traits) else -1

	def value_at(self, x, target=False):
		x_name, x_s1 = self.geo()[0], self.geo()[1]
		x_s0 = x_name + 8
		mid = (x_s0 + x_s1) / 2
		v = (x - mid) / ((x_s1 - x_s0) / 2)
		return self.tsnap(v * self.TRANGE) if target else self.snap(v * self.range)

	def in_slider(self, x):
		x_name, x_s1 = self.geo()[0], self.geo()[1]
		return x_name <= x <= x_s1 + 6

	def in_row_left(self, x):
		return x <= self.geo()[1] + 6

	def mousePressEvent(self, e):
		self.setFocus()
		i = self.row_at(e.position().y())
		if e.button() != Qt.MouseButton.LeftButton:
			return
		if i < 0 or not self.in_row_left(e.position().x()):
			self.focus_row = -1
			self.update()
			return
		self.focus_row = i
		self._press = None
		self._grab = (i, e.position().y()) if self.in_grip(e.position().x()) else None
		if self.in_slider(e.position().x()):
			self.drag = i
			self.set_at(i, e.position().x())
			self._emit_peek()
		self.update()

	def target_card(self, t):
		tid = t["id"]
		tv = self.targets.get(tid)
		b, c = self.base.get(tid), self.current.get(tid)
		dose = self.values.get(tid, 0.0)
		m = self.mult(tid)
		rows = [("", f"+ {t.get('plus', '')}   ↔   − {t.get('minus', '')}", theme.TEXT)]
		rows.append(("target", f"{tv:+.2f}   (the measured value you want)" if tv is not None else "free: no target, its lever keeps its dose", theme.CYAN if tv is not None else theme.MUTED))
		rows.append(("measured", (f"{c:+.2f} now" if c is not None else "not measured with levers yet") + (f"   · baseline {b:+.2f}" if b is not None else ""), theme.TEXT))
		if tv is not None and c is not None:
			gap = tv - c
			rows.append(("gap", f"{gap:+.2f}" + ("   · reached" if abs(gap) <= self.TOL else ""), theme.OK if abs(gap) <= self.TOL else theme.WARN))
		cut = abs(dose) > 1e-3 and abs(m - 1.0) > 1e-3
		capped = abs(dose) >= self.range - 1e-3
		rows.append(("dose", f"{dose:+.2f}" + (f"   → applied {dose * m:+.2f} (lowered by the coherence guard: red)" if cut else "") + (f"   · the maximum dose (±{self.range:.2f}, settings): ALIGN cannot push this lever further" if capped else "   · the lever strength the alignment chose"), theme.ALARM if cut else (theme.WARN if capped else (theme.ACCENT2 if abs(dose) > 1e-3 else theme.MUTED))))
		if tid in self.pending or self.aligning:
			rows.append(("", "aligning: quick measures in progress", theme.NEON))
		rows.append(("controls", f"click or drag: set the target (steps of {self.STEP:g}) · wheel on the selected row: ±{self.STEP:g} · double-click the name: free · right-click: more", theme.DIM))
		return {"title": f"{t['name'].upper()} · TARGET", "rows": rows}

	def sub_card(self, t):
		v = self.values.get(t["id"], 0.0)
		rows = [("dose", f"{v:+.2f}", theme.ACCENT2 if abs(v) > 1e-3 else theme.MUTED), ("", "strength of the goal's trained LoRA weights · the row above doses its vector part", theme.TEXT), ("controls", f"click or drag on the slider: set (steps of {self.STEP:g}) · double-click the name: 0 · right-click: more", theme.DIM)]
		if t["id"] in self.pending:
			rows.insert(1, ("", "applying…", theme.NEON))
		return {"title": f"{t.get('parent_name', '').upper()} · LoRA PART", "rows": rows}

	def peek_card(self, i, part):
		t = self.traits[i]
		if t.get("sub"):
			return self.sub_card(t)
		if self.targeting(t) and part != "measured":
			return self.target_card(t)
		tid = t["id"]
		info = self.info.get(tid) or {}
		v = self.values.get(tid, 0.0)
		words = lambda items: " · ".join(dict.fromkeys(s.strip() for s in items if s.strip()).keys())
		rows = [("", f"+ {t.get('plus', '')}   ↔   − {t.get('minus', '')}", theme.TEXT)]
		if part == "measured":
			b, c = self.base.get(tid), self.current.get(tid)
			if t.get("goal"):
				rows.append(("without it", f"{b:.0%} match" if b is not None else "not checked yet", theme.TEXT))
				if c is not None:
					rows.append(("last check", f"{c:.0%} match", theme.ACCENT2))
				rows.append(("", "scored by the judge: who are you? what do you want? who am I to you? · GOALS window", theme.DIM))
				return {"title": f"{t['name'].upper()} · GOAL", "rows": rows}
			if t.get("lever_only"):
				rows.append(("", "not measured: " + ("its words give no lexicon on this model (none is a single token on one side): add more words, or English ones" if t.get("no_lexicon") else "a lever made from a goal has no word list the lens can read" if t.get("from_goal") else "it needs words on both sides (right-click → EDIT…)"), theme.MUTED))
				return {"title": f"{t['name'].upper()} · MEASURED", "rows": rows}
			rows.append(("baseline", f"{b:+.2f}   (no lever)" if b is not None else "not measured yet", theme.TEXT))
			if c is not None:
				rows.append(("now", f"{c:+.2f}   ({c - (b or 0):+.2f} vs baseline, with the active levers)", theme.ACCENT2))
			rows.append(("", "read-only: measured by the lens on the profile battery", theme.DIM))
			return {"title": f"{t['name'].upper()} · MEASURED", "rows": rows}
		b, c = self.base.get(tid), self.current.get(tid)
		if self.unplaced(t):
			rows.append(("not placed", "this goal has no lever on the model loaded now: it was placed on another model, or not yet", theme.WARN))
			rows.append(("", "open GOALS (right-click → OPEN GOALS…) and TRAIN it here, or ALIGN it; the slider stays inactive until then", theme.TEXT))
			return {"title": f"{t['name'].upper()} · GOAL", "rows": rows}
		if t.get("lever_only"):
			prof = "not measured (" + ("its words give no lexicon on this model" if t.get("no_lexicon") else "made from a goal's character" if t.get("from_goal") else "no words on both sides") + ")"
		elif t.get("goal"):
			prof = "not checked yet" if b is None and c is None else " · ".join(s for s in (f"{c:.0%} match at the last check" if c is not None else "", f"{b:.0%} without it" if b is not None else "") if s)
		elif c is not None:
			prof = f"{c:+.2f} now" + (f"   ({c - b:+.2f} vs baseline {b:+.2f})" if b is not None else "")
		else:
			prof = f"{b:+.2f}   (baseline, no lever active)" if b is not None else "not profiled yet"
		rows.append(("profile", prof, theme.TEXT if b is not None or c is not None else theme.MUTED))
		line = f"{v:+.2f}"
		m = self.mult(tid)
		if tid in self.pending:
			line += "   · staged, not applied (APPLY below)" if self.staged else "   · applying…"
		elif abs(v) > 1e-3 and abs(m - 1.0) > 1e-3:
			why = ([f"coherence guard ×{self.scale:.2f}"] if tid not in self.free and self.scale < 0.999 else []) + ([f"cut ×{self.factors[tid]:.2f}"] if self.blamed(tid) else [])
			line += f"   → applied {v * m:+.2f}   ({' · '.join(why)})"
		rows.append(("your setting", line, theme.ACCENT2 if abs(v) > 1e-3 else theme.MUTED))
		if abs(v) > 1e-3 and self.blamed(tid) and tid not in self.pending:
			rows.append(("guard cut", f"×{self.factors[tid]:.2f}: the guard cut this lever (it hurt coherence the most, or it was judged on its own)", theme.ERR))
		if abs(v) > 1e-3 and tid in self.free:
			rows.append(("guard", "judge overridden: applied as set, outside the coherence guard" if tid in self.exempt else "trained goal lever: applied as trained, outside the coherence guard", theme.MUTED))
		if info.get("layers") and info.get("kind") == "optimized":
			carrier = {"lora": f"LoRA rank {info.get('rank') or '?'}", "both": f"vector + LoRA rank {info.get('rank') or '?'}"}.get(info.get("carrier"), "vector")
			cl = info.get("closed")
			rows.append(("lever", f"trained {carrier} · L{info['layers'][0]}–L{info['layers'][-1]}" + (f" · closes {cl:.0%} of the gap to the character" if cl is not None else ""), theme.TEXT))
			if info.get("carrier") == "both":
				rows.append(("", "this slider doses the vector part · the LoRA part has its own slider just below", theme.DIM))
		elif info.get("layers"):
			gm = info.get("gen_mult")
			rows.append(("lever", f"L{info['layers'][0]}–L{info['layers'][-1]} · gain {info.get('gain', 0) * (gm or 1.0):.3f}" + (f" · calibrated on generations (×{gm:.2f})" if gm else " · not calibrated on generations yet"), theme.TEXT))
			if info.get("reads_as"):
				rows.append(("reads as +", words(info["reads_as"][:10]), theme.ACCENT2))
			if info.get("reads_as_minus"):
				rows.append(("reads as −", words(info["reads_as_minus"][:10]), theme.COOL2))
		else:
			rows.append(("lever", "not calibrated yet: moving the slider extracts it (~10-40 s)", theme.WARN))
		if t.get("edited"):
			rows.append(("definition", "edited by you · right-click: RESTORE DEFAULT", theme.AMBER))
		rows.append(("controls", f"click or drag on the slider: set (steps of {self.STEP:g}) · wheel on the slider of the selected row: ±{self.STEP:g} · double-click the name: 0 · right-click: more", theme.DIM))
		return {"title": t["name"].upper(), "rows": rows}

	def _emit_peek(self, x=None):
		if x is not None:
			self._peek_part = "measured" if x >= self.geo()[3] else "setting"
		i = self.drag if self.drag is not None else self.hover_row
		card = self.peek_card(i, getattr(self, "_peek_part", "setting")) if 0 <= i < len(self.traits) else None
		self.peek.emit(card)

	def mouseMoveEvent(self, e):
		if self.moving is not None:
			self.move_to(e.position().y())
			sa = self.scroll_area()
			if sa is not None:
				sa.ensureVisible(0, int(e.position().y()), 0, self.row_h)
			return
		grab = self._grab
		if grab is not None and e.buttons() & Qt.MouseButton.LeftButton and abs(e.position().y() - grab[1]) >= 4:
			self._grab = None
			blocks = self.blocks()
			self.moving = sum(len(b) for b in blocks[:self.block_of(grab[0], blocks)])
			self.setCursor(Qt.CursorShape.ClosedHandCursor)
			self.peek.emit(None)
			self.update()
			return
		press = getattr(self, "_press", None)
		if self.drag is None and press is not None and e.buttons() & Qt.MouseButton.LeftButton and abs(e.position().x() - press[1]) >= 4:
			self.drag = press[0]
			self._press = None
		if self.drag is not None:
			self.set_at(self.drag, e.position().x())
			self.update()
			self._emit_peek()
			return
		i = self.row_at(e.position().y())
		if i >= 0 and self.in_grip(e.position().x()):
			self.setCursor(Qt.CursorShape.OpenHandCursor)
		else:
			self.unsetCursor()
		if i != self.hover_row:
			self.hover_row = i
			self.hovered.emit(i)
			self.update()
		self._emit_peek(e.position().x())

	def mouseReleaseEvent(self, e):
		self._press = None
		self._grab = None
		if self.moving is not None:
			self.moving = None
			self.unsetCursor()
			self.update()
			self.order_changed.emit([t["id"] for t in self.traits if not t.get("sub")])
			return
		if self.drag is not None:
			t = self.traits[self.drag]
			self.drag = None
			self.update()
			self.emit_row(t)
			if not self.underMouse():
				self.peek.emit(None)

	def mouseDoubleClickEvent(self, e):
		i = self.row_at(e.position().y())
		if i < 0 or e.button() != Qt.MouseButton.LeftButton:
			return
		t = self.traits[i]
		if self.unplaced(t):
			return
		if self.in_slider(e.position().x()):
			self.focus_row = i
			self.drag = None
			self.set_at(i, e.position().x())
			self.update()
			self.emit_row(t)
			self._emit_peek()
			return
		if e.position().x() < self.geo()[0]:
			if self.targeting(t):
				self.targets.pop(t["id"], None)
				self.update()
				self.target_committed.emit(t["id"], None)
			else:
				self.values[t["id"]] = 0.0
				self.update()
				self.committed.emit(t["id"], 0.0)
			self._emit_peek()

	def wheelEvent(self, e):
		i = self.row_at(e.position().y())
		if not (self.hasFocus() and i >= 0 and i == self.focus_row and self.in_slider(e.position().x())):
			e.ignore()
			return
		t = self.traits[i]
		tid = t["id"]
		if self.unplaced(t):
			e.ignore()
			return
		step = self.STEP if e.angleDelta().y() > 0 else -self.STEP
		if self.targeting(t):
			start = self.targets.get(tid)
			self.targets[tid] = self.tsnap((start if start is not None else (self.now_of(tid) or 0.0)) + step)
		else:
			self.values[tid] = self.snap(self.values.get(tid, 0.0) + step)
		self._wheel_row = i
		self._wheel_timer.start(450)
		self.update()
		self._emit_peek()

	def _commit_wheel(self):
		if 0 <= self._wheel_row < len(self.traits):
			self.emit_row(self.traits[self._wheel_row])

	def leaveEvent(self, e):
		self.hover_row = -1
		self.hovered.emit(-1)
		if self.drag is None:
			self.peek.emit(None)
		self.update()

	def focusOutEvent(self, e):
		super().focusOutEvent(e)
		self.update()

	def row_menu(self, i):
		t = self.traits[i]
		tid = t["id"]
		goal = bool(t.get("goal"))
		custom = bool(t.get("custom"))
		menu = QMenu(self)
		menu.setFont(theme.ui(8))
		menu.setToolTipsVisible(True)
		menu_title(menu, (t.get("parent_name", "") + " · LoRA" if t.get("sub") else t["name"]).upper())
		if self.targeting(t):
			if tid in self.targets:
				act = menu.addAction("FREE (NO TARGET)")
				act.setData(("free", tid))
				act.setToolTip("drop the target: the alignment leaves this trait alone, its lever keeps its dose")
			elif self.now_of(tid) is not None:
				act = menu.addAction(f"TARGET ITS MEASURED VALUE ({self.now_of(tid):+.2f})")
				act.setData(("pin", tid))
				act.setToolTip("hold this trait where it is now while the others move")
			menu.addSeparator()
		elif not self.unplaced(t):
			act = menu.addAction("SET A VALUE…")
			act.setData(("value", tid))
			act.setToolTip(f"type this lever's dose, even beyond the slider maximum (±{self.range:g}, settings) without raising it for every lever\nup to ±{DOSE_HARD:g}: the slider then stays at its end with an orange arrow · dragging it brings the value back within the slider")
			if abs(self.values.get(tid, 0.0)) > 1e-9:
				menu.addAction("SET TO 0").setData(("zero", tid))
			menu.addSeparator()
		if goal:
			menu.addAction("OPEN GOALS…").setData(("goals", tid))
			if not t.get("sub"):
				twin = next((r for r in self.traits if r.get("from_goal") and r.get("from_goal") == t.get("goal")), None)
				act = menu.addAction("ADD AS A LEVER" if twin is None else "ALREADY A LEVER")
				act.setData(("as_lever", t.get("goal")))
				act.setEnabled(twin is None)
				act.setToolTip("make a regular lever from this goal's character (character ↔ default assistant), listed with the other levers\nthe persona direction it already has on this model moves to it; the goal itself is then active only through TRAIN or ALIGN" if twin is None else f"its lever is {twin['name']} in the list")
			self.order_menu(menu)
			self.restore_menu(menu)
			return menu
		act = menu.addAction("EDIT…")
		act.setData(("edit", tid))
		if not custom:
			act.setToolTip("change this default lever: name, poles, words, characters\nsaved in your working folder (trait_overrides.json) · RESTORE DEFAULT brings the shipped one back")
		menu.addAction("DUPLICATE…").setData(("duplicate", tid))
		if custom:
			menu.addAction("DELETE…").setData(("delete", tid))
		else:
			act = menu.addAction("REMOVE…")
			act.setData(("delete", tid))
			act.setToolTip("hide this default lever · RESTORE A REMOVED LEVER brings it back")
			if t.get("edited"):
				act = menu.addAction("RESTORE DEFAULT")
				act.setData(("restore", tid))
				act.setToolTip("drop your edits and use the shipped definition again (the lever is rebuilt the next time you move it)")
		menu.addSeparator()
		info = self.info.get(tid) or {}
		if not info:
			act = menu.addAction("EXTRACT NOW")
			act.setData(("extract", tid))
			act.setToolTip("build this lever now (extraction, then calibration), without moving its slider\nplanned in TO DO like any computation")
		act = menu.addAction("RECALIBRATE")
		act.setData(("recalibrate", tid))
		act.setEnabled(bool(info) and info.get("kind") != "optimized")
		act.setToolTip("re-run its calibration on generations with the current judge and test settings" if act.isEnabled() else ("not built yet: EXTRACT NOW, or move its slider" if not info else "a trained vector has nothing to recalibrate"))
		every = menu.addAction("RECALIBRATE EVERY LEVER")
		every.setData(("recalibrate", "*"))
		every.setEnabled(any((v or {}).get("kind") != "optimized" for v in self.info.values()))
		every.setToolTip("re-run the calibration of every built lever, e.g. after changing the judge or the test questions")
		menu.addSeparator()
		ex = menu.addAction("IGNORE THE COHERENCE GUARD")
		ex.setCheckable(True)
		ex.setChecked(tid in self.exempt)
		ex.setData(("exempt", tid))
		ex.setToolTip("override the judge for this lever: it is applied exactly as set, never scaled down by the coherence guard\n(useful when the judge mistakes a wanted style, e.g. deadpan, for incoherence)")
		self.order_menu(menu)
		self.restore_menu(menu)
		return menu

	def order_menu(self, menu):
		menu.addSeparator()
		act = menu.addAction("RESET THE LEVER ORDER")
		act.setData(("order", ""))
		act.setEnabled(self.custom_order)
		act.setToolTip("back to the usual order" if self.custom_order else "drag a lever by its name to move it up or down: the order is kept")

	def restore_menu(self, menu):
		if not self.hidden:
			return
		menu.addSeparator()
		sub = menu.addMenu("RESTORE A REMOVED LEVER")
		sub.setFont(theme.ui(8))
		sub.setToolTipsVisible(True)
		for h in self.hidden:
			act = sub.addAction(f"{h['name'].upper()}   ·   {h.get('plus', '')} ↔ {h.get('minus', '')}")
			act.setData(("restore", h["id"]))
			act.setToolTip("bring this default lever back (as edited, if you had edited it)")

	def menu_action(self, data):
		if not data:
			return
		kind, tid = data
		if kind == "zero":
			self.values[tid] = 0.0
			self.update()
			self.committed.emit(tid, 0.0)
			return
		if kind == "value":
			self.ask_value(tid)
			return
		if kind in ("free", "pin"):
			if kind == "free":
				self.targets.pop(tid, None)
			else:
				self.targets[tid] = self.tsnap(self.now_of(tid) or 0.0)
			self.update()
			self.target_committed.emit(tid, self.targets.get(tid))
			return
		if kind == "goals":
			t = next((r for r in self.traits if r["id"] == tid), None) or {}
			self.goals_requested.emit(t.get("goal") or "")
			return
		if kind == "as_lever":
			self.as_lever_requested.emit(tid)
			return
		if kind == "order":
			self.order_changed.emit([])
			return
		if kind == "exempt":
			self.exempt_requested.emit(tid, tid not in self.exempt)
			return
		{"edit": self.edit_requested, "duplicate": self.duplicate_requested, "delete": self.delete_requested, "restore": self.restore_requested, "recalibrate": self.recalibrate_requested, "extract": self.extract_requested}[kind].emit(tid)

	def ask_value(self, tid):
		t = next((r for r in self.traits if r["id"] == tid), None)
		if t is None:
			return
		name = (t.get("parent_name", "") + " · LoRA" if t.get("sub") else t["name"]).upper()
		dlg = QInputDialog(self)
		dlg.setWindowTitle("Set a value")
		dlg.setLabelText(f"dose of {name}\nthe slider goes to ±{self.range:g}; a typed value may go up to ±{DOSE_HARD:g}")
		dlg.setInputMode(QInputDialog.InputMode.DoubleInput)
		dlg.setDoubleDecimals(2)
		dlg.setDoubleRange(-DOSE_HARD, DOSE_HARD)
		dlg.setDoubleStep(self.STEP)
		dlg.setDoubleValue(float(self.values.get(tid, 0.0)))
		dlg.setStyleSheet(theme.STYLE)
		if not dlg.exec():
			return
		v = round(float(dlg.doubleValue()), 2)
		if abs(v - float(self.values.get(tid, 0.0))) < 1e-9:
			return
		self.values[tid] = v
		self.update()
		self.committed.emit(tid, v)

	def contextMenuEvent(self, e):
		i = self.row_at(e.pos().y())
		if i < 0 and not self.hidden:
			return
		self.peek.emit(None)
		if i < 0:
			menu = QMenu(self)
			menu.setFont(theme.ui(8))
			menu.setToolTipsVisible(True)
			menu_title(menu, "LEVERS")
			self.restore_menu(menu)
		else:
			menu = self.row_menu(i)
		act = menu.exec(e.globalPos())
		data = act.data() if act is not None else None
		menu.deleteLater()
		self.menu_action(data)


class WorkspaceChart(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setMouseTracking(True)
		self.ws = None
		self.band = None
		self.hl = []

	def set_data(self, ws, band, highlight=None):
		self.ws, self.band = ws, band
		self.hl = highlight or []
		self.update()

	def _paint_nerv(self, p):
		auto, a5 = self.ws["autocorr"], self.ws["agree_top5"]
		L = len(auto)
		w, h = self.width(), self.height() - 16
		bw = w / L
		top = max(max(auto), 1e-6)
		lo, hi = self.band or [0, 0]
		band = QRectF(lo * bw, 0, (hi - lo + 1) * bw, h)
		p.fillRect(band, eva.col(theme.ACCENT, 20))
		for l in self.hl:
			p.fillRect(QRectF(l * bw, 0, bw, h), eva.col(theme.ACCENT, 44))
		for l in range(L):
			v = max(0.0, auto[l]) / top
			inb = lo <= l <= hi
			p.fillRect(QRectF(l * bw + 1, h - v * (h - 6), max(bw - 2, 1), v * (h - 6)), eva.col(theme.ACCENT if inb else theme.HANDLE, 235 if inb else 160))
		p.setPen(eva.pen("#000000", 255, 1.0))
		for y in range(3, int(h), 4):
			p.drawLine(QPointF(0, y + 0.5), QPointF(w, y + 0.5))
		p.fillRect(QRectF(0, h, w, 1), eva.col(theme.ACCENT, 150))
		eva.ruler(p, bw / 2, w, h, theme.ACCENT, bw, 5, 3.0, 120)
		eva.brackets(p, band, theme.NEON, 200, 6.0, 1.2)
		p.setPen(eva.pen(theme.COOL2, 235, 1.5))
		pts = [QPointF(l * bw + bw / 2, h - a5[l] * (h - 6)) for l in range(L)]
		for i in range(L - 1):
			p.drawLine(pts[i], pts[i + 1])
		f = eva.font(11, 700)
		p.setFont(f)
		p.setPen(theme.qc(theme.MUTED))
		p.drawText(QPointF(0, h + 12), "L0")
		x_hi = max(lo * bw + 30, hi * bw - 10)
		if w - 24 > x_hi + eva.text_w(f, f"L{hi}") + 4:
			p.drawText(QPointF(w - 24, h + 12), f"L{L - 1}")
		p.setPen(theme.qc(theme.NEON))
		if lo * bw > eva.text_w(f, "L0") + 4:
			p.drawText(QPointF(lo * bw, h + 12), f"L{lo}")
		p.drawText(QPointF(x_hi, h + 12), f"L{hi}")

	def paintEvent(self, e):
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		if not self.ws:
			if eva.on():
				eva.standby(p, QRectF(self.rect()), "load a model to map its workspace", "NO DATA")
			else:
				p.setFont(theme.mono(8))
				p.setPen(theme.qc(theme.DIM))
				p.drawText(self.rect().adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "load a model to map its workspace")
			p.end()
			return
		if eva.on():
			self._paint_nerv(p)
			p.end()
			return
		auto = self.ws["autocorr"]
		a5 = self.ws["agree_top5"]
		L = len(auto)
		w, h = self.width(), self.height() - 16
		bw = w / L
		top = max(max(auto), 1e-6)
		lo, hi = self.band or [0, 0]
		p.fillRect(QRectF(lo * bw, 0, (hi - lo + 1) * bw, h), theme.qc(theme.ACCENT, 18))
		for l in self.hl:
			p.fillRect(QRectF(l * bw, 0, bw, h), theme.qc(theme.ACCENT, 40))
		for l in range(L):
			v = max(0.0, auto[l]) / top
			inb = lo <= l <= hi
			p.fillRect(QRectF(l * bw + 1, h - v * (h - 6), max(bw - 2, 1), v * (h - 6)), theme.qc(theme.ACCENT if inb else theme.MUTED, 200 if inb else 120))
		p.setPen(QPen(theme.qc(theme.COOL2, 220), 1.4))
		pts = [QPointF(l * bw + bw / 2, h - a5[l] * (h - 6)) for l in range(L)]
		for i in range(L - 1):
			p.drawLine(pts[i], pts[i + 1])
		p.setFont(theme.mono(7))
		p.setPen(theme.qc(theme.MUTED))
		fm = p.fontMetrics()
		p.drawText(QPointF(0, h + 12), "L0")
		x_hi = max(lo * bw + 30, hi * bw - 10)
		if w - 24 > x_hi + fm.horizontalAdvance(f"L{hi}") + 4:
			p.drawText(QPointF(w - 24, h + 12), f"L{L - 1}")
		p.setPen(theme.qc(theme.ACCENT2))
		if lo * bw > fm.horizontalAdvance("L0") + 4:
			p.drawText(QPointF(lo * bw, h + 12), f"L{lo}")
		p.drawText(QPointF(x_hi, h + 12), f"L{hi}")
		p.end()

	def mouseMoveEvent(self, e):
		if not self.ws:
			return
		L = len(self.ws["autocorr"])
		l = int(e.position().x() / max(self.width() / L, 1))
		if 0 <= l < L:
			ws = self.ws
			band = self.band
			where = (f"inside the workspace band L{band[0]}–L{band[1]}: the layers where the model holds concepts, levers act there" if band[0] <= l <= band[1] else (f"before the workspace band (L{band[0]}–L{band[1]}): still reading the words" if l < band[0] else f"after the workspace band (L{band[0]}–L{band[1]}): choosing the next words")) if band else ""
			lever = "\nhighlighted: the hovered lever acts on this layer" if l in (self.hl or []) else ""
			QToolTip.showText(e.globalPosition().toPoint(), f"LAYER {l}" + (f" · {where}" if where else "") + f"\nautocorr (abstract persistence) {ws['autocorr'][l]:+.3f}: how much the layer's reading stays the same from token to token\nagreement with the output: top-5 {ws['agree_top5'][l]:.2f} · top-1 {ws['agree_top1'][l]:.2f}\nresidual norm {ws['norm'][l]:.1f} · effective dimension {ws['eff_dim'][l]:.3f}" + lever, self)


class LensHeat(QWidget):
	HISTORY = 4096
	zoomed = Signal(float)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setMouseTracking(True)
		self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
		self.frames = []
		self.band = None
		self.trait_names = []
		self.img = None
		self.n = 0
		self.cell = 7.0

	def clear(self):
		self.frames = []
		self.img = None
		self.n = 0
		self.update()

	def visible(self):
		return max(8, int(self.width() / self.cell))

	def span_text(self):
		return f"{self.n} tok · last {min(self.n, self.visible())} shown" if self.n else f"{self.visible()} tokens wide"

	def _color(self, fr, l):
		tv = fr["tv"][l][0]
		inten = min(1.0, 0.18 + 2.2 * fr["p"][l][0])
		k = abs(tv)
		return theme.mix(theme.PANEL2, theme.ACCENT if tv >= 0 else theme.COOL, min(1.0, k * 3.0) * inten) if k > 0.05 else theme.mix(theme.PANEL2, theme.TEXT, 0.35 * inten)

	def restyle(self):
		frames = list(self.frames)
		self.clear()
		for fr in frames:
			self.add(fr, False)
		self.update()

	def add(self, frame, repaint=True):
		L = len(frame["top"])
		if self.img is None or self.img.height() != L:
			self.img = QImage(self.HISTORY, L, QImage.Format.Format_RGB32)
			self.img.fill(theme.qc(theme.PANEL2))
			self.n = 0
			self.frames = []
		if self.n >= self.HISTORY:
			shifted = self.img.copy(1, 0, self.HISTORY - 1, L)
			self.img.fill(theme.qc(theme.PANEL2))
			q = QPainter(self.img)
			q.drawImage(0, 0, shifted)
			q.end()
			self.n = self.HISTORY - 1
			del self.frames[0]
		for l in range(L):
			self.img.setPixelColor(self.n, L - 1 - l, self._color(frame, l))
		self.n += 1
		self.frames.append(frame)
		if repaint:
			self.update()

	def _span(self):
		vis = self.visible()
		start = max(0, self.n - vis)
		return vis, start, self.n - start, self.width() / vis

	def paintEvent(self, e):
		p = QPainter(self)
		w, h = self.width(), self.height()
		if not self.frames or self.img is None:
			text = f"live lens: chat with the model to see what each layer reads, token by token{chr(10)}{self.visible()} tokens fit this width · wheel (after a click): zoom"
			if eva.on():
				p.setRenderHint(QPainter.RenderHint.Antialiasing)
				eva.standby(p, QRectF(self.rect()), text, "NO SIGNAL")
			else:
				p.setFont(theme.mono(8))
				p.setPen(theme.qc(theme.DIM))
				p.drawText(self.rect().adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
			p.end()
			return
		L = self.img.height()
		vis, start, count, cw = self._span()
		ch = h / L
		p.drawImage(QRectF(0, 0, count * cw, h), self.img, QRectF(start, 0, count, L))
		p.setPen(QPen(theme.qc(theme.PANEL), 1))
		if cw >= 3:
			for c in range(1, count + 1):
				p.drawLine(QPointF(c * cw, 0), QPointF(c * cw, h))
		if ch >= 3:
			for l in range(1, L):
				p.drawLine(QPointF(0, l * ch), QPointF(count * cw, l * ch))
		lo, hi = self.band or [-1, -1]
		p.setPen(eva.pen(theme.ACCENT, 170, 1.0, Qt.PenStyle.DashLine) if eva.on() else QPen(theme.qc(theme.ACCENT, 90), 1))
		if lo >= 0:
			p.drawLine(QPointF(0, h - lo * ch), QPointF(w, h - lo * ch))
			p.drawLine(QPointF(0, h - (hi + 1) * ch), QPointF(w, h - (hi + 1) * ch))
		p.end()

	def wheelEvent(self, e):
		if not (self.hasFocus() and self.underMouse()):
			e.ignore()
			return
		self.cell = max(2.0, min(32.0, self.cell * (1.18 if e.angleDelta().y() > 0 else 1 / 1.18)))
		self.zoomed.emit(self.cell)
		self.update()

	def mouseMoveEvent(self, e):
		if not self.frames:
			return
		L = len(self.frames[-1]["top"])
		vis, start, count, cw = self._span()
		c = start + int(e.position().x() / cw)
		l = L - 1 - int(e.position().y() / (self.height() / L))
		if start <= c < self.n and 0 <= l < L:
			fr = self.frames[c]
			tops = "  ".join(f"{s.strip() or repr(s)} {pp:.2f}" for s, pp in zip(fr["top"][l], fr["p"][l]))
			QToolTip.showText(e.globalPosition().toPoint(), f"token {c + 1} {fr['tok']!r} · L{l}{chr(10)}{tops}", self)


class ConceptList(QWidget):
	hovered = Signal(int)
	clicked = Signal(int)
	overflowed = Signal(int, int)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setMouseTracking(True)
		self.items = []
		self.names = []
		self.hot = -1
		self.fade = False
		self.count = (0, 0)
		self.hidden = []

	def set_items(self, items, names):
		self.items = [dict(i) for i in (items or [])]
		self.names = names
		self.update()

	def set_fade(self, on):
		self.fade = bool(on)
		self.update()

	def _label(self, it):
		s = it["str"].strip() or repr(it["str"])
		return f"{s}·{it['en']}" if it.get("en") else s

	def _report(self, shown, total):
		self.hidden = [self._label(it) for it in self.items if it.get("_rect") is None]
		if (shown, total) != self.count:
			self.count = (shown, total)
			self.overflowed.emit(shown, total)

	def paintEvent(self, e):
		p = QPainter(self)
		nerv = eva.on()
		if not self.items:
			if nerv:
				p.setRenderHint(QPainter.RenderHint.Antialiasing)
				eva.standby(p, QRectF(self.rect()), "profile the model to surface its own signature concepts", "NO DATA")
			else:
				p.setFont(theme.mono(8))
				p.setPen(theme.qc(theme.DIM))
				p.drawText(self.rect().adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "profile the model to surface its own signature concepts")
			p.end()
			self._report(0, 0)
			return
		p.setFont(eva.font(13, 600) if nerv else theme.mono(8))
		fm = p.fontMetrics()
		x, y, line = 2, 14, 0
		placed = []
		for n, it in enumerate(self.items):
			s = self._label(it)
			tw = fm.horizontalAdvance(s) + 12
			if x + tw > self.width() and x > 2:
				x, y, line = 2, y + 17, line + 1
			if y > self.height() - 2:
				it["_rect"] = None
				continue
			it["_rect"] = QRectF(x, y - 11, tw, 15)
			placed.append((n, it, s, x, y, line, tw))
			x += tw
		over = len(placed) < len(self.items)
		last = placed[-1][5] if placed else 0
		lifts = [it["lift"] for _n, it, *_r in placed]
		lo, hi = min(lifts, default=0.0), max(lifts, default=1.0)
		width = max(self.width(), 1)
		for n, it, s, x, y, line, tw in placed:
			v = it.get("tv", 0.0)
			col = theme.mix(theme.TEXT, theme.ACCENT if v >= 0 else theme.COOL, min(1.0, abs(v) * 3)) if abs(v) > 0.05 else theme.qc(theme.TEXT)
			alpha = 0.35 + 0.65 * (it["lift"] - lo) / max(hi - lo, 1e-6) if self.fade else 1.0
			if over and line == last:
				alpha *= 1.0 - 0.75 * min(1.0, (x + tw / 2) / width)
			if n == self.hot:
				p.fillRect(QRectF(x - 3, y - 12, tw - 6, 16), eva.col(theme.ACCENT, 38) if nerv else theme.qc(theme.PANEL2))
				p.setPen(QPen(theme.qc(theme.ACCENT, 200), 1))
				p.drawLine(QPointF(x - 3, y + 4), QPointF(x + tw - 9, y + 4))
				alpha = 1.0
			col.setAlphaF(alpha)
			p.setPen(col)
			p.drawText(QPointF(x, y), s)
		p.end()
		self._report(len(placed), len(self.items))

	def index_at(self, pos):
		return next((n for n, it in enumerate(self.items) if it.get("_rect") and it["_rect"].contains(pos)), -1)

	def mouseMoveEvent(self, e):
		n = self.index_at(e.position())
		tip = ""
		if n < 0 and self.hidden:
			names = self.hidden[:64]
			tip = f"{len(self.hidden)} more concepts do not fit: enlarge the panel, or ◇ CONCEPTS ALL in the iris\n" + "\n".join(" · ".join(names[i:i + 8]) for i in range(0, len(names), 8)) + ("\n…" if len(self.hidden) > len(names) else "")
		if tip != self.toolTip():
			self.setToolTip(tip)
		if n != self.hot:
			self.hot = n
			self.hovered.emit(n)
			if n >= 0:
				self.setCursor(Qt.CursorShape.PointingHandCursor)
			else:
				self.unsetCursor()
			self.update()

	def mousePressEvent(self, e):
		n = self.index_at(e.position())
		if n >= 0 and e.button() == Qt.MouseButton.LeftButton:
			self.clicked.emit(n)

	def leaveEvent(self, e):
		if self.hot != -1:
			self.hot = -1
			self.hovered.emit(-1)
			self.unsetCursor()
			self.update()


class LogView(QTextBrowser):
	trace_requested = Signal(str, float)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setFont(theme.mono(8))
		self.setOpenLinks(False)
		self.restyle()
		self.document().setMaximumBlockCount(400)
		self.count = 0
		self.anchorClicked.connect(self._clicked)
		self.highlighted.connect(self._hover)

	def restyle(self):
		self.setStyleSheet(f"QTextBrowser {{ background: {theme.PANEL}; border: none; color: {theme.MUTED}; }}")

	def add(self, text, level="info", t=None, link=""):
		self.count += 1
		t = t or time.time()
		stamp = time.strftime("%H:%M:%S", time.localtime(t))
		color = {"warn": theme.WARN, "error": theme.ERR, "ok": theme.OK}.get(level, theme.TEXT)
		body = html.escape(text)
		if link:
			body = f'<a href="trace:{link}:{t!r}" style="color:{color}; text-decoration:none">{body} <span style="color:{theme.CYAN}">⌕</span></a>'
		else:
			body = f'<span style="color:{color}">{body}</span>'
		self.append(f'<span style="color:{theme.DIM}">{stamp}</span> {body}')
		self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

	def _clicked(self, url):
		parts = url.toString().split(":")
		if len(parts) == 3 and parts[0] == "trace":
			try:
				self.trace_requested.emit(parts[1], float(parts[2]))
			except ValueError:
				pass

	def _hover(self, url):
		if url.toString().startswith("trace:"):
			QToolTip.showText(QCursor.pos(), "open TRACE on this event: the judge's tests, scores and decision", self)


class ChatPanel(QWidget):
	send = Signal(str)
	stop = Signal()
	clear = Signal()
	retry = Signal()
	system_changed = Signal(str)
	config_restore = Signal(object)
	reply_chosen = Signal(str)
	history_requested = Signal()
	apply_requested = Signal()
	rewind = Signal()

	def __init__(self, parent=None):
		super().__init__(parent)
		lay = QVBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(6)
		self.system = QLineEdit()
		self.system.setFont(theme.mono(8))
		self.system.setPlaceholderText("system prompt (optional)")
		self.system.editingFinished.connect(lambda: self.system_changed.emit(self.system.text()))
		lay.addWidget(self.system)
		self.view = QTextBrowser()
		self.view.setFont(theme.mono(9))
		self.view.setOpenExternalLinks(False)
		self.view.setOpenLinks(False)
		self.view.anchorClicked.connect(self._anchor)
		self.view.highlighted.connect(self._hover_link)
		self.restyle()
		lay.addWidget(self.view, 1)
		self.input = QPlainTextEdit()
		self.input.setFont(theme.mono(9))
		self.input.setPlaceholderText("talk to the model · Enter: send · Shift+Enter: new line")
		self.input.setFixedHeight(64)
		self.input.installEventFilter(self)
		lay.addWidget(self.input)
		row = QHBoxLayout()
		self.meta = QLabel("")
		self.meta.setFont(theme.mono(7.5))
		self.meta.setStyleSheet(f"color: {theme.MUTED};")
		row.addWidget(self.meta, 1)
		self.b_stop = button("STOP", "stop the generation (Esc)")
		self.b_retry = button("↻ RETRY", "regenerate the last answer with the current levers (Ctrl+R or F5)\nthe earlier tries stay: ‹ › on the reply browses them")
		self.b_rewind = button("↶ REWIND", "take back the last message and its reply: the dialogue goes back one exchange, so RETRY works on the message before\nthe message comes back in the input (if it is empty) to edit or send again · its replies stay in the HISTORY")
		self.b_hist = button("HISTORY", "every reply with the configuration that made it, retries and cleared dialogues included (Ctrl+H)")
		self.b_clear = button("CLEAR", "clear the conversation (its replies stay in the HISTORY)")
		self.b_send = button("SEND", "send (Enter): replies at once when nothing waits in TO DO\notherwise the message waits after the operations planned before it (APPLY runs them), it never goes ahead of lever settings\nplan several messages, lever changes and tries this way, come back later and pick the replies you like", accent=True)
		for b in (self.b_hist, self.b_clear, self.b_stop, self.b_rewind, self.b_retry, self.b_send):
			row.addWidget(b)
		lay.addLayout(row)
		self.b_send.clicked.connect(self._send)
		self.b_stop.clicked.connect(self.stop.emit)
		self.b_retry.clicked.connect(self._retry)
		self.b_rewind.clicked.connect(self.rewind.emit)
		self.b_hist.clicked.connect(self.history_requested.emit)
		self.b_clear.clicked.connect(self._clear)
		self.head = QLabel("")
		self.head.setTextFormat(Qt.TextFormat.RichText)
		self.head.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
		self.head.linkActivated.connect(self._anchor)
		self.head.linkHovered.connect(self._hover_link)
		self.head.hide()
		self.streaming = False
		self.stream_i = None
		self.reply_at = 0
		self.flush_timer = QTimer(self)
		self.flush_timer.setSingleShot(True)
		self.flush_timer.timeout.connect(self.flush)
		self.msgs = []
		self.restyle()

	def restyle(self):
		self.view.setStyleSheet(f"QTextBrowser {{ background: {theme.PANEL}; border: none; }}")
		if hasattr(self, "head"):
			self.head.setFont(eva.font(13, 600) if eva.on() else theme.mono(8))
			self.update_head()

	def eventFilter(self, obj, ev):
		if obj is self.input and ev.type() == QEvent.Type.KeyPress:
			if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
				self.apply_requested.emit()
				return True
			if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
				self._send()
				return True
			if ev.key() == Qt.Key.Key_Escape:
				self.stop.emit()
				return True
			if ev.key() == Qt.Key.Key_R and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
				self._retry()
				return True
			if ev.key() == Qt.Key.Key_H and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
				self.history_requested.emit()
				return True
		return False

	def _cfg(self, url):
		s = url.toString() if hasattr(url, "toString") else str(url)
		if not s.startswith("cfg:"):
			return None, None
		try:
			i = int(s[4:])
		except ValueError:
			return None, None
		m = self.msgs[i] if 0 <= i < len(self.msgs) else None
		return (i, m[3]) if m is not None and len(m) > 3 and m[3] else (None, None)

	def _hover_link(self, url):
		s = url.toString() if hasattr(url, "toString") else str(url)
		if s.startswith("alt:"):
			QToolTip.showText(QCursor.pos(), ("earlier try" if s == "alt:-1" else "next try") + " of this reply\nthe one shown is the one the model remembers in the conversation", self.view)
			return
		if s.startswith("hist:"):
			QToolTip.showText(QCursor.pos(), "this reply was tried several times: open the HISTORY to see every try and its configuration", self.view)
			return
		i, cfg = self._cfg(url)
		if cfg is None:
			QToolTip.hideText()
			return
		QToolTip.showText(QCursor.pos(), cfg.get("summary", "") + "\n\nclick: restore it or copy it", self.view)

	def _anchor(self, url):
		s = url.toString() if hasattr(url, "toString") else str(url)
		if s.startswith("alt:"):
			self._switch(-1 if s == "alt:-1" else 1)
			return
		if s.startswith("hist:"):
			self.history_requested.emit()
			return
		i, cfg = self._cfg(url)
		if cfg is None:
			return
		menu = QMenu(self)
		menu.setFont(theme.ui(8))
		menu.setToolTipsVisible(True)
		menu_title(menu, f"REPLY {cfg.get('n', i + 1)} · CONFIGURATION")
		act = menu.addAction("RESTORE THIS CONFIGURATION")
		act.setToolTip("set the levers (and the system prompt) back to what produced this reply\nlevers not used then go back to 0 · the coherence guard checks them again")
		cp = menu.addAction("COPY AS TEXT")
		hist = menu.addAction("SHOW THE HISTORY")
		chosen = menu.exec(QCursor.pos())
		menu.deleteLater()
		if chosen is act:
			self.config_restore.emit(cfg)
		elif chosen is cp:
			QApplication.clipboard().setText(cfg.get("summary", ""))
		elif chosen is hist:
			self.history_requested.emit()

	def _retry(self):
		if not any(m[0] == "user" for m in self.msgs):
			return
		self.retry.emit()

	def take_back(self):
		if self.streaming:
			return False
		i = next((k for k in range(len(self.msgs) - 1, -1, -1) if self.msgs[k][0] == "user"), None)
		if i is None:
			return False
		text = self.msgs[i][1]
		self.msgs = self.msgs[:i]
		self.render()
		if not self.input.toPlainText().strip():
			self.input.setPlainText(text)
		return True

	def _last_assistant(self):
		for i in range(len(self.msgs) - 1, -1, -1):
			if self.msgs[i][0] == "user":
				return None
			if self.msgs[i][0] == "assistant":
				return i
		return None

	def _tries(self, m):
		while len(m) < 4:
			m.append(None)
		if len(m) < 6:
			m[4:] = [[{"text": m[1], "steered": m[2], "cfg": m[3]}], 0]
		return m[4]

	def _switch(self, step):
		if self.streaming:
			return
		i = self._last_assistant()
		if i is None or len(self.msgs[i]) < 6 or len(self.msgs[i][4]) < 2:
			return
		m = self.msgs[i]
		k = (m[5] + step) % len(m[4])
		a = m[4][k]
		m[1], m[2], m[3], m[5] = a["text"], a["steered"], a["cfg"], k
		self.render()
		self.reply_chosen.emit(a["text"])

	def _send(self):
		text = self.input.toPlainText().strip()
		if not text:
			return
		self.input.clear()
		self.send.emit(text)

	def _clear(self):
		self.msgs = []
		self.view.clear()
		self.update_head()
		self.clear.emit()

	def show_history(self, history):
		self.msgs = [[m["role"], m["content"], False] for m in history if m["role"] in ("user", "assistant")]
		self.render()

	def tries_link(self, i, m):
		if len(m) < 6 or len(m[4]) < 2:
			return ""
		n, k = len(m[4]), m[5] + 1
		if i == self._last_assistant() and not self.streaming:
			a = f"color:{theme.ACCENT2}; text-decoration:none;"
			return f' &nbsp;<a href="alt:-1" style="{a}">‹</a> <span style="color:{theme.MUTED}">{k}/{n}</span> <a href="alt:1" style="{a}">›</a>'
		return f' &nbsp;<a href="hist:" style="color:{theme.MUTED}; text-decoration:none;">{k}/{n} tries</a>'

	def config_link(self, i, cfg):
		if not cfg:
			return ""
		return f' <a href="cfg:{i}" style="color:{theme.MUTED}; text-decoration:none;">⟲ CONFIG {cfg.get("n", "")}</a>'

	def _md(self, text):
		from PySide6.QtGui import QTextDocument
		doc = QTextDocument()
		doc.setMarkdown(text)
		body = doc.toHtml()
		body = body[body.find("<body"):]
		return body[body.find(">") + 1:body.rfind("</body>")]

	def last_reply(self):
		return next((i for i in range(len(self.msgs) - 1, -1, -1) if self.msgs[i][0] == "assistant"), None)

	def update_head(self):
		i = self.last_reply()
		m = self.msgs[i] if i is not None else None
		text = (self.config_link(i, m[3] if len(m) > 3 else None) + self.tries_link(i, m)).strip().removeprefix("&nbsp;") if m is not None else ""
		self.head.setText(text)
		self.head.setVisible(bool(text))
		self.head.adjustSize()

	def render(self):
		parts = []
		nerv = eva.on()
		head = self.last_reply()
		for i, m in enumerate(self.msgs):
			role, text, steered = m[:3]
			cfg = m[3] if len(m) > 3 else None
			if role == "note":
				parts.append(f'<p style="margin-top:6px; color:{theme.MUTED}"><i>· {html.escape(text)}</i></p>')
			elif role == "user":
				who = f'<span style="background-color: {theme.LINE2}; color: {theme.NEON};">&nbsp;YOU&nbsp;</span>' if nerv else f'<span style="color:{theme.MUTED}">you ›</span>'
				parts.append(f'<p style="margin-top:10px">{who} <span style="color:{theme.TEXT}">{html.escape(text)}</span></p>')
			else:
				who = f'<span style="background-color: {theme.ACCENT}; color: {theme.ON_ACCENT};">&nbsp;MODEL&nbsp;</span>' if nerv else f'<span style="color:{theme.ACCENT2}">model ›</span>'
				tag = f' <span style="color:{theme.ACCENT}">◆ {"STEERED" if nerv else "steered"}</span>' if steered else ""
				links = "" if i == head else self.config_link(i, cfg) + self.tries_link(i, m)
				parts.append(f'<p style="margin-top:8px">{who}{tag}{links}</p><div style="color:#d8d4cd">{self._md(text)}</div>')
		self.view.setHtml("".join(parts))
		self.update_head()
		self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())

	def add_user(self, text):
		self.msgs = getattr(self, "msgs", [])
		self.msgs.append(["user", text, False])
		self.render()

	def add_note(self, text):
		self.msgs.append(["note", text, False])
		if not self.streaming:
			self.render()

	def start_assistant(self, steered, cfg=None, retry=False):
		self.msgs = getattr(self, "msgs", [])
		i = self._last_assistant() if retry else None
		if i is not None:
			m = self.msgs.pop(i)
			tries = self._tries(m)
			tries.append({"text": "", "steered": steered, "cfg": cfg})
			m[1], m[2], m[3], m[5] = "", steered, cfg, len(tries) - 1
			self.msgs.append(m)
		else:
			self.msgs.append(["assistant", "", steered, cfg])
		self.streaming = True
		self.render()
		self.stream_i = len(self.msgs) - 1
		cur = self.view.textCursor()
		cur.movePosition(QTextCursor.MoveOperation.End)
		self.reply_at = cur.block().position()

	def append_text(self, text):
		i = self.stream_i
		if i is None or i >= len(self.msgs) or self.msgs[i][0] != "assistant":
			return
		self.msgs[i][1] += text
		if not self.flush_timer.isActive():
			self.flush_timer.start(50)

	def flush(self):
		i = self.stream_i
		if not self.streaming or i is None or i >= len(self.msgs) or self.msgs[i][0] != "assistant":
			return
		cur = QTextCursor(self.view.document())
		cur.setPosition(min(self.reply_at - 1, self.view.document().characterCount() - 1))
		cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
		cur.insertHtml(self._md(self.msgs[i][1]))
		cur.setPosition(self.reply_at)
		cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
		fmt = QTextCharFormat()
		fmt.setForeground(QColor("#d8d4cd"))
		cur.mergeCharFormat(fmt)
		self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())

	def drop_failed(self):
		self.streaming = False
		self.flush_timer.stop()
		m = self.msgs[-1] if self.msgs else None
		if m is not None and m[0] == "assistant":
			if len(m) > 5 and len(m[4]) > 1:
				m[4].pop(m[5])
				m[5] = len(m[4]) - 1
				a = m[4][m[5]]
				m[1], m[2], m[3] = a["text"], a["steered"], a["cfg"]
			elif not m[1]:
				self.msgs.pop()
		self.add_note("the reply failed (RUN LOG has the error) · ↻ RETRY asks the same question again")
		self.render()

	def end_assistant(self, stats):
		self.streaming = False
		self.flush_timer.stop()
		m = self.msgs[-1] if self.msgs else None
		if m is not None and m[0] == "assistant" and len(m) > 5:
			m[4][m[5]]["text"] = m[1]
		self.render()
		if stats:
			self.meta.setText(f"{stats.get('tokens', 0)} tok · {stats.get('tps', 0)} tok/s · prompt {stats.get('prompt_tokens', 0)} tok")


def vision_tag(m):
	if not m.get("mmproj"):
		return ""
	return "VISION*" if m.get("mmproj_choice") else "VISION"


class ModelDelegate(QStyledItemDelegate):
	def paint_nerv(self, p, opt, m):
		p.save()
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		r = opt.rect
		selected = bool(opt.state & QStyle.StateFlag.State_Selected)
		if selected or opt.state & QStyle.StateFlag.State_MouseOver:
			p.fillRect(r, eva.col(theme.ACCENT, 34 if selected else 16))
		if selected:
			p.fillRect(QRectF(r.x(), r.y(), 3, r.height()), eva.col(theme.NEON))
		p.fillRect(QRectF(r.x(), r.bottom(), r.width(), 1), eva.col(theme.ACCENT, 40))
		nf, mf = eva.font(15, 700, False), eva.font(12, 600)
		p.setFont(nf)
		p.setPen(theme.qc("#ffffff" if selected else theme.TEXT))
		p.drawText(QPointF(r.x() + 10, r.y() + 17), eva.elide(nf, m.get("label") or m.get("file", "?"), r.width() - 34))
		p.setFont(mf)
		p.setPen(theme.qc(theme.MUTED))
		meta = f"{m.get('arch', '?')} · {m.get('size_label') or '?'} · {m.get('quant', '?')} · {m.get('size', 0) / 2**30:.1f} GB" + (f" · MoE×{m['n_expert']}" if m.get("n_expert") else "")
		vis = vision_tag(m)
		vw = eva.text_w(eva.font(10, 700), vis) + 10 if vis else 0
		p.drawText(QPointF(r.x() + 10, r.y() + 32), eva.elide(mf, meta, r.width() - 24 - vw))
		if vis:
			vf = eva.font(10, 700)
			box = QRectF(r.right() - vw - 8, r.y() + 22, vw, 13)
			eva.box(p, box, theme.COOL, 220, 1.0)
			p.setFont(vf)
			p.setPen(eva.col(theme.COOL))
			p.drawText(box, Qt.AlignmentFlag.AlignCenter, vis)
		lens = m.get("lens_source")
		if lens:
			sq = QRectF(r.right() - 16, r.y() + 8, 7, 7)
			if lens == "hub":
				eva.box(p, sq, theme.NEON, 230, 1.0)
			else:
				p.fillRect(sq, eva.col(theme.NEON))
		p.restore()

	def paint(self, p, opt, index):
		m = index.data(Qt.ItemDataRole.UserRole) or {}
		if eva.on():
			self.paint_nerv(p, opt, m)
			return
		p.save()
		r = opt.rect
		selected = bool(opt.state & QStyle.StateFlag.State_Selected)
		hover = bool(opt.state & QStyle.StateFlag.State_MouseOver)
		if selected or hover:
			p.fillRect(r, theme.qc(theme.PANEL2))
		if selected:
			p.fillRect(QRectF(r.x(), r.y(), 2, r.height()), theme.qc(theme.ACCENT))
		p.setPen(QPen(theme.qc(theme.LINE), 1))
		p.drawLine(r.bottomLeft(), r.bottomRight())
		p.setFont(theme.mono(8.5, selected))
		p.setPen(theme.qc(theme.TEXT))
		fm = p.fontMetrics()
		name = m.get("label") or m.get("file", "?")
		p.drawText(QPointF(r.x() + 10, r.y() + 16), fm.elidedText(name, Qt.TextElideMode.ElideRight, r.width() - 34))
		p.setFont(theme.mono(7.5))
		p.setPen(theme.qc(theme.MUTED))
		meta = f"{m.get('arch', '?')} · {m.get('size_label') or '?'} · {m.get('quant', '?')} · {m.get('size', 0) / 2**30:.1f} GB" + (f" · MoE×{m['n_expert']}" if m.get("n_expert") else "")
		vis = vision_tag(m)
		vw = p.fontMetrics().horizontalAdvance(vis) + 8 if vis else 0
		p.drawText(QPointF(r.x() + 10, r.y() + 31), p.fontMetrics().elidedText(meta, Qt.TextElideMode.ElideRight, int(r.width() - 24 - vw)))
		if vis:
			p.setPen(theme.qc(theme.COOL))
			p.drawText(QPointF(r.right() - vw - 4, r.y() + 31), vis)
		lens = m.get("lens_source")
		if lens:
			c = QPointF(r.right() - 12, r.y() + 12)
			if lens == "hub":
				p.setBrush(Qt.BrushStyle.NoBrush)
				p.setPen(QPen(theme.qc(theme.ACCENT, 200), 1.2))
			else:
				p.setBrush(theme.qc(theme.ACCENT))
				p.setPen(Qt.PenStyle.NoPen)
			p.drawEllipse(c, 3, 3)
		p.restore()

	def sizeHint(self, opt, index):
		return QSize(200, 40)


class PlanView(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.plan = None
		self.gpus = []
		self.pending = ""
		self.t0 = time.perf_counter()
		self.setMinimumHeight(96)
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.update)

	def set_plan(self, plan, gpus):
		self.plan, self.gpus = plan, gpus
		self.pending = ""
		self.setToolTip("\n".join(([self.status_text()] if plan else []) + list((plan or {}).get("notes") or [])))
		self.timer.stop()
		self.update()

	def status_text(self):
		host = (self.plan.get("estimate") or {}).get("Host", {})
		msg = f"CPU spill {host.get('model', 0) / 2**30:.1f} GB — slow" if self.plan.get("spill") else "fully on GPU"
		return f"{msg} · ctx {self.plan.get('n_ctx')} · lens on {self.plan.get('lens_device')}"

	def set_pending(self, text):
		self.plan = None
		self.pending = text
		self.setToolTip(text)
		self.t0 = time.perf_counter()
		self.timer.start(60)
		self.update()

	def _paint_nerv(self, p):
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		f = eva.font(13, 600)
		if self.pending:
			w = self.width()
			evahud.bar(p, QRectF(0, self.height() / 2 - 3, w, 5), None, theme.ACCENT, time.perf_counter(), self.devicePixelRatioF())
			text = eva.elide(f, self.pending, w - 8)
			p.setFont(f)
			p.setPen(eva.col(theme.ACCENT2))
			p.drawText(QPointF((w - eva.text_w(f, text)) / 2, self.height() / 2 + 22), text)
			return
		if not self.plan:
			eva.standby(p, QRectF(self.rect()), "select a model to plan its placement on the GPUs", "NO PLAN")
			return
		sf = eva.font(12, 600)
		est = self.plan.get("estimate") or {}
		y = 4
		w = self.width() - 90
		for i, g in enumerate(self.gpus):
			e = est.get(f"CUDA{i}", {})
			total = g["total"]
			p.setFont(sf)
			p.setPen(theme.qc(theme.NEON))
			p.drawText(QPointF(0, y + 11), f"CUDA{i}")
			x = 50
			eva.box(p, QRectF(x - 2, y + 1, w + 4, 13), theme.ACCENT, 100)
			p.fillRect(QRectF(x, y + 3, w, 9), eva.col(theme.LINE))
			parts = [(max(0, total - g["free"]), theme.DIM), (e.get("model", 0), theme.TEXT), (e.get("context", 0), theme.COOL), (e.get("compute", 0), theme.COOL2)]
			if self.plan.get("lens_device") == f"cuda:{i}":
				parts.append((self.plan.get("lens_bytes", 0), theme.ACCENT))
			for amount, col in parts:
				pw = w * amount / max(total, 1)
				p.fillRect(QRectF(x, y + 3, pw, 9), eva.col(col, 230))
				x += pw
			for gx in range(53, int(50 + w), 4):
				p.fillRect(QRectF(gx, y + 3, 1, 9), eva.col("#000000"))
			p.setPen(theme.qc(theme.TEXT))
			p.drawText(QPointF(w + 58, y + 11), f"{total / 2**30:.0f} GB")
			y += 19
		p.setPen(theme.qc(theme.ALARM if self.plan.get("spill") else theme.PHOSPHOR))
		p.drawText(QPointF(0, y + 11), eva.elide(sf, self.status_text(), self.width() - 4))
		y += 18
		x = 0
		for label, col in zip(["other", "model", "kv", "compute", "lens"], [theme.DIM, theme.TEXT, theme.COOL, theme.COOL2, theme.ACCENT]):
			p.fillRect(QRectF(x, y + 3, 7, 7), eva.col(col))
			p.setPen(theme.qc(theme.MUTED))
			p.drawText(QPointF(x + 11, y + 11), label)
			x += 11 + eva.text_w(sf, label) + 12
		for note in self.plan.get("notes") or []:
			y += 15
			p.setPen(theme.qc(theme.AMBER))
			p.drawText(QPointF(0, y + 11), eva.elide(sf, note, self.width() - 4))

	def paintEvent(self, e):
		p = QPainter(self)
		if eva.on():
			self._paint_nerv(p)
			p.end()
			return
		p.setFont(theme.mono(7.5))
		if self.pending:
			w = self.width()
			x = (time.perf_counter() - self.t0) * 260 % (w + 120) - 120
			p.fillRect(QRectF(0, self.height() / 2 - 1, w, 2), theme.qc(theme.LINE2))
			p.fillRect(QRectF(max(0, x), self.height() / 2 - 1, min(120, w - max(0, x)), 2), theme.qc(theme.ACCENT, 200))
			p.setPen(theme.qc(theme.ACCENT2))
			p.drawText(QRectF(0, self.height() / 2 + 6, w, 16), Qt.AlignmentFlag.AlignHCenter, self.pending)
			p.end()
			return
		if not self.plan:
			p.setPen(theme.qc(theme.DIM))
			p.drawText(self.rect().adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, "select a model to plan its placement on the GPUs")
			p.end()
			return
		est = self.plan.get("estimate") or {}
		y = 4
		w = self.width() - 90
		for i, g in enumerate(self.gpus):
			e = est.get(f"CUDA{i}", {})
			total = g["total"]
			p.setPen(theme.qc(theme.MUTED))
			p.drawText(QPointF(0, y + 10), f"CUDA{i}")
			x = 50
			p.fillRect(QRectF(x, y + 3, w, 9), theme.qc(theme.LINE2))
			used_other = max(0, total - g["free"])
			parts = [(used_other, theme.DIM), (e.get("model", 0), theme.TEXT), (e.get("context", 0), theme.COOL), (e.get("compute", 0), theme.COOL2)]
			if self.plan.get("lens_device") == f"cuda:{i}":
				parts.append((self.plan.get("lens_bytes", 0), theme.ACCENT))
			for amount, col in parts:
				pw = w * amount / max(total, 1)
				p.fillRect(QRectF(x, y + 3, pw, 9), theme.qc(col, 220))
				x += pw
			p.setPen(theme.qc(theme.TEXT))
			p.drawText(QPointF(w + 56, y + 11), f"{total / 2**30:.0f} GB")
			y += 18
		host = est.get("Host", {})
		p.setPen(theme.qc(theme.ERR if self.plan.get("spill") else theme.OK))
		msg = f"CPU spill {host.get('model', 0) / 2**30:.1f} GB — slow" if self.plan.get("spill") else "fully on GPU"
		p.drawText(QPointF(0, y + 10), f"{msg} · ctx {self.plan.get('n_ctx')} · lens on {self.plan.get('lens_device')}")
		y += 16
		p.setPen(theme.qc(theme.MUTED))
		p.drawText(QPointF(0, y + 10), "■ other  ■ model  ■ kv  ■ compute  ■ lens")
		cols = [theme.DIM, theme.TEXT, theme.COOL, theme.COOL2, theme.ACCENT]
		fm = p.fontMetrics()
		x = 0
		for label, col in zip(["other", "model", "kv", "compute", "lens"], cols):
			p.setPen(theme.qc(col))
			p.drawText(QPointF(x, y + 10), "■")
			x += fm.horizontalAdvance(f"■ {label}  ")
		for note in self.plan.get("notes") or []:
			y += 14
			p.setPen(theme.qc(theme.WARN))
			p.drawText(QPointF(0, y + 10), note)
		p.end()


class LensField(QPushButton):
	def __init__(self, parent=None):
		super().__init__(parent)
		tab_focus(self)
		self.setAutoDefault(False)
		self.setCursor(Qt.CursorShape.PointingHandCursor)
		self.setFixedHeight(26)
		self.mode = "auto"
		self.entry = None
		self.setToolTip("choose the lens: best match, any lens from the index, a .pt file, or train one")

	def set_choice(self, mode, entry):
		self.mode, self.entry = mode, entry
		text, _c, where = self.text_parts()
		e = entry or {}
		self.setToolTip(f"{text}{' · ' + where if where else ''}\n{(e.get('repo') or '') + '/' + (e.get('file') or '') if e.get('repo') else e.get('path', '')}\nclick: choose another lens, browse a .pt file or train one".replace("\n\n", "\n"))
		self.update()

	def text_parts(self):
		if self.mode == "logit" or (self.mode == "auto" and not self.entry):
			return ("AUTO · " if self.mode == "auto" else "") + "logit lens (J = I)", theme.MUTED, ""
		e = self.entry or {}
		where = {"hub": "↓ hub", "cached": "cached", "local": "local", "file": "file"}.get(e.get("source"), "")
		return ("AUTO · " if self.mode == "auto" else "") + e.get("label", e.get("base", "?")), theme.TEXT, where

	def paint_nerv(self, p, keyed):
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		w, h = self.width(), self.height()
		text, col, where = self.text_parts()
		hot = self.underMouse()
		evahud.draw_plate(p, QRectF(0, 0, w, h), theme.NEON if hot else (theme.ACCENT2 if keyed else theme.ACCENT), 255 if hot or keyed else 120, 1.0, "#000000", 255, 4.0)
		sf, f = eva.font(12, 700), eva.font(13, 600, False)
		ww = eva.text_w(sf, where) + 10 if where else 0
		if where:
			p.setFont(sf)
			p.setPen(theme.qc(theme.NEON if "hub" in where else theme.MUTED))
			p.drawText(QPointF(w - ww - 14, evahud.base(sf, 0, h)), where)
		p.setFont(f)
		p.setPen(theme.qc(col))
		p.drawText(QPointF(9, evahud.base(f, 0, h)), eva.elide(f, text, w - ww - 34))
		p.setPen(eva.pen(theme.NEON if hot else theme.MUTED, 255, 1.4))
		p.drawPolyline(QPolygonF([QPointF(w - 11, h / 2 - 4), QPointF(w - 7, h / 2), QPointF(w - 11, h / 2 + 4)]))

	def paintEvent(self, e):
		p = QPainter(self)
		keyed = self.hasFocus() and self.window().testAttribute(Qt.WidgetAttribute.WA_KeyboardFocusChange)
		if eva.on():
			self.paint_nerv(p, keyed)
			p.end()
			return
		r = self.rect().adjusted(0, 0, -1, -1)
		p.fillRect(r, theme.qc(theme.PANEL2))
		p.setPen(QPen(theme.qc(theme.ACCENT if self.underMouse() else (theme.ACCENT2 if keyed else theme.LINE2)), 1))
		p.drawRect(r)
		text, col, where = self.text_parts()
		p.setFont(theme.mono(7.5))
		fm = p.fontMetrics()
		ww = fm.horizontalAdvance(where) + 10 if where else 0
		if where:
			p.setPen(theme.qc(theme.ACCENT2 if "hub" in where else theme.MUTED))
			p.drawText(QRectF(r.right() - ww - 14, 0, ww, self.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, where)
		p.setFont(theme.mono(8))
		fm = p.fontMetrics()
		p.setPen(theme.qc(col))
		p.drawText(QRectF(6, 0, r.width() - ww - 26, self.height()), Qt.AlignmentFlag.AlignVCenter, fm.elidedText(text, Qt.TextElideMode.ElideRight, int(r.width() - ww - 26)))
		p.setPen(theme.qc(theme.MUTED))
		p.drawText(QRectF(r.right() - 14, 0, 12, self.height()), Qt.AlignmentFlag.AlignVCenter, "›")
		p.end()


HF_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
QUANT_DIR = re.compile(r"(?i)^(i?q\d\w*|f16|bf16|f32|fp16|ggufs?|quants?)$")
QUANT_TAIL = re.compile(r"(?i)[._ -](i?q\d\w*|f16|bf16|f32|fp16)$")
CONTAINERS = {"models", "model", "gguf", "ggufs", "llm", "llms", "hub", "snapshots", "blobs", "downloads"}


def reveal_path(path):
	if not path or not os.path.exists(path):
		return False
	if sys.platform.startswith("win"):
		subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
		return True
	QDesktopServices.openUrl(QUrl.fromLocalFile(path if os.path.isdir(path) else os.path.dirname(path)))
	return True


def hf_ids(path, roots=()):
	parts = [s for s in os.path.abspath(path).replace("\\", "/").split("/") if s]
	cache = next((s for s in parts if s.startswith("models--")), None)
	if cache:
		bits = cache[len("models--"):].split("--", 1)
		return (bits[0], bits[1]) if len(bits) == 2 and all(HF_PART.match(b) for b in bits) else (None, None)
	low = os.path.normcase(os.path.abspath(path))
	homes = sorted((os.path.normcase(os.path.abspath(r)).rstrip("\\/") for r in roots if r), key=len, reverse=True)
	home = next((r for r in homes if low.startswith(r + os.sep)), None)
	room = len(os.path.relpath(low, home).split(os.sep)) - 1 if home else len(parts) - 2
	dirs = parts[:-1]
	if room >= 3 and QUANT_DIR.match(dirs[-1]):
		dirs, room = dirs[:-1], room - 1
	if room < 2 or len(dirs) < 2:
		return None, None
	author, repo = dirs[-2], dirs[-1]
	if author.lower() in CONTAINERS or not HF_PART.match(author) or not HF_PART.match(repo):
		return None, None
	return author, repo


def hf_query(m):
	stem = os.path.splitext(os.path.basename(m.get("path") or m.get("file") or ""))[0]
	stem = re.sub(r"(?i)-\d{5}-of-\d{5}$", "", stem)
	return QUANT_TAIL.sub("", stem).strip(" .-_")


class ModelDrawer(QWidget):
	selected = Signal(str)
	load = Signal(str, object, int)
	cancel = Signal()
	unload = Signal()
	add_folder = Signal(str)
	refresh_requested = Signal()
	pick_lens = Signal()
	ctx_changed = Signal(int)
	options_changed = Signal(str, object, bool)
	base_requested = Signal(object)
	vision_set = Signal(str, str)
	keep_changed = Signal(str)

	def __init__(self, parent=None):
		super().__init__(parent)
		lay = QHBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(14)
		left = QVBoxLayout()
		left.setSpacing(6)
		right = QVBoxLayout()
		right.setSpacing(8)
		top = QHBoxLayout()
		self.search = QLineEdit()
		self.search.setFont(theme.mono(8))
		self.search.setPlaceholderText("filter models…")
		self.search.textChanged.connect(self._filter)
		top.addWidget(self.search, 1)
		self.b_folder = button("+ FOLDER", "add a folder containing GGUF or safetensors models\n(safetensors: Hugging Face folders or single files such as ComfyUI text encoders, converted to GGUF at the first LOAD)")
		self.b_folder.clicked.connect(self._pick_folder)
		top.addWidget(self.b_folder)
		self.b_refresh = button("↻", "scan the model folders again: models downloaded or removed since (F5)\nthe list is also scanned each time this window opens")
		self.b_refresh.setFixedWidth(30)
		self.b_refresh.clicked.connect(self.refresh_requested.emit)
		top.addWidget(self.b_refresh)
		left.addLayout(top)
		self.list = QListWidget()
		self.list.setItemDelegate(ModelDelegate())
		self.list.setMouseTracking(True)
		self.list.currentItemChanged.connect(self._changed)
		self.list.itemDoubleClicked.connect(lambda _it: self._load())
		self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
		self.list.customContextMenuRequested.connect(self._context)
		self.roots = []
		left.addWidget(self.list, 1)
		right.addWidget(_caption("VRAM PLAN"))
		self.plan = PlanView()
		right.addWidget(self.plan)
		row = QHBoxLayout()
		lab = _caption("LENS")
		row.addWidget(lab)
		self.lens = LensField()
		self.lens.clicked.connect(self.pick_lens.emit)
		row.addWidget(self.lens, 1)
		row.addWidget(_caption("CTX"))
		self.ctx = guard_wheel(QSpinBox())
		self.ctx.setRange(128, 262144)
		self.ctx.setSingleStep(128)
		self.ctx.setValue(4096)
		self.ctx.setFont(theme.mono(8))
		self.ctx.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
		self.ctx.setFixedWidth(70)
		self.ctx.setToolTip("context size (tokens) · wheel when focused")
		self.ctx.editingFinished.connect(lambda: self.ctx_changed.emit(self.ctx.value()))
		row.addWidget(self.ctx)
		right.addLayout(row)
		right.addWidget(_caption("LOAD OPTIONS"))
		self.options = LoadOptionsForm()
		self.options.changed.connect(lambda o, per: self.options_changed.emit(self.current_path() or "", o, per) if self.current_path() else None)
		right.addWidget(self.options)
		right.addWidget(_note("the VRAM plan above is recomputed with these options (llama-fit-params gets the same flags)"))
		row3 = QHBoxLayout()
		row3.addWidget(_caption("LEVERS AT LOAD"))
		self.keep = guard_wheel(QComboBox())
		self.keep.setFont(theme.mono(8))
		self.keep.addItem("last used with this model", "model")
		self.keep.addItem("keep the current ones", "current")
		self.keep.setToolTip("lever settings at LOAD (doses, targets, system prompt; the active levers are the same for every model)\nlast used with this model: what you had the last time this model was loaded\nkeep the current ones: what is set now is carried over and applied to the new model (goals not placed on it stay at 0)")
		self.keep.currentIndexChanged.connect(lambda _i: self.keep_changed.emit(self.keep.currentData()))
		row3.addWidget(self.keep, 1)
		right.addLayout(row3)
		right.addStretch(1)
		row2 = QHBoxLayout()
		self.progress = QLabel("")
		self.progress.setFont(theme.mono(8))
		self.progress.setStyleSheet(f"color: {theme.ACCENT};")
		row2.addWidget(self.progress, 1)
		self.b_unload = button("UNLOAD", "unload the model in use (planned in TO DO when something else is planned)")
		self.b_cancel = button("STOP", "stop the operation that runs now (a load in progress, for instance)")
		self.b_load = button("LOAD", "load the selected model (double-click does the same)\nstarts at once when nothing is loaded or planned, otherwise it is planned in TO DO: APPLY runs it", accent=True)
		for b in (self.b_unload, self.b_cancel, self.b_load):
			row2.addWidget(b)
		right.addLayout(row2)
		box = QWidget()
		box.setLayout(right)
		box.setMinimumWidth(600)
		box.setMaximumWidth(800)
		lay.addLayout(left, 1)
		lay.addWidget(box)
		self.b_load.clicked.connect(self._load)
		self.b_cancel.clicked.connect(self.cancel.emit)
		self.b_unload.clicked.connect(self.unload.emit)
		self.models = []

	def set_models(self, models, lenses=None):
		cur = self.current_path()
		lenses = lenses or {}
		self.models = models
		self.list.blockSignals(True)
		self.list.clear()
		for m in models:
			if m.get("error") or m.get("arch") in ("clip", "mtmd") or "mmproj" in m.get("file", "").lower():
				continue
			label = m.get("name") or m["file"]
			if m["file"].lower().replace(".gguf", "") not in label.lower().replace(" ", "-"):
				label = m["file"].replace(".gguf", "")
			lens = lenses.get(m["path"])
			m = dict(m, label=label, lens_source=lens["source"] if lens else "")
			it = QListWidgetItem()
			it.setData(Qt.ItemDataRole.UserRole, m)
			src = (f"{m['source']}\nconverted → {m['path']}" if m.get("source") else (f"{m['path']}\nsafetensors: converted to GGUF at the first LOAD" if m.get("convert") else m["path"])) + (f"\nconfig + tokenizer: {m['ref']}" if m.get("kind") == "file" and m.get("ref") else ("\nno base model known: right-click → BASE MODEL…" if m.get("kind") == "file" else ""))
			it.setToolTip(f"{src}\n{m.get('arch')} · {int(m.get('n_layer') or 0) - int(m.get('n_nextn') or 0)} layers · d={m.get('n_embd')} · ctx {m.get('n_ctx_train')}\nlens: " + (f"{lens['label']} ({lens['source']})" if lens else "none found (logit lens)") + ("" if m.get("convert") else "\nvision: " + (os.path.basename(m["mmproj"]) + (" (chosen)" if m.get("mmproj_choice") else " (found next to it)") if m.get("mmproj") else ("off (right-click → AUTOMATIC to turn it back on)" if m.get("mmproj_choice") == "none" else "none (right-click → CHOOSE AN MMPROJ…)"))))
			self.list.addItem(it)
			if m["path"] == cur:
				self.list.setCurrentItem(it)
		self.list.blockSignals(False)
		self._filter(self.search.text())

	def current_model(self):
		it = self.list.currentItem()
		return it.data(Qt.ItemDataRole.UserRole) if it else None

	def _filter(self, text):
		t = text.lower().strip()
		for i in range(self.list.count()):
			it = self.list.item(i)
			m = it.data(Qt.ItemDataRole.UserRole)
			it.setHidden(bool(t) and t not in (m.get("label", "") + " " + m.get("arch", "") + " " + m.get("path", "")).lower())

	def current_path(self):
		it = self.list.currentItem()
		return it.data(Qt.ItemDataRole.UserRole)["path"] if it else None

	def select_path(self, path):
		for i in range(self.list.count()):
			it = self.list.item(i)
			if it.data(Qt.ItemDataRole.UserRole)["path"] == path:
				self.list.setCurrentItem(it)
				self.list.scrollToItem(it)
				return

	def _changed(self, cur, prev):
		if cur:
			self.selected.emit(cur.data(Qt.ItemDataRole.UserRole)["path"])

	def set_plan(self, data, gpus):
		plan = dict(data["plan"])
		plan["lens_bytes"] = data.get("lens_bytes", 0)
		self.plan.set_plan(plan, gpus)

	def _load(self):
		path = self.current_path()
		if not path:
			return
		self.load.emit(path, None if self.lens.mode == "logit" else self.lens.entry, self.ctx.value())

	def _pick_folder(self):
		d = QFileDialog.getExistingDirectory(self, "Folder with GGUF or safetensors models")
		if d:
			self.add_folder.emit(d)

	def model_menu(self, m):
		author, repo = hf_ids(m.get("path", ""), self.roots)
		menu = QMenu(self)
		menu.setFont(theme.ui(8))
		menu.addAction("LOAD").setData("load")
		path = m.get("path", "")
		act = menu.addAction("OPEN IN FILE EXPLORER")
		act.setData("reveal")
		act.setEnabled(bool(path) and os.path.exists(path))
		act.setToolTip(path)
		if not m.get("convert"):
			menu.addSeparator()
			menu_title(menu, "VISION (MMPROJ)")
			choice = m.get("mmproj_choice", "")
			found = os.path.basename(m.get("mmproj") or "") if not choice else ""
			for key, text, tip in (("", "AUTOMATIC" + (f" · {found}" if found else " · none found"), "use an mmproj*.gguf found next to the model"), ("pick", "CHOOSE AN MMPROJ…" + (f" · {os.path.basename(choice)}" if choice and choice != "none" else ""), "pick the vision projector (mmproj*.gguf) made for this model, wherever it is"), ("none", "NO VISION", "never load a vision projector for this model")):
				act = menu.addAction(text)
				act.setData(f"vision:{key}")
				act.setCheckable(True)
				act.setChecked(choice == key or (key == "pick" and choice not in ("", "none")))
				act.setToolTip(tip)
		if m.get("source") and os.path.exists(m["source"]):
			act = menu.addAction("OPEN THE SOURCE IN FILE EXPLORER")
			act.setData("reveal_source")
			act.setToolTip(f"the safetensors this GGUF was converted from:\n{m['source']}")
		if m.get("kind") == "file":
			act = menu.addAction("BASE MODEL (config + tokenizer)…")
			act.setData("base_model")
			act.setToolTip("the Hugging Face repo this safetensors file was made from: its config and tokenizer complete the file for the conversion" + (f"\ncurrent: {m['ref']}" if m.get("ref") else ""))
		menu.addSeparator()
		menu_title(menu, "OPEN ON HUGGING FACE")
		if author:
			menu.addAction(f"author page   {author}").setData(f"https://huggingface.co/{author}")
		if author and repo:
			menu.addAction(f"model page    {author}/{repo}").setData(f"https://huggingface.co/{author}/{repo}")
		q = hf_query(m)
		if q:
			menu.addSeparator()
			menu.addAction(f"search the Hub for “{q}”").setData("https://huggingface.co/models?search=" + quote(q))
		return menu

	def _context(self, pos):
		it = self.list.itemAt(pos)
		if it is None:
			return
		m = it.data(Qt.ItemDataRole.UserRole) or {}
		menu = self.model_menu(m)
		act = menu.exec(self.list.viewport().mapToGlobal(pos))
		url = act.data() if act is not None else None
		menu.deleteLater()
		if url == "load":
			self.select_path(m.get("path"))
			self._load()
		elif url == "base_model":
			self.base_requested.emit(m)
		elif url == "reveal":
			reveal_path(m.get("path"))
		elif url in ("vision:", "vision:none"):
			self.vision_set.emit(m.get("path", ""), url.split(":", 1)[1])
		elif url == "vision:pick":
			f, _f = QFileDialog.getOpenFileName(self, "Vision projector for " + os.path.basename(m.get("path", "")), os.path.dirname(m.get("path", "")), "mmproj (*.gguf)")
			if f:
				self.vision_set.emit(m.get("path", ""), os.path.normpath(f))
		elif url == "reveal_source":
			reveal_path(m.get("source"))
		elif url:
			QDesktopServices.openUrl(QUrl(url))


def _human(n):
	return f"{n / 2**30:.2f} GB" if n >= 2**30 else (f"{n / 2**20:.0f} MB" if n else "—")


class LensPicker(EvaDialog):
	train = Signal()

	def __init__(self, model, rows, current, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Lens")
		self.setStyleSheet(theme.STYLE)
		self.resize(820, 560)
		self.choice = None
		self.model = model
		lay = QVBoxLayout(self)
		lay.setSpacing(8)
		lay.addWidget(DialogTitle("LENS FOR", model.get("label") or model.get("file", ""), model.get("path", "")))
		n_layers = int(model.get("n_layer") or 0) - int(model.get("n_nextn") or 0)
		sub = QLabel(f"{model.get('arch', '?')} · {n_layers} layers · d={model.get('n_embd')} · a lens must come from the same base model: shape alone is not enough")
		sub.setFont(theme.mono(7.5))
		sub.setStyleSheet(f"color: {theme.MUTED};")
		sub.setWordWrap(True)
		lay.addWidget(sub)
		self.filter = QLineEdit()
		self.filter.setFont(theme.mono(8))
		self.filter.setPlaceholderText("filter lenses…")
		self.filter.textChanged.connect(self._filter)
		lay.addWidget(self.filter)
		self.tree = QTreeWidget()
		self.tree.setColumnCount(5)
		self.tree.setHeaderLabels(["FIT", "LENS", "PROMPTS", "SIZE", "SOURCE"])
		self.tree.setFont(theme.mono(8))
		self.tree.setRootIsDecorated(False)
		self.tree.setUniformRowHeights(True)
		self.tree.setAlternatingRowColors(False)
		self.tree.header().setFont(theme.ui(7.5))
		self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
		self.tree.header().setStretchLastSection(False)
		for col, w in ((0, 90), (2, 70), (3, 80), (4, 200)):
			self.tree.setColumnWidth(col, w)
		self.tree.itemDoubleClicked.connect(lambda it, _c: self._use(it))
		lay.addWidget(self.tree, 1)
		sections = [("BEST MATCHES — same model family, version and size", [r for r in rows if r["score"] > 0 and r["compat"] is not False]), ("SAME SHAPE — only valid if it is the same base model", [r for r in rows if r["score"] <= 0 and r["compat"] is True]), ("OTHER LENSES", [r for r in rows if r["score"] <= 0 and r["compat"] is not True])]
		self.items = []
		for title, part in sections:
			if not part:
				continue
			sec = QTreeWidgetItem([title])
			sec.setFlags(Qt.ItemFlag.NoItemFlags)
			sec.setForeground(0, theme.qc(theme.ACCENT2))
			self.tree.addTopLevelItem(sec)
			sec.setFirstColumnSpanned(True)
			for r in part:
				e = r["entry"]
				fit = "match" if r["score"] > 0 else {True: "shape ok", False: "no", None: "?"}[r["compat"]]
				src = {"hub": "hub (download)", "cached": "HF cache", "local": "local", "file": "file"}.get(e["source"], e["source"]) + (f" · {e['repo']}" if e.get("repo") and e["repo"] != "neuronpedia/jacobian-lens" else "")
				it = QTreeWidgetItem([fit, e.get("label") or e["base"], str(e.get("n_prompts") or "?"), _human(e.get("size", 0)), src])
				it.setData(0, Qt.ItemDataRole.UserRole, e)
				col = theme.TEXT if r["score"] > 0 else (theme.MUTED if r["compat"] is not False else theme.DIM)
				for c in range(5):
					it.setForeground(c, theme.qc(col))
				if r["score"] > 0:
					it.setForeground(0, theme.qc(theme.OK))
				tip = f"{e.get('repo') or ''}/{e.get('file') or ''}".strip("/") or e.get("path", "")
				tip += f"\n{e.get('path', '')}" if e.get("path") else ""
				tip += f"\nd_model {e['d_model']} · {e['n_src']} layers" if e.get("d_model") else ""
				tip += f"\nvariant: {e['variant']}" if e.get("variant") else ""
				tip += "\nincompatible shape: cannot be loaded on this model" if r["compat"] is False else ""
				for c in range(5):
					it.setToolTip(c, tip)
				if r["compat"] is False:
					it.setFlags(Qt.ItemFlag.ItemIsEnabled)
				self.tree.addTopLevelItem(it)
				self.items.append(it)
				if current and e.get("key") == current.get("key"):
					self.tree.setCurrentItem(it)
		row = QHBoxLayout()
		b_file = button("BROWSE .pt…", "use a lens file from disk")
		b_train = button("TRAIN A LENS…", "fit a Jacobian lens for this model (hours, GPU)")
		row.addWidget(b_file)
		row.addWidget(b_train)
		row.addStretch(1)
		b_cancel = button("CANCEL")
		b_logit = button("LOGIT LENS", "no Jacobian lens: read the residual with the unembedding only")
		b_auto = button("AUTO", "always use the best match for this model")
		b_use = button("USE SELECTED", accent=True)
		b_use.setDefault(True)
		for b in (b_cancel, b_logit, b_auto, b_use):
			row.addWidget(b)
		lay.addLayout(row)
		b_cancel.clicked.connect(self.reject)
		b_logit.clicked.connect(lambda: self._done(("logit", None)))
		b_auto.clicked.connect(lambda: self._done(("auto", None)))
		b_use.clicked.connect(lambda: self._use(self.tree.currentItem()))
		b_file.clicked.connect(self._browse)
		b_train.clicked.connect(lambda: (self.train.emit(), self.reject()))

	def _done(self, choice):
		self.choice = choice
		self.accept()

	def _use(self, it):
		if it is None:
			return
		e = it.data(0, Qt.ItemDataRole.UserRole)
		if e and it.flags() & Qt.ItemFlag.ItemIsSelectable:
			self._done(("entry", e))

	def _browse(self):
		path, _f = QFileDialog.getOpenFileName(self, "Jacobian lens file", "", "PyTorch lens (*.pt)")
		if path:
			self._done(("file", path))

	def _filter(self, text):
		t = text.lower().strip()
		for it in self.items:
			e = it.data(0, Qt.ItemDataRole.UserRole)
			it.setHidden(bool(t) and t not in f"{e.get('label', '')} {e.get('repo', '')} {e.get('file', '')}".lower())


def _clock(s):
	s = int(s or 0)
	return f"{s // 3600}h{s % 3600 // 60:02d}" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class TrainDialog(EvaDialog):
	start = Signal(object, bool)
	stop = Signal()

	def __init__(self, model, gpus, state, loaded, parent=None, band=None):
		super().__init__(parent)
		self.setWindowTitle("Train a lens")
		self.band = list(band) if band else None
		self.setStyleSheet(theme.STYLE)
		self.setMinimumWidth(640)
		self.model = model
		self.gpus = gpus
		lay = QVBoxLayout(self)
		lay.setSpacing(6)
		lay.addWidget(DialogTitle("TRAIN A JACOBIAN LENS", model.get("label") or model.get("file", ""), model.get("path", "")))
		n_layers = int(model.get("n_layer") or 0) - int(model.get("n_nextn") or 0)
		info = QLabel(f"{model.get('arch', '?')} · {n_layers} layers · d={model.get('n_embd')} · {int(model.get('n_params') or 0) / 1e9:.1f}B params. The lens is fitted with Anthropic's jlens (backprop through the model, WikiText prompts). Expect hours: the time per prompt shows once it runs.")
		info.setFont(theme.mono(7.5))
		info.setStyleSheet(f"color: {theme.MUTED};")
		info.setWordWrap(True)
		lay.addWidget(info)
		self.src_gguf = QRadioButton("this GGUF, dequantized by transformers (lens of this exact finetune)")
		self.src_hf = QRadioButton("Hugging Face weights:")
		for r in (self.src_gguf, self.src_hf):
			tab_focus(r)
			r.setFont(theme.mono(8))
		grp = QButtonGroup(self)
		grp.addButton(self.src_gguf)
		grp.addButton(self.src_hf)
		ok_gguf = model.get("arch") in GGUF_ARCHS
		ram = int(model.get("n_params") or 0) * 4 / 2**30
		self.src_gguf.setEnabled(ok_gguf)
		self.src_gguf.setToolTip(f"needs ~{ram:.0f} GB of RAM while loading" if ok_gguf else f"transformers cannot load '{model.get('arch')}' GGUF files: use the Hugging Face weights of the base model")
		lay.addWidget(self.src_gguf)
		row = QHBoxLayout()
		row.addWidget(self.src_hf)
		self.repo = QLineEdit(hf_repo(model))
		self.repo.setFont(theme.mono(8.5))
		self.repo.setPlaceholderText("org/model repo id or a local folder with safetensors")
		self.repo.setToolTip("the original (base) model the GGUF was made from; gated repos need a Hugging Face login")
		row.addWidget(self.repo, 1)
		lay.addLayout(row)
		(self.src_gguf if ok_gguf else self.src_hf).setChecked(True)
		form = QHBoxLayout()
		self.name = QLineEdit(slug(os.path.splitext(model.get("file", "lens"))[0]))
		self.name.setFont(theme.mono(8.5))
		self.name.setToolTip("folder name under data/lenses; the same name resumes an interrupted run")
		self.prompts = guard_wheel(QSpinBox())
		self.prompts.setRange(8, 5000)
		self.prompts.setSingleStep(10)
		self.prompts.setValue(100)
		self.prompts.setToolTip("prompts averaged into the lens (Neuronpedia: 100 to 1000)")
		self.dtype = guard_wheel(QComboBox())
		self.dtype.addItems(["bf16", "nf4"])
		self.dtype.setToolTip("bf16: exact weights · nf4: 4-bit, ~3x less VRAM, slightly noisier lens")
		for label, wdg, stretch in (("NAME", self.name, 3), ("PROMPTS", self.prompts, 1), ("PRECISION", self.dtype, 1)):
			l = QLabel(label)
			theme.tag_font(l, 7.5)
			l.setStyleSheet(f"color: {theme.MUTED};")
			form.addWidget(l)
			wdg.setFont(theme.mono(8.5))
			form.addWidget(wdg, stretch)
		lay.addLayout(form)
		grow = QHBoxLayout()
		self.gpu_boxes = []
		for i, g in enumerate(gpus):
			cb = tab_focus(QCheckBox(f"CUDA{i} {g['name'].split(' ', 1)[-1]} · {g['total'] / 2**30:.0f} GB"))
			cb.setFont(theme.mono(8))
			cb.setChecked(True)
			cb.toggled.connect(self._replan)
			grow.addWidget(cb)
			self.gpu_boxes.append(cb)
		grow.addStretch(1)
		lay.addLayout(grow)
		self.band_only = tab_focus(QCheckBox(f"workspace band only: layers L{self.band[0]}–L{self.band[1]}" if self.band else "workspace band only (load this model first: its band is mapped on load)"))
		self.band_only.setFont(theme.mono(8))
		self.band_only.setEnabled(bool(self.band))
		self.band_only.setToolTip("fit only the layers the levers and the profile work in: faster (the backward pass stops at the band)\nand a smaller lens file; the other layers are then read with the logit lens (J = I)")
		self.band_only.toggled.connect(self._band_name)
		lay.addWidget(self.band_only)
		self.unload = tab_focus(QCheckBox("unload the current model first (frees its VRAM)"))
		self.unload.setFont(theme.mono(8))
		self.unload.setChecked(bool(loaded))
		self.unload.setVisible(bool(loaded))
		lay.addWidget(self.unload)
		self.plan_label = QLabel("")
		self.plan_label.setFont(theme.mono(8))
		self.plan_label.setWordWrap(True)
		lay.addWidget(self.plan_label)
		self.progress = QLabel("")
		self.progress.setFont(theme.mono(8))
		self.progress.setWordWrap(True)
		self.progress.setStyleSheet(f"color: {theme.ACCENT2};")
		lay.addWidget(self.progress)
		btns = QHBoxLayout()
		self.b_stop = button("STOP", "stop the fit (checkpoints are kept: START resumes it)")
		self.b_stop.clicked.connect(self.stop.emit)
		btns.addWidget(self.b_stop)
		btns.addStretch(1)
		close = button("CLOSE", "close this window (the fit keeps running)")
		close.clicked.connect(self.close)
		self.b_start = button("START", accent=True)
		self.b_start.clicked.connect(self._start)
		btns.addWidget(close)
		btns.addWidget(self.b_start)
		lay.addLayout(btns)
		self.dtype.currentTextChanged.connect(self._replan)
		self.src_gguf.toggled.connect(self._replan)
		self.prompts.valueChanged.connect(self._replan)
		self._replan()
		self.set_state(state)

	def restyle(self):
		self.setStyleSheet(theme.STYLE)
		self.progress.setStyleSheet(f"color: {theme.ACCENT2};")
		theme.refont(self)
		self.fit_margins()
		self.update()

	def _devices(self):
		return [i for i, cb in enumerate(self.gpu_boxes) if cb.isChecked()]

	def _band_name(self, on):
		name = self.name.text().strip()
		if on and not name.endswith("-band"):
			self.name.setText(name + "-band")
		elif not on and name.endswith("-band"):
			self.name.setText(name[:-5])

	def _replan(self, *_a):
		self.plan = fit_plan(self.model, self.gpus, self.dtype.currentText(), self._devices())
		ok = self.plan["mode"] != "none"
		n = self.prompts.value()
		eta = f" · rough estimate {_clock(self.plan['spp'] * n + 90)} for {n} prompts" if ok and self.plan.get("spp") else ""
		if self.dtype.currentText() == "bf16" and self.plan["mode"] in ("split", "none"):
			alt = fit_plan(self.model, self.gpus, "nf4", self._devices())
			if alt["mode"] in ("data", "single"):
				eta += f"\nnf4 would fit on each GPU: ~{_clock(alt['spp'] * n + 90)}"
		self.plan_label.setText(("PLAN  " if ok else "CANNOT RUN  ") + self.plan["note"] + eta)
		self.plan_label.setStyleSheet(f"color: {theme.TEXT if ok else theme.WARN};")
		self.b_start.setEnabled(ok and not getattr(self, "busy", False))

	def _start(self):
		if self.plan["mode"] == "none":
			return
		gguf = self.src_gguf.isChecked()
		src = os.path.dirname(self.model["path"]) if gguf else self.repo.text().strip()
		if not src:
			self.plan_label.setText("give a Hugging Face repo id or a local folder")
			return
		params = {"name": self.name.text().strip() or "lens", "source": src, "gguf_file": os.path.basename(self.model["path"]) if gguf else "", "model_id": os.path.splitext(self.model["file"])[0] if gguf else src, "n_prompts": int(self.prompts.value()), "dtype": self.dtype.currentText(), "devices": self.plan["devices"], "dim_batch": self.plan["dim_batch"], "visible": self.plan.get("visible", ""), "max_seq_len": 128, "model_path": self.model["path"]}
		if self.band and self.band_only.isChecked():
			params["source_layers"] = list(range(int(self.band[0]), int(self.band[1]) + 1))
		self.start.emit(params, self.unload.isChecked() and self.unload.isVisible())

	def set_state(self, st):
		st = st or {}
		state = st.get("state", "idle")
		self.busy = state in ("running", "stopping")
		self.b_start.setEnabled(not self.busy and self.plan["mode"] != "none")
		self.b_stop.setEnabled(self.busy)
		if state == "idle":
			self.progress.setText("")
			return
		done, total = st.get("done", 0), st.get("total", 0)
		parts = [f"{st.get('name', '')} · {state} · {st.get('phase', '')}", f"{done}/{total} prompts"]
		if st.get("eta"):
			parts.append(f"ETA {_clock(st['eta'])}")
		parts.append(f"elapsed {_clock(time.time() - st.get('t0', time.time()))}")
		for w in st.get("workers") or []:
			parts.append(f"{w['device']} {w['state']} {w['done']}/{w['total']}")
		if st.get("error"):
			parts.append(f"error: {st['error']}")
		if st.get("lens_path"):
			parts.append(f"saved → {st['lens_path']}")
		self.progress.setText("  ·  ".join(parts))


PREF_DEFAULTS = {"fps_idle": 60, "fps_busy": 30, "spin": 2.0, "concepts": 6, "accent": "ember", "rotate": True, "concepts_mode": "auto", "orbit_button": "left", "pan_button": "right", "concept_count": 120, "tooltip_ms": 700, "concepts_spread": True, "concepts_fade": False, "arm_order": "profile", "center_click": "middle click", "reset_click": "middle double-click", "style": "nerv", "sync_place": "band", "sync_scenes": "both", "sync_height": 150, "sync_warn": True, "sync_ready": True, "scanlines": True, "sync_mode": "sine", "sound": True, "sound_volume": 60, "sound_groups": {k: True for k in SOUND_GROUPS}}
SOUND_PREVIEW = {"lever": 0.6, "target": 0.5, "toggle": "target", "type": 0}
SOUND_DEMO = [("start", None, 0.0), ("boot", None, 0.9), ("lever", -1.0, 3.0), ("lever", 0.0, 0.25), ("lever", 1.0, 0.25), ("magi", None, 0.6), ("ready", None, 1.0), ("toggle", "target", 1.6), ("target", 0.5, 0.5), ("tick", None, 0.6), ("tick", None, 0.35), ("sync", None, 0.6), ("type", 0, 3.4), ("type", 2, 0.07), ("type", 1, 0.07), ("type", 3, 0.07), ("type", 0, 0.07), ("type", 2, 0.07), ("eot", None, 0.12), ("pattern", None, 0.8), ("done", None, 2.6), ("warning", None, 1.6), ("error", None, 1.8), ("abort", None, 0.9), ("click", None, 0.8)]
SYNC_PLACES = {"band": "full-width band between the panels", "levers": "small strip in LEVERS", "off": "hidden"}
SYNC_SCENES = {"both": "sine waves + deviation graph", "sine": "sine waves only", "graph": "deviation graph only"}
ARM_ORDERS = {"profile": "profile order (traits list)", "similar": "similar traits side by side"}
GUARD_MODES = {"uniform": "scale every lever together", "individual": "only the changed levers, the others keep their judged factor", "quick": "quick check, full check only if it fails", "read": "read first, the judge only between the thresholds"}
DISPLAY_KEYS = ("fps_idle", "fps_busy", "spin", "concepts", "concept_count", "tooltip_ms", "sync_height")
STYLE_COMBOS = ("style", "sync_place", "sync_scenes")


def section_style(l):
	if eva.on():
		l.setStyleSheet(f"QLabel {{ color: {theme.NEON}; border: 1px solid {theme.ACCENT}; border-right: 4px solid {theme.ACCENT}; padding: 1px 7px 1px 6px; margin-top: 9px; }}")
	else:
		l.setStyleSheet(f"color: {theme.ACCENT2}; padding-top: 8px;")


class SettingsDialog(EvaDialog):
	refresh_lenses = Signal()
	preview = Signal(object)

	def __init__(self, settings, prefs, parent=None, models=(), gpus=()):
		super().__init__(parent)
		self.setWindowTitle("Settings")
		self.setStyleSheet(theme.STYLE)
		self.settings = settings
		self.prefs = {**PREF_DEFAULTS, **(prefs or {})}
		outer = QVBoxLayout(self)
		outer.setSpacing(8)
		body = QWidget()
		cols = QHBoxLayout(body)
		cols.setContentsMargins(0, 0, 0, 0)
		cols.setSpacing(26)
		self.scroll = QScrollArea()
		self.scroll.setWidget(body)
		self.scroll.setWidgetResizable(True)
		self.scroll.setFrameShape(QFrame.Shape.NoFrame)
		outer.addWidget(self.scroll, 1)
		panes = [QWidget(), QWidget()]
		left, right = QVBoxLayout(panes[0]), QVBoxLayout(panes[1])
		for c, pane in zip((left, right), panes):
			c.setSpacing(5)
			c.setContentsMargins(0, 0, 0, 0)
			cols.addWidget(pane, 1)
		col = [left]
		labels = []
		self.w = {}
		self.sections = []
		samp = settings.get("sampling") or {}

		def section(title):
			l = QLabel(title)
			theme.tag_font(l, 8.5, True)
			section_style(l)
			col[0].addWidget(l, 0, Qt.AlignmentFlag.AlignLeft)
			self.sections.append(l)

		def row(key, label, widget, tip="", extra=None):
			h = QHBoxLayout()
			l = QLabel(label)
			theme.tag_font(l, 8)
			l.setStyleSheet(f"color: {theme.MUTED};")
			l.setToolTip(tip)
			labels.append(l)
			widget.setToolTip(tip)
			widget.setFont(theme.mono(8.5))
			if isinstance(widget, QLineEdit) and tip:
				widget.textChanged.connect(lambda text, w=widget, t=tip: w.setToolTip(f"{text}\n{t}" if text else t))
				widget.setToolTip(f"{widget.text()}\n{tip}" if widget.text() else tip)
			h.addWidget(l)
			h.addWidget(widget, 1)
			if extra is not None:
				h.addWidget(extra)
			col[0].addLayout(h)
			self.w[key] = widget
			return widget

		def note(text, color=theme.DIM):
			l = QLabel(text)
			l.setFont(theme.mono(7.5))
			l.setStyleSheet(f"color: {color};")
			l.setWordWrap(True)
			l.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
			col[0].addWidget(l)
			return l

		def check(text, on, tip=""):
			cb = tab_focus(QCheckBox(text))
			cb.setFont(theme.mono(8))
			cb.setChecked(bool(on))
			cb.setToolTip(tip)
			col[0].addWidget(cb)
			return cb

		def spin(lo, hi, step, value, decimals=None):
			s = QDoubleSpinBox() if decimals else QSpinBox()
			if decimals:
				s.setDecimals(decimals)
			s.setRange(lo, hi)
			s.setSingleStep(step)
			s.setValue(value)
			s.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
			return guard_wheel(s)

		def percent(value):
			s = guard_wheel(StepSlider(1))
			s.setRange(0, 100)
			s.setValue(int(value))
			return s

		section("WORKING FOLDER")
		wd_browse = button("…", "pick the working folder", width=34)
		self.wd = QLineEdit(config.DATA_ENV.replace("\\", "/") if config.DATA_ENV else config.load_workdir())
		self.wd.setPlaceholderText("default: " + config.DEFAULT_DATA_DIR.replace("\\", "/"))
		row("workdir", "working folder (data)", self.wd, "", wd_browse)
		wd_browse.clicked.connect(self._pick_workdir)
		self.wd_hint = note("")
		self.wd_carry = check("bring my settings, window layout, goals and custom levers along", True, "copied into the new folder at the next start, only the files it does not have yet:\nsettings.json, ui.json, goals.json, custom_traits.json\nprofiles, levers, exports, lenses and caches are never copied nor moved")
		if config.DATA_ENV:
			self.wd.setEnabled(False)
			wd_browse.setEnabled(False)
			self.wd_carry.setVisible(False)
		self.wd.textChanged.connect(self._wd_text)
		self._wd_text()
		section("ENGINE")
		browse = button("…", "pick the llama.cpp folder (llama library, llama-fit-params)", width=34)
		row("llama_dir", "llama.cpp folder", QLineEdit(settings.get("llama_dir", "")), "folder with the llama and ggml libraries (llama.dll or libllama.so…) and llama-fit-params", browse)
		browse.clicked.connect(self._pick_llama)
		dirs = QPlainTextEdit("\n".join(settings.get("model_dirs") or []))
		dirs.setFixedHeight(58)
		row("model_dirs", "model folders (one per line)", dirs, "every GGUF and safetensors language model under these folders shows up in MODELS (the Hugging Face cache is always scanned)\nsafetensors: Hugging Face folders (config.json + tokenizer) or single files (ComfyUI text encoders…), converted to GGUF at the first LOAD")
		conv = guard_wheel(QComboBox())
		conv.addItem("16-bit, lossless (BF16 or F16 like the source)", "auto")
		conv.addItem("Q8_0, half the size (near lossless)", "q8_0")
		conv.setCurrentIndex(max(0, conv.findData(settings.get("convert_outtype") or "auto")))
		row("convert_outtype", "safetensors → GGUF", conv, "format of the GGUF written when a safetensors model is loaded the first time (by llama.cpp's convert_hf_to_gguf.py)\nFP8 weights are dequantized first · files go to the converted folder of the working folder\napplies to the next conversions: delete a converted file to convert it again")
		row("n_ctx", "default context (tokens)", spin(1024, 262144, 1024, int(settings.get("n_ctx", 8192))), "context size proposed in MODELS")
		row("vram_reserve_mb", "VRAM reserve on display GPUs (MiB)", spin(256, 8192, 64, int(settings.get("vram_reserve_mb", 900))), "memory left free on the GPUs that drive a screen (Windows, the browser, games grow there)\nwhen loading a model and when training a goal · GPUs without a screen keep 256 MiB")
		load = {**LOAD_DEFAULTS, **(settings.get("load") or {})}
		n_gpu = max(1, len(gpus))
		gpu_names = {i: gpus[i]["name"] if i < len(gpus) else f"CUDA{i}" for i in range(n_gpu)}
		for key, label, items, tips, tip in (
			("split_mode", "GPU split", SPLIT_MODES, SPLIT_TIPS, "how a model is spread over the GPUs (every GPU is used unless split is none)"),
			("main_gpu", "main GPU", list(range(n_gpu)), gpu_names, "with split none: the GPU holding the whole model\notherwise: the GPU for the small shared tensors"),
			("flash_attn", "flash attention", FLASH, FLASH_TIPS, "flash attention in llama.cpp"),
			("kv_type", "KV cache", KV_TYPES, KV_TIPS, "precision of the context memory (keys and values)"),
		):
			c = guard_wheel(QComboBox())
			for k in items:
				c.addItem(str(tips.get(k, k)), k)
			c.setCurrentIndex(max(0, c.findData(load[key])))
			row("load_" + key, label, c, tip + "\ndefault for every model: a model's own options (right-click it in MODELS → LOAD OPTIONS) come first")
		note("GPU dispatch for every model · a model can keep its own (right-click it in MODELS → LOAD OPTIONS)")
		section("GENERATION · dialogue")
		row("temperature", "temperature", spin(0.0, 2.0, 0.05, float(samp.get("temperature", 0.7)), 2), "0 = greedy")
		row("top_p", "top-p", spin(0.0, 1.0, 0.01, float(samp.get("top_p", 0.95)), 2), "nucleus sampling")
		row("top_k", "top-k", spin(0, 500, 1, int(samp.get("top_k", 40))), "0 = off")
		row("repeat_penalty", "repeat penalty", spin(1.0, 2.0, 0.01, float(samp.get("repeat_penalty", 1.05)), 2), "1.0 = off")
		row("max_tokens", "max tokens per answer", spin(16, 8192, 16, int(samp.get("max_tokens", 400))), "")
		section("COHERENCE")
		coh = {**config.DEFAULTS["coherence"], **(settings.get("coherence") or {})}
		self.c_cal = check("calibrate levers on real generations: strength 1 = as strong as stays coherent", coh["calibrate"], "each lever is tuned once on short generations (~10-20 s per lever, the first time it is used)\nunchecked: the older, stronger scale (tuned on a neutral text)")
		self.c_guard = check("coherence guard: test the levers on 2 short replies and scale them down if needed (~2-8 s)", coh["guard"], "the combination of levers is checked on real replies (the judge, or the surprise measure without one);\nif they read as incoherent, the levers are scaled down\nwhen: after each change with each lever change (when APPLY runs it), or with JUDGE NOW (JUDGE panel)\nunchecked (STANDBY): the levers apply exactly as set, at once: no judge, no KL safety, no factor kept from earlier judgements")
		row("apply_delay", "wait before applying the levers (s)", spin(0.0, 10.0, 0.5, float(settings.get("apply_delay", 3.0)), 1), "seconds after the last lever change before the levers are applied\neach change restarts the wait, so several sliders moved in a row are applied together\n0 = at once · APPLY in TO DO, or anything else queued behind (a chat…), does not wait · in STANDBY (no coherence guard) the levers apply at once")
		row("dose_max", "maximum lever dose", spin(1.0, 5.0, 0.25, float(settings.get("dose_max") or 1.5), 2), "how far the lever sliders and ALIGN may push each lever (±)\n1 = its calibrated strength (as strong as stays coherent alone); the hatched zone beyond 1 is past it\nthe coherence guard still has the last word when it is on")
		tol = percent(coh["tolerance"])
		self.tol_label = QLabel("")
		self.tol_label.setFont(theme.mono(7.5))
		self.tol_label.setFixedWidth(170)
		tol.valueChanged.connect(self._tol_text)
		row("tolerance", "tolerance   strict ↔ loose", tol, "how far the levers may push before the guard steps in\nstrict: only clean replies · loose: lets the style get wild\nwheel when focused: ±1", self.tol_label)
		self._tol_text(tol.value())
		ab = percent(coh["absurdity"])
		self.ab_label = QLabel("")
		self.ab_label.setFont(theme.mono(7.5))
		self.ab_label.setFixedWidth(170)
		ab.valueChanged.connect(self._ab_text)
		row("absurdity", "absurdity   sensible ↔ surreal", ab, "how much grammatical but absurd content counts as coherent\nsensible: surreal statements count as incoherent · surreal: only broken text and word salad do\n(with a judge: its level 2 'grammatical but absurd'; without: how much the replies may drift)\nwheel when focused: ±1", self.ab_label)
		self._ab_text(ab.value())
		judge = guard_wheel(QComboBox())
		judge.addItem("none: use the surprise measure", "")
		for m in sorted((m for m in models or [] if not m.get("error") and not m.get("convert") and m.get("arch") not in ("clip", "mtmd")), key=lambda m: m.get("size", 0)):
			judge.addItem(f"{os.path.splitext(m['file'])[0]} · {m.get('size', 0) / 2**30:.1f} GB", m["path"])
		judge.setCurrentIndex(max(0, judge.findData(coh["judge"])))
		row("judge", "judge model", judge, "a second, small model that rates the coherence of test replies (1 = coherent … 3 = word salad),\nread from its probabilities. Prefer a small UNCENSORED model: censored judges tend to call\ncoherent but unaligned text incoherent. It stays loaded on the GPU with the most free VRAM.")
		note("use a 3-8B uncensored or abliterated judge: 1B judges are too lenient, censored ones call unaligned text incoherent")
		ext = {**EXTRA_DEFAULTS, **coh}
		gmode = guard_wheel(QComboBox())
		for key, label in GUARD_MODES.items():
			gmode.addItem(label, key)
		gmode.setCurrentIndex(max(0, gmode.findData({"blame": "individual"}.get(ext["guard_mode"], ext["guard_mode"]))))
		row("guard_mode", "when replies get incoherent", gmode, "uniform: every lever is scaled down together\nindividual: only the levers changed since the last check are tried, the others keep their judged factor; among the changed ones the worst is halved first, then they are scaled together\nquick: a few replies per changed lever, the full check only if it fails\nread: the model reads its own replies with the levers, the judge only between the thresholds")
		ind = percent(ext["individual"])
		self.ind_label = QLabel("")
		self.ind_label.setFont(theme.mono(7.5))
		self.ind_label.setFixedWidth(170)
		ind.valueChanged.connect(self._ind_text)
		row("individual", "lever scale   common ↔ own", ind, "what strength 1 means for each lever\nown (100%): its own coherent maximum, measured on generations when it is calibrated\ncommon (0%): the same KL on a neutral text for every lever (older, stronger scale)\nin between: a blend\nwheel when focused: ±1", self.ind_label)
		self._ind_text(ind.value())
		self.c_gopt = check("guard the trained goal vectors too", ext["guard_optimized"], "unchecked: goal vectors trained by OPTIMIZE are applied exactly as trained, outside the guard\n(the guard's surprise measure reads their change of character as incoherence and cuts them)\nchecked: they are scaled together with the other levers")
		row("judge_temperature", "judge temperature", spin(0.05, 5.0, 0.05, float(ext["judge_temperature"]), 2), "applied to the judge's answer probabilities before the score is read\n1 = as read · above 1 = softer, less sure scores · below 1 = sharper")
		jp = QPlainTextEdit(ext["judge_prompt"] or "")
		jp.setPlaceholderText("empty = automatic: the built-in question that passes the self-test best\ncustom: must contain {reply} (and may use {question})")
		jp.setFixedHeight(58)
		row("judge_prompt", "custom judge question", jp, "your own question for the judge, used when it passes the self-test better than the built-in ones\nmust contain {reply}; {question} is replaced by the test question\nwith lines like \"1 = …\" up to \"4 = …\" the answer is read as 4 levels (1 = coherent … 4 = word salad), otherwise as Yes / No")
		tp = QPlainTextEdit("\n".join(ext["test_prompts"] or []))
		tp.setPlaceholderText("empty = built-in:\n" + "\n".join(COH_PROMPTS[:2]))
		tp.setFixedHeight(58)
		row("test_prompts", "test questions (one per line)", tp, "the questions the levers' test replies answer (calibration and guard)\nempty: two built-in everyday questions")
		row("test_system", "test system prompt", QLineEdit(ext["test_system"] or ""), "system prompt for the test replies (empty = none)\nuseful when you chat with a system prompt: the guard then tests the same situation")
		row("test_tokens", "test reply length (tokens)", spin(12, 200, 4, int(ext["test_tokens"])), "longer replies: more reliable scores, slower")
		row("test_samples", "replies per test question", spin(1, 6, 1, int(ext["test_samples"])), "more replies: steadier scores, slower")
		col[0] = right
		section("STYLE · SYNC DISPLAY")
		look = guard_wheel(QComboBox())
		for key, label in theme.STYLES.items():
			look.addItem(label, key)
		look.setCurrentIndex(max(0, look.findData(self.prefs["style"])))
		row("style", "visual style", look, "NERV: neon military-terminal look (condensed captions, neon frames, MAGI-style panels)\nplain: the sober look\npreviewed as you pick it · CANCEL restores the previous one")
		accent = QComboBox()
		accent.addItems(list(theme.ACCENTS))
		accent.setCurrentText(self.prefs["accent"])
		row("accent", "accent color", guard_wheel(accent), "previewed as you pick it · CANCEL restores the previous one")
		place = guard_wheel(QComboBox())
		for key, label in SYNC_PLACES.items():
			place.addItem(label, key)
		place.setCurrentIndex(max(0, place.findData(self.prefs["sync_place"])))
		row("sync_place", "sync display", place, "where the INTENT / APPLIED / MEASURED synchro display goes\nband: full width between the top and bottom panels (drag the edge above it to resize)\nstrip: the small sine or graph strip at the top of LEVERS (click it to switch)")
		scenes = guard_wheel(QComboBox())
		for key, label in SYNC_SCENES.items():
			scenes.addItem(label, key)
		scenes.setCurrentIndex(max(0, scenes.findData(self.prefs["sync_scenes"])))
		row("sync_scenes", "sync band scenes", scenes, "what the band shows next to the SYNCHRO RATIO readout\n(a click on the band cycles through them too)")
		row("sync_height", "sync band height (px)", spin(64, 420, 10, int(self.prefs["sync_height"])), "height of the sync band · dragging the edge above the band sets it too")
		self.c_warn = check("sync band: flash a WARNING banner when coherence falls below the guard", self.prefs["sync_warn"], "a red WARNING banner flashes for a few seconds when the coherence guard measures the steered replies below its threshold:\nat the strengths you set (it then scales the levers down, the banner shows before → after) or still after its corrections\nthe small WARNING or CAUTION badge stays either way")
		self.c_ready = check("sync band: flash READY when the model is ready to talk", self.prefs.get("sync_ready", True), "a green READY banner shows for a few seconds when a job that changes the model ends (load, levers, calibration, profile, goal) and nothing else is queued · the small READY badge stays either way")
		self.c_scan = check("sync band: CRT scanlines", self.prefs["scanlines"], "thin dark scanlines over the sync band, like an old monitor")
		section("LENSES")
		refresh = button("REFRESH", "re-read the lens list from the Hub now")
		refresh.clicked.connect(self.refresh_lenses.emit)
		row("lens_repos", "Hub lens repositories", QLineEdit(", ".join(settings.get("lens_repos") or ["neuronpedia/jacobian-lens"])), "comma separated Hugging Face repos holding *_jacobian_lens*.pt files", refresh)
		pm = guard_wheel(QComboBox())
		for k, label in (("read_write", "READ + WRITE"), ("read", "READ"), ("write", "WRITE")):
			pm.addItem(label, k)
		mode = settings.get("profile_mode") or ("read_write" if settings.get("profile_read", True) else "write")
		pm.setCurrentIndex(max(0, pm.findData(mode)))
		row("profile_mode", "profile with the levers", pm, "how a profile with levers is measured (MEASURE, after a lever change, ALIGN rounds and lever columns in TARGET); the baseline is always written\nWRITE: the model writes new answers, as it is steered (≈10 s for the quick battery)\nREAD: it reads the answers it wrote for its baseline, levers on, and the change read is added to the baseline (≈2 s); same direction as writing (r ≈ 0.98 on Qwen3.5-4B) but about half the size, and a system prompt shows at about a third\nREAD + WRITE: reads, scaled per lever by one written profile of that lever alone (≈10 s once per lever, kept with the model, redone when the lever changes); a lever not scaled yet uses the average of this model's levers until its own scale is measured, when the TO DO list is idle; in ALIGN (TARGET), a mix of levers is also written once to scale it, and again when its doses move by more than 0.5; with a system prompt other than the baseline's, it writes\nREAD and READ + WRITE need a Jacobian lens: with the logit lens the profile is written")
		section("DISPLAY")
		row("fps_idle", "3D frame rate (idle)", spin(10, 144, 5, int(self.prefs["fps_idle"])), "frames per second of the Persona Iris when nothing runs")
		row("fps_busy", "3D frame rate (busy)", spin(5, 144, 5, int(self.prefs["fps_busy"])), "lower = more CPU left to the engine while it works (the UI shares Python with it)")
		row("spin", "iris rotation (°/s)", spin(-360.0, 360.0, 0.5, float(self.prefs["spin"]), 1), "degrees per second · negative: the other way round · 0 = still\nR in the iris pauses or resumes the rotation")
		arm = guard_wheel(QComboBox())
		for key, label in ARM_ORDERS.items():
			arm.addItem(label, key)
		arm.setCurrentIndex(max(0, arm.findData(self.prefs["arm_order"])))
		row("arm_order", "iris arm order", arm, "profile order: the order of the traits list\nsimilar traits side by side: traits whose profile (layer by layer, in the workspace band)\nlooks alike get neighbouring arms, computed on the baseline so the arms do not move when you measure")
		row("concept_count", "concept clouds in the iris", spin(20, 200, 10, int(self.prefs["concept_count"])), "signature concepts drawn around the iris (the profile keeps up to 200; re-profile older models to get more)")
		row("concepts", "concept labels (AUTO mode)", spin(0, 16, 1, int(self.prefs["concepts"])), "labels shown in AUTO mode; ALL shows every label that fits, zoom in to reveal more")
		self.c_spread = check("AUTO labels spread all around the iris", self.prefs["concepts_spread"], "checked: the circle is cut into as many slices as there are labels, and each slice shows its strongest concept first\nunchecked: the strongest concepts first, wherever they are (they may gather on one arm)")
		self.c_fade = check("CONCEPTS panel: fade the weaker concepts", self.prefs["concepts_fade"], "checked: the less specific a concept, the fainter it is drawn\nunchecked: every concept is drawn fully opaque\neither way, the last row fades out when the list does not fit the panel")
		row("tooltip_ms", "tooltip delay (ms)", spin(0, 5000, 50, int(self.prefs["tooltip_ms"])), "how long the mouse rests on something before its tooltip shows\n700 = Qt's default · 0 = immediately")
		section("NAVIGATION · iris")
		orbit = guard_wheel(QComboBox())
		pan = guard_wheel(QComboBox())
		for combo, key in ((orbit, "orbit_button"), (pan, "pan_button")):
			combo.addItems(["left", "right", "middle"])
			combo.setCurrentText(self.prefs[key])
		row("orbit_button", "orbit and tilt: drag with", orbit, "mouse button that rotates and tilts the iris")
		row("pan_button", "move the view: drag with", pan, "mouse button that moves the view\na left-click without moving still pins cards")
		center = guard_wheel(QComboBox())
		reset = guard_wheel(QComboBox())
		for combo, key in ((center, "center_click"), (reset, "reset_click")):
			combo.addItems(list(CLICKS))
			combo.setCurrentText(self.prefs[key] if self.prefs[key] in CLICKS else "none")
		row("center_click", "re-center: click with", center, "re-center the view on the iris, keeping zoom and tilt (like ◎ or C)\na click = press and release without moving: the drags keep working with the same button")
		row("reset_click", "reset the view: click with", reset, "reset tilt, zoom and position (like ⌖ or Home)\na click = press and release without moving: the drags keep working with the same button")
		note("wheel: zoom (toward the cursor once the view was moved) · ◎ or C: re-center · ⌖ or Home: reset · R: rotation · a left-click without moving pins cards")
		self.nav = {key: self.w[key].currentText() for key in ("orbit_button", "pan_button", "center_click", "reset_click")}
		orbit.currentTextChanged.connect(lambda t: self._nav_changed("orbit_button", "pan_button", t))
		pan.currentTextChanged.connect(lambda t: self._nav_changed("pan_button", "orbit_button", t))
		center.currentTextChanged.connect(lambda t: self._nav_changed("center_click", "reset_click", t))
		reset.currentTextChanged.connect(lambda t: self._nav_changed("reset_click", "center_click", t))
		section("SOUND")
		self.c_sound = check("sound effects: NERV-style blips, alarms and chimes", self.prefs["sound"], "every sound is synthesized by Pendulum (numpy), nothing is borrowed\nplayed through the default Windows output")
		vol = percent(self.prefs["sound_volume"])
		self.vol_label = QLabel("")
		self.vol_label.setFont(theme.mono(7.5))
		self.vol_label.setFixedWidth(170)
		vol.valueChanged.connect(lambda v: self.vol_label.setText(f"  {v}%"))
		vol.sliderReleased.connect(lambda: self.c_sound.isChecked() and self.hear("lever"))
		row("sound_volume", "volume", vol, "loudness of every sound · a sound plays when you let go of the slider\nwheel when focused: ±1", self.vol_label)
		self.vol_label.setText(f"  {vol.value()}%")
		grid = QGridLayout()
		grid.setHorizontalSpacing(14)
		grid.setVerticalSpacing(2)
		self.c_groups = {}
		on = self.prefs.get("sound_groups") or {}
		for i, (k, (label, tip)) in enumerate(SOUND_GROUPS.items()):
			cb = tab_focus(QCheckBox(label.lower()))
			cb.setFont(theme.mono(8))
			cb.setChecked(bool(on.get(k, True)))
			cb.setToolTip(tip)
			grid.addWidget(cb, i // 3, i % 3)
			self.c_groups[k] = cb
		col[0].addLayout(grid)
		pv = QGridLayout()
		pv.setHorizontalSpacing(4)
		pv.setVerticalSpacing(4)
		for i, (name, (_fn, group, label, tip)) in enumerate(SOUNDS.items()):
			b = button("▶ " + label.upper(), f"{label}: {tip}\n({SOUND_GROUPS[group][0].lower()})", click=False)
			b.clicked.connect(lambda _c=False, n=name: self.hear(n))
			pv.addWidget(b, i // 4, i % 4)
		col[0].addLayout(pv)
		srow = QHBoxLayout()
		self.b_demo = button("▶ DEMO", "play every sound one after the other (about 25 s) · click again to stop", accent=True, click=False)
		self.b_demo.clicked.connect(self._demo)
		b_dir = button("SOUND FOLDER", "your own sounds: drop <name>.wav files here to replace the synthesized ones\n" + sound_folder().replace("\\", "/"))
		b_dir.clicked.connect(self._sound_folder)
		srow.addWidget(self.b_demo)
		srow.addWidget(b_dir)
		srow.addStretch(1)
		col[0].addLayout(srow)
		self.sound_note = note("")
		self.demo = []
		self.demo_timer = QTimer(self)
		self.demo_timer.setSingleShot(True)
		self.demo_timer.timeout.connect(self._demo_step)
		self._sound_text()
		left.addStretch(1)
		right.addStretch(1)
		lw = max(l.fontMetrics().horizontalAdvance(l.text()) for l in labels) + 12
		for l in labels:
			l.setFixedWidth(lw)
		for l, fn in ((self.tol_label, self._tol_text), (self.ab_label, self._ab_text), (self.ind_label, self._ind_text), (self.vol_label, lambda v: self.vol_label.setText(f"  {v}%"))):
			keep = l.text()
			texts = []
			for v in range(101):
				fn(v)
				texts.append(l.text())
			l.setText(keep)
			l.setFixedWidth(max(l.fontMetrics().horizontalAdvance(t) for t in texts) + 6)
		cols.setContentsMargins(0, 0, self.scroll.verticalScrollBar().sizeHint().width() + 8, 0)
		self.ensurePolished()
		pw = max(p.sizeHint().width() for p in panes)
		for p in panes:
			p.setMinimumWidth(pw)
		btns = QHBoxLayout()
		defaults = button("DEFAULTS", "restore the default display and generation values")
		defaults.clicked.connect(self._defaults)
		btns.addWidget(defaults)
		setup = button("SETUP…", "the first-start setup again: working folder, llama.cpp (detection and download), model folders, judge\nunsaved changes made here are dropped")
		setup.clicked.connect(self._setup)
		btns.addWidget(setup)
		self.wizard_wanted = False
		btns.addStretch(1)
		cancel = button("CANCEL", "close without saving (display changes are reverted)")
		ok = button("SAVE", "save (Enter)", accent=True)
		ok.setDefault(True)
		cancel.clicked.connect(self.reject)
		ok.clicked.connect(self.accept)
		btns.addWidget(cancel)
		btns.addWidget(ok)
		outer.addLayout(btns)
		for k in DISPLAY_KEYS:
			self.w[k].valueChanged.connect(self._preview)
		accent.currentTextChanged.connect(self._preview)
		arm.currentIndexChanged.connect(self._preview)
		self.c_spread.toggled.connect(self._preview)
		self.c_fade.toggled.connect(self._preview)
		for k in STYLE_COMBOS:
			self.w[k].currentIndexChanged.connect(self._preview)
		self.c_warn.toggled.connect(self._preview)
		self.c_ready.toggled.connect(self._preview)
		self.c_scan.toggled.connect(self._preview)
		self.fit_margins()
		self.resize(self.fit_size())

	def fit_size(self):
		avail = (self.parentWidget().screen() if self.parentWidget() is not None else self.screen()).availableGeometry()
		hint, sh, want = self.layout().sizeHint(), self.scroll.sizeHint(), self.scroll.widget().sizeHint()
		w, h = hint.width() - sh.width() + want.width(), hint.height() - sh.height() + want.height()
		cap_h = avail.height() - 50
		if h > cap_h:
			w += self.scroll.verticalScrollBar().sizeHint().width()
		return QSize(min(w, avail.width() - 40), min(h, cap_h))

	def hear(self, name, key=None):
		SFX.preview(name, self.w["sound_volume"].value() / 100.0, SOUND_PREVIEW.get(name) if key is None else key)
		self._sound_text()

	def _demo(self):
		if self.demo:
			self.demo = []
			self.demo_timer.stop()
			self.b_demo.setText("▶ DEMO")
			return
		self.demo = list(SOUND_DEMO)
		self.b_demo.setText("■ STOP")
		self._demo_step()

	def _demo_step(self):
		if not self.demo:
			return
		name, key, _d = self.demo.pop(0)
		self.hear(name, key)
		if self.demo:
			self.demo_timer.start(int(self.demo[0][2] * 1000))
		else:
			self.b_demo.setText("▶ DEMO")

	def _sound_folder(self):
		os.makedirs(sound_folder(), exist_ok=True)
		QDesktopServices.openUrl(QUrl.fromLocalFile(sound_folder()))

	def _sound_text(self):
		SFX.load_pack()
		if SFX.muted:
			text, color = "sound is off for this run (PENDULUM_NO_SOUND is set)", theme.WARN
		elif SFX.failed:
			text, color = f"no sound output: {SFX.failed}", theme.ERR
		elif SFX.pack_errors:
			text, color = "cannot read " + " · ".join(f"{n}.wav ({e})" for n, e in SFX.pack_errors.items()), theme.WARN
		elif SFX.pack:
			text, color = "your sounds replace: " + ", ".join(SFX.pack), theme.OK
		else:
			text, color = "your own sounds: put <name>.wav (PCM) in the SOUND FOLDER to replace one · names: " + ", ".join(SOUNDS), theme.DIM
		self.sound_note.setText(text)
		self.sound_note.setStyleSheet(f"color: {color};")

	def done(self, r):
		self.demo = []
		self.demo_timer.stop()
		super().done(r)

	def _ab_text(self, v):
		self.ab_label.setText(f"  absurd counts {v}% coherent")

	def _ind_text(self, v):
		self.ind_label.setText("  own coherent maximum" if v >= 100 else ("  common KL scale" if v <= 0 else f"  {v}% own · {100 - v}% common"))

	def _tol_text(self, v):
		self.tol_label.setText(f"  guard below {threshold(v):.0%}")

	def _wd_text(self, *_a):
		if config.DATA_ENV:
			env = config.DATA_ENV.replace("\\", "/")
			self.wd.setToolTip(f"{env}\nset for this run by --data-dir (or PENDULUM_DATA_DIR): workdir.json is neither read nor written")
			self.wd_hint.setText(f"set for this run by --data-dir or PENDULUM_DATA_DIR (workdir.json is not used): {env}")
			return
		target = config.norm_dir(self.wd.text())
		shown = target or config.DEFAULT_DATA_DIR.replace("\\", "/")
		now = config.DATA_DIR.replace("\\", "/")
		self.wd.setToolTip(f"{shown}\nwhere Pendulum keeps profiles, levers, exports, trained lenses, caches, settings and window layout\nempty = the data folder of the project · applies after a restart · existing data is never moved")
		if config.same_dir(target, config.DATA_DIR):
			self.wd_hint.setText(f"in use: {now}" + (f"   ·   {config.DATA_DIR_ERROR}" if config.DATA_DIR_ERROR else ""))
			self.wd_hint.setStyleSheet(f"color: {theme.ERR if config.DATA_DIR_ERROR else theme.DIM};")
			self.wd_carry.setVisible(False)
			return
		self.wd_hint.setText(f"applies after a restart · the data in {now} is not moved: copy it to {shown} yourself if you want it there")
		self.wd_hint.setStyleSheet(f"color: {theme.WARN};")
		self.wd_carry.setVisible(True)

	def _pick_workdir(self):
		d = QFileDialog.getExistingDirectory(self, "Working folder", self.wd.text() or config.DATA_DIR)
		if d:
			self.wd.setText(d.replace("\\", "/"))

	def workdir_choice(self):
		if config.DATA_ENV:
			return None
		target = config.norm_dir(self.wd.text())
		if config.same_dir(target, config.load_workdir()):
			return None
		return target, self.wd_carry.isChecked() and not self.wd_carry.isHidden()

	def accept(self):
		choice = self.workdir_choice()
		if choice is not None:
			path = choice[0] or config.DEFAULT_DATA_DIR
			err = f"{path} is a file, not a folder" if os.path.isfile(path) else ""
			if not err:
				try:
					os.makedirs(path, exist_ok=True)
				except OSError as exc:
					err = f"cannot use {path}: {exc}"
			if err:
				self.wd_hint.setText(err)
				self.wd_hint.setStyleSheet(f"color: {theme.ERR};")
				self.wd.setFocus()
				return
		super().accept()

	def _nav_changed(self, key, other, text):
		prev = self.nav[key]
		if text == self.w[other].currentText() and text != "none":
			self.w[other].blockSignals(True)
			self.w[other].setCurrentText(prev)
			self.w[other].blockSignals(False)
			self.nav[other] = prev
		self.nav[key] = text
		self._preview()

	def _preview(self, *_a):
		self.preview.emit(self.values()[1])
		self.setStyleSheet(theme.STYLE)
		theme.refont(self)
		for l in self.sections:
			section_style(l)
		self.fit_margins()
		self.update()

	def _pick_llama(self):
		d = QFileDialog.getExistingDirectory(self, "llama.cpp folder", self.w["llama_dir"].text())
		if d:
			self.w["llama_dir"].setText(d)

	def _setup(self):
		self.wizard_wanted = True
		self.reject()

	def _defaults(self):
		samp = config.DEFAULTS["sampling"]
		for k in ("temperature", "top_p", "top_k", "repeat_penalty", "max_tokens"):
			self.w[k].setValue(samp[k])
		for k in DISPLAY_KEYS:
			self.w[k].setValue(PREF_DEFAULTS[k])
		self.w["accent"].setCurrentText(PREF_DEFAULTS["accent"])
		self.w["arm_order"].setCurrentIndex(max(0, self.w["arm_order"].findData(PREF_DEFAULTS["arm_order"])))
		self.c_spread.setChecked(PREF_DEFAULTS["concepts_spread"])
		self.c_fade.setChecked(PREF_DEFAULTS["concepts_fade"])
		for k in STYLE_COMBOS:
			self.w[k].setCurrentIndex(max(0, self.w[k].findData(PREF_DEFAULTS[k])))
		self.c_warn.setChecked(PREF_DEFAULTS["sync_warn"])
		self.c_ready.setChecked(PREF_DEFAULTS["sync_ready"])
		self.c_scan.setChecked(PREF_DEFAULTS["scanlines"])
		self.c_sound.setChecked(PREF_DEFAULTS["sound"])
		self.w["sound_volume"].setValue(PREF_DEFAULTS["sound_volume"])
		for k, cb in self.c_groups.items():
			cb.setChecked(PREF_DEFAULTS["sound_groups"].get(k, True))

	def values(self):
		w = self.w
		settings = {
			"llama_dir": w["llama_dir"].text().strip().replace("\\", "/"),
			"model_dirs": [d.strip().replace("\\", "/") for d in w["model_dirs"].toPlainText().splitlines() if d.strip()],
			"n_ctx": int(w["n_ctx"].value()),
			"convert_outtype": w["convert_outtype"].currentData() or "auto",
			"vram_reserve_mb": int(w["vram_reserve_mb"].value()),
			"dose_max": round(float(w["dose_max"].value()), 2),
			"apply_delay": round(float(w["apply_delay"].value()), 1),
			"profile_mode": w["profile_mode"].currentData() or "read_write",
			"load": {k: (int(w["load_" + k].currentData() or 0) if k == "main_gpu" else w["load_" + k].currentData()) for k in ("split_mode", "main_gpu", "flash_attn", "kv_type")},
			"lens_repos": [r.strip() for r in w["lens_repos"].text().split(",") if r.strip()],
			"coherence": {**(self.settings.get("coherence") or {}), "calibrate": self.c_cal.isChecked(), "guard": self.c_guard.isChecked(), "tolerance": int(w["tolerance"].value()), "absurdity": int(w["absurdity"].value()), "judge": w["judge"].currentData() or "", "guard_mode": w["guard_mode"].currentData() or "uniform", "individual": int(w["individual"].value()), "guard_optimized": self.c_gopt.isChecked(), "judge_temperature": round(w["judge_temperature"].value(), 3), "judge_prompt": w["judge_prompt"].toPlainText().strip(), "test_prompts": [l.strip() for l in w["test_prompts"].toPlainText().splitlines() if l.strip()], "test_system": w["test_system"].text().strip(), "test_tokens": int(w["test_tokens"].value()), "test_samples": int(w["test_samples"].value())},
			"sampling": {"temperature": round(w["temperature"].value(), 3), "top_p": round(w["top_p"].value(), 3), "top_k": int(w["top_k"].value()), "repeat_penalty": round(w["repeat_penalty"].value(), 3), "max_tokens": int(w["max_tokens"].value())},
		}
		prefs = {**self.prefs, "style": w["style"].currentData() or "nerv", "sync_place": w["sync_place"].currentData() or "band", "sync_scenes": w["sync_scenes"].currentData() or "both", "sync_height": int(w["sync_height"].value()), "sync_warn": self.c_warn.isChecked(), "sync_ready": self.c_ready.isChecked(), "scanlines": self.c_scan.isChecked(), "sound": self.c_sound.isChecked(), "sound_volume": int(w["sound_volume"].value()), "sound_groups": {k: cb.isChecked() for k, cb in self.c_groups.items()}, "fps_idle": int(w["fps_idle"].value()), "fps_busy": int(w["fps_busy"].value()), "spin": round(w["spin"].value(), 2), "concepts": int(w["concepts"].value()), "accent": w["accent"].currentText(), "rotate": self.prefs.get("rotate", True), "concepts_mode": self.prefs.get("concepts_mode", "auto"), "orbit_button": w["orbit_button"].currentText(), "pan_button": w["pan_button"].currentText(), "concept_count": int(w["concept_count"].value()), "tooltip_ms": int(w["tooltip_ms"].value()), "concepts_spread": self.c_spread.isChecked(), "concepts_fade": self.c_fade.isChecked(), "arm_order": w["arm_order"].currentData() or "profile", "center_click": w["center_click"].currentText(), "reset_click": w["reset_click"].currentText()}
		return settings, prefs


TRAIT_FIELDS = (
	("name", "NAME", "e.g. Pirate", "the lever's name in the LEVERS list"),
	("plus", "PLUS POLE", "e.g. pirate-like", "one or two words for the + side of the slider"),
	("seeds_plus", "PLUS WORDS (optional)", "comma separated: pirate, arr, matey, sea, treasure", "words of the + side: they make the lexicon the lens reads the trait with\nwithout words on both sides the lever still steers, but it is not measured (no MEASURED value, no TARGET)"),
	("desc_plus", "PLUS CHARACTER (optional)", "e.g. a salty old pirate who talks about the sea and says arr", "who the model plays at the + pole: the lever is extracted from replies written in this character\nempty: built from the pole and its words"),
	("minus", "MINUS POLE", "e.g. landlubber", "one or two words for the − side of the slider"),
	("seeds_minus", "MINUS WORDS (optional)", "comma separated: formal, polite, office, standard", "words of the − side\nwithout words on both sides the lever still steers, but it is not measured"),
	("desc_minus", "MINUS CHARACTER (optional)", "e.g. a polite, formal office clerk", "who the model plays at the − pole\nempty: built from the pole and its words"),
)
NEEDED = ("name", "plus", "minus")


def _note(text, color=None):
	l = QLabel(text)
	l.setFont(theme.mono(7.5))
	l.setStyleSheet(f"color: {color or theme.DIM};")
	l.setWordWrap(True)
	return l


def _caption(text):
	l = QLabel(text)
	theme.tag_font(l, 7.5)
	l.setStyleSheet(f"color: {theme.MUTED};")
	return l


POLE_ROLE = Qt.ItemDataRole.UserRole + 1


class ImageDelegate(QStyledItemDelegate):
	def paint(self, p, opt, index):
		super().paint(p, opt, index)
		if not index.data(POLE_ROLE):
			return
		r = QRectF(opt.rect).adjusted(1.5, 1.5, -1.5, -1.5)
		p.save()
		p.setPen(QPen(QColor(theme.ACCENT), 2))
		p.setBrush(Qt.BrushStyle.NoBrush)
		p.drawRect(r)
		f = theme.mono(6.5, True)
		fm = QFontMetrics(f)
		tag = QRectF(r.left(), r.bottom() - fm.height() - 1, fm.horizontalAdvance("POLE") + 8, fm.height() + 1)
		p.fillRect(tag, QColor(theme.ACCENT))
		p.setFont(f)
		p.setPen(QColor(theme.ON_ACCENT))
		p.drawText(tag, Qt.AlignmentFlag.AlignCenter, "POLE")
		p.restore()


class ImageList(QListWidget):
	EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
	THUMB = 64

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setViewMode(QListWidget.ViewMode.IconMode)
		self.setIconSize(QSize(self.THUMB, self.THUMB))
		self.setGridSize(QSize(self.THUMB + 10, self.THUMB + 10))
		self.setResizeMode(QListWidget.ResizeMode.Adjust)
		self.setMovement(QListWidget.Movement.Static)
		self.setFixedHeight(self.THUMB + 26)
		self.setAcceptDrops(True)
		self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
		self.setItemDelegate(ImageDelegate(self))
		self.setToolTip("drop images here, or double-click an empty spot to pick some\nclick an image: POLE or not · Delete removes the selected ones")
		self.itemClicked.connect(self.toggle_pole)

	def paths(self):
		return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

	def poles(self):
		return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count()) if self.item(i).data(POLE_ROLE)]

	def toggle_pole(self, it):
		it.setData(POLE_ROLE, not it.data(POLE_ROLE))
		it.setToolTip(it.data(Qt.ItemDataRole.UserRole) + ("\nPOLE: the lever leans on it most" if it.data(POLE_ROLE) else ""))

	def add(self, paths, poles=()):
		have = set(self.paths())
		marked = {os.path.normcase(os.path.normpath(p)) for p in poles}
		for p in paths:
			p = os.path.normpath(p)
			if not p.lower().endswith(self.EXT) or p in have or not os.path.isfile(p):
				continue
			pm = QPixmap(p)
			icon = QIcon(pm.scaled(self.THUMB, self.THUMB, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)) if not pm.isNull() else QIcon()
			it = QListWidgetItem(icon, "")
			it.setData(Qt.ItemDataRole.UserRole, p)
			it.setData(POLE_ROLE, os.path.normcase(p) in marked)
			it.setToolTip(p + ("\nPOLE: the lever leans on it most" if it.data(POLE_ROLE) else ""))
			self.addItem(it)
			have.add(p)

	def browse(self):
		files, _f = QFileDialog.getOpenFileNames(self, "Images", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
		self.add(files)

	def dragEnterEvent(self, e):
		if e.mimeData().hasUrls():
			e.acceptProposedAction()

	def dragMoveEvent(self, e):
		if e.mimeData().hasUrls():
			e.acceptProposedAction()

	def dropEvent(self, e):
		self.add([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
		e.acceptProposedAction()

	def mouseDoubleClickEvent(self, e):
		if self.itemAt(e.position().toPoint()) is None:
			self.browse()
			return
		super().mouseDoubleClickEvent(e)

	def keyPressEvent(self, e):
		if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
			for it in self.selectedItems():
				self.takeItem(self.row(it))
			return
		super().keyPressEvent(e)


class TraitDialog(EvaDialog):
	MODES = {"new": ("New lever", "CREATE"), "edit": ("Edit lever", "SAVE"), "duplicate": ("Duplicate lever", "CREATE")}

	def __init__(self, parent=None, trait=None, mode="new", vision=None):
		super().__init__(parent)
		title, verb = self.MODES.get(mode, self.MODES["new"])
		self.mode = mode
		self.setWindowTitle(title)
		self.setStyleSheet(theme.STYLE)
		self.setMinimumWidth(520)
		lay = QVBoxLayout(self)
		lay.setSpacing(3)
		lay.addWidget(DialogTitle(title.upper(), (trait or {}).get("name", "") if mode != "new" else ""))
		t = dict(trait or {})
		if mode == "duplicate" and t.get("name"):
			t["name"] = f"{t['name']} copy"
		self.fields = {}
		self._trait = t
		self.switch = SegSwitch((("words", "WORDS", "a lever made from words: name, poles, words and characters"), ("images", "IMAGES", "a lever made from images: the model looks at them (no text is written) and the lever is what sets the + images apart from the − ones")))
		self.switch.set_value("words")
		self.switch.picked.connect(self._page)
		lay.addWidget(self.switch)
		self.words_page = QWidget()
		page = QVBoxLayout(self.words_page)
		page.setContentsMargins(0, 0, 0, 0)
		page.setSpacing(3)
		lay.addWidget(self.words_page)
		self.images_page = self._images_page(vision or {})
		self.images_page.setVisible(False)
		lay.addWidget(self.images_page)
		for key, label, ph, tip in TRAIT_FIELDS:
			l = QLabel(label)
			theme.tag_font(l, 7.5)
			l.setStyleSheet(f"color: {theme.MUTED}; padding-top: 4px;")
			v = t.get(key)
			e = QLineEdit(", ".join(v) if isinstance(v, list) else str(v or ""))
			e.setFont(theme.mono(9))
			e.setPlaceholderText(ph)
			e.setToolTip(tip)
			hscroll(e)
			l.setToolTip(tip)
			page.addWidget(l)
			page.addWidget(e)
			self.fields[key] = e
		self.needed = ("name",) if t.get("lever_only") and not t.get("no_lexicon") else NEEDED
		if t.get("no_lexicon"):
			page.addWidget(_note("not measured on this model: none of the words of one side is a single token for it\nadd more words (or English ones) to have the lens measure it; it steers either way", theme.WARN))
		elif t.get("lever_only") and t.get("from_goal"):
			page.addWidget(_note("made from a goal: no words needed, it steers without being measured\nadd words on both sides to have the lens measure it too"))
		else:
			page.addWidget(_note("words are optional: without words on both sides the lever steers but is not measured (no MEASURED value, no TARGET)"))
		if mode == "edit" and t and not t.get("custom", True):
			self.switch.buttons["images"].setEnabled(False)
			self.switch.buttons["images"].setToolTip("images are for custom levers: DUPLICATE this default lever to make one")
		if t.get("images_plus") or t.get("images_minus"):
			self._page("images")
		if mode == "edit":
			lay.addWidget(_note("saving rebuilds the lever: it goes back to 0 and is extracted again the next time you move it"))
		elif mode == "duplicate":
			lay.addWidget(_note("the copy is a custom lever: edit it freely, the original stays as it is"))
		self.hint = _note("", theme.WARN)
		lay.addWidget(self.hint)
		row = QHBoxLayout()
		row.addStretch(1)
		ok = button(verb, accent=True)
		ok.setDefault(True)
		cancel = button("CANCEL")
		ok.clicked.connect(self._ok)
		cancel.clicked.connect(self.reject)
		row.addWidget(cancel)
		row.addWidget(ok)
		lay.addLayout(row)

	def _caption(self, text, tip=""):
		l = QLabel(text)
		theme.tag_font(l, 7.5)
		l.setStyleSheet(f"color: {theme.MUTED}; padding-top: 4px;")
		l.setToolTip(tip)
		return l

	def _images_page(self, vision):
		w = QWidget()
		v = QVBoxLayout(w)
		v.setContentsMargins(0, 0, 0, 0)
		v.setSpacing(3)
		mm = vision.get("mmproj", "")
		v.addWidget(_note(f"vision: {mm}" if mm else (vision.get("why") or "load a model that has an mmproj*.gguf next to it to see images"), theme.DIM if mm else theme.WARN))
		v.addWidget(self._caption("NAME", "the lever's name in the LEVERS list"))
		self.img_name = QLineEdit(self._trait.get("name", ""))
		self.img_name.setFont(theme.mono(9))
		self.img_name.setPlaceholderText("e.g. Dread")
		hscroll(self.img_name)
		v.addWidget(self.img_name)
		self.img_lists = {}
		for pole, cap in (("plus", "PLUS IMAGES"), ("minus", "MINUS IMAGES")):
			row = QHBoxLayout()
			row.addWidget(self._caption(cap, "images of this side: drop them in, click one to make it a POLE"))
			row.addStretch(1)
			lst = ImageList()
			add = button("ADD…", "pick images for this side (or drop them in the box)")
			add.clicked.connect(lst.browse)
			clr = button("CLEAR", "remove every image of this side")
			clr.clicked.connect(lst.clear)
			row.addWidget(add)
			row.addWidget(clr)
			v.addLayout(row)
			v.addWidget(lst)
			lst.add(self._trait.get(f"images_{pole}") or [], self._trait.get(f"poles_{pole}") or [])
			self.img_lists[pole] = lst
		v.addWidget(_note("click an image to make it a POLE: the lever leans on it most (every image counts, poles included)\nthe model only looks at them, no text is written · one side only: compared with no image\nthe images are copied into the working folder so the lever can be rebuilt on any model that sees"))
		return w

	def _page(self, key):
		self.words_page.setVisible(key == "words")
		self.images_page.setVisible(key == "images")
		self.switch.set_value(key)
		self.adjustSize()

	def kind(self):
		return self.switch.value() or "words"

	def _ok(self):
		v = self.data()
		if self.kind() == "images":
			missing = (["name"] if not v["name"] else []) + (["images on at least one side"] if not (v["images_plus"] or v["images_minus"]) else [])
		else:
			missing = [label.lower() for key, label, _p, _t in TRAIT_FIELDS if key in self.needed and not v[key]]
		if missing:
			self.hint.setText("still needed: " + ", ".join(missing))
			return
		self.accept()

	def data(self):
		if self.kind() == "images":
			v = {k: "" for k in self.fields}
			v["name"] = self.img_name.text().strip()
			v["seeds_plus"], v["seeds_minus"] = [], []
			for pole in ("plus", "minus"):
				v[f"images_{pole}"] = self.img_lists[pole].paths()
				v[f"poles_{pole}"] = self.img_lists[pole].poles()
			return v
		v = {k: e.text().strip() for k, e in self.fields.items()}
		for k in ("seeds_plus", "seeds_minus"):
			v[k] = list(dict.fromkeys(w.strip() for w in v[k].split(",") if w.strip()))
		for pole in ("plus", "minus"):
			v[f"images_{pole}"], v[f"poles_{pole}"] = [], []
		return v

	def values(self):
		v = self.data()
		return v["name"], v["seeds_plus"], v["seeds_minus"], v["plus"], v["minus"]


class ExportDialog(EvaDialog):
	def __init__(self, rows, arch, bakeable, name="", parent=None, lora=(True, "")):
		super().__init__(parent)
		self.arch = (arch or "").strip()
		self.setWindowTitle("Export")
		self.setStyleSheet(theme.STYLE)
		self.setMinimumWidth(560)
		lay = QVBoxLayout(self)
		lay.setSpacing(5)
		lay.addWidget(DialogTitle("EXPORT THE ACTIVE LEVERS"))
		lines = []
		for r in rows:
			eff = f"   → applied {r['applied']:+.2f}" if abs(r["applied"] - r["setting"]) > 1e-3 else ""
			lines.append(f"{r['name'].upper()[:18]:<18} {r['setting']:+.2f}{eff}" + ("   · trained vector" if r.get("optimized") else ""))
		summary = QLabel("\n".join(lines) or "no active lever")
		summary.setFont(theme.mono(8))
		summary.setStyleSheet(f"color: {theme.TEXT};")
		lay.addWidget(summary)
		lay.addWidget(_caption("NAME"))
		self.name = QLineEdit(name)
		self.name.setFont(theme.mono(9))
		self.name.setPlaceholderText("persona name (folder under exports)")
		self.name.setToolTip(f"the model architecture and size are always put before it: {self.arch}_<name> (unless the name already starts with them)")
		lay.addWidget(self.name)
		dose_row = QHBoxLayout()
		dose_row.addWidget(_caption("GLOBAL DOSE"))
		self.dose = guard_wheel(QDoubleSpinBox())
		self.dose.setDecimals(2)
		self.dose.setRange(0.0, 3.0)
		self.dose.setSingleStep(0.05)
		self.dose.setValue(1.0)
		self.dose.setFont(theme.mono(9))
		self.dose.setFixedWidth(80)
		self.dose.setToolTip("multiplies every exported lever at once\n1 = exactly what you hear in the DIALOGUE now\nwith a control vector it is the number after --control-vector-scaled in the .bat: edit it there any time\nwheel when focused: ±0.05")
		dose_row.addWidget(self.dose)
		self.dose_hint = _note("")
		dose_row.addWidget(self.dose_hint, 1)
		lay.addLayout(dose_row)
		self.dose.valueChanged.connect(self._dose_text)
		self._dose_text(1.0)
		self.fmt_cvec = QRadioButton("control vector: .cvec.gguf + a run_llama_server launcher (the model file is untouched)")
		self.fmt_bake = QRadioButton("baked model: a copy of the GGUF with the levers inside (no control vector needed)")
		self.fmt_lora = QRadioButton("LoRA (approximation): .lora.gguf for llama.cpp + .safetensors (Hugging Face, PEFT, ComfyUI…)")
		grp = QButtonGroup(self)
		for r in (self.fmt_cvec, self.fmt_bake, self.fmt_lora):
			tab_focus(r)
			r.setFont(theme.mono(8))
			grp.addButton(r)
			lay.addWidget(r)
		self.fmt_cvec.setChecked(True)
		self.lora_box = QWidget()
		lb = QVBoxLayout(self.lora_box)
		lb.setContentsMargins(22, 0, 0, 2)
		lb.setSpacing(3)
		row = QHBoxLayout()
		row.addWidget(_caption("LAYER NAMES"))
		self.style = guard_wheel(QComboBox())
		self.style.setFont(theme.mono(8))
		for key, (_prefix, label) in LORA_STYLES.items():
			self.style.addItem(label, key)
		self.style.setToolTip("how the layers are named in the .safetensors file: only the prefix before layers.N differs between tools\nthe .lora.gguf for llama.cpp is always written too")
		row.addWidget(self.style, 1)
		lb.addLayout(row)
		row = QHBoxLayout()
		row.addWidget(_caption("PREFIX"))
		self.prefix = QLineEdit(LORA_STYLES["hf"][0])
		self.prefix.setFont(theme.mono(8.5))
		self.prefix.setToolTip("text put before layers.N in every key: edit it if your tool names the layers differently (e.g. model.language_model.)")
		row.addWidget(self.prefix, 1)
		lb.addLayout(row)
		self.example = _note("")
		lb.addWidget(self.example)
		self.lora_note = _note("a lever adds a fixed vector to the model's activations; a LoRA changes weights, so it can only approximate it: rank 1 per layer on the MLP output, fitted in about a minute on sample prompts. How much of the steering it reproduces is measured on unseen prompts and written in pendulum.json (≈ 99% of the next-token gap on Qwen3.5-4B). ComfyUI: LoraLoader with strength_model 0, strength_clip = how strong")
		self.lora_note.setWordWrap(True)
		lb.addWidget(self.lora_note)
		lay.addWidget(self.lora_box)
		self.style.currentIndexChanged.connect(self._style)
		self.prefix.textChanged.connect(self._example)
		self.fmt_lora.toggled.connect(self.lora_box.setVisible)
		self.lora_box.setVisible(False)
		self.fmt_lora.setEnabled(bool(lora[0]))
		self.fmt_lora.setToolTip("rank 1 per layer on the MLP output weight (plus the LoRA of trained goals), fitted so that it adds the levers' vectors, then checked on unseen prompts" if lora[0] else lora[1])
		self._example()
		self.fmt_bake.setEnabled(bool(bakeable))
		self.fmt_bake.setToolTip("writes the levers as FFN output biases: exact, the model then runs anywhere llama.cpp runs\nthe copy is as large as the model" if bakeable else f"'{arch}' models have no FFN output bias in llama.cpp, so the levers cannot be baked exactly into them\n(exact for llama and mistral3 models): use the control vector")
		self.where = _note("")
		self.where.setWordWrap(True)
		lay.addWidget(self.where)
		self.name.textChanged.connect(self._where)
		self._where()
		btns = QHBoxLayout()
		btns.addStretch(1)
		cancel = button("CANCEL")
		ok = button("EXPORT", accent=True)
		ok.setDefault(True)
		cancel.clicked.connect(self.reject)
		ok.clicked.connect(self.accept)
		btns.addWidget(cancel)
		btns.addWidget(ok)
		lay.addLayout(btns)

	def final_name(self):
		name, tag = self.name.text().strip(), self.arch
		if not tag:
			return name
		low, base = name.lower(), tag.split("-")[0].lower()
		if low == tag.lower() or low.startswith(tag.lower() + "_"):
			return name
		if base != tag.lower() and (low == base or low.startswith(base + "_")):
			name = name[len(base):].lstrip("_")
		return f"{tag}_{name}" if name else tag

	def _where(self, _t=""):
		self.where.setText(f"written to {config.EXPORTS_DIR.replace(os.sep, '/')}/{self.final_name()}/ with a manifest (pendulum.json) of the levers, strengths and profiles, a run_llama_server launcher (.bat or .sh) and the LM Studio override (lmstudio_override.txt)")

	def _dose_text(self, v):
		self.dose_hint.setText("  as now" if abs(v - 1.0) < 1e-6 else (f"  {v:.0%} of the current strength" if v < 1 else f"  {v:.2f}× stronger than now: check coherence"))
		self.dose_hint.setStyleSheet(f"color: {theme.WARN if v > 1.0001 else theme.DIM};")

	def _style(self, _i=0):
		self.prefix.setText(LORA_STYLES.get(self.style.currentData(), LORA_STYLES["hf"])[0])
		self._example()

	def _example(self, _t=""):
		key = example_key(self.style.currentData() or "hf", self.prefix.text())
		self.example.setText(f"keys like  {key}")
		self.example.setToolTip(key)

	def values(self):
		kind = "lora" if self.fmt_lora.isChecked() and self.fmt_lora.isEnabled() else ("baked" if self.fmt_bake.isChecked() and self.fmt_bake.isEnabled() else "cvec")
		return self.final_name(), round(float(self.dose.value()), 3), kind, self.style.currentData() or "hf", self.prefix.text()


SPLIT_TIPS = {"layer": "layer: whole layers per GPU (default: best with two GPUs on PCIe)", "row": "row: every matrix split across the GPUs (tensor parallel: needs a fast link, often slower on PCIe)", "none": "none: the whole model on the main GPU"}
FLASH_TIPS = {"auto": "auto: llama.cpp decides", "on": "on: less memory for long contexts", "off": "off"}
KV_TIPS = {"f16": "f16: exact (default)", "q8_0": "q8_0: half the KV memory, nearly lossless (turns flash attention on)", "q4_0": "q4_0: a quarter of the KV memory, slightly lossy (turns flash attention on)"}


class LoadOptionsForm(QWidget):
	changed = Signal(object, bool)

	def __init__(self, parent=None):
		super().__init__(parent)
		grid = QGridLayout(self)
		grid.setContentsMargins(0, 0, 0, 0)
		grid.setHorizontalSpacing(8)
		grid.setVerticalSpacing(5)
		self.w = {}
		self.loading = False
		self.gpu_names = {0: "CUDA0"}
		for k, (key, label, items, tips, tip) in enumerate((
			("split_mode", "SPLIT", SPLIT_MODES, SPLIT_TIPS, "how the model is spread over the GPUs"),
			("main_gpu", "MAIN GPU", [0], self.gpu_names, "with split none: the GPU holding the whole model\notherwise: the GPU for the small shared tensors"),
			("flash_attn", "FLASH ATTN", FLASH, FLASH_TIPS, "flash attention in llama.cpp"),
			("kv_type", "KV CACHE", KV_TYPES, KV_TIPS, "precision of the context memory (keys and values)"),
		)):
			c = guard_wheel(QComboBox())
			c.setFont(theme.mono(8))
			for it in items:
				c.addItem(str(tips.get(it, it)), it)
			c.setToolTip(tip)
			c.currentIndexChanged.connect(lambda _i, c=c, tip=tip: c.setToolTip(f"{tip}\n\nnow: {c.currentText()}"))
			c.currentIndexChanged.connect(self._emit)
			l = _caption(label)
			l.setToolTip(tip)
			grid.addWidget(l, k // 2, (k % 2) * 2)
			grid.addWidget(c, k // 2, (k % 2) * 2 + 1)
			self.w[key] = c
		grid.setColumnStretch(1, 1)
		grid.setColumnStretch(3, 1)
		row = QHBoxLayout()
		self.own = tab_focus(QCheckBox("for this model only"))
		self.own.setFont(theme.mono(8))
		self.own.setToolTip("checked: these options belong to this model\nunchecked: they are the options of every model without its own (also in ⚙ ENGINE)")
		self.own.toggled.connect(self._emit)
		row.addWidget(self.own)
		row.addStretch(1)
		reset = button("DEFAULTS", "llama.cpp defaults: split by layer, CUDA0 main, flash auto, f16 KV")
		reset.clicked.connect(self._defaults)
		row.addWidget(reset)
		grid.addLayout(row, 2, 0, 1, 4)

	def set_gpus(self, gpus):
		names = {i: g.get("name") or f"CUDA{i}" for i, g in enumerate(gpus)} or {0: "CUDA0"}
		if names == self.gpu_names:
			return
		self.gpu_names = names
		c = self.w["main_gpu"]
		cur = c.currentData()
		self.loading = True
		c.clear()
		for i, n in names.items():
			c.addItem(n, i)
		c.setCurrentIndex(max(0, c.findData(cur)))
		self.loading = False

	def set_options(self, options, own):
		o = {**LOAD_DEFAULTS, **(options or {})}
		self.loading = True
		for k, c in self.w.items():
			c.setCurrentIndex(max(0, c.findData(o[k])))
			c.currentIndexChanged.emit(c.currentIndex())
		self.own.setChecked(bool(own))
		self.loading = False

	def values(self):
		opts = {k: c.currentData() for k, c in self.w.items()}
		opts["main_gpu"] = int(opts["main_gpu"] or 0)
		return opts, self.own.isChecked()

	def _defaults(self):
		self.loading = True
		for k, c in self.w.items():
			c.setCurrentIndex(max(0, c.findData(LOAD_DEFAULTS[k])))
		self.loading = False
		self._emit()

	def _emit(self, *_a):
		if not self.loading:
			self.changed.emit(*self.values())
