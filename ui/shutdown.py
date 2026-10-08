import math
import os
import subprocess
import sys
import time

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ui import eva, theme
from ui.sync import lcd_for
from ui.widgets import button

CARDS = ("THE END OF THE LIST", "ALL OPERATIONS COMPLETE", "INTERNAL POWER ONLY", "THE PILOT MAY REST", "UMBILICAL CABLE DISCONNECTED", "おつかれさまでした", "DO YOU WISH TO CONTINUE?")
MAGI = ("MELCHIOR·1", "BALTHASAR·2", "CASPER·3")
DELAYS = (1, 5, 10, 15, 30, 60)
RED = "#ff2a2a"


def serif(px):
	f = QFont()
	f.setFamilies(["Yu Mincho", "MS Mincho", "Times New Roman", "serif"])
	f.setPixelSize(int(px))
	f.setWeight(QFont.Weight.Bold)
	return f


def gothic(px, weight=QFont.Weight.Bold):
	f = QFont()
	f.setFamilies(["Yu Gothic UI", "Yu Gothic", "MS Gothic", "sans-serif"])
	f.setPixelSize(int(px))
	f.setWeight(weight)
	return f


def clock(left):
	left = max(0.0, left)
	m, s = divmod(int(left), 60)
	return f"{m:02d}:{s:02d}:{int((left - int(left)) * 100):02d}"


class Countdown(QWidget):
	aborted = Signal()
	fired = Signal()

	def __init__(self, seconds, parent=None, test=False, force=False):
		super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
		self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
		self.setWindowTitle("Pendulum · shutdown")
		self.total = max(1.0, float(seconds))
		self.t0 = time.monotonic()
		self.done = False
		self.test = test
		self.force = force
		self.setFixedSize(880, 470)
		self.setStyleSheet(theme.STYLE)
		lay = QVBoxLayout(self)
		lay.setContentsMargins(26, 0, 26, 22)
		lay.addStretch(1)
		row = QHBoxLayout()
		row.setSpacing(10)
		row.addStretch(1)
		self.b_now = button("SKIP TO 0" if test else "SHUT DOWN NOW", "end the test now (nothing happens)" if test else "skip the rest of the countdown")
		self.b_abort = button("CLOSE  (Esc)" if test else "ABORT  (Esc)", "close the test" if test else "cancel the shutdown: the computer stays on", accent=True)
		self.b_now.clicked.connect(self.fire)
		self.b_abort.clicked.connect(self.abort)
		for b in (self.b_now, self.b_abort):
			b.setMinimumHeight(30)
			row.addWidget(b)
		lay.addLayout(row)
		self.timer = QTimer(self)
		self.timer.timeout.connect(self._tick)
		self.timer.start(33)

	def left(self):
		return self.total - (time.monotonic() - self.t0)

	def _tick(self):
		if self.left() <= 0:
			self.fire()
			return
		self.update()

	def abort(self):
		if self.done:
			return
		self.done = True
		self.timer.stop()
		self.aborted.emit()
		self.close()

	def fire(self):
		if self.done:
			return
		self.done = True
		self.timer.stop()
		self.fired.emit()
		self.close()

	def keyPressEvent(self, e):
		if e.key() == Qt.Key.Key_Escape:
			self.abort()
			return
		super().keyPressEvent(e)

	def place(self, screen):
		g = screen.availableGeometry()
		self.move(g.center().x() - self.width() // 2, g.center().y() - self.height() // 2)

	def paintEvent(self, e):
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		self.paint(p, time.monotonic() - self.t0, self.left())
		p.end()

	def paint(self, p, t, left):
		W, H = self.width(), self.height()
		frac = max(0.0, left / self.total)
		last = left < 10.0
		beat = 0.5 + 0.5 * math.cos(t * math.pi * 2)
		p.fillRect(QRectF(0, 0, W, H), QColor(4, 0, 0))
		if last:
			p.fillRect(QRectF(0, 0, W, H), eva.col(RED, int(40 * (1 - (left % 1.0)))))
		for y in range(0, H, 3):
			p.fillRect(QRectF(0, y, W, 1), QColor(0, 0, 0, 90))
		band = QRectF(0, 0, W, 30)
		p.save()
		p.setClipRect(band)
		shift = (t * 36) % 24
		eva.hazard(p, QRectF(-24 + shift, 0, W + 48, 30), theme.WARN, 24, 235, "#000000", 255)
		p.restore()
		ef = eva.font(17, 800)
		if int(t * 2) % 2 == 0 or last:
			tw = eva.text_w(ef, "EMERGENCY")
			eva.tag(p, W / 2 - tw / 2 - 10, 5, "EMERGENCY", ef, RED, h=20, pad=10, fill=RED, text_c="#000000")
		box = QRectF(10, 40, W - 20, H - 50)
		eva.box(p, box, RED, int(150 + 105 * beat) if last else 170, 1.6)
		eva.brackets(p, box.adjusted(-5, -5, 5, 5), theme.WARN, 230, 14, 2.0)
		sf, tf = eva.font(12, 600), eva.font(14, 700)
		eva.tag(p, 26, 54, "UMBILICAL CABLE DISCONNECTED", tf, RED, h=20, pad=7)
		p.setFont(sf)
		p.setPen(eva.col(theme.MUTED))
		head = "TEST · NOTHING HAPPENS AT 0" if self.test else "PENDULUM · TO DO LIST FINISHED"
		p.setPen(eva.col(theme.WARN if self.test else theme.MUTED))
		p.drawText(QPointF(W - 26 - eva.text_w(sf, head), 68), head)
		jp = gothic(30)
		p.setFont(jp)
		p.setPen(eva.col("#ffffff"))
		p.drawText(QPointF(30, 118), "内部電源")
		p.setFont(gothic(17, QFont.Weight.DemiBold))
		p.setPen(eva.col(theme.WARN))
		p.drawText(QPointF(30, 142), "INTERNAL POWER")
		p.setFont(gothic(24))
		p.setPen(eva.col("#ffffff"))
		p.drawText(QPointF(30, 186), "活動限界まで")
		p.setFont(gothic(14, QFont.Weight.DemiBold))
		p.setPen(eva.col(theme.WARN))
		p.drawText(QPointF(30, 206), "ACTIVITY LIMIT · あと")
		dpr = self.devicePixelRatioF()
		lcd = lcd_for(96, RED, dpr)
		text = clock(left)
		x0 = 220
		lcd.draw(p, x0, 98, text)
		bar = QRectF(x0 + 4, 212, lcd.width(text) - 10, 12)
		eva.seg_bar(p, bar, frac, RED if last else theme.WARN, 6.0, 2.0)
		self.paint_magi(p, QRectF(30, 240, W - 60, 58), t, frac, left)
		self.paint_card(p, QRectF(30, 310, W - 60, 68), t, left)
		note = eva.font(11, 600, False)
		p.setFont(note)
		p.setPen(eva.col(theme.MUTED))
		if self.test:
			p.drawText(QPointF(30, H - 32), "test run: the computer stays on")
		elif self.force:
			p.setPen(eva.col(RED))
			p.drawText(QPointF(30, H - 32), "FORCED: at 0 the other apps close without saving and the computer shuts down")
		else:
			p.drawText(QPointF(30, H - 32), "at 0 the computer shuts down · an app with unsaved work can still hold it")

	def paint_magi(self, p, r, t, frac, left):
		w = (r.width() - 20) / 3
		f, jf = eva.font(13, 700), gothic(15)
		for k, name in enumerate(MAGI):
			cell = QRectF(r.left() + k * (w + 10), r.top(), w, r.height())
			ok = frac <= 1 - (k + 1) / 3.2 or left < 10
			c = RED if ok else theme.WARN
			cut = 10
			poly = QPolygonF([QPointF(cell.left() + cut, cell.top()), QPointF(cell.right(), cell.top()), QPointF(cell.right(), cell.bottom() - cut), QPointF(cell.right() - cut, cell.bottom()), QPointF(cell.left(), cell.bottom()), QPointF(cell.left(), cell.top() + cut)])
			p.setPen(QPen(eva.col(c, 220), 1.4))
			p.setBrush(eva.col(c, 60 if ok else 18))
			p.drawPolygon(poly)
			p.setFont(f)
			p.setPen(eva.col(theme.MUTED if not ok else "#ffffff"))
			p.drawText(QPointF(cell.left() + 14, cell.top() + 20), name)
			p.setFont(jf)
			if ok:
				p.setPen(eva.col(RED))
				p.drawText(QPointF(cell.left() + 14, cell.bottom() - 12), "承認  SHUTDOWN APPROVED")
			elif int(t * 3 + k) % 3:
				p.setPen(eva.col(theme.WARN))
				p.drawText(QPointF(cell.left() + 14, cell.bottom() - 12), "審議中  DELIBERATING")

	def paint_card(self, p, r, t, left):
		k = int(t / 4.0) % len(CARDS)
		into = t % 4.0
		end = left < 10
		fade = 0.35 + 0.65 * (left % 1.0) if end else max(0.0, min(1.0, (into - 0.15) / 0.45, (4.0 - into) / 0.35))
		p.setFont(serif(40 if end else 34))
		p.setPen(eva.col(RED if end else "#f4efe6", int(255 * fade)))
		p.drawText(r, Qt.AlignmentFlag.AlignCenter, "SHUTDOWN IMMINENT" if end else CARDS[k])


class Shutdown(QObject):
	changed = Signal()

	def __init__(self, win, idle, minutes=15, log=None, sfx=None, force=False):
		super().__init__(win)
		self.win = win
		self.idle = idle
		self.minutes = minutes if minutes in DELAYS else 15
		self.force = bool(force)
		self.log = log or (lambda text, level="info": None)
		self.sfx = sfx or (lambda name: None)
		self.armed = False
		self.had_work = False
		self.overlay = None
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.check)
		self.timer.start(1000)

	def set_armed(self, on):
		self.armed = bool(on)
		self.had_work = self.armed and not self.idle()
		if not self.armed:
			self.close_overlay()
		self.log((f"shutdown armed: when the TO DO list is finished, a {self.minutes} min countdown starts (ABORT or Esc cancels it)" + ("" if self.had_work else " · it waits for the list to run")) if self.armed else "shutdown disarmed", "warn" if self.armed else "info")
		self.changed.emit()

	def set_minutes(self, m):
		self.minutes = int(m)
		self.changed.emit()

	def set_force(self, on):
		self.force = bool(on)
		self.log("shutdown FORCED: at the end of the countdown the other apps are closed without saving" if self.force else "shutdown not forced: an app with unsaved work can hold it", "warn" if self.force else "info")
		self.changed.emit()

	def note_work(self):
		if self.armed:
			self.had_work = True
			if self.overlay is not None and not self.overlay.test:
				self.close_overlay()
				self.log("shutdown countdown stopped: the TO DO list runs again (it starts again when the list is finished)", "warn")

	def check(self):
		if not self.armed:
			return
		if not self.idle():
			self.had_work = True
			if self.overlay is not None and not self.overlay.test:
				self.close_overlay()
				self.log("shutdown countdown stopped: the TO DO list runs again (it starts again when the list is finished)", "warn")
			return
		if self.had_work and self.overlay is None:
			self.start()

	def test(self, seconds=20):
		if self.overlay is not None:
			return
		self.overlay = Countdown(seconds, test=True, force=self.force)
		self.overlay.aborted.connect(self.test_end)
		self.overlay.fired.connect(self.test_end)
		self.show_overlay()
		self.log(f"countdown test ({seconds} s): nothing happens at 0")

	def test_end(self):
		self.overlay = None
		self.log("countdown test over: the computer stays on")

	def show_overlay(self):
		self.overlay.place(self.win.screen())
		self.overlay.show()
		self.overlay.raise_()
		self.overlay.activateWindow()
		self.overlay.setFocus()
		self.sfx("warning")

	def start(self, seconds=None):
		self.had_work = False
		self.overlay = Countdown(seconds if seconds is not None else self.minutes * 60, force=self.force)
		self.overlay.aborted.connect(self.aborted)
		self.overlay.fired.connect(self.fired)
		self.show_overlay()
		self.log(f"TO DO list finished: the computer shuts down in {self.minutes} min (ABORT or Esc in the countdown cancels it)", "warn")

	def close_overlay(self):
		o, self.overlay = self.overlay, None
		if o is not None and not o.done:
			o.done = True
			o.close()

	def aborted(self):
		self.overlay = None
		self.armed = False
		self.sfx("abort")
		self.log("shutdown cancelled: the computer stays on", "ok")
		self.changed.emit()

	def fired(self):
		self.overlay = None
		self.armed = False
		self.changed.emit()
		self.power_off()

	def power_off(self):
		if os.environ.get("PENDULUM_NO_SHUTDOWN"):
			self.log("shutdown (dry run: PENDULUM_NO_SHUTDOWN is set, the computer stays on)", "warn")
			return
		try:
			if sys.platform.startswith("win"):
				subprocess.Popen(["shutdown", "/s"] + (["/f"] if self.force else []) + ["/t", "0"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
			else:
				subprocess.Popen(["shutdown", "-h", "now"])
		except OSError as exc:
			self.log(f"shutdown failed: {exc}", "error")
			return
		self.log("shutting down" + (" (forced: the other apps are closed without saving)" if self.force else ""), "warn")
		QTimer.singleShot(1500, self.win.close)
