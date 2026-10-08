import importlib
import threading
import time

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSplashScreen

from ui import eva, theme

W, H = 440, 96
SECONDS = 5.0
HEAVY = ("numpy", "torch")


def picture():
	pm = QPixmap(W, H)
	pm.fill(QColor(theme.BG))
	p = QPainter(pm)
	p.setRenderHint(QPainter.RenderHint.Antialiasing)
	p.fillRect(QRect(0, 0, 6, H), QColor(theme.ACCENT))
	p.setPen(QPen(QColor(theme.ACCENT), 1.2))
	p.drawRect(QRect(0, 0, W - 1, H - 1))
	p.setFont(theme.mono(17, True))
	p.drawText(QRect(26, 14, W - 40, 36), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "PENDULUM")
	p.end()
	return pm


class Loading(QSplashScreen):
	def __init__(self):
		super().__init__(picture())
		self.t0 = time.perf_counter()
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.repaint)
		self.timer.start(30)

	def drawContents(self, p):
		frac = min(0.98, (time.perf_counter() - self.t0) / SECONDS)
		r = QRectF(26, 62, W - 52, 14)
		p.setPen(QPen(QColor(theme.ACCENT), 1.0))
		p.setBrush(Qt.BrushStyle.NoBrush)
		p.drawRect(r)
		eva.seg_bar(p, r.adjusted(3, 3, -3, -3), frac, theme.ACCENT)


def loading(app):
	s = Loading()
	scr = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
	s.move(scr.availableGeometry().center() - QPoint(W // 2, H // 2))
	s.show()
	app.processEvents()
	return s


def preload(app):
	def work():
		for name in HEAVY:
			try:
				importlib.import_module(name)
			except Exception:
				return

	t = threading.Thread(target=work, daemon=True)
	t.start()
	while t.is_alive():
		app.processEvents()
		t.join(0.02)
