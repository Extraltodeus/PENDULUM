from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget

EDGE = 5
SHAPES = {Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor, Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor, Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor, Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor, Qt.Edge.LeftEdge | Qt.Edge.TopEdge: Qt.CursorShape.SizeFDiagCursor, Qt.Edge.RightEdge | Qt.Edge.BottomEdge: Qt.CursorShape.SizeFDiagCursor, Qt.Edge.RightEdge | Qt.Edge.TopEdge: Qt.CursorShape.SizeBDiagCursor, Qt.Edge.LeftEdge | Qt.Edge.BottomEdge: Qt.CursorShape.SizeBDiagCursor}


class EdgeFilter(QObject):
	def __init__(self, win):
		super().__init__(win)
		self.win = win
		self.shape = None

	def _set(self, shape):
		if shape == self.shape:
			return
		if self.shape is not None:
			QApplication.restoreOverrideCursor()
		if shape is not None:
			QApplication.setOverrideCursor(shape)
		self.shape = shape

	def _edges(self, obj, ev):
		if not isinstance(obj, QWidget) or obj.window() is not self.win or self.win.maxed():
			return Qt.Edge(0)
		pos = self.win.mapFromGlobal(ev.globalPosition().toPoint())
		r = self.win.rect()
		if not r.contains(pos):
			return Qt.Edge(0)
		edges = Qt.Edge(0)
		if pos.x() < EDGE:
			edges |= Qt.Edge.LeftEdge
		if pos.x() >= r.width() - EDGE:
			edges |= Qt.Edge.RightEdge
		if pos.y() < EDGE:
			edges |= Qt.Edge.TopEdge
		if pos.y() >= r.height() - EDGE:
			edges |= Qt.Edge.BottomEdge
		return edges

	def eventFilter(self, obj, ev):
		t = ev.type()
		if t == QEvent.Type.MouseMove:
			if ev.buttons() == Qt.MouseButton.NoButton:
				self._set(SHAPES.get(self._edges(obj, ev)))
		elif t == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
			edges = self._edges(obj, ev)
			if edges and self.win.windowHandle():
				self._set(None)
				self.win.windowHandle().startSystemResize(edges)
				return True
		elif t == QEvent.Type.Leave and obj is self.win:
			self._set(None)
		return False


class Frameless:
	normal_geom = None

	def maxed(self):
		return self.normal_geom is not None

	def toggle_max(self):
		if self.normal_geom is not None:
			self.setGeometry(self.normal_geom)
			self.normal_geom = None
			return
		screen = QGuiApplication.screenAt(self.geometry().center()) or self.screen()
		self.normal_geom = self.geometry()
		self.setGeometry(screen.availableGeometry())

	def restore_for_drag(self, global_pos):
		if self.normal_geom is None:
			return
		g = QRect(self.normal_geom)
		frac = (global_pos.x() - self.x()) / max(self.width(), 1)
		g.moveTopLeft(QPoint(int(global_pos.x() - frac * g.width()), self.y()))
		self.normal_geom = None
		self.setGeometry(g)

	def drag_from(self, global_pos):
		if self.maxed():
			self.restore_for_drag(global_pos)
		if self.windowHandle():
			self.windowHandle().startSystemMove()
