from PySide6.QtCore import QEvent, QObject, QPoint, QTimer

SNAP = 3


class Snapper(QObject):
	def __init__(self, win, band, rows):
		super().__init__(win)
		self.win = win
		self.band = band
		self.rows = rows
		self.links = {}
		self.busy = False
		self.timer = QTimer(self)
		self.timer.setSingleShot(True)
		self.timer.timeout.connect(self.apply)
		for name, sp in rows.items():
			sp.splitterMoved.connect(lambda _pos, i, n=name: self.moved(n, i))
			sp.installEventFilter(self)
		band.installEventFilter(self)

	def eventFilter(self, obj, e):
		if e.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Hide):
			self.timer.start(0)
		return False

	def x0(self, w):
		return w.mapTo(self.win, QPoint(0, 0)).x()

	def handles(self, name):
		sp = self.rows[name]
		if not sp.isVisible():
			return {}
		x = self.x0(sp)
		return {(name, i): x + sp.handle(i).x() for i in range(1, sp.count()) if sp.handle(i).isVisible()}

	def band_edges(self, every=False):
		b = self.band
		if not b.isVisible():
			return {}
		x = self.x0(b)
		info, sine, graph = b.split(b.width(), b.height(), "both")
		out = {}
		if info is not None:
			out[("band", "info")] = x + round(info.right())
		if sine is not None and graph is not None and (every or b.layout_mode == "both"):
			out[("band", "mid")] = x + round(sine.right())
		return out

	def targets(self, name, every=False):
		out = self.band_edges(every)
		for other in self.rows:
			if other != name:
				out.update(self.handles(other))
		return out

	def moved(self, name, i):
		if self.busy:
			return
		key = (name, i)
		x = self.handles(name).get(key)
		if x is None:
			return
		best = min(self.targets(name).items(), key=lambda kv: abs(kv[1] - x), default=None)
		if best is None or abs(best[1] - x) > SNAP:
			self.links.pop(key, None)
			return
		self.links[key] = best[0]
		self.place(name, i, best[1])

	def place(self, name, i, x):
		sp = self.rows[name]
		shift = x - (self.x0(sp) + sp.handle(i).x())
		if not shift:
			return
		a = i - 1
		while a >= 0 and sp.widget(a).isHidden():
			a -= 1
		sizes = sp.sizes()
		if a < 0 or sizes[a] + shift < sp.widget(a).minimumWidth() or sizes[i] - shift < sp.widget(i).minimumWidth():
			return
		sizes[a] += shift
		sizes[i] -= shift
		self.busy = True
		try:
			sp.setSizes(sizes)
		finally:
			self.busy = False

	def apply(self):
		for _round in range(2):
			for (name, i), target in list(self.links.items()):
				x = self.handles(name).get((name, i))
				t = self.targets(name, True).get(target)
				if x is not None and t is not None and t != x:
					self.place(name, i, t)

	def dump(self):
		return [[n, i, tn, tk] for (n, i), (tn, tk) in self.links.items()]

	def load(self, items):
		for item in items or []:
			if isinstance(item, (list, tuple)) and len(item) == 4 and item[0] in self.rows:
				self.links[(item[0], int(item[1]))] = (item[2], item[3] if item[2] == "band" else int(item[3]))
		self.timer.start(0)
