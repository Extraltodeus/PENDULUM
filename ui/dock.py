from PySide6.QtCore import QEvent, QObject, QRect, Qt, QTimer
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QApplication, QMenu, QSplitter, QWidget

from ui import theme
from ui.widgets import menu_title

TITLE = 25
EDGE_V = 0.3
EDGE_H = 0.2


class StayMenu(QMenu):
	def mouseReleaseEvent(self, e):
		a = self.actionAt(e.position().toPoint())
		if a is not None and a.isCheckable() and a.isEnabled():
			a.trigger()
			return
		super().mouseReleaseEvent(e)


class Marker(QWidget):
	def __init__(self, parent):
		super().__init__(parent)
		self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
		self.hide()

	def paintEvent(self, e):
		p = QPainter(self)
		p.fillRect(self.rect(), theme.qc(theme.ACCENT, 46))
		p.setPen(QPen(theme.qc(theme.ACCENT, 230), 2))
		p.drawRect(self.rect().adjusted(1, 1, -1, -1))
		p.end()


class Dock(QObject):
	def __init__(self, win, splitter, fixed, panels, hideable, on_change=None):
		super().__init__(win)
		self.win = win
		self.sp = splitter
		self.fixed = fixed
		self.panels = {key: p for key, p, _s, _m in panels}
		self.stretch = {key: s for key, _p, s, _m in panels}
		self.minw = {key: m for key, _p, _s, m in panels}
		self.slots = [[key] for key, *_rest in panels]
		self.hideable = hideable
		self.on_change = on_change
		self.press = None
		self.target = None
		self.stacks = {}
		self.stack_h = {}
		self.marker = Marker(win)
		for p in self.panels.values():
			p.installEventFilter(self)
		self.arrange()

	@property
	def order(self):
		return [k for s in self.slots for k in s]

	def key_of(self, obj):
		return next((k for k, p in self.panels.items() if p is obj), None)

	def slot_of(self, key):
		return next(i for i, s in enumerate(self.slots) if key in s)

	def holder(self, slot):
		return self.panels[slot[0]] if len(slot) == 1 else self.stacks[tuple(slot)]

	def at(self, gpos):
		for k, p in self.panels.items():
			if not p.isVisible():
				continue
			pos = p.mapFromGlobal(gpos)
			if not p.rect().contains(pos):
				continue
			fx, fy = pos.x() / max(1, p.width()), pos.y() / max(1, p.height())
			if fy < EDGE_V:
				return k, "top"
			if fy > 1 - EDGE_V:
				return k, "bottom"
			if fx < EDGE_H:
				return k, "left"
			if fx > 1 - EDGE_H:
				return k, "right"
			return k, "center"
		return None

	def allowed(self, a, hit):
		if hit is None or hit[0] == a:
			return None
		b, zone = hit
		if zone in ("top", "bottom"):
			slot = self.slots[self.slot_of(b)]
			if len(slot) == 2 and a not in slot:
				return b, "center"
		return hit

	def eventFilter(self, obj, e):
		key = self.key_of(obj)
		if key is None:
			return False
		t = e.type()
		if t == QEvent.Type.ContextMenu:
			self.menu(e.globalPos(), key)
			return True
		if t == QEvent.Type.MouseButtonPress and e.button() == Qt.MouseButton.LeftButton and e.position().y() < TITLE:
			self.press = {"key": key, "at": e.globalPosition().toPoint(), "drag": False}
			return True
		if self.press is None:
			return False
		if t == QEvent.Type.MouseMove:
			g = e.globalPosition().toPoint()
			if not self.press["drag"] and (g - self.press["at"]).manhattanLength() >= QApplication.startDragDistance():
				self.press["drag"] = True
				QApplication.setOverrideCursor(Qt.CursorShape.ClosedHandCursor)
			if self.press["drag"]:
				self.mark(self.allowed(self.press["key"], self.at(g)))
			return True
		if t == QEvent.Type.MouseButtonRelease and e.button() == Qt.MouseButton.LeftButton:
			press, self.press = self.press, None
			if press["drag"]:
				QApplication.restoreOverrideCursor()
				target = self.target
				self.mark(None)
				if target is not None:
					self.drop(press["key"], *target)
			return True
		return False

	def mark(self, hit):
		self.target = hit
		if hit is None:
			self.marker.hide()
			return
		key, zone = hit
		p = self.panels[key]
		r = QRect(p.mapTo(self.win, p.rect().topLeft()), p.size())
		if zone == "top":
			r.setHeight(r.height() // 2)
		elif zone == "bottom":
			r.setTop(r.top() + r.height() // 2)
		elif zone == "left":
			r.setWidth(max(24, r.width() // 4))
		elif zone == "right":
			r.setLeft(r.right() - max(24, r.width() // 4))
		self.marker.setGeometry(r)
		self.marker.raise_()
		self.marker.show()

	def widths(self):
		sizes = self.sp.sizes()
		out = {id(self.sp.widget(i)): sizes[i] for i in range(min(self.fixed, len(sizes)))}
		for n, s in enumerate(self.slots):
			if self.fixed + n < len(sizes):
				w = sizes[self.fixed + n]
				h = self.panels[s[0]] if len(s) == 1 else self.stacks.get(tuple(s))
				if h is not None:
					out[id(h)] = w
				for k in s:
					out[k] = w
		return out

	def drop(self, a, b, zone):
		before = self.widths()
		ia, ib = self.slot_of(a), self.slot_of(b)
		if zone == "center":
			sa, sb = self.slots[ia], self.slots[ib]
			sa[sa.index(a)], sb[sb.index(b)] = b, a
			self.arrange(before)
			return
		self.slots[ia].remove(a)
		if zone in ("top", "bottom"):
			slot = self.slots[ib]
			slot.insert(slot.index(b) + (zone == "bottom"), a)
		else:
			self.slots.insert(ib + (zone == "right"), [a])
		self.slots = [s for s in self.slots if s]
		self.panels[a].show()
		self.arrange(before)

	def unstack(self, key):
		i = self.slot_of(key)
		if len(self.slots[i]) < 2:
			return
		before = self.widths()
		self.slots[i].remove(key)
		self.slots.insert(i + 1, [key])
		self.arrange(before)

	def stack_for(self, slot):
		k = tuple(slot)
		c = self.stacks.get(k)
		if c is None:
			c = QSplitter(Qt.Orientation.Vertical)
			c.setChildrenCollapsible(False)
			c.setHandleWidth(6)
			c.setStyleSheet(f"QSplitter::handle {{ background: transparent; }} QSplitter::handle:hover {{ background: {theme.LINE2}; }}")
			c.splitterMoved.connect(lambda *_a, k=k: self.keep_heights(k))
			self.stacks[k] = c
		for n, key in enumerate(slot):
			c.insertWidget(n, self.panels[key])
		h = self.stack_h.get("+".join(slot))
		if h and len(h) == 2 and min(h) > 40:
			c.setSizes(h)
		else:
			c.setSizes([1000, 1000])
		return c

	def keep_heights(self, k):
		c = self.stacks.get(k)
		if c is not None:
			self.stack_h["+".join(k)] = c.sizes()
			self.changed()

	def arrange(self, before=None):
		before = before if before is not None else self.widths()
		old = dict(self.stacks)
		holders = []
		for slot in self.slots:
			if len(slot) == 1:
				holders.append(self.panels[slot[0]])
			else:
				holders.append(self.stack_for(slot))
		for n, h in enumerate(holders):
			self.sp.insertWidget(self.fixed + n, h)
		for k, c in old.items():
			if list(k) not in self.slots:
				self.stacks.pop(k, None)
				c.hide()
				c.setParent(None)
				c.deleteLater()
		for n, (slot, h) in enumerate(zip(self.slots, holders)):
			self.sp.setStretchFactor(self.fixed + n, max(self.stretch[k] for k in slot))
			h.setMinimumWidth(max(self.minw[k] for k in slot))
			if len(slot) == 2:
				h.setVisible(any(not self.panels[k].isHidden() for k in slot))
		sizes = self.sp.sizes()
		total = sum(sizes)
		if total > 0 and before:
			for i in range(self.fixed):
				if id(self.sp.widget(i)) in before:
					sizes[i] = before[id(self.sp.widget(i))]
			fresh = []
			for n, (slot, h) in enumerate(zip(self.slots, holders)):
				i = self.fixed + n
				if h.isHidden():
					continue
				if id(h) not in before:
					fresh.append(i)
				w = before.get(id(h)) or max([before.get(k) or 0 for k in slot] + [0])
				sizes[i] = max(h.minimumWidth(), w or sizes[i])
			diff = total - sum(sizes)
			if fresh and diff:
				for i in fresh:
					sizes[i] = max(self.sp.widget(i).minimumWidth(), sizes[i] + diff // len(fresh))
			self.sp.setSizes(sizes)
		self.changed()

	def menu(self, gpos, key=None):
		m = StayMenu(self.win)
		m.setFont(theme.ui(8))
		menu_title(m, "PANELS")
		for k, label in self.hideable:
			a = m.addAction(label)
			a.setCheckable(True)
			a.setChecked(not self.panels[k].isHidden())
			a.toggled.connect(lambda on, k=k: self.set_shown(k, on))
		if key is not None and len(self.slots[self.slot_of(key)]) == 2:
			m.addSeparator()
			name = dict(self.hideable).get(key, key.upper())
			u = m.addAction(f"UNSTACK {name}")
			u.setToolTip("put it back in its own place, right of the other one\n(dragging its title onto the left or right edge of a panel does it too)")
			u.triggered.connect(lambda _c=False, k=key: (m.close(), self.unstack(k)))
		m.setToolTipsVisible(True)
		m.exec(gpos)
		m.deleteLater()

	def set_shown(self, key, on):
		p = self.panels[key]
		if on == (not p.isHidden()):
			return
		slot = self.slots[self.slot_of(key)]
		if len(slot) == 2:
			c = self.stacks[tuple(slot)]
			was = not c.isHidden()
			p.setVisible(on)
			now = any(not self.panels[k].isHidden() for k in slot)
			if on and min(c.sizes()) < 40:
				c.setSizes([1000, 1000])
			if now == was:
				self.changed()
				return
			p.setVisible(True)
			self.place(c, now)
			p.setVisible(on)
			self.changed()
			return
		self.place(p, on)
		self.changed()

	def place(self, w, on):
		sizes = self.sp.sizes()
		i = self.sp.indexOf(w)
		if on:
			give = max(w.minimumWidth(), w.property("dock_w") or w.minimumWidth())
			w.show()
			n = max(self.shown(i), key=lambda k: sizes[k], default=0)
			sizes[n] = max(self.sp.widget(n).minimumWidth(), sizes[n] - give)
			sizes[i] = give
		else:
			w.setProperty("dock_w", w.width())
			w.hide()
			near = sorted(self.shown(i), key=lambda k: (abs(k - i), k < i))
			sizes[near[0] if near else 0] += sizes[i]
			sizes[i] = 0
		self.sp.setSizes(sizes)

	def shown(self, i):
		return [k for k in range(self.fixed, self.sp.count()) if k != i and not self.sp.widget(k).isHidden()]

	def changed(self):
		if self.on_change is not None:
			QTimer.singleShot(0, self.on_change)

	def dump(self):
		for k, c in self.stacks.items():
			if c.isVisible() and min(c.sizes()) > 40:
				self.stack_h["+".join(k)] = c.sizes()
		return {"slots": [list(s) for s in self.slots], "order": self.order, "hidden": [k for k, _l in self.hideable if self.panels[k].isHidden()], "stacks": {k: v for k, v in self.stack_h.items() if tuple(k.split("+")) in self.stacks}}

	def load(self, d):
		if not isinstance(d, dict):
			return
		slots = d.get("slots")
		if not isinstance(slots, list) and isinstance(d.get("order"), list):
			slots = [[k] for k in d["order"]]
		if isinstance(slots, list):
			seen = set()
			clean = []
			for s in slots:
				s = [k for k in (s if isinstance(s, list) else []) if k in self.panels and k not in seen][:2]
				seen.update(s)
				if s:
					clean.append(s)
			clean += [[k] for k in self.order if k not in seen]
			self.stack_h = {k: v for k, v in (d.get("stacks") or {}).items() if isinstance(v, list)}
			before = self.widths()
			self.slots = clean
			self.arrange(before)
		for key, _label in self.hideable:
			if key in (d.get("hidden") or []):
				self.set_shown(key, False)
		self.changed()
