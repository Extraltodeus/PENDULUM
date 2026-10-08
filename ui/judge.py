import math
import os
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QAbstractSpinBox, QBoxLayout, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QRadioButton, QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout, QWidget

import config
from persona.coherence import EXTRA_DEFAULTS, LEGACY_MODES, threshold
from ui import eva, theme
from ui.widgets import SegSwitch, StepSlider, button, guard_wheel, tab_focus, toggle

MODES = (("uniform", "UNIFORM", "every lever under the guard is scaled down together until the replies read as coherent", "UNIF"), ("individual", "INDIVIDUAL", "only the levers changed since the last check are tried: the others keep the factor they were judged at\namong the changed ones, the one that hurts coherence the most is halved first (HALVING ROUNDS), then they are scaled together\nJUDGE NOW checks them all again", "INDIV"), ("quick", "QUICK", "a few replies per changed lever: validated at once when they read as coherent,\notherwise the full check runs (QUICK FALLBACK in CONFIG)"), ("read", "READ", "nothing is written first: the model reads its own normal replies with the levers on, and how much less likely they become decides\nsmall drop → validated at once · huge drop → scaled down by reading until it is not · in between → the judge decides (READ FALLBACK in CONFIG)\nalmost free; thresholds in CONFIG"))
STATE_TEXT = {"ready": "READY", "loading": "LOADING", "rejected": "REJECTED", "no_vram": "NO VRAM", "error": "ERROR", "off": "NOT LOADED"}


def fnt(px, weight=600):
	return eva.font(px, weight) if eva.on() else theme.mono(max(7.0, px * 0.62), weight >= 700)


class JudgeView(QWidget):
	def __init__(self, names=None, parent=None):
		super().__init__(parent)
		self.names = names or (lambda k: k)
		self.state = {"state": "off"}
		self.configured = ""
		self.mode = "uniform"
		self.auto = True
		self.guard_on = True
		self.limit = 0.6
		self.seq = None
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.update)
		self.setMinimumHeight(110)

	def set_info(self, state=None, configured=None, mode=None, auto=None, guard_on=None, limit=None):
		for k, v in (("state", state), ("configured", configured), ("mode", mode), ("auto", auto), ("guard_on", guard_on), ("limit", limit)):
			if v is not None:
				setattr(self, k, v)
		self.update()

	def label(self, ph, n):
		m = ph.get("mode")
		if m == "calib":
			return f"{ph.get('step', '?')}/{ph.get('steps', '?')}", f"×{ph.get('scale', 1):.2f}"
		if m == "quick":
			return "QUICK", f"{n} repl{'ies' if n != 1 else 'y'}"
		if m == "read":
			return ("READ" if not ph.get("step") else f"R×{ph.get('scale', 1):.2f}"), "no writing"
		if m == "blame":
			if not ph.get("round"):
				return "AS SET", ""
			return f"R{ph['round']}", f"{self.names(ph.get('lever', ''))[:6]} ½?"
		if not ph.get("step"):
			return "AS SET", f"×{ph.get('scale', 1):.2f}" if ph.get("scale", 1) < 0.999 else ""
		return ("LAST" if ph.get("last") else f"S{ph['step']}"), f"×{ph.get('scale', 1):.2f}"

	def live(self, d):
		k = d.get("kind")
		ph = d.get("phase") or {}
		if k == "start":
			kind = "calib" if ph.get("mode") == "calib" else "guard"
			if self.seq is None or self.seq.get("closed") or self.seq["kind"] != kind:
				self.seq = {"kind": kind, "cells": [], "verdict": None, "closed": False, "t": time.time(), "target": ph.get("target")}
			n = int(d.get("n") or 1)
			top, sub = self.label(ph, n)
			self.seq["cells"].append({"top": top, "sub": sub, "n": n, "replies": [None] * n, "coherence": None, "running": True, "mode": ph.get("mode")})
			if ph.get("threshold") is not None:
				self.seq["limit"] = ph["threshold"]
			self.timer.start(50)
		elif self.seq is not None and self.seq["cells"]:
			c = self.seq["cells"][-1]
			if k == "reply" and 0 <= d.get("i", -1) < c["n"]:
				c["replies"][d["i"]] = "gen"
			elif k == "score" and 0 <= d.get("i", -1) < c["n"]:
				c["replies"][d["i"]] = d.get("coherence") if d.get("coherence") is not None else "gen"
			elif k == "done":
				c["coherence"] = d.get("coherence")
				c["running"] = False
				c["by"] = d.get("by")
				if d.get("drop") is not None:
					c["sub"] = f"drop {d['drop']:.2f}"
		self.update()

	def trace(self, e):
		k = e.get("kind")
		if k == "guard":
			if self.seq is None or self.seq.get("closed"):
				self.seq = {"kind": "guard", "cells": [], "t": time.time()}
			self.seq.update(verdict=dict(e), closed=True, limit=e.get("threshold"))
		elif k == "calibration":
			if self.seq is None or self.seq.get("closed"):
				self.seq = {"kind": "calib", "cells": [], "t": time.time()}
			self.seq.update(verdict=dict(e), closed=True)
		elif k == "measure" and e.get("stopped") and self.seq is not None and not self.seq.get("closed"):
			for c in self.seq["cells"]:
				c["running"] = False
			self.seq.update(verdict={"stopped": True}, closed=True)
		else:
			return
		self.update()

	def running(self):
		return self.seq is not None and any(c["running"] for c in self.seq["cells"])

	def judging(self):
		return self.seq is not None and self.seq.get("kind") == "guard" and not self.seq.get("closed") and bool(self.seq["cells"])

	def ok_color(self, v, limit):
		if v is None:
			return theme.MUTED
		if v >= limit:
			return theme.PHOSPHOR if eva.on() else theme.OK
		return theme.ALARM if eva.on() else theme.ERR

	def paintEvent(self, e):
		if not self.running() and self.timer.isActive():
			self.timer.stop()
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		W, H = self.width(), self.height()
		t = time.perf_counter()
		self.head(p, W)
		area = QRectF(0, 26, W, H - 26)
		if self.seq is None or not (self.seq["cells"] or self.seq.get("verdict")):
			text = "no check yet · it runs when TO DO applies a lever change, or with JUDGE NOW" if self.guard_on else "coherence guard off: levers apply as set"
			if eva.on():
				eva.standby(p, area.adjusted(4, 4, -4, -4), text, "STANDBY")
			else:
				p.setFont(theme.mono(8))
				p.setPen(theme.qc(theme.DIM))
				p.drawText(area, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
			p.end()
			return
		if W < 380:
			bh = 40.0
			self.cells(p, QRectF(area.x() + 2, area.y() + 4, W - 4, area.height() - bh - 12), t)
			self.verdict(p, QRectF(2, area.bottom() - bh - 2, W - 4, bh), t, band=True)
		else:
			vw = min(200.0, W * 0.34)
			self.cells(p, QRectF(area.x() + 2, area.y() + 4, W - vw - 14, area.height() - 6), t)
			self.verdict(p, QRectF(W - vw - 4, area.y() + 4, vw, area.height() - 8), t)
		p.end()

	def head(self, p, W):
		st = self.state.get("state", "off")
		name = self.state.get("name") or os.path.splitext(os.path.basename(self.configured or ""))[0]
		col = {"ready": theme.PHOSPHOR if eva.on() else theme.OK, "loading": theme.AMBER if eva.on() else theme.WARN}.get(st, theme.ALARM if st in ("rejected", "error", "no_vram") else theme.MUTED)
		if not name:
			name, col, st = "NO JUDGE · surprise measure", theme.WARN, ""
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(theme.qc(col))
		p.drawEllipse(QPointF(7, 11), 4, 4)
		f = fnt(12, 600)
		p.setFont(f)
		parts = [STATE_TEXT.get(st, st.upper())] if st else []
		if self.state.get("gpu") is not None and st in ("ready", "loading"):
			parts.append(f"CUDA{self.state['gpu']}")
		if self.state.get("auc") is not None:
			parts.append(f"AUC {self.state['auc']:.2f}")
		x = W - 2
		p.setFont(f)
		p.setPen(theme.qc(theme.TEXT))
		room = x - 18
		line = name + ("   " + " · ".join(parts) if parts else "")
		p.drawText(QPointF(16, 16), eva.elide(f, line, room) if eva.on() else p.fontMetrics().elidedText(line, Qt.TextElideMode.ElideRight, int(room)))
		p.setPen(QPen(theme.qc(theme.LINE), 1))
		p.drawLine(QPointF(0, 23.5), QPointF(W, 23.5))

	def cells(self, p, r, t):
		seq = self.seq
		cells = seq["cells"][-12:]
		calib = seq["kind"] == "calib"
		limit = seq.get("target") if calib and seq.get("target") is not None else (seq.get("limit") or self.limit)
		n = max(len(cells), 4)
		cw = min(64.0, r.width() / n)
		lab_h = 26.0
		dots_h = 10.0
		gt, gb = r.y() + dots_h + 4, r.bottom() - lab_h
		gh = max(10.0, gb - gt)
		ly = gb - limit * gh
		for i, c in enumerate(cells):
			x = r.x() + i * cw
			g = QRectF(x + cw * 0.22, gt, cw * 0.56, gh)
			p.setPen(QPen(theme.qc(theme.LINE2), 1))
			p.setBrush(theme.qc(theme.PANEL2))
			p.drawRect(g)
			v = c["coherence"]
			if v is None and c["running"]:
				done = [s for s in c["replies"] if isinstance(s, float)]
				v = sum(done) / len(done) if done else None
			if v is not None:
				h = max(1.0, min(1.0, v) * gh)
				p.setPen(Qt.PenStyle.NoPen)
				p.setBrush(theme.qc(self.ok_color(v, limit), 120 if c["running"] else 230))
				p.drawRect(QRectF(g.x() + 1, g.bottom() - h, g.width() - 2, h))
			if c["running"]:
				a = 0.5 + 0.5 * math.sin(t * 7)
				if eva.on():
					eva.brackets(p, g.adjusted(-3, -3, 3, 3), theme.NEON, int(90 + 160 * a), 5, 1.4)
				else:
					p.setPen(QPen(theme.qc(theme.ACCENT, int(90 + 160 * a)), 1.4))
					p.setBrush(Qt.BrushStyle.NoBrush)
					p.drawRect(g.adjusted(-2, -2, 2, 2))
			k = len(c["replies"])
			dw = min(7.0, (cw - 4) / max(1, k))
			for j, s in enumerate(c["replies"]):
				dx = x + cw / 2 - k * dw / 2 + j * dw
				rr = QRectF(dx + 1, r.y() + 1, dw - 2, dw - 2)
				if s is None:
					p.setPen(QPen(theme.qc(theme.DIM), 1))
					p.setBrush(Qt.BrushStyle.NoBrush)
				elif s == "gen":
					p.setPen(Qt.PenStyle.NoPen)
					p.setBrush(theme.qc(theme.MUTED))
				else:
					p.setPen(Qt.PenStyle.NoPen)
					p.setBrush(theme.qc(self.ok_color(s, limit)))
				p.drawRect(rr)
			p.setFont(fnt(10, 700))
			p.setPen(theme.qc(theme.TEXT if c["running"] else theme.MUTED))
			p.drawText(QRectF(x, gb + 2, cw, 12), Qt.AlignmentFlag.AlignCenter, c["top"])
			sf = fnt(9, 600)
			p.setFont(sf)
			p.setPen(theme.qc(theme.DIM if not c["running"] else theme.MUTED))
			p.drawText(QRectF(x, gb + 13, cw, 12), Qt.AlignmentFlag.AlignCenter, eva.elide(sf, c["sub"], cw - 2) if eva.on() else p.fontMetrics().elidedText(c["sub"], Qt.TextElideMode.ElideRight, int(cw - 2)))
		p.setPen(QPen(theme.qc(theme.WARN, 220), 1, Qt.PenStyle.DashLine))
		p.drawLine(QPointF(r.x(), ly), QPointF(r.right(), ly))
		f = fnt(10, 700)
		text = f"{'TARGET' if calib else 'LIMIT'} {limit:.0%}"
		tw = eva.text_w(f, text) if eva.on() else p.fontMetrics().horizontalAdvance(text)
		p.setFont(f)
		bx = QRectF(r.right() - tw - 6, ly - 13, tw + 6, 12)
		p.fillRect(bx, theme.qc(theme.PANEL))
		p.setPen(theme.qc(theme.WARN))
		p.drawText(bx, Qt.AlignmentFlag.AlignCenter, text)

	def verdict(self, p, r, t, band=False):
		seq = self.seq
		v = seq.get("verdict")
		big, small, c = "", "", theme.MUTED
		if self.running():
			cell = seq["cells"][-1]
			done = sum(1 for s in cell["replies"] if s is not None and s != "gen")
			gen = sum(1 for s in cell["replies"] if s is not None)
			big = "CALIBRATING" if seq["kind"] == "calib" else "JUDGING"
			small = f"replies {gen}/{cell['n']} · scored {done}/{cell['n']}"
			c = theme.NEON if eva.on() else theme.ACCENT
		elif v is None:
			last = seq["cells"][-1] if seq["cells"] else None
			big, small = "…", f"coherence {last['coherence']:.0%}" if last and last.get("coherence") is not None else ""
		elif v.get("stopped"):
			big, small, c = "STOPPED", "the check was stopped before the end", theme.WARN
		elif seq["kind"] == "calib":
			big, small, c = f"1 = ×{v.get('mult', 1):.2f}", f"{v.get('lever', '')} calibrated · target {v.get('target', 0):.0%}", theme.PHOSPHOR if eva.on() else theme.OK
		else:
			coh, lim = v.get("coherence") or 0.0, v.get("threshold") or self.limit
			blame = v.get("blame") or {}
			if v.get("kept"):
				big, c = "KEPT", theme.PHOSPHOR if eva.on() else theme.OK
				small = f"only lowered from a mix that passed ({coh:.0%})"
			elif coh < lim:
				big, c = "BELOW", theme.ALARM if eva.on() else theme.ERR
				small = f"{coh:.0%} < {lim:.0%} even at the lowest try"
			elif blame:
				big, c = "CUT", theme.AMBER if eva.on() else theme.WARN
				small = " · ".join(f"{k} ×{f:.2f}" for k, f in list(blame.items())[:3])
			elif (v.get("scale") or 1) < 0.999:
				big, c = f"×{v['scale']:.2f}", theme.AMBER if eva.on() else theme.WARN
				small = "levers scaled down " + ("by reading" if v.get("by") == "read" else "together")
			elif v.get("by") == "read":
				big, c = "PASS", theme.PHOSPHOR if eva.on() else theme.OK
				small = f"read: drop {v.get('drop', 0):.2f}, nothing generated"
			else:
				big, c = "PASS", theme.PHOSPHOR if eva.on() else theme.OK
				small = f"coherence {coh:.0%} ≥ {lim:.0%}"
			if v.get("first") is not None and not v.get("kept") and v.get("by") != "read":
				small += f" · as set {v['first']:.0%}"
			if v.get("drop") is not None and v.get("by") != "read":
				small += f" · read drop {v['drop']:.2f}"
		p.setPen(QPen(theme.qc(c, 160), 1))
		p.setBrush(theme.qc(c, 18))
		p.drawRect(r)
		if eva.on():
			eva.brackets(p, r, c, 220, 7, 1.4)
		if band:
			bf = fnt(20, 700)
			p.setFont(bf)
			p.setPen(theme.qc(c))
			bw = (eva.text_w(bf, big) if eva.on() else p.fontMetrics().horizontalAdvance(big)) + 16
			p.drawText(QRectF(r.x() + 4, r.y(), bw, r.height()), Qt.AlignmentFlag.AlignCenter, big)
			p.setFont(fnt(10, 600))
			p.setPen(theme.qc(theme.TEXT))
			p.drawText(QRectF(r.x() + bw + 6, r.y() + 2, r.width() - bw - 12, r.height() - 4), Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, small)
			return
		bf = fnt(26, 700)
		p.setFont(bf)
		p.setPen(theme.qc(c))
		p.drawText(QRectF(r.x(), r.y() + 4, r.width(), r.height() * 0.55), Qt.AlignmentFlag.AlignCenter, big)
		p.setFont(fnt(10, 600))
		p.setPen(theme.qc(theme.TEXT))
		p.drawText(QRectF(r.x() + 6, r.y() + r.height() * 0.55, r.width() - 12, r.height() * 0.45 - 4), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, small)


def slider(value):
	s = guard_wheel(StepSlider(1))
	s.setRange(0, 100)
	s.setValue(int(value))
	return s


def spin(lo, hi, value, step=1, decimals=None, special=""):
	s = QDoubleSpinBox() if decimals else QSpinBox()
	if decimals:
		s.setDecimals(decimals)
	s.setRange(lo, hi)
	s.setSingleStep(step)
	s.setValue(value)
	if special:
		s.setSpecialValueText(special)
	s.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
	s.setFont(theme.mono(8.5))
	return guard_wheel(s)


def cap(text, tip=""):
	l = QLabel(text)
	theme.tag_font(l, 7.5)
	l.setStyleSheet(f"color: {theme.MUTED};")
	l.setToolTip(tip)
	return l


class JudgeBox(QWidget):
	changed = Signal(object, object)
	judge_now = Signal()
	skip_judge = Signal()

	def __init__(self, names=None, parent=None):
		super().__init__(parent)
		self.quiet = False
		self.pending = ({}, {})
		self.models = []
		lay = QVBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(0)
		self.tabs = QWidget()
		tl = QHBoxLayout(self.tabs)
		tl.setContentsMargins(0, 0, 0, 0)
		tl.setSpacing(3)
		self.group = QButtonGroup(self)
		self.group.setExclusive(True)
		self.stack = QStackedWidget()
		for n, (text, tip) in enumerate((("VIEW", "what the judge is doing: each test measure, its replies and the verdict"), ("CONFIG", "the judge, its thresholds and every number: questions per mode, steps, reply length, temperature…"))):
			b = toggle(text, tip, n == 0)
			theme.tag_font(b, 7)
			b.setFixedSize(b.fontMetrics().horizontalAdvance(text) + 26, 18)
			self.group.addButton(b, n)
			tl.addWidget(b)
		self.group.idClicked.connect(self.stack.setCurrentIndex)
		self.tabs.setFixedSize(self.tabs.sizeHint())
		self.view = JudgeView(names)
		page = QWidget()
		pl = QVBoxLayout(page)
		pl.setContentsMargins(0, 0, 0, 0)
		pl.setSpacing(4)
		row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
		row.setSpacing(6)
		self.row = row
		self.b_now = button("JUDGE NOW", "check the levers changed since the last check (planned in TO DO)")
		self.b_now.setFixedHeight(22)
		self.b_now.setMinimumWidth(self.b_now.fontMetrics().horizontalAdvance("JUDGE NOW · 9") + 22)
		self.b_now.clicked.connect(self.judge_now.emit)
		self.b_skip = button("APPLY AS SET", "stop this judgement and apply the levers as you set them, without the guard's verdict\nthey stay 'not judged': JUDGE NOW checks them later", accent=True)
		self.b_skip.setFixedHeight(22)
		self.b_skip.setMinimumWidth(self.b_now.minimumWidth())
		self.b_skip.clicked.connect(self.skip_judge.emit)
		self.b_skip.setVisible(False)
		self.mode = SegSwitch(list(MODES) + [("standby", "STANDBY", "the coherence guard is off: lever changes apply exactly as set, at once\nno judge, no KL safety, no wait, and no factor kept from earlier judgements (they come back when a guard mode is picked again)", "OFF")])
		self.mode.picked.connect(self.mode_picked)
		row.addWidget(self.b_now, 0, Qt.AlignmentFlag.AlignLeft)
		row.addWidget(self.b_skip, 0, Qt.AlignmentFlag.AlignLeft)
		row.addStretch(1)
		row.addWidget(self.mode)
		pl.addLayout(row)
		pl.addWidget(self.view, 1)
		self.stack.addWidget(page)
		self.stack.addWidget(self.config_tab())
		lay.addWidget(self.stack, 1)
		self.save_timer = QTimer(self)
		self.save_timer.setSingleShot(True)
		self.save_timer.timeout.connect(self.flush)
		self.load(config.load_settings())

	def mode_picked(self, key):
		self.edit({"guard": False} if key == "standby" else {"guard": True, "guard_mode": key})

	def set_judging(self, on):
		on = bool(on)
		if on == (not self.b_skip.isHidden()):
			return
		self.b_now.setVisible(not on)
		self.b_skip.setVisible(on)
		self.fit_row()

	def resizeEvent(self, e):
		super().resizeEvent(e)
		self.fit_row()

	def fit_row(self):
		b = self.b_now if self.b_skip.isHidden() else self.b_skip
		wide = self.width() >= max(b.sizeHint().width(), b.minimumWidth()) + self.row.spacing() + self.mode.full_width()
		want = QBoxLayout.Direction.LeftToRight if wide else QBoxLayout.Direction.TopToBottom
		if self.row.direction() != want:
			self.row.setDirection(want)

	def config_tab(self):
		inner = QWidget()
		g = QGridLayout(inner)
		g.setContentsMargins(0, 2, 6, 0)
		g.setHorizontalSpacing(6)
		g.setVerticalSpacing(3)
		self.j_model = guard_wheel(QComboBox())
		self.j_model.setFont(theme.mono(8))
		self.j_model.setMinimumWidth(180)
		self.j_model.setToolTip("the judge: a second, small model that rates the coherence of the test replies\nnone = the surprise measure (no second model)")
		self.j_model.activated.connect(lambda _i: self.edit({"judge": self.j_model.currentData() or ""}))
		jl = QVBoxLayout()
		jl.setSpacing(2)
		jl.addWidget(cap("JUDGE MODEL"))
		jl.addWidget(self.j_model)
		g.addLayout(jl, 0, 0, 1, 3)
		self.s_tol = slider(50)
		self.s_tol.setToolTip("how far the levers may push before the guard steps in\nstrict: only clean replies · loose: lets the style get wild")
		self.l_tol = QLabel("")
		self.l_tol.setFont(theme.mono(7.5))
		self.l_tol.setMinimumWidth(64)
		self.s_tol.valueChanged.connect(lambda v: (self.l_tol.setText(f"limit {threshold(v):.0%}"), self.edit({"tolerance": int(v)})))
		g.addWidget(cap("TOLERANCE", "strict ↔ loose"), 1, 0)
		g.addWidget(self.s_tol, 1, 1)
		g.addWidget(self.l_tol, 1, 2)
		self.s_abs = slider(30)
		self.s_abs.setToolTip("how much grammatical but absurd content counts as coherent\nsensible ↔ surreal")
		self.l_abs = QLabel("")
		self.l_abs.setFont(theme.mono(7.5))
		self.s_abs.valueChanged.connect(lambda v: (self.l_abs.setText(f"{v}%"), self.edit({"absurdity": int(v)})))
		g.addWidget(cap("ABSURDITY", "sensible ↔ surreal"), 2, 0)
		g.addWidget(self.s_abs, 2, 1)
		g.addWidget(self.l_abs, 2, 2)
		self.hint = QLabel("")
		self.hint.setFont(theme.mono(7.5))
		self.hint.setWordWrap(True)
		self.hint.setStyleSheet(f"color: {theme.WARN};")
		g.addWidget(self.hint, 3, 0, 1, 3)
		self.cfg = {}
		rows = (
			("guard_questions", "GUARD QUESTIONS", lambda v: spin(0, 4, int(v), special="all"), "test questions the guard asks for each measure (0 = all the test questions)"),
			("guard_steps", "GUARD STEPS", lambda v: spin(1, 6, int(v)), "UNIFORM: how many scaling tries before giving up"),
			("blame_rounds", "HALVING ROUNDS", lambda v: spin(1, 6, int(v)), "INDIVIDUAL and the 'worst first' fallbacks: how many times the lever that hurts coherence the most may be halved before the rest is scaled together"),
			("quick_questions", "QUICK · PER LEVER", lambda v: spin(1, 6, int(v)), "QUICK: replies per changed lever for the first check"),
			("quick_fallback", "QUICK · IF IT FAILS", None, "QUICK: the full check that runs when the quick one fails"),
			("read_pass", "READ · PASS UNDER", lambda v: spin(0.05, 5.0, float(v), 0.05, 2), "READ: a drop (how much less likely the model finds its own normal replies with the levers, per token) up to this validates at once, nothing generated\nmeasured on Qwen3.5-4B: about 0.8 when the judge still rates ~75%, 2.5 when it varies a lot between levers"),
			("read_fail", "READ · TOO FAR OVER", lambda v: spin(0.5, 12.0, float(v), 0.25, 2), "READ: a drop from this on is surely broken: the levers are scaled down by reading until it is under (on Qwen3.5-4B the judge gave ≤ 8% beyond 4)"),
			("read_fallback", "READ · IN BETWEEN", None, "READ: the full check that runs when the drop lies between the two thresholds"),
			("calib_questions", "CALIBRATION QUESTIONS", lambda v: spin(0, 4, int(v), special="all"), "test questions per calibration step (0 = all)"),
			("calib_steps", "CALIBRATION STEPS", lambda v: spin(2, 8, int(v)), "tries of the calibration search per lever: fewer = faster, less precise (5 before)"),
			("test_samples", "REPLIES PER QUESTION", lambda v: spin(1, 6, int(v)), "replies drawn for each test question: steadier scores, slower"),
			("test_tokens", "REPLY LENGTH", lambda v: spin(12, 200, int(v), 4), "tokens per test reply: longer = more reliable, slower"),
			("judge_temperature", "JUDGE TEMPERATURE", lambda v: spin(0.05, 5.0, float(v), 0.05, 2), "applied to the judge's answer probabilities · 1 = as read"),
			("individual", "SCALE COMMON ↔ OWN", lambda v: spin(0, 100, int(v), 5), "what strength 1 means for each lever: 100 = its own calibrated maximum, 0 = the same KL for all"),
		)
		top = 4
		for n, (key, label, make, tip) in enumerate(rows):
			if key in ("quick_fallback", "read_fallback"):
				w = guard_wheel(QComboBox())
				w.setFont(theme.mono(8))
				w.addItem("worst lever first", "blame")
				w.addItem("uniform", "uniform")
				w.activated.connect(lambda _i, w=w, k=key: self.edit({k: w.currentData()}))
			else:
				w = make(EXTRA_DEFAULTS.get(key, 0))
				w.valueChanged.connect(lambda v, k=key: self.edit({k: round(v, 3) if isinstance(v, float) else int(v)}))
			w.setToolTip(tip)
			g.addWidget(cap(label, tip), top + n, 0)
			g.addWidget(w, top + n, 1, 1, 2)
			self.cfg[key] = w
		n = top + len(rows)
		self.c_cal = tab_focus(QCheckBox("calibrate levers on real generations"))
		self.c_cal.setFont(theme.mono(8))
		self.c_cal.toggled.connect(lambda on: self.edit({"calibrate": bool(on)}))
		g.addWidget(self.c_cal, n, 0, 1, 3)
		self.c_gopt = tab_focus(QCheckBox("guard the trained goal vectors too"))
		self.c_gopt.setFont(theme.mono(8))
		self.c_gopt.toggled.connect(lambda on: self.edit({"guard_optimized": bool(on)}))
		g.addWidget(self.c_gopt, n + 1, 0, 1, 3)
		self.e_sys = QLineEdit()
		self.e_sys.setFont(theme.mono(8))
		self.e_sys.setPlaceholderText("none")
		self.e_sys.editingFinished.connect(lambda: self.edit({"test_system": self.e_sys.text().strip()}))
		g.addWidget(cap("TEST SYSTEM PROMPT"), n + 2, 0)
		g.addWidget(self.e_sys, n + 2, 1, 1, 2)
		self.e_prompts = QPlainTextEdit()
		self.e_prompts.setFont(theme.mono(8))
		self.e_prompts.setFixedHeight(54)
		self.e_prompts.setPlaceholderText("one per line · empty = built-in everyday questions")
		self.e_prompts.textChanged.connect(lambda: self.edit({"test_prompts": [l.strip() for l in self.e_prompts.toPlainText().splitlines() if l.strip()]}))
		g.addWidget(cap("TEST QUESTIONS"), n + 3, 0, Qt.AlignmentFlag.AlignTop)
		g.addWidget(self.e_prompts, n + 3, 1, 1, 2)
		self.e_jq = QPlainTextEdit()
		self.e_jq.setFont(theme.mono(8))
		self.e_jq.setFixedHeight(54)
		self.e_jq.setPlaceholderText("empty = automatic · custom: must contain {reply}")
		self.e_jq.textChanged.connect(lambda: self.edit({"judge_prompt": self.e_jq.toPlainText().strip()}))
		g.addWidget(cap("CUSTOM JUDGE QUESTION"), n + 4, 0, Qt.AlignmentFlag.AlignTop)
		g.addWidget(self.e_jq, n + 4, 1, 1, 2)
		self.dose = spin(1.0, 5.0, 1.5, 0.25, 2)
		self.dose.setToolTip("how far the lever sliders and ALIGN may push each lever (±)")
		self.dose.valueChanged.connect(lambda v: self.edit({}, {"dose_max": round(v, 2)}))
		g.addWidget(cap("MAX LEVER DOSE"), n + 5, 0)
		g.addWidget(self.dose, n + 5, 1, 1, 2)
		g.setColumnStretch(1, 1)
		g.setRowStretch(n + 6, 1)
		sc = QScrollArea()
		sc.setWidgetResizable(True)
		sc.setFrameShape(QFrame.Shape.NoFrame)
		sc.setWidget(inner)
		return sc

	def set_models(self, models, current):
		self.models = [m for m in models or [] if not m.get("error") and not m.get("convert") and m.get("arch") not in ("clip", "mtmd")]
		self.quiet = True
		self.j_model.clear()
		self.j_model.addItem("none: surprise measure", "")
		for m in sorted(self.models, key=lambda m: m.get("size", 0)):
			self.j_model.addItem(f"{os.path.splitext(m['file'])[0]} · {m.get('size', 0) / 2**30:.1f} GB", m["path"])
			self.j_model.setItemData(self.j_model.count() - 1, m["path"], Qt.ItemDataRole.ToolTipRole)
		i = self.j_model.findData(current or "")
		if i < 0 and current:
			self.j_model.addItem(os.path.basename(current), current)
			i = self.j_model.count() - 1
		self.j_model.setCurrentIndex(max(0, i))
		self.j_model.view().setMinimumWidth(self.j_model.view().sizeHintForColumn(0) + 24)
		self.quiet = False

	def load(self, st):
		coh = {**config.DEFAULTS["coherence"], **EXTRA_DEFAULTS, **(st.get("coherence") or {})}
		self.quiet = True
		try:
			if self.j_model.count() == 0 or self.j_model.findData(coh["judge"]) < 0:
				self.set_models(self.models, coh["judge"])
				self.quiet = True
			self.j_model.setCurrentIndex(max(0, self.j_model.findData(coh["judge"])))
			self.mode.set_value(LEGACY_MODES.get(coh["guard_mode"], coh["guard_mode"]) if coh["guard"] else "standby")
			self.s_tol.setValue(int(coh["tolerance"]))
			self.s_abs.setValue(int(coh["absurdity"]))
			self.l_tol.setText(f"limit {threshold(coh['tolerance']):.0%}")
			self.l_abs.setText(f"{int(coh['absurdity'])}%")
			for k, w in self.cfg.items():
				if isinstance(w, QComboBox):
					w.setCurrentIndex(max(0, w.findData(coh.get(k))))
				else:
					w.setValue(coh.get(k) if coh.get(k) is not None else w.value())
			self.c_cal.setChecked(bool(coh["calibrate"]))
			self.c_gopt.setChecked(bool(coh["guard_optimized"]))
			if self.e_sys.text() != (coh["test_system"] or ""):
				self.e_sys.setText(coh["test_system"] or "")
			if [l.strip() for l in self.e_prompts.toPlainText().splitlines() if l.strip()] != list(coh["test_prompts"] or []):
				self.e_prompts.setPlainText("\n".join(coh["test_prompts"] or []))
			if self.e_jq.toPlainText().strip() != (coh["judge_prompt"] or ""):
				self.e_jq.setPlainText(coh["judge_prompt"] or "")
			self.dose.setValue(float(st.get("dose_max") or 1.5))
		finally:
			self.quiet = False
		self.view.set_info(configured=coh["judge"], mode=LEGACY_MODES.get(coh["guard_mode"], coh["guard_mode"]), guard_on=bool(coh["guard"]), limit=threshold(coh["tolerance"]))

	def edit(self, coh, top=None):
		if self.quiet:
			return
		self.pending = ({**self.pending[0], **coh}, {**self.pending[1], **(top or {})})
		if "guard_mode" in coh or "guard" in coh or "tolerance" in coh or "judge" in coh:
			self.view.set_info(mode=coh.get("guard_mode"), guard_on=coh.get("guard"), limit=threshold(coh["tolerance"]) if "tolerance" in coh else None, configured=coh.get("judge"))
		if any(k in coh for k in ("judge", "tolerance", "absurdity", "guard", "guard_mode", "judge_prompt", "judge_temperature")):
			self.hint.setText("changed: the next check uses it · JUDGE NOW checks the levers again")
		self.save_timer.start(500)

	def flush(self):
		coh, top = self.pending
		self.pending = ({}, {})
		if coh or top:
			self.changed.emit(coh, top)

	def set_unjudged(self, names):
		self.b_now.setText(f"JUDGE NOW · {len(names)}" if names else "JUDGE NOW")
		self.b_now.setProperty("accent", bool(names))
		self.b_now.style().unpolish(self.b_now)
		self.b_now.style().polish(self.b_now)
		self.b_now.setToolTip("check the levers changed since the last check (planned in TO DO)" + ("\nnot judged yet: " + ", ".join(names) if names else "\nevery lever was judged as it is: JUDGE NOW checks them all again"))
