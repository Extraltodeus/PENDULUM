import html
import json
import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QSplitter, QTextBrowser, QVBoxLayout

import config
from ui import theme
from ui.widgets import EvaDialog, button, tab_focus

KINDS = {"measure": "JUDGE TESTS", "guard": "GUARD", "calibration": "CALIBRATION", "judge": "JUDGE SELF-TEST", "goal": "GOAL CHECKS", "align": "ALIGNMENT", "matrix": "LEVER EFFECTS", "read_gain": "READ SCALE", "goal_scan": "GOAL READING"}
SHORT = {"measure": "TEST", "guard": "GUARD", "calibration": "CALIBRATION", "judge": "SELF-TEST", "goal": "GOAL", "align": "ALIGN", "matrix": "EFFECTS", "read_gain": "READ SCALE", "goal_scan": "READ GOAL"}
LEVELS = {"1": "sensible", "2": "absurd but grammatical", "3": "broken", "4": "word salad"}


def pct(v):
	return "—" if v is None else f"{float(v):.0%}"


def esc(text):
	return html.escape(str(text or "")).replace(chr(10), "<br>")


def tone(v, good=0.65, bad=0.4):
	if v is None:
		return theme.MUTED
	return theme.OK if v >= good else (theme.WARN if v >= bad else theme.ERR)


def headline(e):
	k = e.get("kind")
	if k == "measure" and e.get("stopped"):
		return "stopped before the end"
	if k == "measure" and e.get("running"):
		return "running…" + (f" · {e['note']}" if e.get("note") else "")
	if k == "measure":
		return f"{pct(e.get('coherence'))} {e.get('by', '')}" + (f" · {e['note']}" if e.get("note") else "")
	if k == "guard":
		if e.get("kept"):
			return f"kept {pct(e.get('coherence'))}: levers only lowered"
		if e.get("first") is not None:
			return f"{pct(e['first'])} < {pct(e.get('threshold'))} → " + (f"×{e['scale']:.2f} " if e.get("scale", 1) < 0.999 else "") + ("cut " if e.get("blame") else "") + f"→ {pct(e.get('coherence'))}"
		return f"passed {pct(e.get('coherence'))} (threshold {pct(e.get('threshold'))})"
	if k == "calibration":
		return f"'{e.get('lever')}': strength 1 = ×{e.get('mult', 0):.2f}"
	if k == "judge":
		return f"{e.get('name', '')}: AUC {e.get('auc', 0):.2f} · " + ("accepted" if e.get("accepted") else "rejected")
	if k == "goal":
		return f"match {pct(e.get('match'))} · sense {pct(e.get('sense'))}"
	if k == "align":
		return f"round {e.get('round', 0)} · largest gap {e.get('gap', 0):.2f}"
	if k == "matrix":
		top = sorted((e.get("moves") or {}).items(), key=lambda kv: -abs(kv[1]))[:3]
		return f"'{e.get('lever')}' moves " + ", ".join(f"{n} {v:+.2f}" for n, v in top)
	if k == "goal_scan":
		sc = [s for s in e.get("scan") or [] if s.get("pull") is not None]
		top = max(sc, key=lambda s: s["pull"]) if sc else None
		return f"'{e.get('goal')}': {len(e.get('scan') or [])} strengths read" + (f" · strongest pull {top['pull']:.0%} at {top['strength']:+.2f}" if top else "")
	if k == "read_gain" and e.get("mix"):
		return f"mix {e.get('lever')}: reading ×{e.get('total', e.get('gain', 1.0)):.2f}" + (f" · r = {e['corr']:.2f}" if e.get("corr") is not None else "")
	if k == "read_gain":
		return f"'{e.get('lever')}': reading ×{e.get('gain', 1.0):.2f}" + (f" · r = {e['corr']:.2f}" if e.get("corr") is not None else "")
	return ""


class TraceDialog(EvaDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Trace")
		self.setMinimumSize(1040, 620)
		self.entries = []
		self.shown = []
		lay = QVBoxLayout(self)
		lay.setSpacing(6)
		lay.addWidget(self.frameless("TRACE", "EVERY TEST OF THE JUDGE, ITS ANSWERS AND WHAT IT DECIDED"))
		top = QHBoxLayout()
		self.search = QLineEdit()
		self.search.setFont(theme.mono(8.5))
		self.search.setPlaceholderText("filter: words of a prompt, a reply, a lever…")
		self.search.textChanged.connect(self.fill)
		top.addWidget(self.search, 1)
		self.kinds = {}
		for k, label in KINDS.items():
			c = QCheckBox(label.lower())
			tab_focus(c)
			c.setFont(theme.mono(8))
			c.setChecked(True)
			c.toggled.connect(self.fill)
			self.kinds[k] = c
			top.addWidget(c)
		self.count = QLabel("")
		self.count.setFont(theme.mono(8))
		self.count.setStyleSheet(f"color: {theme.MUTED};")
		top.addWidget(self.count)
		lay.addLayout(top)
		split = QSplitter(Qt.Orientation.Horizontal)
		self.quiet = False
		self.list = QListWidget()
		self.list.setFont(theme.mono(8))
		self.list.currentRowChanged.connect(self.show_entry)
		self.list.verticalScrollBar().valueChanged.connect(self._scrolled)
		self.view = QTextBrowser()
		self.view.setFont(theme.mono(8.5))
		split.addWidget(self.list)
		split.addWidget(self.view)
		split.setSizes([430, 610])
		lay.addWidget(split, 1)
		btns = QHBoxLayout()
		note = QLabel("kept for this run of the app (the last 1500 events), newest at the bottom: the list follows them while it is scrolled to the bottom · SAVE writes them to a JSON file")
		note.setWordWrap(True)
		note.setToolTip(note.text())
		note.setFont(theme.mono(7.5))
		note.setStyleSheet(f"color: {theme.MUTED};")
		btns.addWidget(note, 1)
		b_copy = button("COPY", "copy the selected event as text")
		b_copy.clicked.connect(self.copy)
		b_save = button("SAVE…", "write every event to a JSON file")
		b_save.clicked.connect(self.save)
		b_clear = button("CLEAR", "forget the events gathered so far")
		b_clear.clicked.connect(self.clear)
		for b in (b_copy, b_save, b_clear):
			btns.addWidget(b)
		lay.addLayout(btns)

	def set_entries(self, entries):
		self.entries = entries
		self.fill()

	def add(self, e):
		if self.isVisible():
			self.fill(keep_last=True)

	def match(self, e, words):
		c = self.kinds.get(e.get("kind"))
		if c is not None and not c.isChecked():
			return False
		if not words:
			return True
		hay = json.dumps(e, ensure_ascii=False).lower()
		return all(w in hay for w in words)

	def at_bottom(self):
		bar = self.list.verticalScrollBar()
		return self.list.count() == 0 or bar.value() >= bar.maximum() - 2

	def _scrolled(self, v):
		if self.quiet or not self.shown:
			return
		if v >= self.list.verticalScrollBar().maximum() - 2 and self.list.currentRow() != len(self.shown) - 1:
			self.list.setCurrentRow(len(self.shown) - 1)

	def fill(self, *_a, keep_last=False):
		self.quiet = True
		try:
			self._fill(keep_last)
		finally:
			self.quiet = False

	def _fill(self, keep_last):
		words = self.search.text().lower().split()
		cur = self.current()
		last = self.shown[-1] if self.shown else None
		follow = self.at_bottom()
		pos = self.list.verticalScrollBar().value()
		self.shown = [e for e in self.entries if self.match(e, words)]
		self.list.blockSignals(True)
		self.list.clear()
		for e in self.shown:
			k = e.get("kind")
			v = e.get("coherence") if k in ("measure", "guard") else (e.get("match") if k == "goal" else None)
			it = QListWidgetItem(f"{time.strftime('%H:%M:%S', time.localtime(e.get('t', 0)))}  {SHORT.get(k, k):<11} {headline(e)}")
			it.setToolTip((e.get("context") or "") + ("\n" + headline(e)))
			it.setForeground(theme.qc(tone(v) if v is not None else (theme.TEXT if k != "judge" else (theme.OK if e.get("accepted") else theme.ERR))))
			self.list.addItem(it)
		row = next((i for i, e in enumerate(self.shown) if e is cur), -1)
		if self.shown and (row < 0 or (keep_last and follow and cur is last)):
			row = len(self.shown) - 1
		if self.shown:
			self.list.setCurrentRow(row)
		self.list.blockSignals(False)
		self.count.setText(f"{len(self.shown)} / {len(self.entries)}")
		self.list.doItemsLayout()
		if follow:
			self.list.scrollToBottom()
		else:
			self.list.verticalScrollBar().setValue(pos)
		self.show_entry(self.list.currentRow())

	def current(self):
		i = self.list.currentRow() if hasattr(self, "list") else -1
		return self.shown[i] if 0 <= i < len(self.shown) else None

	def focus(self, e):
		c = self.kinds.get(e.get("kind"))
		if c is not None and not c.isChecked():
			c.setChecked(True)
		if self.search.text():
			self.search.clear()
		self.fill()
		i = next((i for i, x in enumerate(self.shown) if x is e), -1)
		if i >= 0:
			self.quiet = True
			self.list.setCurrentRow(i)
			self.list.scrollToItem(self.list.item(i), QAbstractItemView.ScrollHint.PositionAtCenter)
			self.quiet = False

	def show_entry(self, _i=-1):
		e = self.current()
		self.view.setHtml(self.render(e) if e is not None else f'<p style="color:{theme.MUTED}">nothing yet: the events show up as the levers are applied, calibrated or checked by the judge</p>')

	def render(self, e):
		k = e.get("kind")
		mute = lambda t: f'<span style="color:{theme.MUTED}">{t}</span>'
		out = [f'<p>{mute(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.get("t", 0))))} · <span style="color:{theme.NEON}">{KINDS.get(k, k)}</span><br><span style="color:{theme.TEXT}">{esc(e.get("context") or "")}</span></p>']
		if k == "measure" and e.get("running"):
			judge = f"the judge {esc(e.get('judge_name'))} ('{esc(e.get('variant'))}' question)" if e.get("judge_name") else "the surprise measure"
			out.append(f"<p><span style='color:{theme.NEON}'>RUNNING</span>{(' · ' + esc(e.get('note'))) if e.get('note') else ''}<br>{mute('generating the test replies, then scored by ' + judge + ' · this entry is replaced by the result when the measure ends')}</p>")
			return "".join(out)
		if k == "measure":
			c = e.get("coherence")
			judge = ("judge " + pct(e.get("judge")) + " (" + esc(e.get("judge_name")) + ", '" + esc(e.get("variant")) + "' question) · ") if e.get("judge_name") else ""
			sur = "surprise %.2f vs %.2f unsteered (+%.2f) · absurdity tolerance %s" % (e.get("surprise", 0), e.get("base", 0), e.get("excess", 0), pct(e.get("absurd")))
			note = (" · " + esc(e.get("note"))) if e.get("note") else ""
			out.append(f"<p><span style='color:{tone(c)}'>COHERENCE {pct(c)}</span> by the {esc(e.get('by'))}{note}<br>{mute(judge + sur)}</p>")
			for i, s in enumerate(e.get("samples") or []):
				j = s.get("judge") or {}
				probs = j.get("p") or {}
				pj = " · ".join(f"{d} {LEVELS.get(d, d)} {pct(probs.get(d))}" for d in "1234" if d in probs and (probs.get(d) or d in ("1", "4")))
				reply = esc(s.get("reply")) or mute("(empty)")
				verdict = f"<br>{mute('judge:')} {pj} → <span style='color:{tone(j.get('coherence'))}'>{pct(j.get('coherence'))}</span>{mute(' · answer mass ' + pct(j.get('mass')))}" if j else ""
				out.append(f"<p>{mute('TEST ' + str(i + 1))} <span style='color:{theme.TEXT}'>{esc(s.get('prompt'))}</span><br><span style='color:#d8d4cd'>{reply}</span>{verdict}</p>")
		elif k == "guard":
			lines = [f"threshold {pct(e.get('threshold'))} · mode {esc(e.get('mode'))}"]
			if e.get("kept"):
				lines.append(f"levers only lowered from a mix that passed ({pct(e.get('coherence'))}): no new test")
			elif e.get("first") is not None:
				lines.append(f"at the strengths you set: <span style='color:{theme.ERR}'>{pct(e['first'])}</span>, below the threshold: the guard stepped in")
				if e.get("blame"):
					lines.append("halved: " + ", ".join(f"{esc(n)} ×{f:.2f}" for n, f in e["blame"].items()))
				if e.get("scale", 1) < 0.999:
					lines.append(f"every guarded lever scaled ×{e['scale']:.2f}")
				lines.append(f"after: <span style='color:{tone(e.get('coherence'))}'>{pct(e.get('coherence'))}</span>")
			else:
				lines.append(f"<span style='color:{tone(e.get('coherence'))}'>passed at {pct(e.get('coherence'))}</span>: applied as set")
			out.append("<p>" + "<br>".join(lines) + "</p>")
		elif k == "calibration":
			why = "the strongest dose whose test replies stay above " + pct(e.get("target")) + " coherence (" + esc(e.get("by")) + "); its tests are listed just before"
			out.append(f"<p>lever <span style='color:{theme.TEXT}'>{esc(e.get('lever'))}</span>: strength 1 now = ×{e.get('mult', 0):.2f} of the old scale<br>{mute(why)}</p>")
		elif k == "judge":
			rep = e.get("report") or {}
			rows = "".join(f"<tr><td>{esc(n)}</td><td>{r.get('auc', 0):.2f}</td><td>{pct(r.get('good'))}</td><td>{pct(r.get('bad'))}</td></tr>" for n, r in rep.items() if isinstance(r, dict))
			out.append(f"<p>AUC {e.get('auc', 0):.2f} (minimum {e.get('minimum', 0):.2f}): <span style='color:{theme.OK if e.get('accepted') else theme.ERR}'>{'accepted' if e.get('accepted') else 'rejected'}</span> · question '{esc(e.get('variant'))}' · scale {(e.get('anchors') or [0, 1])[0]:.2f}→{(e.get('anchors') or [0, 1])[1]:.2f} · absurdity cap {pct(e.get('cap'))}" + (f" · word salad rated {pct(rep.get('salad_at_full_absurdity'))} at full absurdity" if 'salad_at_full_absurdity' in rep else "") + "</p>")
			if rows:
				out.append(f"<table cellspacing='0' cellpadding='3'><tr>{''.join(f'<th align=left>{mute(h)}</th>' for h in ('question', 'AUC', 'coherent rated', 'salad rated'))}</tr>{rows}</table>")
		elif k == "goal":
			out.append(f"<p>match <span style='color:{tone(e.get('match'), 0.66)}'>{pct(e.get('match'))}</span> · sense {pct(e.get('sense'))}" + (f" · goal lever {e['strength']:+.2f}" if e.get("strength") is not None else "") + "</p>")
			for r in e.get("rows") or []:
				out.append(f"<p>{mute(esc(str(r.get('lang', '')).upper() + ' · ' + str(r.get('key', ''))))} <span style='color:{theme.TEXT}'>{esc(r.get('question'))}</span><br><span style='color:#d8d4cd'>{esc(r.get('reply'))}</span><br>{mute('sense')} <span style='color:{tone(r.get('sense'), 0.8, 0.6)}'>{pct(r.get('sense'))}</span> · {mute('match')} <span style='color:{tone(r.get('match'), 0.66)}'>{pct(r.get('match'))}</span></p>")
		elif k == "align":
			rows = "".join(f"<tr><td>{esc(r.get('name'))}</td><td style='color:{theme.CYAN}'>{r.get('target', 0):+.2f}</td><td style='color:{tone(1.0 - abs(r.get('target', 0) - r.get('now', 0)) * 10, 0.5, 0.0)}'>{r.get('now', 0):+.2f}</td><td>{r.get('dose', 0):+.2f}</td><td>{r.get('applied', 0):+.2f}</td></tr>" for r in e.get("rows") or [])
			out.append(f"<p>largest gap between target and measure: {e.get('gap', 0):.2f}" + (" · reached" if e.get("gap", 1) <= 0.05 else "") + "</p>")
			out.append(f"<table cellspacing='0' cellpadding='3'><tr>{''.join(f'<th align=left>{mute(h)}</th>' for h in ('trait', 'target', 'measured', 'dose asked', 'dose applied'))}</tr>{rows}</table>")
			out.append(f"<p>{mute('the next round solves for the doses with the measured effects and the interactions, applies them under the coherence guard, measures again and corrects the effects')}</p>")
		elif k == "matrix":
			moves = sorted((e.get("moves") or {}).items(), key=lambda kv: -abs(kv[1]))
			out.append(f"<p>lever <span style='color:{theme.TEXT}'>{esc(e.get('lever'))}</span> alone at {e.get('dose', 0):+.2f}: change of each measured trait</p>")
			out.append("<p>" + "<br>".join(f"<span style='color:{theme.TEXT if abs(v) >= 0.03 else theme.MUTED}'>{esc(n)}</span> <span style='color:{theme.ACCENT2 if v >= 0 else theme.COOL2}'>{v:+.3f}</span>" for n, v in moves) + "</p>")
			out.append(f"<p>{mute('its own trait should move most; the others are the interactions the alignment accounts for · kept per model, measured again when the lever changes')}</p>")
		elif k == "goal_scan":
			out.append(f"<p>goal <span style='color:{theme.TEXT}'>{esc(e.get('goal'))}</span>: the model reads your target answers with the goal lever at each strength, in the order tried (nothing is written)</p>")
			rows = "".join(f"<tr><td>×{s.get('strength', 0):g}</td><td>{pct(s.get('pull'))}</td><td>{s.get('pull_raw', 0):+.3f}</td><td style='color:{theme.ERR if s.get('broken') else theme.TEXT}'>{'—' if s.get('drop') is None else format(s['drop'], '.2f') + (' broken' if s.get('broken') else '')}</td></tr>" for s in e.get("scan") or [])
			out.append(f"<table cellspacing='0' cellpadding='3'><tr>{''.join(f'<th align=left>{mute(h)}</th>' for h in ('strength', 'pull', 'log-prob per token', 'reading drop'))}</tr>{rows}</table>")
			out.append(f"<p>{mute('pull = how much more likely the targets are than with the bare model, as a share of what the character prompt does · reading drop = how much less likely the bare model replies become (the READ threshold of the guard marks them broken) · seek: from 1, doubled until the pull reaches REACH or the replies break, then halved between the last two')}</p>")
		elif k == "read_gain":
			wr, rd = e.get("written") or {}, e.get("read") or {}
			if e.get("mix"):
				out.append(f"<p>mix <span style='color:{theme.TEXT}'>{esc(e.get('lever'))}</span> (" + ", ".join(f"{esc(n)} {v:+.2f}" for n, v in (e.get("doses") or {}).items()) + f"): written once and compared with its reading · ×{e.get('gain', 1.0):.2f} now, ×{e.get('total', e.get('gain', 1.0)):.2f} on top of the levers' own scales" + (f" · agreement r = {e['corr']:.2f}" if e.get("corr") is not None else "") + "</p>")
			else:
				out.append(f"<p>lever <span style='color:{theme.TEXT}'>{esc(e.get('lever'))}</span> alone at {e.get('dose', 0):+.2f}: profile written once and read on the same prompts · reading scaled ×{e.get('gain', 1.0):.2f}" + (f" · agreement r = {e['corr']:.2f}" if e.get("corr") is not None else "") + "</p>")
			rows = "".join(f"<tr><td>{esc(n)}</td><td>{wr.get(n, 0):+.3f}</td><td>{rd.get(n, 0):+.3f}</td><td>{rd.get(n, 0) * e.get('gain', 1.0):+.3f}</td></tr>" for n in sorted(wr, key=lambda n: -abs(wr[n])))
			out.append(f"<table cellspacing='0' cellpadding='3'><tr>{''.join(f'<th align=left>{mute(h)}</th>' for h in ('trait', 'written', 'read', 'read × scale'))}</tr>{rows}</table>")
			out.append(f"<p>{mute('READ + WRITE: profiles with this lever are read and multiplied by this scale (a mix of levers uses their scales weighted by dose) · kept per model, measured again when the lever changes')}</p>")
		return "".join(out)

	def copy(self):
		e = self.current()
		if e is not None:
			QApplication.clipboard().setText(json.dumps(e, ensure_ascii=False, indent=1))

	def save(self):
		path, _f = QFileDialog.getSaveFileName(self, "Save the trace", os.path.join(config.DATA_DIR, time.strftime("trace_%Y%m%d_%H%M%S.json")), "JSON (*.json)")
		if path:
			with open(path, "w", encoding="utf-8") as f:
				json.dump(self.entries, f, ensure_ascii=False, indent=1)

	def clear(self):
		self.entries.clear()
		self.fill()
