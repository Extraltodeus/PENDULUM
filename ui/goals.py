import html
import os
import time

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QPainter, QPen, QTextCursor
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QScrollArea, QSpinBox, QSplitter, QTextBrowser, QVBoxLayout, QWidget

from persona import goals as GO
from persona import optimize as OPT
from ui import eva, theme
from ui.widgets import EvaDialog, Panel, button, guard_wheel

PLACEHOLDERS = {"who": "e.g. Bart, a grumpy old sailor who has sailed every sea", "want": "e.g. to find the treasure of his sunken ship", "me": "e.g. his cabin boy, whom he teases but protects"}
WEIGHTS = (("auto", "auto", "the fastest way that fits in the free VRAM, from the sizes of the model and estimated step times"), ("bf16", "bf16", "always a full bf16 copy: fastest steps, the most VRAM (gradient checkpointing or layers in RAM if needed)"), ("quant", "quantized", "the frozen weights stay quantized as in the GGUF and are unpacked on the fly: about 2.4 times slower steps, close to the GGUF size in VRAM"))
CARRIERS = (("vector", "vector", "one direction per layer: light, exported as a control vector"), ("lora", "LoRA", "small trained MLP weights: more precise for names and facts, exported as a LoRA adapter"), ("both", "vector + LoRA", "the two together"))
AS_LEVER_TIP = "make a regular lever from this goal's character (character ↔ default assistant), listed with the other levers (also in the right-click menu of a goal lever)\nthe persona direction the goal already has on this model moves to it; the goal itself is then active only through TRAIN or ALIGN"
TRAIN_CAP = "trains a goal lever on the character's answers · vector, LoRA or both (CARRIER) · a few minutes"
ALIGN_CAP = "a contrast lever (character ↔ default assistant), its strength found by reading and/or the judge (ALIGN BY)"
AS_LEVER_CAP = "a regular lever made from the character, listed with the others · built the first time you move it"
ALIGN_BY = (("read_judge", "READ + JUDGE", "the goal lever is read at many strengths (almost free: the model reads your target answers, nothing is written), then the judge checks only the best few strengths, with the coherence guard, and keeps the best one that still makes sense"), ("read", "READ", "the strength is picked by reading alone: the smallest one that reaches the REACH share of the strongest pull; then a normal CHECK (with the coherence guard, and the judge if there is one)\nworks without a judge and with models that refuse to write the targets: reading does not ask for their opinion"), ("judge", "JUDGE", "the former way: the model answers at 0 / 0.5 / 1 / 1.5 and the judge scores each answer (needs a judge)"))
JUDGE_TEXT = {"ready": "ready", "loading": "loading…", "rejected": "rejected by its self-test", "no_vram": "not enough free VRAM", "error": "error"}


class Rule(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setFixedWidth(9)

	def paintEvent(self, _e):
		p = QPainter(self)
		x = self.width() / 2
		p.setPen(eva.pen(theme.ACCENT, 60, 1.0, Qt.PenStyle.DotLine) if eva.on() else QPen(theme.qc(theme.LINE2), 1))
		p.drawLine(QPointF(x, 3), QPointF(x, self.height() - 3))
		p.end()


def pct(v):
	return "—" if v is None else f"{v:.0%}"


def score_color(v, good=0.66, bad=0.4):
	if v is None:
		return theme.DIM
	if eva.on():
		return theme.PHOSPHOR if v >= good else (theme.AMBER if v >= bad else theme.ALARM)
	return theme.OK if v >= good else (theme.WARN if v >= bad else theme.ERR)


def pull_tip(r):
	if r.get("pull_raw") is None:
		return ""
	text = f"log-probability of the target per token: {r['pull_raw']:+.2f} with the levers"
	if r.get("twin") is not None:
		text += f", {r['twin']:+.2f} with the character prompt (bare model = 0)"
	return html.escape(text)


def tip_edit(text="", hint=""):
	e = QLineEdit(text)
	e.setFont(theme.mono(8.5))
	e.setPlaceholderText(hint)
	e.setToolTip(text)
	e.textChanged.connect(e.setToolTip)
	return e


class QAEditor(QWidget):
	ROW_H = 30
	SHOWN = 8
	counted = Signal()

	def __init__(self, parent=None):
		super().__init__(parent)
		self.rows = []
		lay = QVBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(3)
		head = QHBoxLayout()
		title = QLabel("CUSTOM QUESTIONS   ·   asked as written, each with the answer you expect")
		theme.tag_font(title, 7.5)
		title.setStyleSheet(f"color: {theme.MUTED};")
		head.addWidget(title, 1)
		self.count = QLabel("")
		self.count.setFont(theme.mono(7.5))
		self.count.setStyleSheet(f"color: {theme.MUTED};")
		head.addWidget(self.count)
		b_add = button("+ QUESTION", "add a question and the answer you expect")
		b_add.clicked.connect(lambda: self.add_row().setFocus())
		self.b_bulk = button("BULK…", "all the questions as text, one per line: question | expected answer\nopens with the current ones written in it: edit them all at once, copy them, or paste new ones (handy to have an LLM write them)")
		self.b_bulk.setCheckable(True)
		self.b_bulk.toggled.connect(self._bulk)
		head.addWidget(b_add)
		head.addWidget(self.b_bulk)
		lay.addLayout(head)
		self.box = QWidget()
		self.box_lay = QVBoxLayout(self.box)
		self.box_lay.setContentsMargins(0, 0, 4, 0)
		self.box_lay.setSpacing(3)
		self.empty = QLabel("no custom question yet · + QUESTION, or BULK… to paste many")
		self.empty.setFont(theme.mono(8))
		self.empty.setStyleSheet(f"color: {theme.DIM};")
		self.box_lay.addWidget(self.empty)
		self.box_lay.addStretch(1)
		self.scroll = QScrollArea()
		self.scroll.setWidgetResizable(True)
		self.scroll.setFrameShape(QFrame.Shape.NoFrame)
		self.scroll.setWidget(self.box)
		lay.addWidget(self.scroll)
		self.bulk = QWidget()
		bl = QVBoxLayout(self.bulk)
		bl.setContentsMargins(0, 2, 0, 0)
		bl.setSpacing(3)
		self.text = QPlainTextEdit()
		self.text.setFont(theme.mono(8.5))
		self.text.setPlaceholderText("What is your ship called? | the Rusty Gull\nWhere were you born? | in Saint-Malo\n(one per line · => or a tab work too · list numbers and dashes are dropped)")
		self.text.setFixedHeight(110)
		bl.addWidget(self.text)
		row = QHBoxLayout()
		self.note = QLabel(self.NOTE)
		self.note.setFont(theme.mono(7.5))
		self.note.setStyleSheet(f"color: {theme.MUTED};")
		row.addWidget(self.note, 1)
		b_text = button("ROWS → TEXT", "write the current questions here again\nwhen the text box holds other lines, it asks first (and Ctrl+Z brings them back)")
		b_text.clicked.connect(self.to_text)
		b_copy = button("COPY", "copy the text to the clipboard")
		b_copy.clicked.connect(self.copy_text)
		b_rep = button("REPLACE ALL", "the questions become exactly these lines (the current ones are replaced)")
		b_rep.clicked.connect(self.replace_text)
		b_imp = button("ADD", "add these lines after the current questions (questions already there are skipped)", accent=True)
		b_imp.clicked.connect(self.import_text)
		for b in (b_text, b_copy, b_rep, b_imp):
			row.addWidget(b)
		bl.addLayout(row)
		self.bulk.setVisible(False)
		lay.addWidget(self.bulk)
		self._fit()

	NOTE = "one per line: question | expected answer"

	def _bulk(self, on):
		if on and self.rows and not self.text.toPlainText().strip():
			self.to_text()
		self.bulk.setVisible(on)
		self._fit()
		if on:
			self.text.setFocus()

	def _fit(self):
		n = len(self.rows)
		self.empty.setVisible(n == 0)
		self.scroll.setMinimumHeight(self.ROW_H + 2)
		self.count.setText(f"{n} question{'s' if n != 1 else ''}  " if n else "")
		self.counted.emit()

	def want_height(self):
		n = min(self.SHOWN // 2 if self.b_bulk.isChecked() else self.SHOWN, max(1, len(self.rows)))
		h = self.b_bulk.sizeHint().height() + 3 + n * self.ROW_H + 2
		if self.bulk.isVisible():
			h += self.bulk.sizeHint().height() + 3
		return h

	def add_row(self, q="", a=""):
		w = QWidget()
		h = QHBoxLayout(w)
		h.setContentsMargins(0, 0, 0, 0)
		h.setSpacing(4)
		qe = tip_edit(q, "question, e.g. What is your ship called?")
		ae = tip_edit(a, "expected answer, e.g. the Rusty Gull")
		x = button("X", "remove this question")
		x.setFixedWidth(30)
		h.addWidget(qe, 3)
		h.addWidget(ae, 2)
		h.addWidget(x)
		self.box_lay.insertWidget(len(self.rows), w)
		self.rows.append((w, qe, ae))
		x.clicked.connect(lambda _=False, w=w: self.remove(w))
		self._fit()
		QTimer.singleShot(0, lambda: self.scroll.ensureWidgetVisible(w))
		return qe

	def remove(self, w):
		self.rows = [r for r in self.rows if r[0] is not w]
		w.setParent(None)
		w.deleteLater()
		self._fit()

	def set_items(self, items):
		for w, _q, _a in self.rows:
			w.setParent(None)
			w.deleteLater()
		self.rows = []
		for c in items or []:
			self.add_row(c.get("q", ""), c.get("target", ""))
		self._fit()

	def items(self):
		out = []
		for _w, qe, ae in self.rows:
			q, a = qe.text().strip(), ae.text().strip()
			if q and a:
				out.append({"q": q, "target": a})
		return out

	def incomplete(self):
		return sum(1 for _w, qe, ae in self.rows if bool(qe.text().strip()) != bool(ae.text().strip()))

	def to_text(self):
		text = GO.custom_text({"custom": self.items()})
		now = self.text.toPlainText().strip()
		other = bool(now) and now != text.strip()
		if other:
			n = sum(1 for l in now.splitlines() if l.strip())
			box = QMessageBox(QMessageBox.Icon.Question, "Rows → text", f"The text box holds {n} line(s) that are not the current questions.\n\nReplace them with the current questions?\n(Ctrl+Z in the text box brings them back)", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, self)
			box.setStyleSheet(theme.STYLE)
			box.setDefaultButton(QMessageBox.StandardButton.Cancel)
			if box.exec() != QMessageBox.StandardButton.Yes:
				return
		cur = self.text.textCursor()
		cur.select(QTextCursor.SelectionType.Document)
		cur.insertText(text)
		self.say((f"{len(self.items())} questions written above · edit them, then REPLACE ALL" + (" · Ctrl+Z brings the previous text back" if other else "")) if self.items() else self.NOTE)

	def say(self, text, color=None):
		self.note.setText(text)
		self.note.setStyleSheet(f"color: {color or theme.MUTED};")

	def copy_text(self):
		text = self.text.toPlainText()
		QGuiApplication.clipboard().setText(text)
		self.say(f"{sum(1 for l in text.splitlines() if l.strip())} lines copied", theme.OK)

	def replace_text(self):
		items, left, seen = [], [], set()
		for line in [l for l in self.text.toPlainText().splitlines() if l.strip()]:
			qa = GO.split_qa(line)
			if qa is None:
				left.append(line)
			elif qa[0].lower() not in seen:
				seen.add(qa[0].lower())
				items.append({"q": qa[0], "target": qa[1]})
		if not items:
			self.say("nothing to use: write lines like question | expected answer", theme.WARN)
			return
		self.set_items(items)
		self.text.setPlainText(chr(10).join(left))
		self.say(f"{len(items)} questions" + (f" · {len(left)} line{'s' if len(left) != 1 else ''} without a | separator, left above" if left else ""), theme.WARN if left else theme.OK)

	def import_text(self):
		lines = [l for l in self.text.toPlainText().splitlines() if l.strip()]
		have = {c["q"].lower() for c in self.items()}
		added = dup = 0
		left = []
		for line in lines:
			qa = GO.split_qa(line)
			if qa is None:
				left.append(line)
				continue
			if qa[0].lower() in have:
				dup += 1
				continue
			have.add(qa[0].lower())
			self.add_row(*qa)
			added += 1
		self.text.setPlainText(chr(10).join(left))
		self.say(f"{added} added" + (f" · {dup} already there" if dup else "") + (f" · {len(left)} line{'s' if len(left) != 1 else ''} without a | separator, left above" if left else ""), theme.WARN if left else theme.OK)


class GoalsDialog(EvaDialog):
	saved = Signal(object)
	compare_changed = Signal(bool)
	read_changed = Signal(object)
	check = Signal(object)
	align = Signal(object)
	optimize = Signal(object)
	delete = Signal(str)
	stop = Signal()
	as_lever = Signal(str)

	def __init__(self, goals, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Goals")
		self.setStyleSheet(theme.STYLE)
		self.setMinimumSize(860, 640)
		self.goals = list(goals)
		self.current = ""
		self.busy = False
		self.trials = []
		lay = QVBoxLayout(self)
		lay.setSpacing(6)
		lay.addWidget(self.frameless("GOALS", "WHO ARE YOU?  WHAT DO YOU WANT?  WHO AM I TO YOU?"))
		info = QLabel("Write who the model should be, then TRAIN: the model, prompted with your character, answers a few dozen questions (the judge keeps its best answers), then a goal lever is trained so that the bare model answers like it, and the judge checks the result. It takes a few minutes (a trainable copy of the model is loaded in torch meanwhile). CHECK only asks the questions with the active levers: the judge checks that each answer makes sense, then that it matches your target. ALIGN is the older, lighter way: a contrast lever whose strength is picked by reading your targets and/or from the judge's scores (ALIGN BY). PULL: how much the active levers make the model read your targets as likely, next to what your character prompt does.")
		info.setFont(theme.mono(7.5))
		info.setStyleSheet(f"color: {theme.MUTED};")
		info.setWordWrap(True)
		lay.addWidget(info)
		self.judge_label = QLabel("")
		self.judge_label.setFont(theme.mono(8))
		self.judge_label.setWordWrap(True)
		lay.addWidget(self.judge_label)
		row = QHBoxLayout()
		self.pick = guard_wheel(QComboBox())
		self.pick.setFont(theme.mono(8.5))
		self.pick.setMinimumWidth(220)
		self.pick.activated.connect(self._picked)
		b_new = button("NEW", "start a new goal")
		b_new.clicked.connect(self.new)
		self.b_dup = button("DUPLICATE", "save a copy of this goal under a new name (what is written below, unsaved edits included) and switch to it")
		self.b_dup.clicked.connect(self._duplicate)
		self.b_ren = button("RENAME…", "rename this goal (only its name: its lever and results stay)")
		self.b_ren.clicked.connect(self._rename)
		self.name = QLineEdit()
		self.name.setFont(theme.mono(8.5))
		self.name.setPlaceholderText("goal name, e.g. Captain Bart")
		for w, s in ((self._label("GOAL"), 0), (self.pick, 2), (b_new, 0), (self.b_dup, 0), (self.b_ren, 0), (self._label("NAME"), 0), (self.name, 3)):
			row.addWidget(w, s)
		lay.addLayout(row)
		self.fields = {}
		for key in GO.KEYS:
			lay.addWidget(self._label(GO.LABELS[key].upper()))
			e = QPlainTextEdit()
			e.setFont(theme.mono(8.5))
			e.setPlaceholderText(PLACEHOLDERS[key])
			e.setFixedHeight(46)
			lay.addWidget(e)
			self.fields[key] = e
		self.qa = QAEditor()
		self.split = QSplitter(Qt.Orientation.Vertical)
		self.split.setChildrenCollapsible(False)
		self.split.setHandleWidth(7)
		self.split.addWidget(self.qa)
		low = QWidget()
		lay_low = QVBoxLayout(low)
		lay_low.setContentsMargins(0, 2, 0, 0)
		lay_low.setSpacing(6)
		self.split.addWidget(low)
		self.split.setStretchFactor(1, 1)
		self.split.splitterMoved.connect(self._qa_moved)
		self.split.handle(1).setToolTip("drag to give the custom questions more or less room")
		self.qa_sized = False
		self.qa_bulk = False
		self.qa.counted.connect(lambda: QTimer.singleShot(0, self._qa_fit))
		lay.addWidget(self.split, 1)
		btns = QHBoxLayout()
		self.b_delete = button("DELETE", "delete this goal (its lever too)")
		self.b_delete.clicked.connect(self._delete)
		self.b_stop = button("STOP", "stop the check or the alignment (Esc in the main window)")
		self.b_stop.clicked.connect(self.stop.emit)
		self.b_save = button("SAVE", "save the goal (changing a target rebuilds its lever)")
		self.b_save.clicked.connect(self._save)
		self.b_check = button("CHECK", "ask the three questions with the active levers and let the judge score the answers")
		self.b_check.clicked.connect(lambda: self._run(self.check))
		self.b_align = button("ALIGN", "lighter: build a contrast lever (your character against the default assistant), try it at several strengths and keep the best one that still makes sense\nhow the strength is picked: ALIGN BY below (READ + JUDGE, READ or JUDGE)")
		self.b_align.clicked.connect(lambda: self._run(self.align))
		self.b_optimize = button("TRAIN", "train a goal lever so that the bare model answers like the model prompted with your character, then check it (a few minutes)", accent=True)
		self.b_optimize.clicked.connect(self._optimize)
		self.b_as_lever = button("AS A LEVER", AS_LEVER_TIP)
		self.b_as_lever.clicked.connect(self._as_lever)
		self.twins = {}
		self.o_layers = guard_wheel(QComboBox())
		self.o_layers.addItems(list(OPT.LAYER_MODES))
		self.o_layers.setToolTip("layers the goal is trained on, for the vector and the LoRA alike:\nearly = the first 9 layers of the workspace band (best for a vector in the tests)\nband = the whole band (default for a LoRA: more room for names and facts, about the same training time)\nmiddle = 9 layers around its middle\nchanging the carrier picks its default until you choose the layers yourself")
		self.layers_auto = True
		self.o_layers.activated.connect(self._layers_picked)
		self.o_steps = guard_wheel(QSpinBox())
		self.o_steps.setRange(20, 1000)
		self.o_steps.setSingleStep(10)
		self.o_steps.setValue(OPT.DEFAULTS["steps"])
		self.o_steps.setToolTip("training steps (~2 s each on a 4B model)")
		self.o_lam = guard_wheel(QDoubleSpinBox())
		self.o_lam.setRange(0.0, 10.0)
		self.o_lam.setSingleStep(0.1)
		self.o_lam.setDecimals(2)
		self.o_lam.setValue(OPT.DEFAULTS["lam"])
		self.o_lam.setToolTip("PRESERVATION: how hard the training protects the model's normal behavior\nwhile the goal is learned, a neutral text that has nothing to do with the character must keep being predicted as before:\nthis number weighs that penalty (the drift on the neutral text, a KL divergence) against learning the character's answers\nhigher = the model stays itself outside the role (fewer side effects), but the goal sinks in less\nlower = the goal sinks in more, with more risk of odd or off-topic text\n1 = default · LoRA: used as set · vector: starting value, then adjusted during training to hold the drift near a fixed level")
		self.o_samples = guard_wheel(QSpinBox())
		self.o_samples.setRange(1, 6)
		self.o_samples.setValue(OPT.DEFAULTS["samples"])
		self.o_samples.setToolTip("answers of the character kept for each teaching prompt\nwith a judge, the identity questions get one more draw and the judge keeps the best ones")
		self.o_carrier = guard_wheel(QComboBox())
		for key, label, tip in CARRIERS:
			self.o_carrier.addItem(label, key)
			self.o_carrier.setItemData(self.o_carrier.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
		self.o_carrier.setCurrentIndex(max(0, self.o_carrier.findData(OPT.DEFAULTS["carrier"])))
		self.o_layers.setCurrentText(OPT.LAYER_DEFAULT[OPT.DEFAULTS["carrier"]])
		self.o_carrier.currentIndexChanged.connect(self._carrier_changed)
		self.o_carrier.setToolTip("what carries the goal:\nvector: one direction per layer, added to the model's thoughts (light, exported as a control vector)\nLoRA: small trained weights in the MLP of the same layers (more precise for names and facts, exported as a LoRA adapter)\nboth: the two together")
		self.o_weights = guard_wheel(QComboBox())
		for key, label, tip in WEIGHTS:
			self.o_weights.addItem(label, key)
			self.o_weights.setItemData(self.o_weights.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
		self.o_weights.setToolTip("how the trainable copy of the model holds its weights:\nauto = the fastest way that fits in the free VRAM (default)\nbf16 = full bf16 copy: fastest steps, needs the most VRAM\nquantized = frozen weights kept quantized as in the GGUF and unpacked on the fly: about 2.4 times slower steps, but close to the GGUF size in VRAM (the llama.cpp copy can often stay loaded)\nthe result is the same, only speed and memory change")
		for w in (self.o_layers, self.o_steps, self.o_lam, self.o_samples, self.o_carrier, self.o_weights):
			w.setFont(theme.mono(8))
		self.compare = QCheckBox("compare with the bare model")
		self.compare.setFont(theme.mono(8))
		self.compare.setToolTip("CHECK with active levers also asks the model without them, and the judge says which answer matches your target better (easier for small judges)")
		self.compare.toggled.connect(self.compare_changed.emit)
		self.r_pull = QCheckBox("PULL: read your targets")
		self.r_pull.setFont(theme.mono(8))
		self.r_pull.setChecked(GO.READ_DEFAULTS["pull"])
		self.r_pull.setToolTip("CHECK also reads each target answer with the active levers and the system prompt (almost free: nothing is written)\nPULL = how much more likely your target becomes than with the bare model, as a share of how much your character prompt makes it likely (100% = as much as the model prompted with the character)\nit shows the direction even when the answers do not say it yet, and a model that refuses to write something still reads it")
		self.r_align = guard_wheel(QComboBox())
		for key, label, tip in ALIGN_BY:
			self.r_align.addItem(label, key)
			self.r_align.setItemData(self.r_align.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
		self.r_align.setToolTip("how ALIGN picks the strength of the goal lever:\n" + "\n".join(f"{label}: {tip}" for _k, label, tip in ALIGN_BY))
		self.r_step = guard_wheel(QSpinBox())
		self.r_step.setRange(1, 8)
		self.r_step.setValue(GO.READ_DEFAULTS["halvings"])
		self.r_step.setToolTip("READ and READ + JUDGE seek the strength of the goal lever: it is read at 1, then doubled (2, 4, 8…) until it pulls REACH of the character's pull, pulls less than the step before, or the replies break (reading drop past the coherence guard's READ threshold)\nthe answer is then between the last two: this many more readings narrow it down (more = finer, ~1-2 s each)\nthe strength found is folded into the lever: dose 1 then means that strength")
		self.r_verify = guard_wheel(QSpinBox())
		self.r_verify.setRange(1, 6)
		self.r_verify.setValue(GO.READ_DEFAULTS["verify"])
		self.r_verify.setToolTip("READ + JUDGE: how many strengths the judge checks, from half the strength found by reading up to it")
		self.r_reach = guard_wheel(QSpinBox())
		self.r_reach.setRange(30, 100)
		self.r_reach.setSuffix(" %")
		self.r_reach.setValue(int(GO.READ_DEFAULTS["reach"] * 100))
		self.r_reach.setToolTip("how much the goal lever must pull, as a share of the pull of the model prompted with your character (100% = as much)\nthe seek keeps the smallest strength that reaches it; when the pull fades or the replies break first, the strength of the strongest pull\nlower = gentler levers, higher = closer to the target (the pull keeps rising even when the answers start to break: the reading drop, the judge and the guard watch that)")
		for w in (self.r_align, self.r_step, self.r_verify, self.r_reach):
			w.setFont(theme.mono(8))
		self.r_pull.toggled.connect(self._read_changed)
		self.r_align.currentIndexChanged.connect(self._read_changed)
		for w in (self.r_step, self.r_verify, self.r_reach):
			w.valueChanged.connect(self._read_changed)
		self.reading = False
		btns.addWidget(self.b_delete)
		btns.addSpacing(8)
		btns.addWidget(self.b_save)
		btns.addStretch(1)
		btns.addWidget(self.compare)
		btns.addSpacing(12)
		btns.addWidget(self.r_pull)
		btns.addSpacing(8)
		btns.addWidget(self.b_check)
		btns.addWidget(self.b_stop)
		lay_low.addLayout(btns)
		self.add_box = Panel("ADD LEVERS")
		self.add_box.set_info("three ways to turn this goal into a lever")
		trio = QHBoxLayout()
		trio.setSpacing(12)
		train = ((("CARRIER", self.o_carrier), ("STEPS", self.o_steps)), (("LAYERS", self.o_layers), ("ANSWERS", self.o_samples)), (("WEIGHTS", self.o_weights), ("PRESERVATION", self.o_lam)))
		align = ((("ALIGN BY", self.r_align), ("REACH", self.r_reach)), (("HALVINGS", self.r_step), ("JUDGED", self.r_verify)))
		self.caps = {}
		for i, (b, text, rows, stretch) in enumerate(((self.b_optimize, TRAIN_CAP, train, 5), (self.b_align, ALIGN_CAP, align, 4), (self.b_as_lever, AS_LEVER_CAP, (), 3))):
			if i:
				trio.addWidget(Rule())
			col = QVBoxLayout()
			col.setSpacing(4)
			col.addWidget(b)
			cap = QLabel(text)
			cap.setFont(theme.mono(7.5))
			cap.setStyleSheet(f"color: {theme.MUTED};")
			cap.setWordWrap(True)
			col.addWidget(cap)
			col.addSpacing(4)
			self.caps[b] = cap
			form = QGridLayout()
			form.setHorizontalSpacing(6)
			form.setVerticalSpacing(4)
			for r, pairs in enumerate(rows):
				for k, (label, w) in enumerate(pairs):
					l = self._label(label)
					l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
					form.addWidget(l, r, k * 2)
					form.addWidget(w, r, k * 2 + 1)
					if isinstance(w, QComboBox):
						w.setMinimumWidth(w.sizeHint().width())
			form.setColumnStretch(1, 3)
			form.setColumnStretch(3, 2)
			col.addLayout(form)
			col.addStretch(1)
			trio.addLayout(col, stretch)
		tall = max(b.sizeHint().height() for b in self.caps)
		for b in self.caps:
			b.setMinimumHeight(tall)
		self.add_box.body.addLayout(trio)
		lay_low.addWidget(self.add_box)
		self.scan = []
		self.status = QLabel("")
		self.status.setFont(theme.mono(8.5, True))
		self.status.setWordWrap(True)
		lay_low.addWidget(self.status)
		self.trials_label = QLabel("")
		self.trials_label.setFont(theme.mono(8))
		self.trials_label.setWordWrap(True)
		self.trials_label.setStyleSheet(f"color: {theme.MUTED};")
		lay_low.addWidget(self.trials_label)
		self.rows = []
		self.view = QTextBrowser()
		self.view.setOpenLinks(False)
		self.view.setFont(theme.mono(8))
		self.view.setMinimumHeight(80)
		lay_low.addWidget(self.view, 1)
		self.set_goals(self.goals)
		self.set_judge({"state": "off"}, "")
		self._enable()
		room = QGuiApplication.primaryScreen().availableGeometry().height() - 60
		self.resize(1080, max(640, min(900, room)))

	def _qa_moved(self, *_a):
		self.qa_sized = True

	def _qa_fit(self):
		total = sum(self.split.sizes())
		if total <= 0:
			return
		bulk = self.qa.b_bulk.isChecked()
		if self.qa_sized and bulk == self.qa_bulk:
			return
		self.qa_bulk = bulk
		top = max(self.qa.minimumSizeHint().height(), min(self.qa.want_height(), total - 220))
		self.split.setSizes([top, total - top])

	def showEvent(self, e):
		super().showEvent(e)
		QTimer.singleShot(0, self._qa_fit)

	def free_id(self, name, skip=""):
		base = GO.make(name, "", "", "")["id"]
		taken = {g["id"] for g in self.goals if g["id"] != skip}
		gid, k = base, 2
		while gid in taken:
			gid, k = f"{base}_{k}", k + 1
		return gid

	def _duplicate(self):
		v = self.values()
		if not v["name"]:
			self.set_status("give the goal a name first", theme.WARN)
			return
		names = {g["name"].strip().lower() for g in self.goals}
		name, k = f"{v['name']} copy", 2
		while name.strip().lower() in names:
			name, k = f"{v['name']} copy {k}", k + 1
		new = {**v, "name": name, "id": self.free_id(name)}
		if not self.valid(new):
			return
		self.current = new["id"]
		self.name.setText(name)
		self.saved.emit(new)
		self.set_rows([])
		self.trials = []
		self.trials_label.setText("")
		self.set_status(f"copy saved as {name}: the original stays as it was", theme.OK)

	def _rename(self):
		g = next((g for g in self.goals if g["id"] == self.current), None)
		if g is None:
			self.set_status("nothing to rename yet: save the goal first (its name is the NAME field)", theme.WARN)
			return
		dlg = QInputDialog(self)
		dlg.setWindowTitle("Rename goal")
		dlg.setLabelText(f"new name for the goal '{g['name']}'")
		dlg.setTextValue(g["name"])
		dlg.setStyleSheet(theme.STYLE)
		if not dlg.exec():
			return
		name = dlg.textValue().strip()
		if not name or name == g["name"]:
			return
		if any(x["name"].strip().lower() == name.lower() and x["id"] != g["id"] for x in self.goals):
			self.set_status(f"another goal is already called {name}", theme.WARN)
			return
		self.name.setText(name)
		self.saved.emit({**g, "name": name, "custom": list(g.get("custom") or [])})
		self.set_status(f"renamed to {name} (lever and results kept)", theme.OK)

	def _label(self, text):
		l = QLabel(text)
		theme.tag_font(l, 7.5)
		l.setStyleSheet(f"color: {theme.MUTED};")
		return l

	def restyle(self):
		self.setStyleSheet(theme.STYLE)
		theme.refont(self)
		self.fit_margins()
		self.render()
		self.update()

	def values(self):
		v = {"id": self.current, "name": self.name.text().strip(), **{k: e.toPlainText().strip() for k, e in self.fields.items()}, "custom": self.qa.items()}
		if not v["id"]:
			v["id"] = self.free_id(v["name"]) if v["name"] else ""
		return v

	def valid(self, v):
		if not v["name"]:
			self.set_status("give the goal a name", theme.WARN)
			return False
		if self.qa.incomplete():
			self.set_status(f"{self.qa.incomplete()} custom question(s) without their answer (or the reverse): fill them in or remove them", theme.WARN)
			return False
		if not any(v[k] for k in GO.KEYS) and not v["custom"]:
			self.set_status("write at least one target answer", theme.WARN)
			return False
		return True

	def dirty(self, v):
		g = next((g for g in self.goals if g["id"] == v["id"]), None)
		return g is None or any((g.get(k) or "") != v[k] for k in ("name",) + GO.KEYS) or (g.get("custom") or []) != v["custom"]

	def _save(self):
		v = self.values()
		if self.valid(v):
			self.current = v["id"]
			self.saved.emit(v)

	def _run(self, signal):
		v = self.values()
		if not self.valid(v):
			return
		if self.dirty(v):
			self.saved.emit(v)
		self.current = v["id"]
		self._select(self.current)
		self.set_rows([])
		self.trials = []
		self.scan = []
		self.trials_label.setText("")
		self.set_status("asking the model…", theme.ACCENT2)
		signal.emit(v)

	def read_options(self):
		return {"pull": self.r_pull.isChecked(), "align": self.r_align.currentData() or "read_judge", "halvings": int(self.r_step.value()), "verify": int(self.r_verify.value()), "reach": round(self.r_reach.value() / 100.0, 3)}

	def set_read_options(self, opts):
		r = {**GO.READ_DEFAULTS, **(opts or {})}
		self.reading = True
		self.r_pull.setChecked(bool(r["pull"]))
		self.r_align.setCurrentIndex(max(0, self.r_align.findData(r["align"])))
		self.r_step.setValue(int(r["halvings"]))
		self.r_verify.setValue(int(r["verify"]))
		self.r_reach.setValue(int(round(float(r["reach"]) * 100)))
		self.reading = False
		self._read_enable()

	def _read_enable(self):
		mode = self.r_align.currentData()
		self.r_step.setEnabled(mode != "judge")
		self.r_verify.setEnabled(mode == "read_judge")
		self.r_reach.setEnabled(mode != "judge")

	def _read_changed(self, *_a):
		self._read_enable()
		if not self.reading:
			self.read_changed.emit(self.read_options())

	def _layers_picked(self, _i=0):
		self.layers_auto = self.o_layers.currentText() == OPT.LAYER_DEFAULT.get(self.o_carrier.currentData() or "lora")

	def _carrier_changed(self, _i=0):
		if self.layers_auto:
			self.o_layers.setCurrentText(OPT.LAYER_DEFAULT.get(self.o_carrier.currentData() or "lora", "early"))

	def options(self):
		return {"layers": self.o_layers.currentText(), "layers_auto": self.layers_auto, "steps": int(self.o_steps.value()), "lam": round(float(self.o_lam.value()), 3), "samples": int(self.o_samples.value()), "draws": int(self.o_samples.value()) + 1, "carrier": self.o_carrier.currentData() or "vector", "weights": self.o_weights.currentData() or "auto"}

	def set_options(self, opts):
		if not opts:
			return
		self.layers_auto = bool(opts.get("layers_auto", opts.get("layers") in (None, "", "early")))
		if opts.get("carrier") in OPT.CARRIERS:
			self.o_carrier.setCurrentIndex(max(0, self.o_carrier.findData(opts["carrier"])))
		if self.o_weights.findData(opts.get("weights")) >= 0:
			self.o_weights.setCurrentIndex(self.o_weights.findData(opts["weights"]))
		if not self.layers_auto and opts.get("layers") in OPT.LAYER_MODES:
			self.o_layers.setCurrentText(opts["layers"])
		else:
			self._carrier_changed()
		for key, w in (("steps", self.o_steps), ("lam", self.o_lam), ("samples", self.o_samples)):
			if opts.get(key) is not None:
				w.setValue(opts[key])

	def _optimize(self):
		v = self.values()
		if not self.valid(v):
			return
		if self.dirty(v):
			self.saved.emit(v)
		self.current = v["id"]
		self._select(self.current)
		self.set_rows([])
		self.trials = []
		self.trials_label.setText("")
		self._t_step = None
		self.set_status("the character answers the teaching prompts…", theme.ACCENT2)
		self.optimize.emit({**v, "options": self.options()})

	def _delete(self):
		g = next((g for g in self.goals if g["id"] == self.current), None)
		if g is None:
			self.new()
			return
		if QMessageBox.question(self, "Delete goal", f"Delete the goal '{g['name']}' and its lever?") != QMessageBox.StandardButton.Yes:
			return
		self.delete.emit(g["id"])
		self.new()

	def new(self):
		self.current = ""
		self.pick.setCurrentIndex(-1)
		self.name.clear()
		for e in self.fields.values():
			e.clear()
		self.qa.set_items([])
		self.set_rows([])
		self.status.setText("")
		self.trials_label.setText("")
		self._enable()

	def show_goal(self, gid):
		i = next((i for i, g in enumerate(self.goals) if g["id"] == gid), -1)
		if i < 0 or gid == self.current:
			return
		v = self.values()
		if (v["name"] or any(v[k] for k in GO.KEYS) or v["custom"]) and self.dirty(v):
			box = QMessageBox(QMessageBox.Icon.Question, "Open a goal", f"'{v['name'] or 'this goal'}' has unsaved edits.\n\nOpen '{self.goals[i]['name']}' and drop them?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, self)
			box.setStyleSheet(theme.STYLE)
			box.setDefaultButton(QMessageBox.StandardButton.Cancel)
			if box.exec() != QMessageBox.StandardButton.Yes:
				return
		self._select(gid)
		self._picked(i)

	def _select(self, gid):
		self.pick.blockSignals(True)
		self.pick.setCurrentIndex(next((i for i, g in enumerate(self.goals) if g["id"] == gid), -1))
		self.pick.blockSignals(False)

	def _picked(self, i):
		gid = self.pick.itemData(i)
		g = next((g for g in self.goals if g["id"] == gid), None)
		if g is None:
			return
		self.current = g["id"]
		self.name.setText(g["name"])
		for k, e in self.fields.items():
			e.setPlainText(g.get(k) or "")
		self.qa.set_items(g.get("custom") or [])
		self.set_rows([])
		self.status.setText("")
		self.trials_label.setText("")
		self._enable()

	def set_goals(self, goals):
		self.goals = list(goals)
		self.pick.blockSignals(True)
		self.pick.clear()
		for g in self.goals:
			self.pick.addItem(g["name"], g["id"])
			self.pick.setItemData(self.pick.count() - 1, "\n".join(f"{GO.LABELS[k]}  {g.get(k) or '—'}" for k in GO.KEYS), Qt.ItemDataRole.ToolTipRole)
		idx = next((i for i, g in enumerate(self.goals) if g["id"] == self.current), -1)
		self.pick.setCurrentIndex(idx)
		self.pick.blockSignals(False)
		if idx < 0 and self.current and not self.name.text():
			self.current = ""
		if idx >= 0 and not self.name.text():
			self._picked(idx)
		self._enable()

	def set_judge(self, state, configured=""):
		st = (state or {}).get("state", "off")
		name = (state or {}).get("name") or os.path.basename(configured or "")
		if st == "off" and not configured:
			text, col = "JUDGE  none — pick one in ⚙ settings · coherence. Without a judge, CHECK only shows the answers.", theme.WARN
		elif st == "off":
			text, col = f"JUDGE  {name} · not loaded yet (it loads on the first CHECK or lever change)", theme.MUTED
		else:
			parts = [f"JUDGE  {name}", JUDGE_TEXT.get(st, st)]
			if state.get("gpu") is not None and st in ("ready", "loading"):
				parts.insert(1, f"CUDA{state['gpu']}")
			if state.get("auc") is not None:
				parts.append(f"self-test AUC {state['auc']:.2f}")
			if state.get("size"):
				parts.append(f"{state['size'] / 2**30:.1f} GB")
			if state.get("error"):
				parts.append(str(state["error"])[:120])
			text, col = " · ".join(parts), {"ready": theme.OK, "loading": theme.ACCENT2}.get(st, theme.ERR if st in ("rejected", "error") else theme.WARN)
		self.judge_label.setText(text)
		self.judge_label.setStyleSheet(f"color: {col};")

	def set_busy(self, busy):
		self.busy = bool(busy)
		self._enable()

	def _enable(self):
		self.b_stop.setEnabled(self.busy)
		self.b_ren.setEnabled(any(g["id"] == self.current for g in self.goals))
		twin = self.twins.get(self.current) if self.current else None
		self.b_as_lever.setEnabled(not twin)
		self.b_as_lever.setToolTip(f"its lever is {twin} in LEVERS" if twin else AS_LEVER_TIP)
		self.caps[self.b_as_lever].setText(f"already a lever: {twin} in LEVERS" if twin else AS_LEVER_CAP)

	def set_twins(self, twins):
		self.twins = dict(twins or {})
		self._enable()

	def _as_lever(self):
		v = self.values()
		if not self.valid(v):
			return
		if self.dirty(v):
			self.saved.emit(v)
		self.current = v["id"]
		self._select(self.current)
		self.as_lever.emit(v["id"])
		self.set_status(f"{v['name']}: made a regular lever (LEVERS), built the first time you move it", theme.OK)

	def set_status(self, text, color=None):
		self.status.setText(text)
		self.status.setStyleSheet(f"color: {color or theme.TEXT};")

	def add_row(self, row):
		self.rows.append(row)
		self.render()

	def set_rows(self, rows):
		self.rows = list(rows)
		self.render()

	def render(self):
		if not self.rows:
			self.view.clear()
			return
		cell = "padding: 3px 6px; vertical-align: top;" + (f" border-bottom: 1px solid {theme.LINE};" if eva.on() else "")
		cmp = any(r.get("vs_base") is not None for r in self.rows)
		pull = any("pull_raw" in r for r in self.rows)
		models = list(dict.fromkeys(r["model"] for r in self.rows if r.get("model")))
		head = "".join(f'<th align="left" style="{cell} color: {theme.NEON if eva.on() else theme.MUTED}; font-weight: normal;"{tip}>{h}</th>' for h, tip in (("", ""), ("QUESTION", ""), ("ANSWER", ""), ("SENSE", ""), ("MATCH", "")) + ((("VS BARE", ""),) if cmp else ()) + ((("PULL", ' title="how much more likely the target answer becomes with the active levers and system prompt, as a share of what the character prompt does (100% = as much)"'),) if pull else ()))
		body = []
		for r in self.rows:
			sensible = r["sense"] is None or r["sense"] >= GO.SENSE_MIN
			reply = html.escape(" ".join(r["reply"].split()) or "(empty)")
			body.append("<tr>" + f'<td style="{cell} color: {theme.MUTED}; white-space: nowrap;">{r["key"]}</td>' + f'<td style="{cell} color: {theme.TEXT}; white-space: nowrap;">{html.escape(r["question"])}</td>' + f'<td style="{cell} color: {theme.TEXT if sensible else theme.MUTED};">{reply}</td>' + f'<td style="{cell} color: {score_color(r["sense"], 0.8, 0.6)};">{pct(r["sense"])}</td>' + f'<td style="{cell} color: {score_color(r["match"] if sensible else 0.0)};">{pct(r["match"])}</td>' + (f'<td style="{cell} color: {score_color(r.get("vs_base"), 0.6, 0.45)};" title="{html.escape(" ".join((r.get("base") or "").split()))}">{pct(r.get("vs_base"))}</td>' if cmp else "") + (f'<td style="{cell} color: {score_color(r.get("pull"), 0.6, 0.25)}; white-space: nowrap;" title="{pull_tip(r)}">{pct(r.get("pull"))}</td>' if pull else "") + "</tr>")
		top = f'<div style="color: {theme.NEON if eva.on() else theme.ACCENT}; padding: 2px 6px 4px 6px;">MODEL&nbsp;&nbsp;{html.escape(" · ".join(models))}</div>' if models else ""
		self.view.setHtml(f'{top}<table width="100%" cellspacing="0" style="border-collapse: collapse;"><tr>{head}</tr>{"".join(body)}</table>')

	def add_trial(self, t):
		self.trials.append(t)
		self.set_rows(t.get("rows") or [])
		self._trials_text()
		self.set_status(f"goal lever {t['strength']:+.2f}: match {pct(t.get('match'))} · sense {pct(t.get('sense'))}…", theme.ACCENT2)

	def set_scan(self, p):
		if p.get("goal") != self.current:
			return
		self.scan = list(p.get("scan") or [])
		self._trials_text()
		if not p.get("done") and self.scan:
			s = self.scan[-1]
			self.set_status(f"reading the targets: goal lever {s['strength']:+.2f} → pull {pct(s.get('pull'))}…", theme.ACCENT2)

	def _scan_text(self, chosen=None):
		if not self.scan:
			return ""
		top = max((s["pull"] for s in self.scan if s.get("pull") is not None), default=None)
		parts = []
		for s in self.scan:
			mark = "✓" if chosen is not None and abs(s["strength"] - chosen) < 1e-6 else ""
			parts.append(f"×{s['strength']:g} {pct(s.get('pull'))}" + (" broken" if s.get("broken") else "") + mark)
		return "READ  " + " · ".join(parts) + (f"   (strongest {pct(top)})" if top is not None else "")

	def _trials_text(self, chosen=None):
		parts = []
		for t in self.trials:
			mark = "  ✓" if chosen is not None and abs(t["strength"] - chosen) < 1e-6 else ""
			if t.get("sense") is not None and t["sense"] < GO.SENSE_MIN:
				parts.append(f"{t['strength']:+.2f} → nonsense{mark}")
			else:
				parts.append(f"{t['strength']:+.2f} → {pct(t.get('match'))}" + (f" (guard ×{t['scale']:.2f})" if t["strength"] and t.get("scale", 1) < 0.999 else "") + mark)
		lines = [x for x in (self._scan_text(chosen), ("TRIED  " + "   ·   ".join(parts)) if parts else "") if x]
		self.trials_label.setText("\n".join(lines))
		self._fit_trials()

	def _fit_trials(self):
		w = self.trials_label.width() if self.trials_label.width() > 50 else max(400, self.width() - 40)
		self.trials_label.setMinimumHeight(self.trials_label.heightForWidth(w) if self.trials_label.text() else 0)

	def resizeEvent(self, e):
		super().resizeEvent(e)
		if hasattr(self, "trials_label"):
			QTimer.singleShot(0, self._fit_trials)

	def set_result(self, res):
		if res.get("goal") != self.current:
			return
		self.set_rows(res.get("rows") or [])
		if res.get("match") is None:
			self.set_status("answers only: no judge to score them", theme.WARN)
			return
		lever = f" · goal lever {res['strength']:+.2f}" + (f" (guard ×{res['scale']:.2f})" if res["strength"] and res.get("scale", 1) < 0.999 else "") if res.get("strength") else " · goal lever off"
		others = res.get("levers") or {}
		extra = f" · other levers: {', '.join(f'{k} {v:+.2f}' for k, v in others.items())}" if others else ""
		vs = (f"   VS BARE {pct(res['vs_base'])}" if res.get("vs_base") is not None else "") + (f"   PULL {pct(res['pull'])}" if res.get("pull") is not None else "")
		self.set_status((f"{res['model']}   " if res.get("model") else "") + f"MATCH {pct(res['match'])}   SENSE {pct(res['sense'])}" + vs + (f"   {res['nonsense']} nonsense answer(s)" if res.get("nonsense") else "") + lever + extra, score_color(res["match"]))

	def set_step(self, p):
		if p.get("goal") != self.current:
			return
		now = time.time()
		if p["step"] <= 1 or not getattr(self, "_t_step", None):
			self._t_step = (now, p["step"])
		t0, s0 = self._t_step
		left = (now - t0) / (p["step"] - s0) * (p["steps"] - p["step"]) if p["step"] > s0 else None
		eta = f" · about {left / 60:.0f} min left" if left is not None and left >= 90 else (f" · about {left:.0f} s left" if left is not None else "")
		self.set_status(f"training the goal lever: step {p['step']}/{p['steps']}{eta}", theme.ACCENT2)
		self.trials_label.setText(f"TRAINING  fit to the character {p['ce']:.2f}  ·  drift on neutral text {p['kl']:.3f}")
		self._fit_trials()

	def set_optimized(self, p):
		if p.get("goal") != self.current:
			return
		t = p.get("trained") or {}
		closed = p.get("closed")
		carrier = {"lora": f"LoRA rank {t.get('rank')}", "both": f"vector + LoRA rank {t.get('rank')}"}.get(t.get("carrier"), "vector")
		parts = [f"TRAINED  {carrier} on L{p['layers'][0]}–L{p['layers'][-1]}", f"{t.get('steps', '?')} steps in {t.get('seconds', 0):.0f} s"]
		if t.get("teacher_match") is not None:
			parts.append(f"teaching answers {t['teacher_match']:.0%} on target")
		if closed is not None:
			parts.append(f"closes {closed:.0%} of the gap to the character")
		parts.append(f"KL on neutral text {p.get('kl', 0):.2f}")
		self.trials_label.setText("   ·   ".join(parts))
		self._fit_trials()
		self.set_status("goal lever set to +1.00: checking…", theme.ACCENT2)

	def set_aligned(self, p):
		if p.get("goal") != self.current:
			return
		if p.get("trials") and not self.trials:
			self.trials = p["trials"]
		if p.get("scan"):
			self.scan = p["scan"]
		self._trials_text(p.get("chosen", p.get("strength")) if p.get("found") else None)
		if p.get("fold"):
			self.set_status(f"goal lever ×{p['fold']:.2f} folded into the lever: its dose 1 is now that strength", theme.OK)
