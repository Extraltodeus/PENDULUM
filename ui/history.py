import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QCheckBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QSplitter, QTextBrowser, QVBoxLayout

from hypno import history as HI
from ui import theme
from ui.widgets import EvaDialog, button, tab_focus


def short(text, n=70):
	t = " ".join((text or "").split())
	return t if len(t) <= n else t[:n - 1] + "…"


class HistoryDialog(EvaDialog):
	restore = Signal(object)

	def __init__(self, describe, parent=None):
		super().__init__(parent)
		self.describe = describe
		self.setWindowTitle("History")
		self.setMinimumSize(980, 600)
		self.model = ""
		self.entries = []
		self.shown = []
		lay = QVBoxLayout(self)
		lay.setSpacing(6)
		lay.addWidget(self.frameless("HISTORY", "EVERY REPLY WITH THE CONFIGURATION THAT MADE IT"))
		top = QHBoxLayout()
		self.search = QLineEdit()
		self.search.setFont(theme.mono(8.5))
		self.search.setPlaceholderText("filter: words of the message, of the reply, a lever or a date")
		self.search.textChanged.connect(self.fill)
		self.only = QCheckBox("this model only")
		tab_focus(self.only)
		self.only.setFont(theme.mono(8))
		self.only.setChecked(True)
		self.only.toggled.connect(self.fill)
		self.retries = QCheckBox("retries")
		tab_focus(self.retries)
		self.retries.setFont(theme.mono(8))
		self.retries.setChecked(True)
		self.retries.setToolTip("show the replies replaced by ↻ RETRY too")
		self.retries.toggled.connect(self.fill)
		self.count = QLabel("")
		self.count.setFont(theme.mono(8))
		self.count.setStyleSheet(f"color: {theme.MUTED};")
		b_reload = button("RELOAD", "read the history file again")
		b_reload.clicked.connect(self.reload)
		top.addWidget(self.search, 1)
		for w in (self.only, self.retries, self.count, b_reload):
			top.addWidget(w)
		lay.addLayout(top)
		split = QSplitter(Qt.Orientation.Horizontal)
		self.list = QListWidget()
		self.list.setFont(theme.mono(8))
		self.list.setWordWrap(False)
		self.list.currentRowChanged.connect(self.show_entry)
		self.view = QTextBrowser()
		self.view.setFont(theme.mono(8.5))
		self.view.setOpenLinks(False)
		split.addWidget(self.list)
		split.addWidget(self.view)
		split.setSizes([400, 560])
		lay.addWidget(split, 1)
		self.note = QLabel("")
		self.note.setFont(theme.mono(8))
		self.note.setWordWrap(True)
		lay.addWidget(self.note)
		btns = QHBoxLayout()
		self.b_reply = button("COPY THE REPLY")
		self.b_reply.clicked.connect(lambda: self.copy("reply"))
		self.b_cfg = button("COPY THE CONFIGURATION")
		self.b_cfg.clicked.connect(lambda: self.copy("config"))
		self.b_restore = button("RESTORE THIS CONFIGURATION", "set the levers and the system prompt back to what produced this reply\nlevers not used then go back to 0 · the coherence guard checks them again", accent=True)
		self.b_restore.clicked.connect(self._restore)
		btns.addWidget(self.b_reply)
		btns.addWidget(self.b_cfg)
		btns.addStretch(1)
		btns.addWidget(self.b_restore)
		lay.addLayout(btns)
		self.reload()

	def set_model(self, name):
		if name != self.model:
			self.model = name or ""
			self.only.setText(f"this model only ({short(self.model, 28)})" if self.model else "this model only")
			self.fill()

	def reload(self):
		cur = self.current()
		self.entries = HI.load()
		self.fill()
		if cur is not None:
			self.select(cur.get("id"))

	def select(self, gid):
		for i, e in enumerate(self.shown):
			if e.get("id") == gid:
				self.list.setCurrentRow(i)
				return True
		return False

	def match(self, e, words):
		if self.only.isChecked() and self.model and e.get("model") != self.model:
			return False
		if e.get("retry") and not self.retries.isChecked():
			return False
		if not words:
			return True
		cfg = e.get("config") or {}
		hay = " ".join([e.get("t", ""), e.get("prompt", ""), e.get("reply", ""), cfg.get("system") or "", e.get("model", "")] + list((cfg.get("strengths") or {}).keys())).lower()
		return all(w in hay for w in words)

	def fill(self, *_a):
		words = self.search.text().lower().split()
		cur = self.current()
		self.shown = [e for e in reversed(self.entries) if self.match(e, words)]
		self.list.blockSignals(True)
		self.list.clear()
		for e in self.shown:
			lev = (e.get("config") or {}).get("strengths") or {}
			mark = "↻ " if e.get("retry") else ""
			it = QListWidgetItem(f"{e.get('t', '')[5:16]}  {mark}{short(e.get('prompt'), 34)}  →  {short(e.get('reply'), 40)}" + (f"   [{len(lev)} lever{'s' if len(lev) != 1 else ''}]" if lev else ""))
			it.setToolTip(f"{e.get('t', '')} · {e.get('model', '')}\nyou: {short(e.get('prompt'), 300)}\nreply: {short(e.get('reply'), 300)}")
			if e.get("retry"):
				it.setForeground(theme.qc(theme.MUTED))
			self.list.addItem(it)
		self.list.blockSignals(False)
		total = sum(1 for e in self.entries if not self.only.isChecked() or not self.model or e.get("model") == self.model)
		self.count.setText(f"{len(self.shown)} / {total}")
		if not (cur is not None and self.select(cur.get("id"))) and self.shown:
			self.list.setCurrentRow(0)
		self.show_entry(self.list.currentRow())

	def current(self):
		i = self.list.currentRow() if hasattr(self, "list") else -1
		return self.shown[i] if 0 <= i < len(self.shown) else None

	def show_entry(self, _i=-1):
		e = self.current()
		for b in (self.b_reply, self.b_cfg, self.b_restore):
			b.setEnabled(e is not None)
		if e is None:
			self.view.setHtml(f'<p style="color:{theme.MUTED}">no reply yet' + (" with this model" if self.only.isChecked() and self.model else "") + "</p>")
			self.note.setText("")
			return
		cfg = e.get("config") or {}
		st = e.get("stats") or {}
		head = f'<p style="color:{theme.MUTED}">{html.escape(e.get("t", ""))} · {html.escape(e.get("model", ""))}' + (" · ↻ retry" if e.get("retry") else "") + (" · stopped" if e.get("stopped") else "") + (f" · turn {e.get('turn', 0) + 1}" if e.get("turn") is not None else "") + (f" · {st.get('tokens', 0)} tokens, {st.get('tps', 0)} tok/s" if st else "") + "</p>"
		you = f'<p><span style="color:{theme.NEON}">YOU</span><br><span style="color:{theme.TEXT}">{html.escape(e.get("prompt", "")).replace(chr(10), "<br>")}</span></p>'
		reply = f'<p><span style="color:{theme.ACCENT}">MODEL</span><br><span style="color:#d8d4cd">{html.escape(e.get("reply", "")).replace(chr(10), "<br>")}</span></p>'
		conf = f'<p style="color:{theme.MUTED}">{html.escape(self.describe(cfg, "CONFIGURATION")).replace(chr(10), "<br>")}</p>'
		self.view.setHtml(head + you + reply + conf)
		other = self.model and e.get("model") != self.model
		self.note.setText(f"made with {e.get('model')}: load that model before restoring, the levers are calibrated per model" if other else "")
		self.note.setStyleSheet(f"color: {theme.WARN};")

	def copy(self, what):
		e = self.current()
		if e is None:
			return
		QApplication.clipboard().setText(e.get("reply", "") if what == "reply" else self.describe(e.get("config") or {}, f"REPLY OF {e.get('t', '')} · CONFIGURATION"))

	def _restore(self):
		e = self.current()
		if e is None:
			return
		self.restore.emit(dict(e.get("config") or {}, n=e.get("t", "?")))
