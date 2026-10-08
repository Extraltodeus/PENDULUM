import html
import os
import time

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMessageBox, QSplitter, QTextBrowser, QVBoxLayout

import config
from hypno import presets as PS
from ui import theme
from ui.widgets import EvaDialog, button


class PresetsDialog(EvaDialog):
	save_requested = Signal(str)
	use_requested = Signal(object, bool)

	def __init__(self, lever_name, parent=None):
		super().__init__(parent)
		self.lever_name = lever_name
		self.setWindowTitle("Presets")
		self.setMinimumSize(860, 520)
		self.items = []
		self.extra = []
		lay = QVBoxLayout(self)
		lay.setSpacing(6)
		lay.addWidget(self.frameless("PRESETS", "SAVE THE LEVERS, GOALS AND SYSTEM PROMPT · LOAD THEM BACK"))
		info = QLabel("A preset keeps the high-level settings: your lever settings (the intent, before the coherence guard), the DOSE / TARGET mode with its targets, the goals they use with their sentences, the system prompt and the levers that ignore the guard. Loading it switches to its mode; in TARGET the levers are aligned on its targets again. PLAN IN TO DO puts the sliders in place and adds the changes to TO DO without running them: adjust, then APPLY there (✕ on a change drops it). An export's hypnolens.json can be loaded too (IMPORT A FILE), in DOSE mode.")
		info.setFont(theme.mono(7.5))
		info.setStyleSheet(f"color: {theme.MUTED};")
		info.setWordWrap(True)
		lay.addWidget(info)
		split = QSplitter(Qt.Orientation.Horizontal)
		self.list = QListWidget()
		self.list.setFont(theme.mono(8.5))
		self.list.currentRowChanged.connect(self.show_item)
		self.list.itemDoubleClicked.connect(lambda _it: self._use(True))
		self.view = QTextBrowser()
		self.view.setFont(theme.mono(8.5))
		split.addWidget(self.list)
		split.addWidget(self.view)
		split.setSizes([330, 500])
		lay.addWidget(split, 1)
		self.note = QLabel("")
		self.note.setFont(theme.mono(8))
		self.note.setWordWrap(True)
		lay.addWidget(self.note)
		btns = QHBoxLayout()
		b_save = button("SAVE THE CURRENT SETTINGS AS…", "write the lever settings, the mode and its targets, their goals and the system prompt into a preset (presets folder of the working folder)")
		b_save.clicked.connect(self._save)
		b_imp = button("IMPORT A FILE…", "load a preset file or an export's hypnolens.json from anywhere")
		b_imp.clicked.connect(self._import)
		b_dir = button("OPEN THE FOLDER", "the presets folder: rename or delete presets there")
		b_dir.clicked.connect(self._folder)
		self.b_stage = button("PLAN IN TO DO", "put the sliders and the system prompt in place and add the changes to TO DO without running them\nadjust, then APPLY in TO DO (✕ on a change drops it)")
		self.b_stage.clicked.connect(lambda: self._use(False))
		self.b_apply = button("APPLY", "load this preset and apply it now: its changes go to TO DO and run at once (double-click does the same)", accent=True)
		self.b_apply.clicked.connect(lambda: self._use(True))
		for b in (b_save, b_imp, b_dir):
			btns.addWidget(b)
		btns.addStretch(1)
		btns.addWidget(self.b_stage)
		btns.addWidget(self.b_apply)
		lay.addLayout(btns)
		self.reload()

	def reload(self, select=""):
		cur = self.current()
		self.items = self.extra + PS.load_all()
		self.list.blockSignals(True)
		self.list.clear()
		for d in self.items:
			n = len(d.get("strengths") or {})
			k = len(d.get("targets") or {})
			when = d.get("created") or time.strftime("%Y-%m-%d %H:%M", time.localtime(d.get("mtime", 0)))
			it = QListWidgetItem(("⇪ " if d.get("export") or d in self.extra else "") + f"{d.get('name', '?')}   ·   " + (("TARGET " + (f"{k} target{'s' if k != 1 else ''}, " if k else "· ")) if d.get("mode") == "target" else "") + f"{n} lever{'s' if n != 1 else ''}   ·   {when[:16]}")
			it.setToolTip(f"{it.text()}\n{d.get('path', '')}\nmade with {d.get('model') or '?'}")
			self.list.addItem(it)
		self.list.blockSignals(False)
		want = select or (cur or {}).get("path", "")
		idx = next((i for i, d in enumerate(self.items) if want and d.get("path") == want), 0 if self.items else -1)
		self.list.setCurrentRow(idx)
		self.show_item(idx)

	def current(self):
		i = self.list.currentRow() if hasattr(self, "list") else -1
		return self.items[i] if 0 <= i < len(self.items) else None

	def show_item(self, _i=-1):
		d = self.current()
		self.b_apply.setEnabled(d is not None)
		self.b_stage.setEnabled(d is not None)
		self.note.setText("")
		if d is None:
			self.view.setHtml(f'<p style="color:{theme.MUTED}">no preset yet: set the levers, then SAVE THE CURRENT SETTINGS AS…</p>')
			return
		rows = [f'<p style="color:{theme.MUTED}">{html.escape(d.get("created", ""))} · made with {html.escape(d.get("model") or "?")}<br>{html.escape(d.get("path", ""))}</p>']
		mode = d.get("mode")
		tg = d.get("targets") or {}
		if mode == "target":
			rows.append(f'<p><span style="color:{theme.NEON}">TARGET MODE</span> <span style="color:{theme.MUTED}">· loading it switches to TARGET and aligns the levers on ' + (f"its {len(tg)} target{'s' if len(tg) != 1 else ''}" if tg else "no target (every trait FREE, the levers as saved)") + "</span></p>")
		elif mode == "dose":
			rows.append(f'<p><span style="color:{theme.NEON}">DOSE MODE</span> <span style="color:{theme.MUTED}">· loading it switches to DOSE: the levers are set as saved</span></p>')
		if tg and mode != "dose":
			rows.append(f'<p><span style="color:{theme.NEON}">TARGETS</span> <span style="color:{theme.MUTED}">· measured values wanted' + ("" if mode else ", used if you are in TARGET mode when you load it") + "</span><br>" + "<br>".join(f'<span style="color:{theme.TEXT}">{html.escape(self.lever_name(t))}</span> <span style="color:{theme.CYAN}">◆ {v:+.2f}</span>' for t, v in sorted(tg.items(), key=lambda kv: self.lever_name(kv[0]))) + "</p>")
		lev = d.get("strengths") or {}
		goal_ids = {f"goal_{g.get('id')}" for g in d.get("goals") or []}
		if lev:
			rows.append(f'<p><span style="color:{theme.NEON}">LEVERS</span>' + (f' <span style="color:{theme.MUTED}">· the doses found for these targets, where the alignment starts</span>' if mode == "target" and tg else "") + "<br>" +"<br>".join(f'<span style="color:{theme.TEXT}">{html.escape(self.lever_name(t))}</span> <span style="color:{theme.ACCENT2 if v >= 0 else theme.COOL2}">{v:+.2f}</span>' + (" · goal" if t in goal_ids or t.startswith("goal") else "") + (" · ignores the guard" if t in (d.get("exempt") or []) else "") for t, v in sorted(lev.items(), key=lambda kv: self.lever_name(kv[0]))) + "</p>")
		else:
			rows.append(f'<p style="color:{theme.MUTED}">no lever set: loading it puts every lever back to 0</p>')
		for g in d.get("goals") or []:
			rows.append(f'<p><span style="color:{theme.NEON}">GOAL</span> <span style="color:{theme.TEXT}">{html.escape(g.get("name", ""))}</span><br>' + "<br>".join(f'<span style="color:{theme.MUTED}">{k}</span> {html.escape(g.get(k) or "—")}' for k in ("who", "want", "me")) + "".join(f'<br><span style="color:{theme.MUTED}">Q</span> {html.escape(c.get("q", ""))} <span style="color:{theme.MUTED}">→</span> {html.escape(c.get("target", ""))}' for c in g.get("custom") or []) + "</p>")
		if d.get("traits"):
			rows.append(f'<p style="color:{theme.MUTED}">custom levers carried with it: ' + ", ".join(html.escape(t.get("name", "")) for t in d["traits"]) + " (recreated if missing)</p>")
		sysp = d.get("system") or ""
		rows.append(f'<p><span style="color:{theme.NEON}">SYSTEM PROMPT</span><br>' + (html.escape(sysp).replace(chr(10), "<br>") if sysp else f'<span style="color:{theme.MUTED}">none (loading it clears the current one)</span>') + "</p>")
		self.view.setHtml("".join(rows))

	def set_model(self, name):
		d = self.current()
		if d is not None and name and d.get("model") and d["model"] != name:
			self.note.setText(f"made with {d['model']}: the levers are calibrated per model, the same settings can feel different here")
			self.note.setStyleSheet(f"color: {theme.WARN};")

	def _save(self):
		name, ok = QInputDialog.getText(self, "Save a preset", "preset name:")
		name = (name or "").strip()
		if not ok or not name:
			return
		if os.path.exists(PS.path_for(name)) and QMessageBox.question(self, "Replace the preset", f"A preset named '{name}' exists. Replace it?") != QMessageBox.StandardButton.Yes:
			return
		self.save_requested.emit(name)

	def _import(self):
		path, _f = QFileDialog.getOpenFileName(self, "Load a preset or an export manifest", config.EXPORTS_DIR if os.path.isdir(config.EXPORTS_DIR) else config.DATA_DIR, "JSON (*.json)")
		if not path:
			return
		try:
			d = PS.read(path)
		except (OSError, ValueError) as exc:
			self.note.setText(f"cannot read {os.path.basename(path)}: {exc}")
			self.note.setStyleSheet(f"color: {theme.ERR};")
			return
		self.extra = [x for x in self.extra if x.get("path") != d.get("path")] + [d]
		self.reload(d.get("path", ""))

	def _folder(self):
		os.makedirs(PS.folder(), exist_ok=True)
		QDesktopServices.openUrl(QUrl.fromLocalFile(PS.folder()))

	def _use(self, apply):
		d = self.current()
		if d is not None:
			self.use_requested.emit(d, apply)
