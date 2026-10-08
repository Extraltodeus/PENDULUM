import os
import threading

from PySide6.QtCore import QObject, QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QGuiApplication, QPainter
from PySide6.QtWidgets import QButtonGroup, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QStackedWidget, QVBoxLayout, QWidget

import config
from engine import llamadist as LD
from hypno import catalog
from ui import eva, theme
from ui.widgets import EvaDialog, button, guard_wheel

STEPS = (("folder", "WORKING FOLDER", "one folder for your work"), ("llama", "LLAMA.CPP", "runs the models"), ("models", "MODELS", "your GGUF files"), ("judge", "JUDGE", "optional"))
CHECKS = (("lib", "llama library", "runs the models (required)"), ("backends", "GPU backend", "CUDA, Vulkan or Metal: without one, the CPU does everything"), ("cudart", "CUDA runtime", "needed by a CUDA build"), ("fit", "llama-fit-params", "plans how a model spreads over the GPUs"), ("server", "llama-server", "used by the launchers written with each export"), ("src", "llama.cpp sources", "gguf-py and convert_hf_to_gguf.py: to convert safetensors models (a pip gguf can stand in for reading GGUF files)"))
JUDGE_TIP = "a small instruct model of 2 to 4 B (a Qwen3.5 4B or 2B, a Gemma 4 E2B…) judges well; 1 B models are too lenient"
GB = 2**30


def heading(text):
	l = QLabel(text)
	theme.tag_font(l, 13, True)
	return l


def caption(text, color=None):
	l = QLabel(text)
	l.setWordWrap(True)
	l.setFont(theme.mono(8))
	l.setStyleSheet(f"color: {color or theme.MUTED};")
	return l


def path_row(edit, pick):
	row = QHBoxLayout()
	row.setSpacing(6)
	edit.setFont(theme.mono(8.5))
	row.addWidget(edit, 1)
	b = button("BROWSE…", "pick the folder")
	b.clicked.connect(pick)
	row.addWidget(b)
	return row


def wanted():
	return not os.path.exists(config.SETTINGS_PATH)


class Rail(QWidget):
	def __init__(self):
		super().__init__()
		self.at = 0
		self.setFixedWidth(224)

	def set_at(self, i):
		self.at = i
		self.update()

	def paintEvent(self, _e):
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		nerv = eva.on()
		big = eva.font(30, 700) if nerv else theme.mono(18, True)
		name = eva.font(14, 700) if nerv else theme.mono(8.5, True)
		sub = theme.mono(7.5)
		left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
		y = 4.0
		for i, (_k, title, hint) in enumerate(STEPS):
			now = i == self.at
			c = theme.NEON if now else (theme.TEXT if i < self.at else theme.DIM)
			if now:
				p.fillRect(QRectF(0, y + 6, 3, 34), theme.qc(theme.NEON))
			p.setFont(big)
			p.setPen(theme.qc(c))
			p.drawText(QRectF(12, y, 56, 46), left, f"{i + 1:02d}")
			p.setFont(name)
			p.drawText(QRectF(70, y + 4, self.width() - 70, 20), left, title)
			p.setFont(sub)
			p.setPen(theme.qc(theme.MUTED if now else theme.DIM))
			p.drawText(QRectF(70, y + 23, self.width() - 70, 16), left, hint)
			y += 62
		p.end()


class Bridge(QObject):
	builds = Signal(object)
	progress = Signal(float)
	said = Signal(str)
	failed = Signal(str)
	installed = Signal(object)
	counted = Signal(str, object)


class Wizard(EvaDialog):
	def __init__(self, settings=None, parent=None, first=True):
		super().__init__(parent)
		self.setWindowTitle("Pendulum setup")
		self.settings = dict(settings or config.load_settings())
		self.first = first
		self.bridge = Bridge()
		self.bridge.builds.connect(self._got_builds)
		self.bridge.progress.connect(self._progress)
		self.bridge.said.connect(lambda t: self.dl_note.setText(t))
		self.bridge.failed.connect(self._failed)
		self.bridge.installed.connect(self._installed)
		self.bridge.counted.connect(self._counted)
		self.cancel = None
		self.release = None
		self.found = {}
		lay = QVBoxLayout(self)
		lay.setSpacing(10)
		lay.addWidget(self.frameless("SETUP", "FIRST START" if first else "PENDULUM", "four short steps; everything here can be changed later in ⚙ settings"))
		body = QHBoxLayout()
		body.setSpacing(22)
		self.rail = Rail()
		body.addWidget(self.rail)
		self.pages = QStackedWidget()
		for make in (self._folder_page, self._llama_page, self._models_page, self._judge_page):
			self.pages.addWidget(make())
		body.addWidget(self.pages, 1)
		body.setContentsMargins(0, 0, 6, 0)
		lay.addLayout(body, 1)
		nav = QHBoxLayout()
		self.b_skip = button("SKIP SETUP" if first else "CANCEL", "start with the defaults: llama.cpp found automatically, models in the Hugging Face cache, no judge · this setup stays in ⚙ settings" if first else "close without changing anything")
		self.b_skip.clicked.connect(self.reject)
		self.b_back = button("◂ BACK")
		self.b_back.clicked.connect(lambda: self.go(self.pages.currentIndex() - 1))
		self.b_next = button("NEXT ▸", accent=True)
		self.b_next.clicked.connect(self._next)
		nav.addWidget(self.b_skip)
		nav.addStretch(1)
		nav.addWidget(self.b_back)
		nav.addWidget(self.b_next)
		lay.addLayout(nav)
		self.resize(820, 520)
		self.go(0)

	def _page(self, title, text):
		w = QWidget()
		v = QVBoxLayout(w)
		v.setContentsMargins(0, 4, 0, 0)
		v.setSpacing(8)
		v.addWidget(heading(title))
		v.addWidget(caption(text))
		v.addSpacing(6)
		return w, v

	def _folder_page(self):
		w, v = self._page("WHERE PENDULUM KEEPS ITS WORK", "Profiles, levers, goals, trained LoRAs, lenses, exports and settings all live in one working folder. Lenses and LoRAs can take a few GB.")
		self.wd_default = QRadioButton(f"next to the app: {config.DEFAULT_DATA_DIR.replace(os.sep, '/')}")
		self.wd_own = QRadioButton("a folder of my own:")
		for r in (self.wd_default, self.wd_own):
			r.setFont(theme.mono(8.5))
		group = QButtonGroup(w)
		group.addButton(self.wd_default)
		group.addButton(self.wd_own)
		self.wd_edit = QLineEdit(config.load_workdir())
		self.wd_edit.setPlaceholderText("e.g. D:/Pendulum")
		self.wd_edit.textEdited.connect(lambda _t: self.wd_own.setChecked(True))
		(self.wd_own if self.wd_edit.text() else self.wd_default).setChecked(True)
		v.addWidget(self.wd_default)
		v.addWidget(self.wd_own)
		v.addLayout(path_row(self.wd_edit, self._pick_workdir))
		if config.DATA_ENV:
			for x in (self.wd_default, self.wd_own, self.wd_edit):
				x.setEnabled(False)
			v.addWidget(caption(f"this run uses {config.DATA_ENV.replace(os.sep, '/')} (--data-dir or HYPNOLENS_DATA_DIR): the setup writes there and leaves the usual working folder alone", theme.WARN))
		elif not self.first:
			v.addWidget(caption("a new folder applies after a restart; your settings, goals and custom levers are copied there, the rest stays where it is"))
		v.addStretch(1)
		return w

	def _pick_workdir(self):
		d = QFileDialog.getExistingDirectory(self, "Working folder", self.wd_edit.text() or config.ROOT)
		if d:
			self.wd_edit.setText(d.replace("\\", "/"))
			self.wd_own.setChecked(True)

	def workdir(self):
		return self.wd_edit.text().strip() if self.wd_own.isChecked() else ""

	def _llama_page(self):
		w, v = self._page("THE ENGINE: LLAMA.CPP", "Pendulum runs your models through the llama.cpp library. Point to a llama.cpp folder (a release or your own build), or let Pendulum fetch the build that suits this computer.")
		self.ll_edit = QLineEdit(self.settings.get("llama_dir") or LD.guess(config.DATA_DIR))
		self.ll_edit.setPlaceholderText("folder with llama.dll / libllama, e.g. D:/llama.cpp")
		self.ll_edit.textChanged.connect(lambda _t: self._llama_status())
		v.addLayout(path_row(self.ll_edit, self._pick_llama))
		grid = QGridLayout()
		grid.setHorizontalSpacing(10)
		grid.setVerticalSpacing(3)
		self.marks = {}
		for r, (key, label, tip) in enumerate(CHECKS):
			mark = QLabel("")
			mark.setFont(theme.mono(9, True))
			mark.setFixedWidth(16)
			name = QLabel(label)
			name.setFont(theme.mono(8.5))
			name.setToolTip(tip)
			note = QLabel("")
			note.setFont(theme.mono(7.5))
			note.setStyleSheet(f"color: {theme.MUTED};")
			note.setToolTip(tip)
			grid.addWidget(mark, r, 0)
			grid.addWidget(name, r, 1)
			grid.addWidget(note, r, 2)
			self.marks[key] = (mark, note)
		grid.setColumnStretch(2, 1)
		v.addLayout(grid)
		v.addSpacing(8)
		dl = QHBoxLayout()
		dl.setSpacing(6)
		self.b_builds = button("FIND A BUILD", "ask GitHub for the latest llama.cpp release and suggest the build that fits this computer (nothing is downloaded yet)")
		self.b_builds.clicked.connect(self._find_builds)
		self.build = guard_wheel(QComboBox())
		self.build.setFont(theme.mono(8))
		self.build.setEnabled(False)
		self.b_get = button("DOWNLOAD", "download the chosen build (checked by its sha256) with the llama.cpp sources, into the llama.cpp folder of the working folder")
		self.b_get.setEnabled(False)
		self.b_get.clicked.connect(self._download)
		dl.addWidget(self.b_builds)
		dl.addWidget(self.build, 1)
		dl.addWidget(self.b_get)
		v.addLayout(dl)
		self.bar = Progress()
		v.addWidget(self.bar)
		note = QHBoxLayout()
		self.dl_note = caption("")
		note.addWidget(self.dl_note, 1)
		link = button("RELEASES ↗", "open the llama.cpp releases page in your browser", click=False)
		link.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(LD.PAGE)))
		note.addWidget(link, 0, Qt.AlignmentFlag.AlignTop)
		v.addLayout(note)
		v.addStretch(1)
		self._llama_status()
		return w

	def _pick_llama(self):
		d = QFileDialog.getExistingDirectory(self, "llama.cpp folder", self.ll_edit.text() or config.ROOT)
		if d:
			self.ll_edit.setText(d.replace("\\", "/"))

	def _llama_status(self):
		st = LD.status(self.ll_edit.text().strip())
		self.llama = st
		rows = {
			"lib": (st["lib"], st["bin"].replace(os.sep, "/") if st["lib"] else "not found in this folder"),
			"backends": (bool(st["backends"]), " · ".join(st["backends"]) if st["backends"] else "CPU only"),
			"cudart": (bool(st["cudart"]) or "CUDA" not in st["backends"], os.path.basename(st["cudart"]) if st["cudart"] else ("not needed" if "CUDA" not in st["backends"] else "missing: install the CUDA runtime or download it with the build")),
			"fit": (st["fit"], "found" if st["fit"] else "missing: VRAM planning falls back on estimates"),
			"server": (st["server"], "found" if st["server"] else "missing: exports still work, their launcher needs it"),
			"src": (bool(st["src"]) or bool(st["gguf"]), st["src"].replace(os.sep, "/") if st["src"] else (f"not found · {st['gguf']} reads GGUF files, safetensors cannot be converted" if st["gguf"] else "missing: GGUF files cannot be read")),
		}
		for key, (good, text) in rows.items():
			mark, note = self.marks[key]
			mark.setText("✓" if good else "✗")
			mark.setStyleSheet(f"color: {theme.OK if good else (theme.ERR if key in ('lib', 'src') else theme.WARN)};")
			note.setText(text)
		if self.pages.currentIndex() == 1:
			self._nav()

	def _find_builds(self):
		self.b_builds.setEnabled(False)
		self.dl_note.setText("asking GitHub for the latest release…")

		def work():
			try:
				rels = LD.releases()
				rel, names = LD.builds(rels)
				self.bridge.builds.emit((rels, rel, names, LD.pick(rels)))
			except Exception as exc:
				self.bridge.failed.emit(f"no answer from GitHub ({exc}): download a build from the releases page and point to its folder")

		threading.Thread(target=work, daemon=True).start()

	def _got_builds(self, got):
		_rels, rel, names, best = got
		self.b_builds.setEnabled(True)
		self.release = rel
		self.build.clear()
		if rel is None or not names:
			self.dl_note.setText("no build of the latest releases fits this system: see the releases page")
			return
		sizes = {a["name"]: int(a.get("size") or 0) for a in rel.get("assets") or []}
		pick = best["files"][0]["name"] if best and best["tag"] == rel["tag_name"] else ""
		for n in names:
			self.build.addItem(f"{LD.label(rel['tag_name'], n)} · {sizes.get(n, 0) / 2**20:.0f} MB" + ("  (suggested)" if n == pick else ""), n)
		if pick:
			self.build.setCurrentIndex(max(0, self.build.findData(pick)))
		self.build.setEnabled(True)
		self.b_get.setEnabled(True)
		self.dl_note.setText(f"llama.cpp {rel['tag_name']}" + (f" · suggested: {best['why']}" if best and pick else "") + " · pick another build if you know better")

	def _download(self):
		if self.cancel is not None:
			self.cancel.set()
			return
		plan = LD.plan_for(self.release, self.build.currentData())
		root = os.path.join((self.workdir() or config.DEFAULT_DATA_DIR) if self.first and not config.DATA_ENV else config.DATA_DIR, "llama.cpp")
		self.cancel = threading.Event()
		self.b_get.setText("CANCEL")
		self.b_builds.setEnabled(False)
		self.build.setEnabled(False)
		self.bar.set(0.0)
		cancel = self.cancel

		def work():
			try:
				res = LD.install(plan, root, on_progress=self.bridge.progress.emit, cancel=cancel, log=self.bridge.said.emit)
				self.bridge.installed.emit(res)
			except LD.Cancelled:
				self.bridge.failed.emit("download cancelled")
			except Exception as exc:
				self.bridge.failed.emit(f"download failed: {exc}")

		threading.Thread(target=work, daemon=True).start()

	def _progress(self, v):
		self.bar.set(float(v))

	def _done_download(self):
		self.cancel = None
		self.b_get.setText("DOWNLOAD")
		self.b_builds.setEnabled(True)
		self.build.setEnabled(self.build.count() > 0)

	def _failed(self, text):
		self._done_download()
		self.b_builds.setEnabled(True)
		self.bar.set(-1.0)
		self.dl_note.setText(text)

	def _installed(self, res):
		self._done_download()
		self.bar.set(1.0)
		self.ll_edit.setText(res["bin"].replace(os.sep, "/"))
		self.dl_note.setText(f"llama.cpp {res['tag']} installed in {os.path.dirname(res['bin']).replace(os.sep, '/')}")

	def _models_page(self):
		w, v = self._page("YOUR MODELS", "Pendulum lists the GGUF files it finds (and Hugging Face safetensors folders, converted to GGUF once when loaded). The Hugging Face cache is always searched.")
		self.hub = catalog.default_dirs()
		hub = (self.hub[0] if self.hub else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")).replace(os.sep, "/")
		self.md_hub = QRadioButton("the Hugging Face cache only")
		self.md_hub.setToolTip(hub)
		self.md_own = QRadioButton("a folder of mine as well:")
		for r in (self.md_hub, self.md_own):
			r.setFont(theme.mono(8.5))
		group = QButtonGroup(w)
		group.addButton(self.md_hub)
		group.addButton(self.md_own)
		dirs = list(self.settings.get("model_dirs") or [])
		self.md_edit = QLineEdit(dirs[0] if dirs else "")
		self.md_edit.setPlaceholderText("e.g. D:/models")
		self.md_edit.textEdited.connect(lambda _t: self.md_own.setChecked(True))
		self.md_edit.editingFinished.connect(self._count)
		self.md_extra = dirs[1:]
		(self.md_own if dirs else self.md_hub).setChecked(True)
		self.md_hub.toggled.connect(lambda _on: self._count())
		v.addWidget(self.md_hub)
		v.addWidget(caption(hub))
		v.addWidget(self.md_own)
		v.addLayout(path_row(self.md_edit, self._pick_models))
		self.md_note = caption("")
		v.addWidget(self.md_note)
		if self.md_extra:
			v.addWidget(caption(f"also kept from your settings: {', '.join(d.replace(os.sep, '/') for d in self.md_extra)}"))
		v.addStretch(1)
		return w

	def _pick_models(self):
		d = QFileDialog.getExistingDirectory(self, "Models folder", self.md_edit.text() or os.path.expanduser("~"))
		if d:
			self.md_edit.setText(d.replace("\\", "/"))
			self.md_own.setChecked(True)
			self._count()

	def model_dirs(self):
		own = self.md_edit.text().strip()
		return ([own] if self.md_own.isChecked() and own else []) + list(self.md_extra)

	def _count(self):
		dirs = self.model_dirs() + list(self.hub)
		key = tuple(dirs)
		self.md_note.setText("looking for GGUF files…")

		def work():
			try:
				files = [(p, os.path.getsize(p)) for p in catalog.gguf_files(dirs)]
			except OSError:
				files = []
			self.bridge.counted.emit(repr(key), files)

		threading.Thread(target=work, daemon=True).start()

	def _counted(self, key, files):
		if key != repr(tuple(self.model_dirs() + list(self.hub))):
			return
		self.found = {p: s for p, s in files}
		self.md_note.setText(f"{len(files)} GGUF file{'s' if len(files) != 1 else ''} found" + ("" if files else " · you can add folders later in MODELS or ⚙ settings"))
		self.md_note.setStyleSheet(f"color: {theme.OK if files else theme.WARN};")
		self._fill_judges()

	def _judge_page(self):
		w, v = self._page("A JUDGE · OPTIONAL", "A second, small model reads the test replies and says whether they still make sense, and whether a goal is reached. Without one, Pendulum measures how surprised the model is by its own replies: it works, only more cautiously. No need to decide now: the JUDGE panel and ⚙ settings change it any time.")
		self.judge = guard_wheel(QComboBox())
		self.judge.setFont(theme.mono(8.5))
		v.addWidget(self.judge)
		v.addWidget(caption(JUDGE_TIP))
		v.addStretch(1)
		self._fill_judges()
		return w

	def _fill_judges(self):
		if not hasattr(self, "judge"):
			return
		keep = self.judge.currentData() or (self.settings.get("coherence") or {}).get("judge") or ""
		self.judge.clear()
		self.judge.addItem("none for now: the surprise measure", "")
		for p, s in sorted(self.found.items(), key=lambda kv: (kv[1], kv[0].lower())):
			self.judge.addItem(f"{os.path.basename(p)}  ·  {s / GB:.1f} GB", p)
		if keep and self.judge.findData(keep) < 0 and os.path.isfile(keep):
			self.judge.addItem(f"{os.path.basename(keep)}  ·  {os.path.getsize(keep) / GB:.1f} GB", keep)
		self.judge.setCurrentIndex(max(0, self.judge.findData(keep)))

	def go(self, i):
		i = max(0, min(self.pages.count() - 1, i))
		self.pages.setCurrentIndex(i)
		self.rail.set_at(i)
		if STEPS[i][0] == "models" and not self.found:
			self._count()
		self._nav()

	def _nav(self):
		i = self.pages.currentIndex()
		last = i == self.pages.count() - 1
		self.b_back.setEnabled(i > 0)
		self.b_next.setText("START ▸" if last and self.first else ("SAVE" if last else "NEXT ▸"))
		self.b_next.setToolTip("" if STEPS[i][0] != "llama" or self.llama["lib"] else "no llama library in this folder yet: you can still go on and set it later in ⚙ settings")

	def _next(self):
		if self.pages.currentIndex() < self.pages.count() - 1:
			self.go(self.pages.currentIndex() + 1)
			return
		self.accept()

	def values(self):
		return {"workdir": self.workdir(), "llama_dir": (self.llama["bin"] or self.ll_edit.text().strip()).replace(os.sep, "/"), "model_dirs": self.model_dirs(), "judge": self.judge.currentData() or ""}

	def reject(self):
		if self.cancel is not None:
			self.cancel.set()
		super().reject()


class Progress(QWidget):
	def __init__(self):
		super().__init__()
		self.frac = -1.0
		self.setFixedHeight(12)

	def set(self, v):
		self.frac = v
		self.update()

	def paintEvent(self, _e):
		if self.frac < 0:
			return
		p = QPainter(self)
		r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
		p.setPen(theme.qc(theme.LINE2))
		p.drawRect(r)
		eva.seg_bar(p, r.adjusted(2, 2, -2, -2), max(0.0, min(1.0, self.frac)), theme.ACCENT)
		p.end()


def apply_first(values):
	if not config.DATA_ENV:
		config.use_data_dir(values["workdir"])
	settings = config.load_settings()
	settings["llama_dir"] = values["llama_dir"] or settings.get("llama_dir") or ""
	settings["model_dirs"] = list(values["model_dirs"])
	settings["coherence"] = {**settings["coherence"], "judge": values["judge"]}
	settings["wizard_done"] = True
	config.save_settings(settings)


def skip_first():
	settings = config.load_settings()
	settings["wizard_done"] = True
	config.save_settings(settings)


def first_start():
	dlg = Wizard(first=True)
	scr = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
	dlg.move(scr.availableGeometry().center() - dlg.rect().center())
	if dlg.exec():
		apply_first(dlg.values())
	else:
		skip_first()
