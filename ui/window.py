import copy
import ctypes
import json
import math
import os
import re
import sys
import time

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QApplication, QCheckBox, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox, QScrollArea, QSplitter, QToolTip, QVBoxLayout, QWidget

import config
from lens.match import best, file_entry, lens_vram, rank
from persona import goals as GO
from ui import icon, theme
from ui.goals import GoalsDialog
from ui.history import HistoryDialog
from ui.active import ActiveLeversDialog
from ui.frame import EDGE, EdgeFilter, Frameless
from ui.dock import Dock
from ui.shutdown import DELAYS, Shutdown
from ui.snap import Snapper
from ui.models import ModelsWindow
from ui.plan import CHAT, INFO, INSTANT, Eta, Plan
from ui.presets import PresetsDialog
from ui.sfx import SFX
from ui.todo import TodoPanel
from ui.trace import TraceDialog
from ui.sync import SyncBand, SyncView
from ui.iris import IrisView, demo_state
from ui.wizard import Wizard
from ui.widgets import PREF_DEFAULTS, menu_title, Breath, ChatPanel, ConceptList, ExportDialog, HeaderBar, LensHeat, LensPicker, LeverTable, LogView, ModeSwitch, Panel, SettingsDialog, TrainDialog, TraitDialog, WorkspaceChart, button, set_tooltip_delay, tab_focus, toggle
from lens.fit import Fitter
from ui.worker import Planner, Relay, Worker

UI_PATH = os.path.join(config.DATA_DIR, "ui.json")
TRACE_LINKS = (
	("guard", ("guard",), re.compile(r"checking coherence|^coherence \d|levers only lowered|^blame:|^worst lever:|^INDIVIDUAL: (checking|coherence)|^scaling the levers|coherence guard cut|^JUDGE: (checking|coherence)", re.I)),
	("calibration", ("calibration",), re.compile(r"^calibrating '|strength 1 = ", re.I)),
	("judge", ("judge",), re.compile(r"^judge (ready|rejected)|rates word salad", re.I)),
	("goal", ("goal",), re.compile(r"^goal '.*': (asking|match|goal lever)", re.I)),
	("align", ("align",), re.compile(r"^aligning|^round \d+:|^targets (partly )?reached|closest result was round", re.I)),
	("matrix", ("matrix",), re.compile(r"^measuring what '", re.I)),
)
UNDO_MAX = 60
PROFILE_TIP = "measure the personality\nwith levers set: a quick measure of the steered model (6 prompts, informative: a chat sent meanwhile goes first)\nwithout levers: the full baseline profile\nShift+click: the full profile of the steered model · right-click: every choice"
SKIP_JUDGE = ("levers", "judge", "exempt", "apply_settings", "active")
STAGE_OF = {"levers_as_set": "LEVERS", "align": "LEVERS", "load": "MODEL", "unload": "MODEL", "map": "WORKSPACE", "profile": "PROFILE", "measure": "PROFILE", "read_scale": "PROFILE", "levers": "LEVERS", "calibrate": "LEVERS", "add_trait": "LEVERS", "chat": "DIALOGUE", "goal_check": "PROFILE", "goal_align": "LEVERS", "goal_optimize": "LEVERS", "goal_save": "LEVERS", "goal_delete": "LEVERS", "recalibrate": "LEVERS", "trait_edit": "LEVERS", "load_options": "MODEL", "exempt": "LEVERS", "goal_as_lever": "LEVERS", "active": "LEVERS", "judge": "LEVERS"}
JOB_LABELS = {"levers_as_set": "apply the levers as set (judgement skipped)", "judge": "judge the changed levers", "rebuild": "rebuild the lever list", "drop": "drop an outdated goal lever", "apply_settings": "apply the new settings to the levers", "load": "load a model", "unload": "unload the model", "catalog": "scan models and lenses", "plan": "plan the VRAM placement", "map": "re-map the workspace band", "profile": "full personality profile", "measure": "measure the steered model (quick profile)", "read_scale": "measure the reading scale of new levers (READ + WRITE)", "levers": "apply the lever settings", "calibrate": "calibrate levers", "recalibrate": "recalibrate levers", "add_trait": "build a new lever", "trait_edit": "save a lever", "load_options": "save the load options", "system": "set the system prompt", "chat": "reply in the dialogue", "export": "export", "goal_optimize": "TRAIN a goal", "goal_check": "check a goal", "goal_align": "align on a goal", "goal_save": "save a goal", "goal_delete": "delete a goal", "prefs": "save a setting", "settings": "apply the settings", "reset_chat": "clear the dialogue", "chat_pick": "use the reply shown in the dialogue", "align": "align the levers on the targets", "preset_save": "save a preset", "preset_load": "load a preset", "exempt": "change a guard exemption", "goal_as_lever": "make a lever from a goal", "active": "set the active levers", "shot": "screenshot"}
READY_AFTER = {"levers_as_set": "LEVERS APPLIED", "judge": "LEVERS JUDGED", "align": "TARGETS ALIGNED", "load": "MODEL LOADED", "levers": "LEVERS APPLIED", "calibrate": "LEVERS CALIBRATED", "recalibrate": "LEVERS CALIBRATED", "exempt": "LEVERS APPLIED", "profile": "PROFILE MEASURED", "map": "WORKSPACE MAPPED", "settings": "SETTINGS APPLIED", "add_trait": "NEW LEVER BUILT", "trait_edit": "LEVERS UPDATED", "goal_align": "GOAL ALIGNED", "goal_optimize": "GOAL TRAINED", "active": "ACTIVE LEVERS SET"}
TITLES = {"levers_as_set": "APPLYING LEVERS AS SET", "judge": "JUDGING THE LEVERS", "guard": "CHECKING COHERENCE", "rebuild": "REBUILDING THE LEVER LIST", "drop": "DROPPING A GOAL LEVER", "apply_settings": "APPLYING THE SETTINGS", "load": "LOADING", "convert": "CONVERTING TO GGUF", "lens download": "DOWNLOADING THE LENS", "workspace": "MAPPING THE WORKSPACE", "profile": "PROFILING THE PERSONALITY", "map": "MAPPING THE WORKSPACE", "measure": "MEASURING THE STEERED MODEL", "read_scale": "SCALING THE READING", "levers": "APPLYING LEVERS", "calibrate": "CALIBRATING LEVERS", "add_trait": "BUILDING THE NEW LEVER", "chat": "GENERATING", "plan": "PLANNING", "catalog": "SCANNING MODELS AND LENSES", "export": "EXPORTING", "unload": "UNLOADING", "reset_chat": "CLEARING THE DIALOGUE", "chat_pick": "KEEPING THE REPLY SHOWN", "align": "ALIGNING ON THE TARGETS", "preset_save": "SAVING THE PRESET", "preset_load": "LOADING THE PRESET", "system": "SETTING THE SYSTEM PROMPT", "fit": "TRAINING A LENS", "goal_check": "CHECKING THE GOAL", "goal_align": "ALIGNING ON THE GOAL", "goal_optimize": "TRAINING THE GOAL LEVER", "goal_save": "SAVING THE GOAL", "goal_delete": "DELETING THE GOAL", "recalibrate": "RECALIBRATING LEVERS", "trait_edit": "EDITING LEVERS", "prefs": "SAVING SETTINGS", "settings": "APPLYING THE SETTINGS", "load_options": "SAVING LOAD OPTIONS", "exempt": "APPLYING LEVERS", "goal_as_lever": "MAKING A LEVER FROM THE GOAL", "active": "SETTING THE ACTIVE LEVERS"}


def dark_title(hwnd):
	if not sys.platform.startswith("win"):
		return
	on = ctypes.c_int(1)
	for attr in (20, 19):
		if ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
			return


class DarkFrames(QObject):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.seen = set()

	def eventFilter(self, obj, ev):
		if ev.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow() and obj.windowType() in (Qt.WindowType.Window, Qt.WindowType.Dialog) and not obj.windowFlags() & Qt.WindowType.FramelessWindowHint:
			hwnd = int(obj.winId())
			if hwnd not in self.seen:
				self.seen.add(hwnd)
				try:
					dark_title(hwnd)
				except Exception:
					pass
		return False


def _load_ui():
	try:
		with open(UI_PATH, encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return {}


def _nvml():
	try:
		import pynvml

		pynvml.nvmlInit()
		return pynvml
	except Exception:
		return None


class MainWindow(Frameless, QMainWindow):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.setWindowTitle("Pendulum")
		self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
		self.ui = _load_ui()
		self.prefs = {**PREF_DEFAULTS, **(self.ui.get("prefs") or {})}
		set_tooltip_delay(self.prefs.get("tooltip_ms", 700))
		theme.apply_accent(self.prefs.get("accent", "ember"))
		self.setStyleSheet(theme.STYLE)
		self.edge_filter = EdgeFilter(self)
		QApplication.instance().installEventFilter(self.edge_filter)
		self.dark_frames = DarkFrames(self)
		QToolTip.setFont(theme.mono(8))
		QApplication.instance().installEventFilter(self.dark_frames)
		self.normal_geom = None
		self.persist_ui = not (args.shot or args.quit or os.environ.get("PENDULUM_NO_UI_SAVE"))
		self.snap = {"loaded": False}
		self.summaries = {}
		self.levers = {}
		self.inactive = set(config.load_settings().get("inactive_levers") or [])
		self.strengths = {}
		self.applied = {}
		self.apply_pending = set()
		self.calib_pending = set()
		self.reply_n = 0
		self.type_n = 0
		self.superseded = set()
		self.requeue = {}
		self.carry = None
		self.ref_odds = None
		self.catalog = []
		self.lens_index = []
		self.lens_best = {}
		self.llama_dir = ""
		self.margin = 900
		self.loaded_plan = None
		self.plan_key = None
		self.gpus = []
		self.busy = None
		self.last_ok = None
		self.script = []
		self.profiling = None
		self.tps = []
		self.nv = _nvml()
		self.activity = {"name": None}
		self.goal_defs = GO.load()
		self.goal_scores = {}
		self.judge_state = {"state": "off"}
		self.goals_dlg = None
		self.notice_box = None
		self.todo_plan = Plan(Eta(os.path.join(config.DATA_DIR, "eta.json")))
		self.todo_plan.auto = bool(self.prefs.get("todo_auto", False))
		self._build()
		if config.DATA_DIR_ERROR:
			self.log.add(config.DATA_DIR_ERROR, "error")
		elif not config.DATA_ENV and not config.same_dir(config.DATA_DIR, config.DEFAULT_DATA_DIR):
			self.log.add(f"working folder: {config.DATA_DIR}" + (f" · brought along from the previous one: {', '.join(config.CARRIED)}" if config.CARRIED else ""))
		self.worker = Worker()
		self.worker.event.connect(self.on_event)
		self.planner = Planner()
		self.planner.planned.connect(self.on_planned)
		self.fit_relay = Relay()
		self.fit_relay.fired.connect(self.on_fit_state)
		self.fitter = Fitter(self.fit_relay.fired.emit)
		self.fit_state = {"state": "idle"}
		self.pending_fit = None
		self.train_dlg = None
		self.hold_timer = QTimer(self)
		self.hold_timer.setSingleShot(True)
		self.hold_timer.timeout.connect(self.plan_dispatch)
		self.gpu_timer = QTimer(self)
		self.gpu_timer.timeout.connect(self.poll_gpus)
		self.gpu_timer.start(1500)
		self.poll_gpus()
		self.apply_prefs()
		self.refresh_iris()
		self.update_hud()
		self._restore_geometry()
		QTimer.singleShot(700, lambda: SFX.play("start"))

	def _build(self):
		theme.apply_style(self.prefs.get("style", "nerv"))
		self.setStyleSheet(theme.STYLE)
		root = QWidget()
		self.setCentralWidget(root)
		self.setContentsMargins(EDGE, EDGE, EDGE, EDGE)
		v = QVBoxLayout(root)
		v.setContentsMargins(0, 0, 0, 0)
		v.setSpacing(0)
		self.header = HeaderBar()
		for key, text, tip in (("models", "MODELS", "the models: pick one, its VRAM plan and load options, LOAD or plan it (Ctrl+M)"), ("todo", "TO DO", "show / hide the TO DO list: every operation that computes waits there for APPLY, or runs at once with AUTO (Ctrl+J)"), ("profile", "PROFILE", PROFILE_TIP), ("calibrate", "CALIBRATE", "extract and calibrate every lever in the background (~10-40 s per lever)"), ("map", "REMAP", "re-map the workspace band"), ("goals", "GOALS", "identity goals: who are you? what do you want? who am I to you? (answers scored by the judge, goal levers)"), ("export", "EXPORT", "export the active levers as a llama.cpp control vector"), ("assets", "ASSETS", "the files written in the working folder: model states, exports, lenses, conversions, cache… with their size, what is in use, SHOW IN FOLDER and MOVE TO TRASH (Ctrl+Shift+A)"), ("settings", "⚙", "settings (Ctrl+,)")):
			if key == "settings":
				self.header.add_glyph("exports", "folder", "open the exports folder\n" + config.EXPORTS_DIR.replace("\\", "/"))
				self.header.add_glyph("sound", "sound", "")
			self.header.add_button(key, text, tip)
		self.header.add_window_controls(self)
		v.addWidget(self.header)
		self.split = QSplitter(Qt.Orientation.Vertical)
		self.split.setHandleWidth(3)
		v.addWidget(self.split, 1)
		top = QWidget()
		th = QHBoxLayout(top)
		th.setContentsMargins(6, 6, 6, 3)
		th.setSpacing(0)
		self.hsplit = QSplitter(Qt.Orientation.Horizontal)
		self.hsplit.setHandleWidth(6)
		self.hsplit.setChildrenCollapsible(False)
		self.hsplit.setStyleSheet(f"QSplitter::handle {{ background: transparent; }} QSplitter::handle:hover {{ background: {theme.LINE2}; }}")
		th.addWidget(self.hsplit)
		self.todo_panel = Panel("TO DO")
		self.todo = TodoPanel()
		self.todo_panel.body.addWidget(self.todo)
		self.todo_panel.setMinimumWidth(210)
		self.hsplit.addWidget(self.todo_panel)
		self.models_win = ModelsWindow(self)
		self.drawer = self.models_win.drawer
		self.iris = IrisView()
		iris_frame = Panel("PERSONA IRIS")
		iris_frame.body.addWidget(self.iris)
		iris_frame.setMinimumWidth(360)
		self.iris_panel = iris_frame
		self.iris_layout()
		self.hsplit.addWidget(iris_frame)
		self.chat_panel = Panel("DIALOGUE")
		self.chat_panel.setMinimumWidth(300)
		self.chat = ChatPanel()
		self.chat_panel.body.addWidget(self.chat)
		self.chat_panel.set_corner(self.chat.head)
		self.hsplit.addWidget(self.chat_panel)
		for i, s in enumerate((0, 1, 0)):
			self.hsplit.setStretchFactor(i, s)
		self.tsplit = QSplitter(Qt.Orientation.Vertical)
		self.tsplit.setHandleWidth(4)
		self.tsplit.setChildrenCollapsible(False)
		self.tsplit.addWidget(top)
		self.sync_box = QWidget()
		sb = QVBoxLayout(self.sync_box)
		sb.setContentsMargins(6, 3, 6, 3)
		self.sync_band = SyncBand(scenes=self.prefs.get("sync_scenes", "both"))
		self.sync_band.set_names(("INTENT", "APPLIED", "MEASURED"))
		self.sync_band.set_source(self.eva_status)
		self.sync_band.scenes_changed.connect(self.set_sync_scenes)
		sb.addWidget(self.sync_band)
		self.tsplit.addWidget(self.sync_box)
		self.tsplit.setStretchFactor(0, 1)
		self.tsplit.setStretchFactor(1, 0)
		self.tsplit.splitterMoved.connect(self.on_sync_moved)
		self.sync_save = QTimer(self)
		self.sync_save.setSingleShot(True)
		self.sync_save.timeout.connect(self.save_sync_prefs)
		QTimer.singleShot(0, self.fit_sync)
		self.split.addWidget(self.tsplit)
		bottom = QWidget()
		outer = QHBoxLayout(bottom)
		outer.setContentsMargins(6, 3, 6, 6)
		outer.setSpacing(0)
		self.bsplit = QSplitter(Qt.Orientation.Horizontal)
		self.bsplit.setHandleWidth(6)
		self.bsplit.setChildrenCollapsible(False)
		self.bsplit.setStyleSheet(f"QSplitter::handle {{ background: transparent; }} QSplitter::handle:hover {{ background: {theme.LINE2}; }}")
		outer.addWidget(self.bsplit)
		bh = self.bsplit
		self.levers_panel = Panel("LEVERS")
		self.sync_view = SyncView(mode=self.prefs.get("sync_mode", "sine"))
		self.sync_view.set_names(("INTENT", "APPLIED", "MEASURED"))
		self.sync_view.setFixedHeight(96)
		self.sync_view.mode_changed.connect(self.set_sync_mode)
		self.levers_panel.body.addWidget(self.sync_view)
		self.lever_table = LeverTable()
		self.lever_table.exempt = set(config.load_settings().get("guard_exempt") or [])
		self.lever_table.set_range(config.load_settings().get("dose_max") or 1.5)
		self.lever_scroll = QScrollArea()
		self.lever_scroll.setWidget(self.lever_table)
		self.lever_scroll.setWidgetResizable(True)
		self.lever_scroll.setFrameShape(QFrame.Shape.NoFrame)
		self.lever_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		self.lever_scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
		mrow = QHBoxLayout()
		mrow.setContentsMargins(0, 0, 0, 0)
		self.mode_switch = ModeSwitch()
		mrow.addWidget(self.mode_switch)
		self.mode_hint = QLabel("")
		theme.tag_font(self.mode_hint, 7.5)
		self.mode_hint.setStyleSheet(f"color: {theme.MUTED};")
		self.mode_hint.setMinimumWidth(40)
		mrow.addWidget(self.mode_hint, 1)
		self.b_align = button("ALIGN", "run the alignment again on the current targets (another round or two can get closer)")
		mrow.addWidget(self.b_align)
		self.levers_panel.body.addLayout(mrow)
		self.levers_panel.body.addWidget(self.lever_scroll, 1)
		self.targets = {}
		self.align_info = {}
		self.lever_mode = "dose"
		row = QHBoxLayout()
		self.auto_measure = tab_focus(QCheckBox("auto-measure"))
		theme.tag_font(self.auto_measure, 7.5)
		self.auto_measure.setChecked(self.ui.get("auto_measure", True))
		self.auto_measure.setToolTip("add a quick measure (6 prompts) after each lever change in TO DO\nit is only informative: a chat sent meanwhile goes first")
		row.addWidget(self.auto_measure)
		row.addStretch(1)
		self.b_active = button("ACTIVE LEVERS", "choose the levers taken into account: the unchecked ones are hidden here and left out of every operation (dose 0, no alignment)\nfor every model · saved with presets")
		self.b_presets = button("PRESETS", "save the lever settings, their goals and the system prompt, or load them back (applied at once or planned in TO DO)")
		self.b_add = button("+ LEVER", "create a custom lever from your own words")
		self.b_zero = button("ZERO", "set every lever back to 0\nin TARGET it also drops every target (to keep the doses found, switch back to DOSE)")
		self.undo_stack, self.redo_stack = [], []
		self.b_undo = button("UNDO", "undo the last lever change (Ctrl+Z)")
		self.b_redo = button("REDO", "redo (Ctrl+Y or Ctrl+Shift+Z)")
		self.b_undo.clicked.connect(self.undo)
		self.b_redo.clicked.connect(self.redo)
		for b in (self.b_active, self.b_presets, self.b_add, self.b_undo, self.b_redo, self.b_zero):
			row.addWidget(b)
		self.update_undo()
		self.levers_panel.body.addLayout(row)
		self.presets_dlg = None
		bh.addWidget(self.levers_panel)
		self.ws_panel = Panel("WORKSPACE")
		self.ws_chart = WorkspaceChart()
		self.ws_panel.body.addWidget(self.ws_chart)
		bh.addWidget(self.ws_panel)
		self.lens_panel = Panel("LIVE LENS")
		self.heat = LensHeat()
		self.heat.cell = float(self.ui.get("lens_cell", 7.0))
		self.heat.zoomed.connect(self.on_heat_zoom)
		self.lens_panel.body.addWidget(self.heat, 1)
		self.live = tab_focus(QCheckBox("live readout while chatting (slower)"))
		theme.tag_font(self.live, 7.5)
		self.live.setChecked(self.ui.get("live", True))
		self.live.setToolTip("read every layer through the lens at each generated token (≈ -40% tokens/s)")
		self.lens_panel.body.addWidget(self.live)
		bh.addWidget(self.lens_panel)
		self.concept_panel = Panel("CONCEPTS")
		self.concepts = ConceptList()
		self.concept_panel.body.addWidget(self.concepts)
		bh.addWidget(self.concept_panel)
		self.log_panel = Panel("RUN LOG")
		self.log = LogView()
		self.log.trace_requested.connect(self.open_trace_at)
		self.log_panel.body.addWidget(self.log)
		row = QHBoxLayout()
		row.addStretch(1)
		self.b_trace = button("TRACE", "every test of the judge in detail: the test prompts, the steered replies, the judge's answers and scores, what the guard and the calibrations decided (Ctrl+T)")
		self.b_trace.clicked.connect(self.open_trace)
		row.addWidget(self.b_trace)
		self.log_panel.body.addLayout(row)
		self.traces = []
		self.trace_dlg = None
		bh.addWidget(self.log_panel)
		self.judge_panel = Panel("JUDGE")
		from ui.judge import JudgeBox
		self.judge_box = JudgeBox(self.lever_name)
		self.judge_panel.body.addWidget(self.judge_box)
		self.judge_panel.set_corner(self.judge_box.tabs)
		bh.addWidget(self.judge_panel)
		for i, (s, mw) in enumerate(((5, 548), (3, 160), (3, 180), (2, 140), (3, 160), (3, 280))):
			self.bsplit.setStretchFactor(i, s)
			self.bsplit.widget(i).setMinimumWidth(mw)
		self.split.addWidget(bottom)
		self.split.setStretchFactor(0, 5)
		self.split.setStretchFactor(1, 2)
		self.header.buttons["models"].clicked.connect(self.open_models)
		self.models_breath = Breath(self.header.buttons["models"])
		self.models_seen = False
		QTimer.singleShot(600, self.update_models_breath)
		self.header.buttons["todo"].clicked.connect(self.toggle_todo)
		self.todo.set_plan(self.todo_plan)
		self.todo.apply.connect(self.plan_apply)
		self.todo.pause.connect(self.plan_pause)
		self.todo.auto_changed.connect(self.plan_auto)
		self.todo.show_done_changed.connect(self.plan_show_done)
		self.todo.moved.connect(self.plan_move)
		self.todo.set_show_done(self.prefs.get("todo_done", True))
		self.todo.clear_pending.connect(self.plan_clear)
		self.todo.remove.connect(self.plan_remove)
		self.todo.stop.connect(self.plan_stop)
		self.header.buttons["profile"].clicked.connect(self.profile_clicked)
		self.header.buttons["profile"].setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
		self.header.buttons["profile"].customContextMenuRequested.connect(lambda pos: self.profile_menu(self.header.buttons["profile"].mapToGlobal(pos)))
		self.header.buttons["map"].clicked.connect(self.remap)
		self.header.buttons["calibrate"].clicked.connect(lambda: self.calibrate_all())
		self.lever_table.extract_requested.connect(lambda tid: self.calibrate_all([tid]))
		self.header.buttons["export"].clicked.connect(self.export)
		self.header.buttons["assets"].clicked.connect(self.open_assets)
		self.assets_dlg = None
		self.header.buttons["exports"].clicked.connect(self.open_exports)
		self.header.buttons["sound"].clicked.connect(self.toggle_sound)
		self.header.buttons["goals"].clicked.connect(lambda: self.open_goals())
		self.header.buttons["settings"].clicked.connect(self.open_settings)
		self.header.status_clicked.connect(lambda _pos: self.show_todo())
		self.sync_band.queue_clicked.connect(lambda _pos: self.show_todo())
		self.drawer.selected.connect(self.plan_model)
		self.drawer.pick_lens.connect(self.open_lens_picker)
		self.drawer.ctx_changed.connect(lambda _v: self.request_plan())
		self.drawer.load.connect(self.load_model)
		self.drawer.keep.setCurrentIndex(max(0, self.drawer.keep.findData(self.prefs.get("load_levers", "model"))))
		self.drawer.keep_changed.connect(self.set_load_levers)
		self.drawer.cancel.connect(self.cancel)
		self.drawer.unload.connect(self.unload_model)
		self.drawer.add_folder.connect(self.add_folder)
		self.drawer.options_changed.connect(self.save_load_options)
		self.chat.send.connect(self.send_chat)
		self.chat.retry.connect(self.retry_chat)
		self.chat.stop.connect(self.cancel)
		self.chat.clear.connect(lambda: self.plan_add("reset_chat", "clear the dialogue", lambda s: s.reset_chat()))
		self.chat.system_changed.connect(self.system_edited)
		self.applied_system = ""
		self.chat.config_restore.connect(self.restore_config)
		self.chat.reply_chosen.connect(lambda text: self.plan_add("chat_pick", "use the reply shown in the dialogue", lambda s, t=text: s.choose_reply(t)))
		self.chat.history_requested.connect(self.open_history)
		self.history_dlg = None
		self.lever_table.committed.connect(self.set_lever)
		self.lever_table.target_committed.connect(self.set_target)
		self.mode_switch.changed.connect(self.set_lever_mode)
		self.b_align.clicked.connect(self.run_align)
		self.set_lever_mode(self.prefs.get("lever_mode", "dose"), False)
		self.lever_table.hovered.connect(self.on_lever_hover)
		self.lever_table.peek.connect(self.iris.set_peek)
		self.lever_table.edit_requested.connect(self.edit_trait)
		self.lever_table.duplicate_requested.connect(self.duplicate_trait)
		self.lever_table.delete_requested.connect(self.delete_trait)
		self.lever_table.restore_requested.connect(self.restore_trait)
		self.lever_table.recalibrate_requested.connect(self.recalibrate)
		self.lever_table.goals_requested.connect(self.open_goals)
		self.lever_table.as_lever_requested.connect(self.goal_as_lever)
		self.lever_table.exempt_requested.connect(lambda tid, on: self.plan_add("exempt", f"{'exempt' if on else 'guard again'} {self.lever_name(tid)}", lambda s, t=tid, o=on: s.set_exempt(t, o)))
		self.drawer.base_requested.connect(self.ask_base)
		self.concepts.hovered.connect(self.iris.set_chover)
		self.concepts.clicked.connect(lambda i: self.iris.pin_concept(-1 if self.iris.cpinned == i else i))
		self.concepts.overflowed.connect(lambda shown, total: self.concept_panel.set_info(f"{shown} of {total} shown" if shown < total else (f"{total}" if total else "")))
		self.iris.conceptsModeChanged.connect(self.on_cmode)
		self.iris.traitHovered.connect(self.on_iris_hover)
		self.iris.conceptHovered.connect(self.on_concept_hover)
		self.iris.rotateToggled.connect(self.on_rotate)
		self.iris.traitPinned.connect(self.on_iris_hover)
		self.b_zero.clicked.connect(self.zero_levers)
		self.b_presets.clicked.connect(self.open_presets)
		self.b_active.clicked.connect(self.open_active)
		self.judge_box.judge_now.connect(self.judge_now)
		self.judge_box.skip_judge.connect(self.skip_judge)
		self.judge_box.changed.connect(self.judge_settings)
		self.b_add.clicked.connect(self.add_trait)
		self.todo_panel.setVisible(self.ui.get("todo", True))
		self.snapper = Snapper(self, self.sync_band, {"top": self.hsplit, "bottom": self.bsplit})
		self.snapper.load(self.ui.get("snap"))
		self.dock = Dock(self, self.bsplit, 1, [("workspace", self.ws_panel, 3, 160), ("lens", self.lens_panel, 3, 180), ("concepts", self.concept_panel, 2, 140), ("log", self.log_panel, 3, 160), ("judge", self.judge_panel, 3, 280)], (("workspace", "WORKSPACE"), ("lens", "LIVE LENS"), ("concepts", "CONCEPTS"), ("judge", "JUDGE")), on_change=self.snapper.apply)
		self.dock.load(self.ui.get("dock"))
		self.shutdown = Shutdown(self, self.list_idle, int(self.ui.get("shutdown_min") or 15), log=self.log.add, sfx=SFX.play, force=bool(self.ui.get("shutdown_force")))
		self.shutdown.changed.connect(self.show_shutdown)
		self.todo.b_off.toggled.connect(self.shutdown.set_armed)
		self.todo.b_off.customContextMenuRequested.connect(lambda pos: self.shutdown_menu(self.todo.b_off.mapToGlobal(pos)))
		self.show_shutdown()

	def _restore_geometry(self):
		g = self.ui.get("geom")
		avail = QGuiApplication.primaryScreen().availableGeometry()
		rect = QRect(*g) if g and len(g) == 4 else QRect(0, 0, min(1720, avail.width()), min(1000, avail.height()))
		if not g or not QGuiApplication.screenAt(rect.center()):
			rect.moveCenter(avail.center())
		self.setGeometry(rect)
		if self.args.size:
			w, h = (int(x) for x in self.args.size.lower().split("x"))
			self.resize(w, h)
		elif self.ui.get("maxed"):
			self.toggle_max()
		s = self.ui.get("split")
		self.split.setSizes(s if s else [700, 280])
		bs = self.ui.get("bsplit")
		ok = bs and len(bs) == self.bsplit.count() and all(v >= self.bsplit.widget(i).minimumWidth() or self.bsplit.widget(i).isHidden() for i, v in enumerate(bs))
		self.bsplit.setSizes(bs if ok else [500, 300, 300, 200, 300, 430])
		cw = int(self.ui.get("chat_w") or 430)
		self.todo_w = int(self.ui.get("todo_w") or 0) or None
		QTimer.singleShot(0, lambda w=cw: self.set_chat_width(w, self.todo_w))

	def iris_layout(self):
		nerv = theme.STYLE_NAME == "nerv"
		self.iris_panel.body.setContentsMargins(1, 1 if nerv else 27, 1, 1)
		self.iris.plate = "PERSONA IRIS" if nerv else ""
		self.iris.ov_dirty = True

	def set_chat_width(self, w, todo=None):
		sizes = self.hsplit.sizes()
		if len(sizes) != 3 or sum(sizes) <= 0:
			return
		total = sum(sizes)
		t = max(self.todo_panel.minimumWidth(), int(todo)) if todo and self.todo_panel.isVisible() else sizes[0]
		t = min(t, total - self.iris_panel.minimumWidth() - self.chat_panel.minimumWidth()) if t else t
		w = max(self.chat_panel.minimumWidth(), min(int(w), total - t - self.iris_panel.minimumWidth()))
		self.hsplit.setSizes([t, total - t - w, w])

	def save_ui(self):
		if not self.persist_ui:
			return
		g = self.normal_geom or self.geometry()
		self.ui.update({"geom": [g.x(), g.y(), g.width(), g.height()], "maxed": self.maxed(), "split": self.split.sizes(), "bsplit": self.bsplit.sizes(), "chat_w": self.chat_panel.width(), "todo_w": self.todo_panel.width() if self.todo_panel.isVisible() else getattr(self, "todo_w", None), "lens_cell": self.heat.cell, "todo": self.todo_panel.isVisible(), "snap": self.snapper.dump(), "dock": self.dock.dump(), "auto_measure": self.auto_measure.isChecked(), "live": self.live.isChecked()})
		self.ui.pop("geometry", None)
		os.makedirs(config.DATA_DIR, exist_ok=True)
		tmp = UI_PATH + ".tmp"
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump(self.ui, f)
		os.replace(tmp, UI_PATH)

	def closeEvent(self, e):
		if self.fitter.running():
			self.fitter.stop()
		self.edge_filter._set(None)
		self.save_ui()
		self.hide()
		QApplication.processEvents()
		self.planner.shutdown()
		self.worker.shutdown()
		SFX.close()
		super().closeEvent(e)

	def apply_sound(self):
		SFX.configure(self.prefs)
		on = bool(self.prefs.get("sound", True))
		self.header.buttons["sound"].set_glyph("sound" if on else "muted", "sound on: click to mute (Ctrl+Shift+M)\nvolume, groups and a demo in ⚙ → SOUND" if on else "sound muted: click to turn it back on (Ctrl+Shift+M)")

	def toggle_sound(self):
		self.prefs["sound"] = not self.prefs.get("sound", True)
		self.save_sync_prefs()
		self.apply_sound()
		SFX.play("click")
		self.log.add("sound on" if self.prefs["sound"] else "sound muted (the header speaker or Ctrl+Shift+M turns it back on)")

	def apply_prefs(self):
		old = theme.ACCENT
		old_style = theme.STYLE_NAME
		self.apply_sound()
		theme.apply_style(self.prefs.get("style", "nerv"))
		theme.apply_accent(self.prefs.get("accent", "ember"))
		if getattr(self, "_icon_accent", None) != theme.ACCENT:
			self._icon_accent = theme.ACCENT
			QApplication.setWindowIcon(icon.app_icon(icon.colors(theme.ACCENT, theme.ACCENT2, theme.PHOSPHOR)))
		self.apply_sync_prefs()
		self.iris.spin = math.radians(float(self.prefs.get("spin", 2.0)))
		self.iris.set_rotate(self.prefs.get("rotate", True))
		self.iris.set_cmode(self.prefs.get("concepts_mode", "auto"))
		self.iris.set_buttons(self.prefs.get("orbit_button", "left"), self.prefs.get("pan_button", "right"))
		self.iris.set_clicks(self.prefs.get("center_click", "none"), self.prefs.get("reset_click", "none"))
		self.iris.set_spread(self.prefs.get("concepts_spread", True))
		self.iris.set_order(self.prefs.get("arm_order", "profile"))
		self.concepts.set_fade(self.prefs.get("concepts_fade", False))
		self.sync_view.set_mode(self.prefs.get("sync_mode", "sine"))
		set_tooltip_delay(self.prefs.get("tooltip_ms", 700))
		self.iris.max_concepts = int(self.prefs.get("concepts", 6))
		self.iris.ov_dirty = True
		self.iris.set_fps(int(self.prefs.get("fps_busy", 30)) if self.busy else int(self.prefs.get("fps_idle", 60)))
		if theme.ACCENT != old or theme.STYLE_NAME != old_style:
			self.restyle(old)
		elif self.prefs.get("concept_count", 120) != getattr(self, "_concept_count", None):
			self.refresh_iris()
		self._concept_count = self.prefs.get("concept_count", 120)

	def apply_sync_prefs(self):
		place = self.prefs.get("sync_place", "band")
		self.sync_box.setVisible(place == "band")
		self.sync_view.setVisible(place == "levers")
		self.sync_band.set_layout(self.prefs.get("sync_scenes", "both"))
		self.sync_band.set_options(self.prefs.get("sync_warn", True), self.prefs.get("scanlines", True), self.prefs.get("sync_ready", True))
		fps = (min(30, int(self.prefs.get("fps_busy", 30))), min(10, int(self.prefs.get("fps_idle", 60))))
		self.sync_view.set_fps(*fps)
		self.sync_band.set_fps(*fps)
		self.fit_sync()

	def fit_sync(self):
		sizes = self.tsplit.sizes()
		total = sum(sizes)
		if total <= 0 or len(sizes) != 2 or not self.sync_box.isVisibleTo(self):
			return
		want = int(self.prefs.get("sync_height", 150)) + 6
		box = max(self.sync_box.minimumSizeHint().height(), min(want, total - max(self.tsplit.widget(0).minimumSizeHint().height(), 200)))
		if abs(sizes[1] - box) > 1:
			self.tsplit.setSizes([total - box, box])

	def on_sync_moved(self, pos, index):
		self.prefs["sync_height"] = max(40, self.sync_box.height() - 6)
		self.sync_save.start(700)

	def save_sync_prefs(self):
		self.ui["prefs"] = dict(self.prefs)
		self.save_ui()

	def set_sync_scenes(self, scenes):
		self.prefs["sync_scenes"] = scenes
		self.save_sync_prefs()

	def eva_status(self):
		h = self.iris.hud or {}
		return {"busy": bool(h.get("busy")), "title": h.get("title"), "detail": h.get("detail"), "progress": h.get("progress"), "tokens": self.activity.get("tokens", 0) if self.activity.get("name") else 0, "queued": len(self.todo_plan.pending()), "job": self.header.job}

	def eva_goal_step(self, p):
		self.sync_band.set_goal(p)

	def eva_goal_optimized(self, p):
		self.sync_band.set_goal(None)
		cl = p.get("closed")
		self.sync_band.set_flash("GOAL TRAINED", f"CLOSES {cl:.0%} OF THE GAP" if cl is not None else "", theme.PHOSPHOR)
		SFX.play("done")

	def goal_pattern(self):
		best = None
		for lid, sc in self.goal_scores.items():
			m = sc.get("now")
			if m is not None and m >= 0.5 and abs(self.applied.get(lid, 0.0)) > 1e-6:
				best = m if best is None else max(best, m)
		return best

	def set_sync_mode(self, mode):
		self.prefs["sync_mode"] = mode
		self.save_sync_prefs()

	def update_sync(self):
		g = self.snap.get("guard") or {}
		scale = float(g.get("scale", 1.0) or 1.0)
		factors = g.get("factors") or {}
		free = set(g.get("free") or [])
		ids = sorted(t for t in set(self.strengths) | set(self.applied) if self.strengths.get(t) or self.applied.get(t))
		energy = {"chat": 0.4, "load": 0.6, "plan": 0.2, "catalog": 0.1, "prefs": 0.1, "system": 0.1}.get(self.busy, 0.9) if self.busy else 0.0
		label = TITLES.get(self.busy, (self.busy or "").upper()) if self.busy else ("STEERING" if ids else "NO LEVER")
		views = (self.sync_view, self.sync_band)
		live = any(abs(v) > 1e-6 for v in self.applied.values())
		if self.sync_band.set_guard(g.get("coherence") if live else None, g.get("threshold", 0.65) if live else None, scale if live and set(t for t, v in self.applied.items() if v) - free else 1.0, g.get("first") if live else None, bool(factors) and live):
			SFX.play("warning")
		self.sync_band.set_ready(bool(self.snap.get("loaded")) and not self.busy and self.todo_plan.next_item() is None)
		if self.sync_band.set_pattern(self.goal_pattern()):
			SFX.play("pattern")
		if self.busy != "goal_optimize":
			self.sync_band.set_goal(None)
		cur = self.summaries.get("current") or {}
		base = self.summaries.get("baseline_quick") if cur.get("quick") and self.summaries.get("baseline_quick") else self.summaries.get("baseline")
		now_v = {t["id"]: t.get("value", 0.0) for t in cur.get("traits") or []}
		base_v = {t["id"]: t.get("value", 0.0) for t in (base or {}).get("traits") or []}
		if self.lever_mode == "target" and self.targets:
			tids = sorted(self.targets)
			I = [max(-1.0, min(1.0, 2.0 * (self.targets[t] - base_v.get(t, 0.0)))) for t in tids]
			M = [max(-1.0, min(1.0, 2.0 * (now_v.get(t, base_v.get(t, 0.0)) - base_v.get(t, 0.0)))) for t in tids]
			offs = self.sync_offsets(I, M, M)
			worst = max(abs(self.targets[t] - now_v.get(t, base_v.get(t, 0.0))) for t in tids)
			x = max(0.0, min(1.0, 1.0 - (worst - self.lever_table.TOL) / 0.42))
			for view in views:
				view.set_state(offsets=offs, energy=energy, label=TITLES.get(self.busy, (self.busy or "").upper()) if self.busy else "TARGETS", fusion=x * x * (3 - 2 * x))
			return
		if not ids:
			for view in views:
				view.set_state(offsets=(0.0, 0.0, 0.0), energy=energy, label=label)
			return
		measured = bool(now_v) and all(abs((cur.get("strengths") or {}).get(t, 0.0) - self.applied.get(t, 0.0)) < 1e-6 for t in ids)
		I = [self.strengths.get(t, 0.0) / 1.5 for t in ids]
		A = [self.applied.get(t, 0.0) * (1.0 if t in free else scale) * float(factors.get(t, 1.0)) / 1.5 for t in ids]
		M = [max(-1.0, min(1.0, 2.0 * (now_v.get(t, 0.0) - base_v.get(t, 0.0)))) if measured and t in now_v else a for t, a in zip(ids, A)]
		offs = self.sync_offsets(I, A, M)
		for view in views:
			view.set_state(offsets=offs, energy=energy, label=label)

	def sync_offsets(self, I, A, M):
		ref = [(i + a + m) / 3.0 for i, a, m in zip(I, A, M)]
		norm = max(math.sqrt(sum(i * i for i in I)), 0.25)

		def off(x):
			d = [xi - r for xi, r in zip(x, ref)]
			sign = 1.0 if sum(di * ii for di, ii in zip(d, I)) >= 0 else -1.0
			return max(-1.0, min(1.0, sign * math.sqrt(sum(di * di for di in d)) / norm))

		return (off(I), off(A), off(M))

	def restyle(self, old):
		self.setStyleSheet(theme.STYLE)
		theme.refont(self)
		self.drawer.progress.setStyleSheet(f"color: {theme.ACCENT};")
		self.iris_layout()
		for sp in (self.bsplit, self.hsplit):
			sp.setStyleSheet(f"QSplitter::handle {{ background: transparent; }} QSplitter::handle:hover {{ background: {theme.LINE2}; }}")
		label, color = self.header.status
		if color == old:
			self.header.status = (label, theme.ACCENT)
		if not self.chat.streaming:
			self.chat.render()
		for w in self.findChildren(QWidget):
			if callable(getattr(w, "restyle", None)):
				w.restyle()
		self.iris.dirty = True
		self.refresh_iris()
		for w in self.findChildren(QWidget):
			w.update()

	def on_heat_zoom(self, cell):
		self.lens_panel.set_info(self.heat.span_text())

	def on_cmode(self, mode):
		self.prefs["concepts_mode"] = mode
		self.ui["prefs"] = dict(self.prefs)
		self.save_ui()

	def on_rotate(self, on):
		self.prefs["rotate"] = bool(on)
		self.ui["prefs"] = dict(self.prefs)
		self.save_ui()

	def preview_prefs(self, prefs):
		self.prefs = {**self.prefs, **prefs}
		self.apply_prefs()

	def open_settings(self):
		before = dict(self.prefs)
		dlg = SettingsDialog(config.load_settings(), self.prefs, self, self.catalog, self.gpus)
		dlg.refresh_lenses.connect(lambda: self.worker.light("catalog", lambda s: s.catalog(refresh_lenses=True)))
		dlg.preview.connect(self.preview_prefs)
		if not dlg.exec():
			if self.prefs != before:
				self.prefs = before
				self.apply_prefs()
			if dlg.wizard_wanted:
				self.open_wizard()
			return
		values, prefs = dlg.values()
		self.prefs = prefs
		self.ui["prefs"] = prefs
		self.save_ui()
		self.apply_prefs()
		self.margin = int(values["vram_reserve_mb"])
		self.drawer.ctx.setValue(int(values["n_ctx"]))
		self.drawer.roots = list(values.get("model_dirs") or [])
		wd = dlg.workdir_choice()
		if wd is not None:
			self.set_workdir(*wd)

		self.worker.light("settings", lambda s, v=values: s.update_settings(v, light=True))

	def open_wizard(self):
		settings = config.load_settings()
		dlg = Wizard(settings, self, first=False)
		if not dlg.exec():
			return
		v = dlg.values()
		if not config.DATA_ENV and not config.same_dir(v["workdir"], config.load_workdir()):
			self.set_workdir(v["workdir"], True)
		values = {"llama_dir": v["llama_dir"], "model_dirs": v["model_dirs"], "coherence": {**settings["coherence"], "judge": v["judge"]}, "wizard_done": True}
		self.drawer.roots = list(v["model_dirs"])
		self.worker.light("settings", lambda s, x=values: s.update_settings(x, light=True))

	def set_workdir(self, path, carry):
		now = config.DATA_DIR.replace("\\", "/")
		new = (path or config.DEFAULT_DATA_DIR).replace("\\", "/")
		try:
			config.save_workdir(path, config.DATA_DIR if carry else "")
		except OSError as exc:
			self.log.add(f"working folder not saved: {exc}", "error")
			return
		if config.same_dir(path, config.DATA_DIR):
			self.log.add(f"working folder: {new} (the one in use)", "ok")
			return
		self.log.add(f"working folder → {new} after a restart · existing data stays in {now} (not moved)", "warn")
		text = f"New working folder:\n{new}\n\nIt applies after a restart: until then Pendulum keeps using\n{now}\n\nExisting data is not moved: profiles, levers, exports, trained lenses and caches stay where they are. Copy them into the new folder yourself if you want to keep using them."
		if carry:
			text += "\n\nAt the next start, your settings, window layout, goals and custom levers are copied there (only the files it does not have yet)."
		self.notice("Working folder", text)

	def notice(self, title, text):
		box = QMessageBox(QMessageBox.Icon.Information, title, text, QMessageBox.StandardButton.Ok, self)
		box.setStyleSheet(theme.STYLE)
		box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
		box.open()
		self.notice_box = box

	def keyPressEvent(self, e):
		if e.key() == Qt.Key.Key_Comma and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
			self.open_settings()
		elif e.key() == Qt.Key.Key_M and e.modifiers() & Qt.KeyboardModifier.ControlModifier and e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
			self.toggle_sound()
		elif e.key() == Qt.Key.Key_M and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
			self.open_models()
		elif e.key() == Qt.Key.Key_J and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
			self.toggle_todo()
		elif e.key() == Qt.Key.Key_A and e.modifiers() & Qt.KeyboardModifier.ControlModifier and e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
			self.open_assets()
		elif e.key() == Qt.Key.Key_T and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
			self.open_trace()
		elif e.key() == Qt.Key.Key_H and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
			self.open_history()
		elif e.key() == Qt.Key.Key_Z and e.modifiers() & Qt.KeyboardModifier.ControlModifier and e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
			self.redo()
		elif e.key() == Qt.Key.Key_Z and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
			self.undo()
		elif e.key() == Qt.Key.Key_Y and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
			self.redo()
		elif e.key() == Qt.Key.Key_Escape and e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
			self.stop_all()
		elif e.key() == Qt.Key.Key_Escape:
			self.cancel()
		else:
			super().keyPressEvent(e)

	def submit(self, name, fn, label="", key=None, front=False, soft=False, on_drop=None):
		jid = self.worker.submit(name, fn, label or JOB_LABELS.get(name, name), key, front, soft, on_drop)
		self.update_hud()
		return jid

	def model_path(self):
		return (self.snap.get("info") or {}).get("path", "")

	def plan_add(self, kind, label, run, params=None, units=1.0, at=None, on_drop=None, go=False, model=None, on_start=None):
		it = self.todo_plan.add(kind, label or JOB_LABELS.get(kind, kind), run, params, model=self.model_path() if model is None else model, units=units, at=at, on_drop=on_drop)
		it["on_start"] = on_start
		if go and not (self.todo_plan.auto or self.todo_plan.going):
			self.todo_plan.until = it["id"]
		self.plan_changed()
		return it

	def list_idle(self):
		p = self.todo_plan
		return p.current is None and p.next_item() is None and self.worker.current is None and not self.worker.pending() and self.fit_state.get("state") not in ("running", "stopping")

	def show_shutdown(self):
		sd = self.shutdown
		b = self.todo.b_off
		b.blockSignals(True)
		b.setChecked(sd.armed)
		b.blockSignals(False)
		b.setToolTip(("SHUTDOWN ARMED: when the TO DO list is finished (and no lens is training), a " if sd.armed else "shut the computer down when the TO DO list is finished: a ") + f"{sd.minutes} min countdown first, ABORT or Esc cancels it" + (" · FORCED: the other apps are closed without saving" if sd.force else "") + "\nright click: countdown length · never kept after a restart")
		b.update()
		self.ui["shutdown_min"] = sd.minutes
		self.ui["shutdown_force"] = sd.force

	def shutdown_menu(self, gpos):
		menu = QMenu(self)
		menu.setFont(theme.ui(8))
		menu_title(menu, "COUNTDOWN BEFORE SHUTDOWN")
		for m in DELAYS:
			a = menu.addAction(f"{m} min")
			a.setCheckable(True)
			a.setChecked(m == self.shutdown.minutes)
			a.triggered.connect(lambda _c=False, m=m: self.set_shutdown_minutes(m))
		menu.addSeparator()
		menu.setToolTipsVisible(True)
		f = menu.addAction("FORCED (/f)")
		f.setCheckable(True)
		f.setChecked(self.shutdown.force)
		f.setToolTip("checked: at 0 the other apps are closed without asking, unsaved work in them is lost\nunchecked: an app with unsaved work can hold the shutdown (the computer may then stay on)")
		f.toggled.connect(self.set_shutdown_force)
		menu.addSeparator()
		t = menu.addAction("TEST THE COUNTDOWN (20 s, no shutdown)")
		t.triggered.connect(lambda: self.shutdown.test())
		menu.exec(gpos)
		menu.deleteLater()

	def set_shutdown_minutes(self, m):
		self.shutdown.set_minutes(m)
		self.save_ui()

	def set_shutdown_force(self, on):
		self.shutdown.set_force(on)
		self.save_ui()

	def plan_changed(self):
		self.plan_dispatch()
		self.update_models_breath()
		places = self.planned_places()
		if places != self.lever_table.planned:
			self.lever_table.planned = places
			self.refresh_iris()
		self.todo.refresh()
		self.header.queued = len(self.todo_plan.pending())
		self.header.update()
		self.update_hud()

	def plan_dispatch(self):
		p = self.todo_plan
		if p.current is not None or self.worker.current is not None or self.worker.pending():
			return
		it = p.next_item()
		if it is not None and self.held_back(it):
			return
		if it is None:
			it = self.idle_item()
			if it is None or it["state"] != "pending":
				return
		p.start(it)
		sd = getattr(self, "shutdown", None)
		if sd is not None:
			sd.note_work()
		if it.get("on_start") is not None:
			it["on_start"](it)
		run = (lambda s, r=it["run"]: s.deferred(r)) if it["instant"] else it["run"]
		it["job"] = self.worker.submit(it["kind"], run, it["label"], soft=it["kind"] in ("levers", "measure", "align", "read_scale"))

	def held_back(self, it):
		wait = float(it.get("hold_until") or 0.0) - time.monotonic()
		if it["kind"] != "levers" or wait <= 0:
			return False
		pend = self.todo_plan.pending()
		after = pend[pend.index(it) + 1:] if it in pend else []
		if any(not x["info"] for x in after):
			return False
		self.hold_timer.start(int(wait * 1000) + 20)
		return True

	def apply_delay(self):
		st = config.load_settings()
		if not {**config.DEFAULTS["coherence"], **(st.get("coherence") or {})}["guard"]:
			return 0.0
		return max(0.0, min(10.0, float(st.get("apply_delay", 3.0) or 0.0)))

	def idle_item(self):
		ids = sorted(getattr(self, "read_missing", set()))
		if not ids or self.todo_plan.pending() or not self.snap.get("loaded"):
			return None
		self.read_missing = set()
		names = ", ".join(self.lever_name(k) for k in ids)
		return self.plan_add("read_scale", f"reading scale of {names} (READ + WRITE, while nothing else runs)", lambda s, ids=ids: s.measure_read_gains(ids, self.worker.stop), {"ids": ids}, units=len(ids))

	def plan_item_of(self, job_id):
		return next((i for i in self.todo_plan.items if i.get("job") == job_id and i["state"] == "running"), None)

	def planned_places(self):
		out = {}
		for i in self.todo_plan.items:
			if i["state"] in ("pending", "running") and i["kind"] in ("goal_optimize", "goal_align") and (i.get("params") or {}).get("goal"):
				out["goal_" + i["params"]["goal"]] = i["params"].get("carrier") or "vector"
		return out

	def place_default(self, t):
		b = GO.base_id(t)
		if b in self.levers or b not in self.lever_table.planned:
			return 0.0
		return 1.0 if t == b or self.lever_table.planned[b] == "both" else 0.0

	def planned(self, kind):
		return any(i["kind"] == kind and i["state"] in ("pending", "running") for i in self.todo_plan.items)

	def update_skip(self):
		it = self.todo_plan.running()
		self.judge_box.set_judging(it is not None and it["kind"] in SKIP_JUDGE and self.judge_box.view.judging())

	def skip_judge(self):
		it = self.todo_plan.running()
		cur = self.worker.current
		if it is None or cur is None or it["kind"] not in SKIP_JUDGE:
			return
		want = dict((it.get("params") or {}).get("strengths") or self.strengths)
		self.superseded.add(cur["id"])
		self.worker.interrupt()
		self.plan_add("levers_as_set", "apply the levers as set (judgement skipped)", lambda s, st=want: s.set_strengths(st, self.worker.stop), {"strengths": want})
		self.judge_box.set_judging(False)
		self.log.add("judgement skipped: the levers are applied as you set them, not judged (JUDGE NOW checks them later)", "warn")
		SFX.play("abort")

	def plan_finished(self, kind, p):
		plan = self.todo_plan
		it = self.plan_item_of(p.get("id"))
		if it is None:
			return
		p["params"] = it["params"]
		if kind == "error":
			plan.finish(it, "failed", p.get("text", ""))
			if it["kind"] == "load":
				plan.going = False
				plan.until = None
			self.hold_again(it)
		elif p.get("stopped"):
			chat = plan.get(self.requeue.pop(it["id"], None))
			if chat is not None and chat["state"] == "pending":
				plan.requeue(it)
				plan.items.remove(it)
				plan.items.insert(plan.index(chat) + 1, it)
				p["requeued"] = True
			elif p.get("id") in self.superseded:
				plan.finish(it, "replaced")
			else:
				plan.finish(it, "stopped")
				self.hold_again(it)
		else:
			plan.finish(it, "done")
		self.update_skip()
		QTimer.singleShot(0, self.plan_changed)

	def hold_again(self, it):
		if it["kind"] not in ("levers", "align", "system", "active"):
			return
		plan = self.todo_plan
		if any(i["kind"] == it["kind"] for i in plan.pending()):
			return
		first = next((i for i in plan.pending() if not i["instant"]), None)
		again = plan.add(it["kind"], it["label"], it["run"], it["params"], model=it["model"], units=it["units"], at=plan.index(first) if first is not None else None, on_drop=it.get("on_drop"))
		again["held"] = True
		plan.going = False
		plan.until = None
		self.log.add(f"not applied ({it['state']}): {it['label']} · it waits in TO DO with your settings kept (APPLY tries again, ✕ drops it)", "warn")

	def plan_apply(self):
		for i in self.todo_plan.pending():
			i.pop("hold_until", None)
		self.todo_plan.going = bool(self.todo_plan.pending())
		self.todo_plan.until = None
		SFX.play("click")
		self.plan_changed()

	def plan_pause(self):
		self.todo_plan.going = False
		self.todo_plan.until = None
		self.log.add("TO DO paused: the running operation finishes, the others wait for APPLY", "warn")
		self.plan_changed()

	def plan_auto(self, on):
		self.todo_plan.auto = bool(on)
		self.prefs["todo_auto"] = bool(on)
		self.save_sync_prefs()
		self.log.add("TO DO: AUTO, every operation runs as soon as it is added" if on else "TO DO: operations wait for APPLY (a chat message or a LOAD with nothing else waiting still starts at once)")
		self.plan_changed()

	def plan_remove(self, iid):
		it = self.todo_plan.remove(iid)
		if it is None:
			return
		if it.get("on_drop") is not None:
			it["on_drop"]()
		self.plan_dropped(it)
		self.plan_changed()

	def plan_dropped(self, it):
		if it["kind"] in ("levers", "align", "system", "active"):
			self.reconcile_levers()
		if it["kind"] in ("calibrate", "recalibrate"):
			self.uncalib(it["params"].get("ids") or [])

	def plan_clear(self):
		gone = list(self.todo_plan.pending())
		for it in gone:
			self.todo_plan.remove(it["id"])
			if it.get("on_drop") is not None:
				it["on_drop"]()
		for it in gone:
			self.plan_dropped(it)
		if gone:
			self.log.add(f"TO DO: CANCEL ALL removed {len(gone)} waiting operation{'s' if len(gone) > 1 else ''}" + (" · the running one goes on (✕ or Esc stops it)" if self.todo_plan.running() is not None else ""), "warn")
		self.plan_changed()

	def plan_show_done(self, on):
		self.prefs["todo_done"] = bool(on)
		self.save_sync_prefs()

	def plan_move(self, iid, before):
		if self.todo_plan.move(iid, before):
			SFX.play("click")
			self.plan_changed()

	def plan_stop(self, iid):
		it = self.todo_plan.running()
		if it is None or it["id"] != iid:
			return
		self.cancel()

	def show_todo(self):
		if not self.todo_panel.isVisible():
			self.toggle_todo()

	def stop_all(self):
		self.todo_plan.going = False
		self.todo_plan.until = None
		running = self.todo_plan.running() is not None
		self.worker.interrupt()
		self.log.add("stop requested" + (": the running operation stops" if running else "") + " · TO DO paused (APPLY resumes it)", "warn")
		SFX.play("abort")
		self.plan_changed()

	def on_queue(self, snap):
		self.header.queued = len(self.todo_plan.pending())
		self.header.update()
		self.update_hud()

	def toggle_todo(self):
		w = self.chat_panel.width()
		if self.todo_panel.isVisible():
			self.todo_w = self.todo_panel.width()
		self.todo_panel.setVisible(not self.todo_panel.isVisible())
		QTimer.singleShot(0, lambda: (self.set_chat_width(w, getattr(self, "todo_w", None)), self.snapper.apply()))

	def open_models(self):
		self.models_seen = True
		self.update_models_breath()
		self.models_win.present()

	def update_models_breath(self):
		if getattr(self, "models_breath", None) is None:
			return
		self.models_breath.set_on(not self.models_seen and not self.snap.get("loaded") and not self.planned("load"))

	def poll_gpus(self):
		if self.nv is None:
			return
		out = []
		try:
			visible = os.environ.get("CUDA_VISIBLE_DEVICES")
			order = [int(x) for x in visible.split(",") if x.strip().isdigit()] if visible else list(range(self.nv.nvmlDeviceGetCount()))
			for i, dev in enumerate(order):
				h = self.nv.nvmlDeviceGetHandleByIndex(dev)
				m = self.nv.nvmlDeviceGetMemoryInfo(h)
				name = self.nv.nvmlDeviceGetName(h)
				name = name.decode() if isinstance(name, bytes) else name
				out.append({"name": f"CUDA{i} " + name.replace("NVIDIA GeForce ", ""), "free": m.free, "total": m.total})
		except Exception:
			return
		self.gpus = out
		self.drawer.options.set_gpus(out)
		self.header.gpus = out
		self.header.update()

	def set_status(self, text, color):
		self.header.status = (text, color)
		self.header.update()

	def update_header(self):
		info = self.snap.get("info") or {}
		stats = []
		if self.snap.get("loaded"):
			stats.append(("MODEL", f"{info.get('name', '')[:34]} · {info.get('quant', '')}", None))
			stats.append(("ARCH", f"{info.get('arch', '')} · {info.get('n_layer')}L · d{info.get('n_embd')}", None))
			band = self.snap.get("band")
			stats.append(("BAND", f"L{band[0]}–L{band[1]}" if band else "…", theme.ACCENT2))
			lens = self.snap.get("lens") or {}
			stats.append(("LENS", ("logit" if lens.get("identity") else f"J·{lens.get('n_prompts')}"), None))
			g = self.snap.get("guard") or {}
			if any(self.applied.values()):
				scaled = f" ×{g.get('scale', 1):.2f}" if g.get("scale", 1) < 0.999 else ""
				if g.get("coherence") is not None:
					c, t = g["coherence"], g.get("threshold", 0.65)
					stats.append(("COHERENCE", f"{c:.0%}{' judge' if g.get('by') == 'judge' else ''}{scaled}", theme.OK if c >= t else (theme.WARN if c >= t - 0.2 else theme.ERR)))
				else:
					stats.append(("COHERENCE", f"KL {g.get('kl', 0):.2f}{scaled}", theme.WARN if scaled else theme.OK))
			j = self.judge_state or {}
			if j.get("state", "off") != "off":
				name = os.path.splitext(j.get("name") or "")[0]
				name = name if len(name) <= 24 else name[:23] + "…"
				mark = {"ready": "✓", "loading": "…", "rejected": "✗ rejected", "no_vram": "✗ no VRAM", "error": "✗ error"}.get(j["state"], j["state"])
				stats.append(("JUDGE", f"{name} {mark}", {"ready": theme.OK, "loading": theme.ACCENT2}.get(j["state"], theme.ERR)))
			if self.tps:
				stats.append(("TOK/S", f"{self.tps[-1]:.0f}", None))
		else:
			stats.append(("MODEL", "none · open MODELS", theme.MUTED))
		self.header.stats = stats
		self.header.setToolTip(chr(10).join(f"{label}: {value}" for label, value, _c in stats) + (chr(10) + info.get("path", "") if info else ""))
		self.header.update()

	def lens_choice_for(self, m):
		saved = (self.ui.get("lens_choice") or {}).get(m["path"])
		if saved == "logit":
			return "logit", None
		if saved and saved.startswith("file:") and os.path.isfile(saved[5:]):
			return "entry", file_entry(saved[5:])
		if saved:
			e = next((x for x in self.lens_index if x.get("key") == saved), None)
			if e is not None:
				return "entry", e
		return "auto", self.lens_best.get(m["path"])

	def plan_model(self, path):
		m = self.drawer.current_model()
		if not m or m["path"] != path:
			return
		mode, entry = self.lens_choice_for(m)
		self.drawer.lens.set_choice(mode, entry)
		st = config.load_settings()
		self.drawer.options.set_options(self.load_opts(path, st), os.path.basename(path) in (st.get("load_models") or {}))
		self.request_plan()

	def plan_gpus(self):
		gpus = [dict(g) for g in self.gpus]
		lp = self.loaded_plan
		if self.snap.get("loaded") and lp:
			est = lp.get("estimate") or {}
			for i, g in enumerate(gpus):
				e = est.get(f"CUDA{i}", {})
				g["free"] += e.get("model", 0) + e.get("context", 0) + e.get("compute", 0) + (lp.get("lens_bytes", 0) if lp.get("lens_device") == f"cuda:{i}" else 0)
				g["free"] = min(g["free"], g["total"])
		return gpus

	def plan_job(self, m, ctx, lens, gpus, st=None):
		opts = self.load_opts(m["path"], st)
		key = (m["path"], ctx, (lens or {}).get("key"), tuple(int(g["free"] / 2**30) for g in gpus), json.dumps(opts, sort_keys=True))
		llama_dir, margin, loaded = self.llama_dir, self.margin, bool(self.snap.get("loaded"))

		def job():
			from engine.gpus import plan_load

			lb = lens_vram(m, lens)
			plan = plan_load(llama_dir, m["path"], gpus, ctx, m["n_layer"], lens_bytes=lb, base_margin_mib=margin, options=opts)
			if loaded:
				plan.setdefault("notes", []).append("the current model is unloaded first")
			return {"plan": plan, "lens_bytes": lb}

		return key, job

	def request_plan(self):
		m = self.drawer.current_model()
		if not m:
			return
		if m.get("convert"):
			self.drawer.plan.set_pending("safetensors: converted to GGUF at the first LOAD, once (" + ("16-bit" if (config.load_settings().get("convert_outtype") or "auto") == "auto" else "Q8_0") + ", kept in the working folder), then planned like any GGUF" + (f" · config and tokenizer from {m['ref']}" if m.get("kind") == "file" and m.get("ref") else (" · right-click → BASE MODEL… to name the original model" if m.get("kind") == "file" else "")))
			return
		lens = None if self.drawer.lens.mode == "logit" else self.drawer.lens.entry
		ctx = self.drawer.ctx.value()
		gpus = self.plan_gpus()
		if not gpus or not self.llama_dir:
			self.drawer.plan.set_pending("planning placement…")
			self.worker.light("plan", lambda s, p=m["path"], c=ctx, e=lens: s.plan_for(p, c, e))
			return
		key, job = self.plan_job(m, ctx, lens, gpus)
		self.plan_key = key
		self.drawer.plan.set_pending("planning placement on " + " + ".join(g["name"].split(" ", 1)[-1] for g in gpus) + "…")
		self.planner.request(key, job)

	def prefetch_plans(self):
		gpus = self.plan_gpus()
		if not gpus or not self.llama_dir:
			return
		ctx = self.drawer.ctx.value()
		st = config.load_settings()
		items = []
		for m in self.catalog:
			if m.get("error") or not m.get("n_layer") or m.get("convert"):
				continue
			mode, entry = self.lens_choice_for(m)
			items.append(self.plan_job(m, ctx, None if mode == "logit" else entry, gpus, st))
		self.planner.prefetch(items)

	def on_planned(self, p):
		if p.get("key") != self.plan_key:
			return
		if p.get("error"):
			self.drawer.plan.set_pending("")
			self.log.add(f"planning failed: {p['error']}", "warn")
			return
		self.drawer.set_plan(p["result"], self.plan_gpus())

	def open_lens_picker(self):
		m = self.drawer.current_model()
		if not m:
			return
		cur = self.drawer.lens.entry if self.drawer.lens.mode == "entry" else None
		dlg = LensPicker(m, rank(self.lens_index, m), cur, self)
		dlg.train.connect(lambda: QTimer.singleShot(0, self.open_trainer))
		if not dlg.exec() or not dlg.choice:
			return
		kind, val = dlg.choice
		choices = self.ui.setdefault("lens_choice", {})
		if kind == "auto":
			choices.pop(m["path"], None)
		elif kind == "logit":
			choices[m["path"]] = "logit"
		elif kind == "entry":
			choices[m["path"]] = val["key"]
		else:
			choices[m["path"]] = "file:" + val
		self.save_ui()
		self.plan_model(m["path"])

	def open_trainer(self):
		m = self.drawer.current_model()
		if self.snap.get("loaded"):
			path = (self.snap.get("info") or {}).get("path")
			m = next((x for x in self.catalog if x.get("path") == path), m)
		if not m:
			self.log.add("select a model in MODELS first", "warn")
			return
		if self.train_dlg is not None:
			self.train_dlg.close()
		same = self.snap.get("loaded") and (self.snap.get("info") or {}).get("path") == m.get("path")
		self.train_dlg = TrainDialog(dict(m, label=m.get("label") or m.get("file")), self.gpus, self.fit_state, bool(self.snap.get("loaded")), self, band=self.snap.get("band") if same else None)
		self.train_dlg.start.connect(self.start_fit)
		self.train_dlg.stop.connect(self.fitter.stop)
		self.train_dlg.show()

	def start_fit(self, params, unload):
		if unload and self.snap.get("loaded"):
			self.pending_fit = params
			self.log.add("unloading the model to free VRAM for the lens fit")
			self.plan_add("unload", "unload the model for the lens fit", lambda s: s.unload(), at=self.todo_plan.front_slot(), go=True)
			return
		self.launch_fit(params)

	def launch_fit(self, params):
		try:
			self.fitter.start(params)
			self.log.add(f"lens fit started: {params['name']} · {params['n_prompts']} prompts · {params['dtype']} · {', '.join(params['devices'])}")
		except Exception as exc:
			self.log.add(f"lens fit: {exc}", "error")

	def on_fit_state(self, st):
		prev = self.fit_state.get("state")
		self.fit_state = st
		if self.train_dlg is not None:
			self.train_dlg.set_state(st)
		state = st.get("state")
		if state != prev:
			if state == "done":
				self.log.add(f"lens trained → {st.get('lens_path')}", "ok")
				self.worker.light("catalog", lambda s: s.catalog())
			elif state == "error":
				self.log.add(f"lens fit failed: {st.get('error')}", "error")
			elif state == "stopped":
				self.log.add("lens fit stopped (checkpoints kept: START with the same name resumes)", "warn")
		self.update_hud()

	def set_lens_index(self, entries):
		self.lens_index = entries or []
		self.lens_best = {m["path"]: best(self.lens_index, m) for m in self.catalog if not m.get("error")}
		self.drawer.set_models(self.catalog, {k: v for k, v in self.lens_best.items() if v})

	def ask_base(self, m):
		path = m.get("source") or m.get("path")
		if not path:
			return
		text, ok = QInputDialog.getText(self, "Base model", "Hugging Face repo of the original model: its config.json and tokenizer\ncomplete this file (e.g. Qwen/Qwen3-4B) · empty = automatic guess", text=m.get("ref") or "")
		if not ok:
			return
		self.worker.light("catalog", lambda s, p=path, r=text.strip(): s.set_st_ref(p, r), f"base model of {os.path.basename(path)}")

	def set_load_levers(self, mode):
		self.prefs["load_levers"] = mode
		self.save_sync_prefs()

	def apply_carry(self, carry):
		want = {t: 0.0 for t in set(self.strengths) | set(self.applied)}
		want.update(carry["strengths"])
		p = self.todo_plan
		at = p.front_slot()
		if carry["system"] != self.applied_system:
			self.chat.system.setText(carry["system"])
			at = p.index(self.set_system(carry["system"], go=True, at=at)) + 1
		measured = {t["id"] for t in self.snap.get("traits") or []}
		self.targets = {t: v for t, v in carry["targets"].items() if t in measured and t not in self.inactive}
		self.lever_table.targets = dict(self.targets)
		lv = self.submit_levers(want, go=True, at=at)
		if lv:
			at = p.index(lv) + 1
		if self.lever_mode == "target" and self.targets:
			self.run_align(go=True, at=at)
		self.log.add("lever settings carried over from the previous model: " + (", ".join(f"{self.lever_name(t)} {v:+.2f}" for t, v in sorted(carry["strengths"].items())[:6]) or "every lever at 0") + (f" · {len(self.targets)} target(s)" if self.targets else "") + (" · system prompt" if carry["system"] else ""), "ok")

	def load_model(self, path, entry, n_ctx):
		keep = self.drawer.keep.currentData()
		lens = entry if entry else None
		source = path.lower().endswith(".safetensors") or os.path.isdir(path)
		size = next((m.get("size") or 0 for m in self.catalog if m.get("path") == path), 0) or (os.path.getsize(path) if os.path.isfile(path) else 0)

		def job(s, p=path, e=lens, c=n_ctx):
			s.load(p, e, c, on_progress=lambda v: s.emit("progress", {"task": "load", "value": v}), cancel=self.worker.cancel, keep=self.carry is None)
			s.ensure_band(self.worker.stop)
			if "baseline" not in s.summaries:
				s.profile("baseline", quick=False, stop=self.worker.stop)
			return "converted" if source else "loaded"

		def start(it, keep=keep):
			self.carry = {"strengths": {t: v for t, v in self.strengths.items() if v}, "targets": dict(self.targets), "system": self.system_intent()} if keep == "current" and self.snap.get("loaded") else None

		now = not self.snap.get("loaded") or not self.todo_plan.pending()
		self.plan_add("load", f"load {os.path.basename(path)}", job, {"path": path}, units=max(0.2, size / 2**30), go=now, model=path, on_start=start)
		self.log.add(f"load {'starts' if now or self.todo_plan.auto or self.todo_plan.going else 'planned in TO DO (APPLY to run it)'}: {os.path.basename(path)}")

	def unload_model(self):
		now = not self.todo_plan.pending()
		self.plan_add("unload", "unload the model", lambda s: s.unload(), go=now)
		if not now and not (self.todo_plan.auto or self.todo_plan.going):
			self.log.add("unload planned in TO DO (APPLY to run it)")

	def cancel(self):
		it = self.todo_plan.running()
		self.worker.interrupt()
		left = len(self.todo_plan.pending())
		self.log.add("stop requested" + (f": {it['label']}" if it else "") + (f" · {left} operation{'s' if left > 1 else ''} still in TO DO (Shift+Esc also pauses it)" if left else ""), "warn")
		SFX.play("abort")

	def add_folder(self, d):
		def job(s, d=d):
			dirs = list(s.settings.get("model_dirs") or [])
			if d not in dirs:
				s.settings = dict(s.settings, model_dirs=dirs + [d])
				config.save_settings(s.settings)
			return s.catalog()

		self.worker.light("catalog", job)

	def calibrate_all(self, only=None):
		gen = config.load_settings()["coherence"]["calibrate"]
		todo = [t["id"] for t in list(self.snap.get("traits") or []) + list(self.snap.get("extra_levers") or []) if t["id"] not in self.inactive and (only is None or t["id"] in only) and (t["id"] not in self.levers or (gen and self.levers[t["id"]].get("gen_mult") is None))]
		if only is not None and not todo:
			return
		if not todo:
			self.recalibrate("*")
			return
		self.calib_pending |= set(todo)
		self.sync_pending()

		def job(s, ids=todo):
			s.calibrate_all(ids, self.worker.stop)
			return len(ids)

		self.plan_add("calibrate", f"calibrate {len(todo)} lever{'s' if len(todo) > 1 else ''}: " + ", ".join(self.lever_name(t) for t in todo[:4]) + ("…" if len(todo) > 4 else ""), job, {"ids": todo}, units=len(todo))

	def uncalib(self, ids):
		self.calib_pending.difference_update(ids)
		self.sync_pending()

	def lever_name(self, tid):
		if tid.endswith(GO.LORA):
			return self.lever_name(GO.base_id(tid)) + " · LoRA"
		return next((t["name"] for t in self.lever_rows() if t["id"] == tid), tid)

	def set_lever_mode(self, mode, save=True):
		self.lever_mode = mode if mode in ("dose", "target") else "dose"
		self.mode_switch.set_mode(self.lever_mode)
		self.lever_table.set_mode(self.lever_mode)
		self.lever_table.targets = dict(self.targets)
		self.b_align.setVisible(self.lever_mode == "target")
		self.update_mode_hint()
		self.refresh_iris()
		self.update_sync()
		if save:
			SFX.play("toggle", self.lever_mode)
			self.prefs["lever_mode"] = self.lever_mode
			self.save_sync_prefs()
			self.log.add("levers: TARGET mode, set the measured value you want and the doses are found for you" if self.lever_mode == "target" else "levers: DOSE mode, each slider sets its lever's strength")

	def update_mode_hint(self):
		if self.lever_mode != "target":
			self.mode_hint.setText("  each slider = how hard its lever pushes")
			self.mode_hint.setToolTip("each slider = how hard its lever pushes\nDOSE mode: each slider sets its lever's strength · switch to TARGET to set the result instead")
			self.mode_hint.setStyleSheet(f"color: {theme.MUTED};")
			return
		n = len(self.targets)
		a = self.align_info or {}
		if self.planned("align"):
			st = a.get("step") or {}
			text, col = "  aligning" + (f" · round {st.get('round', 0) + 1} · largest gap {st['gap']:.2f}" if st.get("gap") is not None else "…"), theme.NEON
		elif not n:
			text, col = "  drag a slider to set the result you want", theme.MUTED
		elif a.get("report"):
			miss = [x for x in a["report"] if not x.get("ok")]
			text, col = (f"  {n} target{'s' if n != 1 else ''} reached" if not miss else f"  {len(a['report']) - len(miss)}/{len(a['report'])} reached · " + ", ".join(x["name"] for x in miss[:2]) + " out of reach"), (theme.OK if not miss else theme.WARN)
		else:
			text, col = f"  {n} target{'s' if n != 1 else ''}", theme.MUTED
		self.mode_hint.setText(text)
		self.mode_hint.setStyleSheet(f"color: {col};")
		rep = "\n".join(f"{x['name']}: target {x['target']:+.2f} · now {x['now']:+.2f} · dose {x['dose']:+.2f}" + ("" if x.get("ok") else (" · dose limit" if x.get("capped") else (" · held back by the coherence guard" if x.get("guarded") else " · out of reach"))) for x in a.get("report") or [])
		self.mode_hint.setToolTip(text.strip() + "\n\nTARGET mode: each slider sets the measured value you want (◆ target, ● measured now)\nthe alignment measures what each lever moves, solves for the doses with the interactions, applies them under the coherence guard, measures again and corrects (up to 4 rounds)" + (f"\n\n{rep}" if rep else ""))

	def set_target(self, tid, value):
		self.remember()
		if value is None:
			self.targets.pop(tid, None)
			SFX.play("click")
		else:
			self.targets[tid] = float(value)
			SFX.play("target", round(float(value) * 20) / 20)
		self.lever_table.targets = dict(self.targets)
		self.run_align()

	def judge_settings(self, coh, top):
		st = config.load_settings()
		values = dict(top)
		if coh:
			values["coherence"] = {**st["coherence"], **coh}
		if values:
			self.worker.light("judge_settings", lambda s, v=values: s.update_settings(v, light=True))

	def judge_now(self):
		names = [self.lever_name(t) for t in (self.snap.get("guard") or {}).get("unjudged") or []]
		label = ("judge the changed levers: " + ", ".join(names[:4]) + ("…" if len(names) > 4 else "")) if names else "judge the levers (the ones changed since the last check, else all)"
		it = self.todo_plan.mergeable("judge")
		if it is not None:
			it["label"] = label
			self.plan_changed()
			return
		self.plan_add("judge", label, lambda s: s.judge_levers(self.worker.stop))

	def update_judge(self):
		g = self.snap.get("guard") or {}
		names = [self.lever_name(t) for t in g.get("unjudged") or []] if self.snap.get("loaded") else []
		self.judge_box.set_unjudged(names)

	def align_verdict(self, p):
		miss = [x for x in p.get("report") or [] if not x.get("ok")]
		if not miss:
			return ""
		out = []
		for x in miss:
			if x.get("capped"):
				why = f"out of reach at the max dose {x['dose']:+.2f}"
			elif x.get("guarded"):
				why = f"held back by the coherence guard ({x['dose']:+.2f} → {x['applied']:+.2f})"
			else:
				why = f"not reached yet at {x['dose']:+.2f}: ALIGN again can get closer"
			out.append(f"{x['name']} {x['now']:+.2f} for {x['target']:+.2f}, {why}")
		return ("nothing to change · " if not p.get("rounds") else "") + " · ".join(out)

	def job_part(self, a, sub):
		j = a.get("job")
		if not j or sub is None:
			return sub
		return min(1.0, (float(j.get("done", 0.0)) + float(j.get("w", 0.0)) * max(0.0, min(1.0, float(sub)))) / max(float(j.get("total", 1.0)), 1e-6))

	def run_align(self, go=False, at=None):
		n = len(self.targets)
		self.align_info = {}
		self.lever_table.aligning = bool(n)
		self.lever_table.update()
		label = (f"align the levers on {n} target{'s' if n != 1 else ''}: " + ", ".join(f"{self.lever_name(k)} {v:+.2f}" for k, v in sorted(self.targets.items())[:3]) if n else "drop the targets")
		run = lambda s, t=dict(self.targets): s.align_targets(t, self.worker.stop)
		it = self.todo_plan.mergeable("align") if at is None else None
		if it is not None:
			it.update(label=label, run=run, params={"targets": dict(self.targets)})
			if go and not (self.todo_plan.auto or self.todo_plan.going):
				self.todo_plan.until = it["id"]
			self.plan_changed()
		else:
			it = self.plan_add("align", label, run, {"targets": dict(self.targets)}, units=max(1, n), go=go, at=at, on_drop=self.align_dropped)
		self.supersede_for(it, ("align", "levers", "measure"))
		self.update_mode_hint()

	def supersede_for(self, it, names):
		cur = self.todo_plan.running()
		if cur is None or cur["kind"] not in names or self.todo_plan.next_item() is not it:
			return
		sup = self.worker.supersede(names)
		if sup:
			self.superseded.add(sup)

	def align_dropped(self):
		if not any(i["kind"] == "align" for i in self.todo_plan.pending()) and not ((self.todo_plan.running() or {}).get("kind") == "align"):
			self.lever_table.aligning = False
			self.lever_table.update()
			self.update_mode_hint()

	def remap(self):
		self.plan_add("map", "re-map the workspace band", lambda s: s.analyze_workspace(self.worker.stop))

	def profile_clicked(self):
		steered = any(self.strengths.values())
		if steered and QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
			self.run_profile(full=True)
		elif steered:
			self.measure()
		else:
			self.run_profile(full=True)

	def profile_menu(self, gpos):
		steered = any(self.strengths.values())
		menu = QMenu(self)
		menu.setFont(theme.ui(8))
		menu.setToolTipsVisible(True)
		menu_title(menu, "PROFILE")
		quick = menu.addAction("QUICK MEASURE OF THE STEERED MODEL")
		quick.setToolTip("6 prompts with the levers as applied (informative: a chat sent meanwhile goes first)")
		quick.setEnabled(steered)
		quick.triggered.connect(lambda: self.measure())
		full = menu.addAction("FULL PROFILE OF THE STEERED MODEL")
		full.setToolTip("every prompt of the battery with the levers as applied")
		full.setEnabled(steered)
		full.triggered.connect(lambda: self.run_profile(full=True))
		base = menu.addAction("BASELINE AGAIN (LEVERS OFF)")
		base.setToolTip("the full profile of the bare model, the reference the levers are measured against")
		base.triggered.connect(self.reprofile)
		menu.exec(gpos)

	def run_profile(self, full=True):
		self.plan_add("profile", "full personality profile" if full else "quick personality profile", lambda s: s.profile("current" if any(s.strengths.values()) else "baseline", quick=not full and any(s.strengths.values()), stop=self.worker.stop))

	def reprofile(self):
		self.plan_add("profile", "profile the model again (baseline, levers off)", lambda s: s.profile("baseline", quick=False, stop=self.worker.stop))
		if any(self.applied.values()):
			self.measure()

	def measure(self, at=None):
		return self.plan_add("measure", "measure the steered model (quick, informative)", lambda s: s.profile("current", quick=True, stop=self.worker.stop) if any(s.strengths.values()) else None, at=at)

	def sync_pending(self):
		self.lever_table.pending = self.apply_pending | self.calib_pending
		self.lever_table.update()
		self.update_sync()

	def submit_levers(self, strengths, go=False, at=None):
		skip = sorted(t for t, v in strengths.items() if v and GO.is_goal(t) and GO.base_id(t) not in self.levers and GO.base_id(t) not in self.lever_table.planned)
		if skip:
			strengths = {t: (0.0 if t in skip else v) for t, v in strengths.items()}
			self.log.add("not placed on this model: " + ", ".join(dict.fromkeys(self.lever_name(GO.base_id(t)) for t in skip)) + " · TRAIN or ALIGN it in GOALS first (kept at 0 here)", "warn")
		changed = {t: float(v) for t, v in strengths.items() if abs(self.strengths.get(t, self.place_default(t)) - float(v)) > 1e-6}
		for t, v in changed.items():
			self.lever_table.values[t] = v
		if not changed:
			self.lever_table.update()
			return False
		self.strengths.update(changed)
		self.apply_pending = {t for t in set(self.strengths) | set(self.applied) if abs(self.strengths.get(t, 0.0) - self.applied.get(t, 0.0)) > 1e-6}
		self.sync_pending()
		return self.plan_levers(go, at)

	def plan_levers(self, go=False, at=None):
		snap = {t: float(v) for t, v in self.strengths.items()}
		todo = sorted(self.apply_pending, key=lambda t: self.lever_name(t))
		label = "apply levers: " + ", ".join(f"{self.lever_name(t)} {snap.get(t, 0.0):+.2f}" for t in todo[:4]) + ("…" if len(todo) > 4 else "") if todo else "apply the lever settings"
		run = lambda s, st=dict(snap): s.set_strengths(st, self.worker.stop)
		it = self.todo_plan.mergeable("levers") if at is None else None
		if it is not None:
			it.update(label=label, run=run, params={"strengths": snap})
		else:
			it = self.todo_plan.add("levers", label, run, {"strengths": snap}, model=self.model_path(), at=at)
		delay = 0.0 if go else self.apply_delay()
		it["hold_until"] = time.monotonic() + delay if delay > 0 else 0.0
		if go and not (self.todo_plan.auto or self.todo_plan.going):
			self.todo_plan.until = it["id"]
		if self.auto_measure.isChecked() and any(snap.values()) and self.todo_plan.after(it, "measure") is None:
			self.measure(at=self.todo_plan.index(it) + 1)
		self.supersede_for(it, ("levers", "measure"))
		self.plan_changed()
		return it

	def open_active(self):
		dlg = ActiveLeversDialog(self.lever_rows(everything=True), self.inactive, self.strengths, self)
		if not dlg.exec():
			return
		self.set_active(dlg.inactive())

	def set_active(self, off, go=False):
		if set(off) == self.inactive:
			return
		if not self.snap.get("loaded") and not self.planned("load"):
			self.worker.light("active", lambda s, o=list(off): s.set_active(o), f"set the active levers ({len(off)} inactive)")
			return
		self.plan_add("active", f"set the active levers ({len(off)} inactive)", lambda s, o=list(off): s.set_active(o), {"inactive": sorted(off)}, go=go)

	def open_presets(self):
		if self.presets_dlg is None:
			self.presets_dlg = PresetsDialog(self.lever_name, self)
			self.presets_dlg.save_requested.connect(self.save_preset)
			self.presets_dlg.use_requested.connect(self.load_preset)
		self.presets_dlg.reload()
		self.presets_dlg.set_model(os.path.basename((self.snap.get("info") or {}).get("path", "")))
		self.presets_dlg.show()
		self.presets_dlg.raise_()
		self.presets_dlg.activateWindow()

	def save_preset(self, name):
		self.worker.light("preset_save", lambda s, n=name, st=dict(self.strengths), sp=self.chat.system.text(), tg=dict(self.targets) if self.lever_mode == "target" else {}, m=self.lever_mode: s.save_preset(n, st, sp, tg, m), f"save the preset {name}")

	def load_preset(self, data, apply):
		self.plan_add("preset_load", f"{'apply' if apply else 'plan'} the preset {data.get('name', '')}".strip(), lambda s, d=dict(data), a=bool(apply): dict(s.prepare_preset(d), apply=a))

	def use_preset(self, res):
		self.remember()
		want = {t: 0.0 for t in set(self.strengths) | set(self.applied)}
		want.update(res.get("strengths") or {})
		go = bool(res.get("apply"))
		off = res.get("inactive")
		if off is not None:
			off = set(off)
			want = {t: (0.0 if GO.base_id(t) in off else v) for t, v in want.items()}
			if off != self.inactive:
				self.set_active(sorted(off), go=go)
		notes = ([f"levers recreated: {', '.join(res['added'])}"] if res.get("added") else []) + ([f"goals added: {', '.join(res['goals_added'])}"] if res.get("goals_added") else []) + ([f"goals kept as you have them (the preset's version differs): {', '.join(res['goals_kept'])}"] if res.get("goals_kept") else []) + ([f"unknown levers skipped: {', '.join(res['missing'])}"] if res.get("missing") else [])
		before = (self.lever_mode, dict(self.targets))
		mode = res.get("mode")
		if res.get("targets") is not None and mode != "dose":
			self.targets = dict(res.get("targets") or {})
			self.lever_table.targets = dict(self.targets)
		if mode and mode != self.lever_mode:
			self.set_lever_mode(mode)
		moved = self.targets != before[1]
		if res.get("system", "") != self.chat.system.text() or res.get("system", "") != self.applied_system:
			self.chat.system.setText(res.get("system", ""))
			self.set_system(res.get("system", ""), go=go)
		changed = self.submit_levers(want, go=go)
		if self.lever_mode == "target" and (moved or (self.targets and (changed or self.lever_mode != before[0]))):
			self.run_align(go=go)
		running = go or self.todo_plan.auto or self.todo_plan.going
		self.log.add(f"preset '{res.get('name')}' " + ("applied" if running else "planned in TO DO: adjust the sliders if you want, then APPLY") + ("" if changed or moved else " (the levers already matched)") + ("" if not notes else " · " + " · ".join(notes)), "warn" if res.get("missing") or res.get("goals_kept") else "ok")

	def reconcile_levers(self):
		p = self.todo_plan
		planned = [i for i in p.items if i["state"] in ("pending", "running") and i["kind"] == "levers"]
		want = dict(planned[-1]["params"]["strengths"]) if planned else {}
		keys = set(self.strengths) | set(self.applied) | set(want)
		reverted = sorted(t for t in keys if abs(self.strengths.get(t, 0.0) - want.get(t, self.applied.get(t, 0.0))) > 1e-6)
		for t in reverted:
			self.strengths[t] = want.get(t, self.applied.get(t, 0.0))
		if reverted and not planned:
			self.log.add("lever change not applied (stopped, removed or failed): " + ", ".join(self.lever_name(t) for t in reverted), "warn")
		self.apply_pending = {t for t in set(self.strengths) | set(self.applied) if abs(self.strengths.get(t, 0.0) - self.applied.get(t, 0.0)) > 1e-6}
		self.sync_pending()
		self.refresh_iris()

	def goal_as_lever(self, gid):
		name = next((g.get("name") for g in self.goal_defs if g.get("id") == gid), gid)
		self.worker.light("goal_as_lever", lambda s, g=gid: s.goal_lever_def(g), f"make a lever from the goal {name}")

	def set_lever(self, tid, value):
		SFX.play("lever", round(float(value) * 20) / 20)
		self.remember()
		self.submit_levers({tid: value})

	def lever_snapshot(self):
		return {"strengths": {k: round(float(v), 4) for k, v in self.strengths.items() if abs(float(v)) > 1e-9}, "targets": {k: round(float(v), 4) for k, v in self.targets.items()}}

	def lever_summary(self, st):
		parts = [f"{self.lever_name(k)} {v:+.2f}" for k, v in sorted(st["strengths"].items(), key=lambda kv: self.lever_name(kv[0]))]
		text = ", ".join(parts[:6]) + (f" and {len(parts) - 6} more" if len(parts) > 6 else "") if parts else "every lever at 0"
		if st["targets"]:
			text += " · targets: " + ", ".join(f"{self.lever_name(k)} {v:+.2f}" for k, v in sorted(st["targets"].items(), key=lambda kv: self.lever_name(kv[0])))
		return text

	def remember(self):
		now = self.lever_snapshot()
		if self.undo_stack and self.undo_stack[-1] == now:
			return
		self.undo_stack.append(now)
		del self.undo_stack[:-UNDO_MAX]
		self.redo_stack = []
		self.update_undo()

	def undo(self):
		self.step_lever(self.undo_stack, self.redo_stack)

	def redo(self):
		self.step_lever(self.redo_stack, self.undo_stack)

	def step_lever(self, take, put):
		now = self.lever_snapshot()
		while take and take[-1] == now:
			take.pop()
		if not take:
			self.update_undo()
			return
		st = take.pop()
		put.append(now)
		if st["targets"] != now["targets"]:
			self.targets = dict(st["targets"])
			self.lever_table.targets = dict(self.targets)
			self.run_align()
		self.submit_levers({k: st["strengths"].get(k, 0.0) for k in set(self.strengths) | set(self.applied) | set(st["strengths"])})
		SFX.play("click")
		self.update_undo()

	def update_undo(self):
		self.b_undo.setEnabled(bool(self.undo_stack))
		self.b_redo.setEnabled(bool(self.redo_stack))
		self.b_undo.setToolTip("undo the last lever change (Ctrl+Z)" + (f"\nback to: {self.lever_summary(self.undo_stack[-1])}" if self.undo_stack else "\nnothing to undo yet"))
		self.b_redo.setToolTip("redo (Ctrl+Y or Ctrl+Shift+Z)" + (f"\nforward to: {self.lever_summary(self.redo_stack[-1])}" if self.redo_stack else "\nnothing to redo"))

	def zero_levers(self):
		self.remember()
		if self.targets:
			self.targets = {}
			self.lever_table.targets = {}
			self.run_align()
		self.submit_levers({t: 0.0 for t in set(self.strengths) | set(self.applied)})

	def add_trait(self):
		dlg = TraitDialog(self)
		if dlg.exec():
			v = dlg.data()
			self.plan_add("add_trait", f"build the lever {v['name']}", lambda s, v=v: s.add_trait(v["name"], v["seeds_plus"], v["seeds_minus"], v["plus"], v["minus"], v["desc_plus"], v["desc_minus"]))

	def trait_row(self, tid):
		return next((t for t in list(self.snap.get("traits") or []) + list(self.snap.get("extra_levers") or []) if t["id"] == tid), None)

	def edit_trait(self, tid):
		t = self.trait_row(tid)
		if t is None:
			return
		dlg = TraitDialog(self, t, "edit")
		if dlg.exec():
			self.plan_add("trait_edit", f"save the lever {t['name']}", lambda s, i=tid, v=dlg.data(): s.update_trait(i, v))

	def duplicate_trait(self, tid):
		t = self.trait_row(tid)
		if t is None:
			return
		dlg = TraitDialog(self, t, "duplicate")
		if dlg.exec():
			self.plan_add("trait_edit", f"duplicate the lever {t['name']}", lambda s, i=tid, v=dlg.data(): s.duplicate_trait(i, v))

	def delete_trait(self, tid):
		t = self.trait_row(tid)
		if t is None:
			return
		if t.get("custom"):
			box = QMessageBox(QMessageBox.Icon.Question, "Delete lever", f"Delete the custom lever '{t['name']}'?\n\nIts definition is removed from your custom levers (custom_traits.json) and its calibration is dropped." + ("\nThe goal it was made from stays in GOALS (ADD AS A LEVER makes it again)." if t.get("from_goal") else ""), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, self)
		else:
			box = QMessageBox(QMessageBox.Icon.Question, "Remove lever", f"Remove the default lever '{t['name']}'?\n\nIt leaves the LEVERS list and its calibration is dropped. Right-click any lever → RESTORE A REMOVED LEVER brings it back.\nThe model is profiled again afterwards (the set of levers changed).", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, self)
		box.setStyleSheet(theme.STYLE)
		box.setDefaultButton(QMessageBox.StandardButton.Cancel)
		if box.exec() == QMessageBox.StandardButton.Yes:
			self.plan_add("trait_edit", f"delete the lever {t['name']}", lambda s, i=tid: s.delete_trait(i))

	def restore_trait(self, tid):
		name = next((h["name"] for h in self.snap.get("hidden") or [] if h["id"] == tid), None) or self.lever_name(tid)
		self.plan_add("trait_edit", f"restore the lever {name}", lambda s, i=tid: s.restore_trait(i))

	def recalibrate(self, tid):
		ids = None if tid == "*" else [tid]
		todo = [t for t in (ids or list(self.levers)) if (self.levers.get(t) or {}).get("kind") != "optimized"]
		self.calib_pending |= set(todo)
		self.sync_pending()

		def job(s, ids=ids):
			s.recalibrate_levers(ids, self.worker.stop)
			return len(todo)

		self.plan_add("recalibrate", "recalibrate every lever" if ids is None else f"recalibrate {self.lever_name(tid)}", job, {"ids": todo}, units=max(1, len(todo)))

	def load_opts(self, path, st=None):
		st = st if st is not None else config.load_settings()
		return {**(st.get("load") or {}), **((st.get("load_models") or {}).get(os.path.basename(path)) or {})}

	def save_load_options(self, path, opts, per):
		self.log.add(f"load options for {'this model' if per else 'every model'}: split {opts['split_mode']} · main CUDA{opts['main_gpu']} · flash {opts['flash_attn']} · KV {opts['kv_type']}" + (" · applies at the next load" if self.snap.get("loaded") and (self.snap.get("info") or {}).get("path") == path else ""))
		self.worker.light("load_options", lambda s, p=path, o=dict(opts), per=per: (s.set_load_options(p, o, per), "load_options")[1])

	def chat_config(self, cfg):
		if not cfg:
			return None
		self.reply_n += 1
		return {**cfg, "n": self.reply_n, "summary": self.config_text(cfg, f"REPLY {self.reply_n} · CONFIGURATION")}

	def config_text(self, cfg, title):
		g = cfg.get("guard") or {}
		scale = float(g.get("scale", 1.0) or 1.0)
		factors = g.get("factors") or {}
		free = set(g.get("free") or [])
		kinds = cfg.get("kinds") or {}
		rows = [title, f"model: {cfg.get('model') or '?'}"]
		levers = []
		for t, v in sorted((cfg.get("strengths") or {}).items(), key=lambda kv: self.lever_name(kv[0])):
			eff = v * (1.0 if t in free else scale) * float(factors.get(t, 1.0))
			kind = kinds.get(t) or {}
			extra = " (trained " + ({"lora": "LoRA", "both": "vector + LoRA"}.get(kind.get("carrier"), "vector")) + ")" if kind.get("kind") == "optimized" else ""
			levers.append(f"  {self.lever_name(t)} {v:+.2f}" + (f" → applied {eff:+.2f}" if abs(eff - v) > 1e-3 else "") + extra + (" · outside the guard" if t in free else ""))
		rows += ["levers:"] + levers if levers else ["levers: none (bare model)"]
		if levers:
			coh = g.get("coherence")
			rows.append("coherence guard: " + (f"×{scale:.2f}" if scale < 0.999 else "no scaling") + (f" · coherence {coh:.0%} ({g.get('by', '?')})" if coh is not None else f" · KL {g.get('kl', 0):.2f}"))
		sysp = cfg.get("system") or ""
		rows.append("system prompt: " + (f"«{sysp if len(sysp) <= 160 else sysp[:159] + '…'}»" if sysp else "none"))
		if cfg.get("targets"):
			rows.append("targets: " + ", ".join(f"{self.lever_name(t)} {v:+.2f}" for t, v in sorted(cfg["targets"].items())))
		sm = cfg.get("sampling") or {}
		rows.append("sampling: " + " · ".join(f"{k} {sm[k]}" for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "max_tokens") if k in sm) + f" · seed {cfg.get('seed')}")
		return "\n".join(rows)

	def open_history(self):
		if self.history_dlg is None:
			self.history_dlg = HistoryDialog(self.config_text, self)
			self.history_dlg.restore.connect(self.restore_config)
		self.history_dlg.set_model((self.snap.get("info") or {}).get("path") and os.path.basename(self.snap["info"]["path"]) or "")
		self.history_dlg.reload()
		self.history_dlg.show()
		self.history_dlg.raise_()
		self.history_dlg.activateWindow()

	def restore_config(self, cfg):
		known = {t["id"] for t in self.lever_rows()}
		saved = {t: float(v) for t, v in (cfg.get("strengths") or {}).items()}
		missing = [t for t in saved if t not in known]
		want = {t: 0.0 for t in set(self.strengths) | set(self.applied)}
		want.update({t: v for t, v in saved.items() if t in known})
		sysp = cfg.get("system") or ""
		if sysp != self.system_intent():
			self.chat.system.setText(sysp)
			self.set_system(sysp)
		changed = self.submit_levers(want)
		self.log.add(f"configuration of reply {cfg.get('n', '?')} " + ("restored" if self.todo_plan.auto or self.todo_plan.going else "planned in TO DO (APPLY to use it)") + ("" if changed else " (the levers already matched)") + (f" · these levers no longer exist: {', '.join(missing)}" if missing else ""), "warn" if missing else "ok")
		self.chat.add_note(f"configuration of reply {cfg.get('n', '?')} restored")

	def system_intent(self):
		planned = [i for i in self.todo_plan.items if i["state"] in ("pending", "running") and i["kind"] == "system"]
		return planned[-1]["params"]["text"] if planned else self.applied_system

	def set_system(self, text, go=False, at=None):
		short = " ".join(text.split())
		label = "system prompt: " + (f"«{short if len(short) <= 40 else short[:39] + '…'}»" if short else "none")
		run = lambda s, t=text: setattr(s, "system_prompt", t)
		it = self.todo_plan.mergeable("system") if at is None else None
		if it is not None:
			it.update(label=label, run=run, params={"text": text})
			if go and not (self.todo_plan.auto or self.todo_plan.going):
				self.todo_plan.until = it["id"]
			self.plan_changed()
			return it
		return self.plan_add("system", label, run, {"text": text}, go=go, at=at, on_drop=lambda: self.chat.system.setText(self.system_intent()))

	def system_edited(self, text):
		if text != self.system_intent():
			self.set_system(text)

	def nothing_waits(self):
		return not any(not i["info"] for i in self.todo_plan.pending())

	def send_chat(self, text, now=None):
		now = self.nothing_waits() if now is None else now
		live = self.live.isChecked()
		short = " ".join(text.split())
		label = "reply: " + (short if len(short) <= 48 else short[:47] + "…")
		it = self.plan_add("chat", label, lambda s, t=text, lv=live: s.chat(t, stop=self.worker.stop, live=lv), {"text": text}, at=self.todo_plan.chat_slot(), go=now, on_drop=lambda: self.chat.add_note("not sent: removed from TO DO"))
		self.preempt_info(it)
		if not now and not (self.todo_plan.auto or self.todo_plan.going):
			self.chat.add_note("waits in TO DO after what is planned before it (APPLY runs it): " + label)

	def retry_chat(self, now=None):
		if not self.snap.get("loaded"):
			return
		now = self.nothing_waits() if now is None else now
		live = self.live.isChecked()
		label = "another try of the last reply"
		it = self.plan_add("chat", label, lambda s, lv=live: s.retry(stop=self.worker.stop, live=lv), {"text": None}, at=self.todo_plan.chat_slot(), go=now)
		self.preempt_info(it)

	def preempt_info(self, it):
		cur = self.todo_plan.running()
		if cur is None or not cur["info"] or self.todo_plan.next_item() is not it:
			return
		self.requeue[cur["id"]] = it["id"]
		self.worker.interrupt()

	def export(self):
		waits = bool(self.apply_pending) or bool(self.lever_table.planned)
		use = dict(self.strengths) if waits else self.applied
		for b, carrier in self.lever_table.planned.items():
			if b not in self.levers:
				for k in (b, GO.lora_key(b)) if carrier == "both" else (b,):
					use.setdefault(k, 1.0)
		if not any(use.values()):
			self.log.add("move at least one lever before exporting", "warn")
			return
		from hypno.export import bakeable, export, model_tag
		from hypno.lorakeys import arch_ok

		g = self.snap.get("guard") or {}
		scale = float(g.get("scale", 1.0) or 1.0)
		factors = g.get("factors") or {}
		free = set(g.get("free") or [])
		names = {t["id"]: t["name"] for t in self.lever_rows()}
		rows = [{"name": names.get(t, t) if not t.endswith(GO.LORA) else names.get(GO.base_id(t), t) + " · LoRA", "setting": v, "applied": v * (1.0 if t in free else scale) * float(factors.get(t, 1.0)), "optimized": (self.levers.get(GO.base_id(t)) or {}).get("kind") == "optimized" or GO.base_id(t) in self.lever_table.planned} for t, v in use.items() if v]
		info = self.snap.get("info") or {}
		arch = info.get("arch", "")
		dlg = ExportDialog(rows, model_tag(info), bakeable(arch), "_".join(r["name"].lower().replace(" ", "-") for r in rows[:3]), self, arch_ok(arch, info.get("n_expert", 0)))
		if not dlg.exec():
			return
		name, dose, kind, style, prefix = dlg.values()
		self.plan_add("export", f"export {name}" + (" as a LoRA" if kind == "lora" else ""), lambda s, n=name, d=dose, k=kind, st=style, px=prefix: export(s, n, d, k, on_progress=lambda v: s.emit("progress", {"task": "export", "value": v}), style=st, prefix=px))

	def lmstudio_note(self, res):
		if res.get("lmstudio"):
			self.log.add("LM Studio override written to lmstudio_override.txt in the export folder: copy its ```json block, then use the llama.cpp override import button when loading the model in LM Studio", "ok")

	def open_exports(self):
		try:
			os.makedirs(config.EXPORTS_DIR, exist_ok=True)
		except OSError as exc:
			self.log.add(f"exports folder unavailable: {exc}", "error")
			return
		QDesktopServices.openUrl(QUrl.fromLocalFile(config.EXPORTS_DIR))

	def optimize_goal(self, v):
		opts = dict(v.get("options") or {})
		st = config.load_settings()
		st["optimize"] = opts
		config.save_settings(st)
		self.plan_add("goal_optimize", f"TRAIN the goal {v.get('name') or v['id']}", lambda s, g=v["id"], o=opts: s.optimize_goal(g, o, self.worker.stop), {"goal": v["id"], "carrier": opts.get("carrier") or "vector"}, units=max(0.1, int(opts.get("steps") or 100) / 100))

	def save_goal(self, v):
		self.worker.light("goal_save", lambda s, v=dict(v): s.store_goal(v), f"save the goal {v.get('name') or ''}".strip())

	def delete_goal(self, gid):
		self.worker.light("goal_delete", lambda s, g=gid: s.forget_goal(g))

	def on_light(self, p):
		name, res = p.get("name"), p.get("result")
		if p.get("error"):
			self.log.add(f"{name}: {p['error']}", "error")
			return
		if name == "catalog" and isinstance(res, dict):
			self.use_catalog(res)
		elif name == "defs" and isinstance(res, dict):
			if not res.get("loaded") and not self.snap.get("loaded"):
				self.use_unloaded(res)
				self.refresh_iris()
		elif name == "plan" and isinstance(res, dict):
			self.drawer.set_plan(res, self.plan_gpus())
		elif name == "preset_save" and isinstance(res, dict):
			if self.presets_dlg is not None:
				self.presets_dlg.reload(res.get("path", ""))
		elif name == "judge_settings" and isinstance(res, dict):
			self.lever_table.set_range(config.load_settings().get("dose_max") or 1.5)
			if res.get("stale"):
				self.plan_add("apply_settings", "unload the previous judge", lambda s: s.apply_settings())
		elif name == "settings" and isinstance(res, dict):
			self.judge_box.load(config.load_settings())
			self.lever_table.set_range(config.load_settings().get("dose_max") or 1.5)
			if res.get("rescan") or res.get("relens"):
				self.worker.light("catalog", lambda s, r=bool(res.get("relens")): s.catalog(refresh_lenses=r))
			if (res.get("coherence") and self.snap.get("loaded") and any(self.applied.values())) or res.get("stale"):
				self.plan_add("apply_settings", "apply the new settings to the levers (judge, coherence guard)" if self.snap.get("loaded") else "unload the previous judge", lambda s: s.apply_settings())
			self.request_plan()
		elif name == "load_options":
			self.request_plan()
		elif name == "goal_as_lever" and isinstance(res, dict):
			self.snap["extra_levers"] = res["extra_levers"]
			self.refresh_iris()
			if res.get("move"):
				self.plan_add("goal_as_lever", f"move the persona direction of the goal {res.get('name')} to its new lever", lambda s, g=res["goal"]: s.goal_lever_move(g))
		elif name in ("goal_save", "goal_delete") and isinstance(res, dict) and res.get("drop"):
			self.plan_add("drop", "drop the outdated lever of the goal " + (res.get("name") or ""), lambda s, t=res["drop"]: s.drop_lever(t))

	def use_catalog(self, res):
		self.catalog = res["models"]
		self.judge_box.set_models(self.catalog, config.load_settings()["coherence"].get("judge", ""))
		self.drawer.roots = list(config.load_settings().get("model_dirs") or [])
		self.llama_dir = res.get("llama_dir") or ""
		self.margin = int(res.get("margin") or 900)
		self.set_lens_index(res.get("lenses"))
		n_lens = sum(1 for v in self.lens_best.values() if v)
		self.log.add(f"{len([m for m in self.catalog if not m.get('error')])} models found" + (f" ({sum(1 for m in self.catalog if m.get('format') == 'safetensors')} safetensors)" if any(m.get("format") == "safetensors" for m in self.catalog) else "") + f" · {len(self.lens_index)} lenses indexed · {n_lens} models with a matching lens")
		last = self.args.load or (config.load_settings().get("last_model") or "")
		if last:
			self.drawer.select_path(last)
		if self.args.load and not self.script:
			self.start_script()
		else:
			self.prefetch_plans()

	def set_goal_compare(self, on):
		st = config.load_settings()
		st["goal_compare"] = bool(on)
		config.save_settings(st)
		self.worker.light("prefs", lambda s, on=bool(on): s.set_setting("goal_compare", on))

	def set_goal_read(self, opts):
		st = config.load_settings()
		st["goal_read"] = dict(opts)
		config.save_settings(st)
		self.worker.light("prefs", lambda s, o=dict(opts): s.set_setting("goal_read", o))

	def open_goals(self, gid=""):
		if self.goals_dlg is None:
			self.goals_dlg = GoalsDialog(self.goal_defs, self)
			self.goals_dlg.saved.connect(self.save_goal)
			self.goals_dlg.check.connect(lambda v: self.plan_add("goal_check", f"check the goal {v.get('name') or v['id']}", lambda s, g=v["id"]: s.check_goal(g, self.worker.stop), {"goal": v["id"]}))
			self.goals_dlg.align.connect(lambda v: self.plan_add("goal_align", f"align on the goal {v.get('name') or v['id']}", lambda s, g=v["id"]: s.align_goal(g, self.worker.stop), {"goal": v["id"]}))
			self.goals_dlg.optimize.connect(self.optimize_goal)
			self.goals_dlg.compare_changed.connect(self.set_goal_compare)
			st = config.load_settings()
			self.goals_dlg.compare.setChecked(bool(st.get("goal_compare", False)))
			self.goals_dlg.set_read_options(st.get("goal_read") or {})
			self.goals_dlg.read_changed.connect(self.set_goal_read)
			self.goals_dlg.set_options(st.get("optimize") or {})
			self.goals_dlg.delete.connect(self.delete_goal)
			self.goals_dlg.stop.connect(self.cancel)
			self.goals_dlg.as_lever.connect(self.goal_as_lever)
		self.goals_dlg.set_judge(self.judge_state, config.load_settings()["coherence"].get("judge", ""))
		self.goals_dlg.set_busy(str(self.busy or "").startswith("goal_"))
		self.goals_dlg.set_twins(self.goal_twins())
		self.goals_dlg.show()
		self.goals_dlg.raise_()
		self.goals_dlg.activateWindow()
		if gid:
			self.goals_dlg.show_goal(gid)

	def lever_rows(self, everything=False):
		keep = (lambda r: True) if everything else (lambda r: r["id"] not in self.inactive)
		rows = [r for r in list(self.snap.get("traits") or []) + list(self.snap.get("extra_levers") or []) if keep(r)]
		goals = [g for g in self.snap.get("goals") or [] if keep(g)]
		plain = {r["name"].strip().lower() for r in rows}
		twins = {g["name"].strip().lower() for g in goals} & plain
		if twins:
			rows = [dict(r, name=r["name"] + " (lever)") if r["name"].strip().lower() in twins else r for r in rows]
		for g in goals:
			g = dict(g, name=g["name"] + " (goal)") if g["name"].strip().lower() in twins else g
			rows.append(g)
			if (self.levers.get(g["id"]) or {}).get("carrier") == "both" or (g["id"] not in self.levers and self.lever_table.planned.get(g["id"]) == "both"):
				rows.append({"id": GO.lora_key(g["id"]), "name": "LoRA", "goal": g.get("goal"), "sub": True, "parent": g["id"], "parent_name": g["name"], "plus": g.get("plus", ""), "minus": g.get("minus", ""), "custom": True, "lexicon": {}})
		return rows

	def on_lever_hover(self, i):
		rows = self.lever_table.traits
		tid = rows[i]["id"] if 0 <= i < len(rows) else None
		self.iris.hover = next((k for k, t in enumerate(self.snap.get("traits") or []) if t["id"] == tid), -1)
		self.iris.lines_dirty = self.iris.ov_dirty = True
		self.highlight_layers(tid)

	def on_iris_hover(self, i):
		i = i if i >= 0 else self.iris.pinned
		traits = self.snap.get("traits") or []
		tid = traits[i]["id"] if 0 <= i < len(traits) else None
		self.lever_table.highlight = next((k for k, t in enumerate(self.lever_table.traits) if t["id"] == tid), -1)
		self.lever_table.update()
		self.highlight_layers(tid)

	def on_concept_hover(self, ci):
		concepts = self.iris.state.get("concepts") or []
		traits = self.iris.state.get("traits") or []
		c = concepts[ci] if 0 <= ci < len(concepts) else None
		k = self.iris.axis_of(c) if c is not None else -1
		i = self.lever_table.set_concept(traits[k].get("id") if 0 <= k < len(traits) else None, float(c.get("tv", 0.0) or 0.0) if c is not None else 0.0)
		if i >= 0:
			y = self.lever_table.top + i * self.lever_table.row_h
			self.lever_scroll.ensureVisible(0, y + self.lever_table.row_h // 2, 0, self.lever_table.row_h)

	def highlight_layers(self, tid):
		lv = self.levers.get(tid) if tid else None
		self.ws_chart.set_data(self.snap.get("workspace"), self.snap.get("band"), lv["layers"] if lv else [])

	def use_unloaded(self, p):
		p = p or {}
		self.snap = dict(p, loaded=False)
		self.inactive = set(p.get("inactive") or [])
		self.goal_defs = p.get("goal_defs") or self.goal_defs
		self.judge_state = p.get("judge") or self.judge_state

	def refresh_iris(self):
		if not self.snap.get("loaded"):
			self.iris.set_state(demo_state())
			self.fill_levers(None, None)
			return
		traits = self.snap.get("traits") or []
		info = self.snap["info"]
		base = self.summaries.get("baseline")
		cur = self.summaries.get("current") if any(self.applied.values()) else None
		main = cur or base
		ref = self.summaries.get("baseline_quick") if self.summaries.get("baseline_quick") and ((cur and cur.get("quick")) or self.lever_mode == "target") else base
		values = {t["id"]: t["value"] for t in main["traits"]} if main else {}
		basev = {t["id"]: t["value"] for t in ref["traits"]} if ref and cur else {}
		base_all = {t["id"]: t["value"] for t in ref["traits"]} if ref else {}
		cur_all = {t["id"]: t["value"] for t in cur["traits"]} if cur else {}
		act = np.asarray(main["delta"], dtype=np.float32) if main and main.get("delta") else None
		scale = float((self.snap.get("guard") or {}).get("scale", 1.0)) if any(self.applied.values()) else 1.0
		concepts = (main or {}).get("concepts", [])
		n_concepts = int(self.prefs.get("concept_count", 120))
		cards = []
		for k, t in enumerate(traits):
			lv = self.levers.get(t["id"]) or {}
			lex = t.get("lexicon") or {}
			row = act[k] if act is not None and k < len(act) else None
			peaks = sorted(int(l) for l in np.argsort(-np.abs(row))[:3]) if row is not None and len(row) else []
			cards.append({"id": t["id"], "name": t["name"], "plus": t["plus"], "minus": t["minus"], "value": values.get(t["id"], 0.0), "base": basev.get(t["id"]), "lever": self.strengths.get(t["id"]) or None, "reads": lv.get("reads_as"), "reads_minus": lv.get("reads_as_minus"), "layers": lv.get("layers"), "gain": lv.get("gain"), "setting": self.strengths.get(t["id"], 0.0), "applied": self.applied.get(t["id"], 0.0) * scale, "v_base": base_all.get(t["id"]), "v_now": cur_all.get(t["id"]), "peaks": peaks, "lex_plus": lex.get("plus", [])[:8], "lex_minus": lex.get("minus", [])[:8], "concepts": [c["str"].strip() + (f"·{c['en']}" if c.get("en") else "") for c in concepts if c.get("trait") == k and abs(c.get("tv") or 0.0) >= 0.005][:6]})
		st = {
			"L": info["n_layer"],
			"band": self.snap.get("band") or [int(info["n_layer"] * 0.33), int(info["n_layer"] * 0.88)],
			"traits": cards,
			"act": main["delta"] if main else None,
			"order_act": base["delta"] if base and base.get("delta") else None,
			"concepts": [dict(c) for c in concepts[:n_concepts]],
			"ws": {"autocorr": (self.snap.get("workspace") or {}).get("autocorr"), "agree_top5": (self.snap.get("workspace") or {}).get("agree_top5")},
		}
		self.iris.set_state(st)
		self.fill_levers(ref, cur)
		self.concepts.set_items(concepts[:n_concepts], [t["name"] for t in traits])
		self.heat.band = self.snap.get("band")
		self.ws_chart.set_data(self.snap.get("workspace"), self.snap.get("band"))
		self.update_header()

	def goal_twins(self):
		return {r["from_goal"]: r["name"] for r in list(self.snap.get("traits") or []) + list(self.snap.get("extra_levers") or []) if r.get("from_goal")}

	def fill_levers(self, ref, cur):
		rows = self.lever_rows()
		if self.goals_dlg is not None:
			self.goals_dlg.set_twins(self.goal_twins())
		self.lever_table.set_traits(rows)
		self.lever_table.values = {t["id"]: self.strengths[t["id"]] if t["id"] in self.strengths else self.strengths.get(t["parent"], 0.0) if t.get("sub") and t["parent"] in self.levers else self.place_default(t["id"]) for t in rows}
		self.lever_table.targets = dict(self.targets)
		self.lever_table.info = self.levers
		self.lever_table.base = {t["id"]: t["value"] for t in ref["traits"]} if ref else {}
		self.lever_table.current = {t["id"]: t["value"] for t in cur["traits"]} if cur else {}
		for lid, sc in self.goal_scores.items():
			if sc.get("base") is not None:
				self.lever_table.base[lid] = sc["base"]
			if sc.get("now") is not None:
				self.lever_table.current[lid] = sc["now"]
		self.lever_table.scale = float((self.snap.get("guard") or {}).get("scale", 1.0)) if any(self.applied.values()) else 1.0
		self.lever_table.update()
		n_active = sum(1 for t, v in self.strengths.items() if v and not t.endswith(GO.LORA))
		self.levers_panel.set_info(f"{n_active} steering" if n_active else "drag to steer", theme.ACCENT if n_active else theme.MUTED)

	def on_event(self, kind, p):
		if kind == "judge_live":
			self.judge_box.view.live(p)
			self.update_skip()
			return
		if kind == "read_missing":
			self.read_missing = set(p.get("ids") or [])
			QTimer.singleShot(0, self.plan_dispatch)
			return
		if kind in ("loaded", "profile", "lever", "levers", "workspace", "goals", "goal_result", "goal_trial", "goal_aligned", "goal_optimized", "goal_step", "goal_scan"):
			p = copy.deepcopy(p)
		self.track(kind, p)
		it = self.todo_plan.running()
		if it is not None and kind in ("progress", "job_progress", "align_step", "profile_prompt", "profile_response", "text", "log", "goal_step") and self.activity.get("name"):
			it["progress"] = self.activity.get("progress")
			it["pt0"] = self.activity.get("pt0")
			it["detail"] = self.activity.get("detail", "")
		self.dispatch(kind, p)
		self.update_hud()

	def track(self, kind, p):
		a = self.activity
		if kind == "busy":
			name = p.get("name")
			if name:
				now = time.time()
				self.activity = {"name": name, "t0": now, "pt0": now, "stage": STAGE_OF.get(name), "detail": "", "progress": None, "tokens": 0}
			else:
				a["name"] = None
		elif not a.get("name"):
			return
		elif kind == "log":
			a["detail"] = p.get("text", "")
		elif kind == "job_progress":
			a["job"] = dict(p)
			a["progress"] = min(1.0, float(p.get("done", 0.0)) / max(float(p.get("total", 1.0)), 1e-6))
		elif kind == "align_step":
			a["detail"] = f"round {p.get('round', 0) + 1} · largest gap {p.get('gap', 0):.2f}"
			if not a.get("job"):
				a["progress"] = min(1.0, p.get("round", 0) / max(p.get("rounds", 1), 1))
		elif kind == "progress":
			task = p.get("task", "")
			if task != a.get("task") and not a.get("job"):
				a["pt0"] = time.time()
			a["progress"] = self.job_part(a, p.get("value"))
			a["stage"] = "WORKSPACE" if task == "workspace" else ("LEVERS" if task.startswith("lever") or task == "guard" else ("MODEL" if task in ("load", "lens download", "plan", "convert") else a.get("stage")))
			a["task"] = task
			if task == "lens download":
				a["detail"] = f"downloading the lens · {p.get('value', 0) * 100:.0f}%"
			elif task == "convert":
				a["detail"] = f"safetensors → GGUF · {p.get('value', 0) * 100:.0f}%"
		elif kind == "profile_prompt":
			a["stage"], a["task"] = "PROFILE", "profile"
			a["progress"] = self.job_part(a, p["index"] / max(p.get("total") or 1, 1))
			a["detail"] = f"probe {p['index'] + 1}/{p.get('total', '?')} · {p['prompt']}"
		elif kind == "profile_response":
			a["progress"] = self.job_part(a, (p["index"] + 1) / max((self.profiling or {}).get("total") or 1, 1))
		elif kind == "chat_start":
			a["stage"], a["task"], a["progress"], a["tokens"] = "DIALOGUE", "chat", None, 0
			a["detail"] = f"prompt {p.get('prompt_tokens', 0)} tokens"
		elif kind in ("text", "frame") and a.get("task") == "chat":
			if kind == "text":
				a["tokens"] = a.get("tokens", 0) + 1
				dt = max(time.time() - a["t0"], 1e-3)
				a["detail"] = f"{a['tokens']} tokens · {a['tokens'] / dt:.0f} tok/s"

	def hud_state(self):
		loaded = bool(self.snap.get("loaded"))
		a = self.activity
		busy = a.get("name")
		stage = a.get("stage") if busy else None
		info = self.snap.get("info") or {}
		lens = self.snap.get("lens") or {}
		band = self.snap.get("band")
		n_active = sum(1 for v in self.applied.values() if v)
		coh = (self.snap.get("guard") or {}).get("coherence")

		def st(label, done):
			return "active" if stage == label else ("done" if done else "idle")

		stages = [
			("MODEL", st("MODEL", loaded), (f"{info.get('quant', '')} · " + ("logit lens" if lens.get("identity") else f"J·{lens.get('n_prompts') or '?'}")) if loaded else ""),
			("WORKSPACE", st("WORKSPACE", bool(band)), f"L{band[0]}–L{band[1]}" if band else ""),
			("PROFILE", st("PROFILE", "baseline" in self.summaries), ("baseline + now" if "current" in self.summaries and n_active else "baseline") if "baseline" in self.summaries else ""),
			("LEVERS", st("LEVERS", n_active > 0), (f"{n_active} steering · coherence {coh:.0%}" if n_active and coh is not None else f"{n_active} steering · {len(self.levers)} ready") if self.levers else ""),
			("DIALOGUE", st("DIALOGUE", bool(self.chat.msgs)), f"{self.tps[-1]:.0f} tok/s" if self.tps else ""),
		]
		hud = {"stages": stages}
		if busy:
			task = a.get("task") or ""
			title = TITLES.get(task if busy == "load" and task else busy, busy.upper())
			if task.startswith("lever:"):
				tid = task.split(":", 1)[1]
				title = "EXTRACTING LEVER · " + next((t["name"] for t in self.snap.get("traits") or [] if t["id"] == tid), tid).upper()
			hud.update(busy=True, title=title, detail=a.get("detail", ""), progress=a.get("progress"), t0=a.get("t0"), queued=len(self.todo_plan.pending()))
		elif not loaded:
			hud.update(title="NO MODEL", detail="open MODELS (Ctrl+M), pick a GGUF and LOAD")
		else:
			hud.update(title="READY", detail="drag a lever · talk to the model · MEASURE to profile the steered model")
		fs = self.fit_state
		if fs.get("state") in ("running", "stopping"):
			eta = fs.get("eta")
			hud["side"] = {"title": "LENS FIT", "detail": f"{fs.get('name', '')} · {fs.get('phase', '')} · {fs.get('done', 0)}/{fs.get('total', 0)} prompts" + (f" · ETA {int(eta // 3600)}h{int(eta % 3600 // 60):02d}" if eta else ""), "progress": fs.get("done", 0) / max(fs.get("total", 1), 1)}
		return hud

	def update_hud(self):
		self.iris.hud = self.hud_state()

	def add_trace(self, e):
		if e.get("kind") == "guard" and e.get("first") is None and not e.get("kept") and (e.get("coherence") or 0) >= (e.get("threshold") or 0) and self.busy != "align":
			SFX.play("magi")
		i = next((i for i in range(len(self.traces) - 1, max(-1, len(self.traces) - 50), -1) if e.get("id") and self.traces[i].get("id") == e["id"]), -1)
		if i >= 0:
			self.traces[i] = e
		else:
			self.traces.append(e)
		if len(self.traces) > 1500:
			del self.traces[:len(self.traces) - 1500]
		if self.trace_dlg is not None:
			self.trace_dlg.add(e)
		self.judge_box.view.trace(e)
		self.update_skip()

	def assets_context(self):
		return {"models": [m for m in self.catalog or [] if m.get("path")], "loaded": (self.snap.get("info") or {}).get("path", "") if self.snap.get("loaded") else "", "lens": (self.snap.get("lens") or {}).get("name", "") if self.snap.get("loaded") else "", "fitting": self.fitter.running()}

	def open_assets(self):
		if self.assets_dlg is None:
			from ui.assets import AssetsDialog
			self.assets_dlg = AssetsDialog(self.assets_context, self.log.add, self, lambda: self.llama_dir)
		else:
			self.assets_dlg.reload()
		self.assets_dlg.show()
		self.assets_dlg.raise_()
		self.assets_dlg.activateWindow()

	def open_trace(self):
		if self.trace_dlg is None:
			self.trace_dlg = TraceDialog(self)
		self.trace_dlg.set_entries(self.traces)
		self.trace_dlg.show()
		self.trace_dlg.raise_()
		self.trace_dlg.activateWindow()

	def open_trace_at(self, family, t):
		kinds = next((k for f, k, _rx in TRACE_LINKS if f == family), ())
		after = [e for e in self.traces if e.get("kind") in kinds and t - 1.0 <= e.get("t", 0) <= t + 1800]
		if not after:
			self.log.add("not in TRACE yet: the check is still running (TRACE keeps the events of this run only)", "warn")
			return
		self.open_trace()
		self.trace_dlg.focus(min(after, key=lambda e: e.get("t", 0)))

	def dispatch(self, kind, p):
		if kind == "trace":
			self.add_trace(p)
			return
		if kind == "ready":
			self.log.add("session ready")
			st = p.get("settings") or {}
			self.drawer.ctx.setValue(int(st.get("n_ctx", 8192)))
			self.margin = int(st.get("vram_reserve_mb", 900))
			self.worker.light("catalog", lambda s: s.catalog())
			self.worker.light("defs", lambda s: s.snapshot())
			return
		if kind == "log":
			text = p["text"].strip()
			self.log.add(p["text"], p.get("level", "info"), p.get("t"), next((f for f, _k, rx in TRACE_LINKS if rx.search(text)), ""))
			self.log_panel.set_info(str(self.log.count))
			return
		if kind == "light":
			self.on_light(p)
			return
		if kind == "busy":
			prev = self.busy
			self.busy = p.get("name")
			self.update_sync()
			idle = not self.busy and self.todo_plan.current is None and self.todo_plan.next_item() is None
			if idle and self.snap.get("loaded") and prev in READY_AFTER and self.last_ok == prev:
				sub = READY_AFTER[prev]
				rep = (self.align_info or {}).get("report") if prev == "align" else None
				if rep:
					ok = sum(1 for x in rep if x.get("ok"))
					sub = "TARGETS REACHED" if ok == len(rep) else f"{ok}/{len(rep)} TARGETS REACHED"
				self.sync_band.flash_ready(sub)
				SFX.play("boot" if prev == "load" else ("sync" if sub == "TARGETS REACHED" else "ready"))
			self.planner.pause(self.busy == "load")
			self.iris.set_fps(int(self.prefs.get("fps_busy", 30)) if self.busy else int(self.prefs.get("fps_idle", 60)))
			if self.goals_dlg is not None:
				self.goals_dlg.set_busy(str(self.busy or "").startswith("goal_"))
			label = {"judge": "JUDGING", "align": "ALIGNING", "calibrate": "CALIBRATING", "load": "LOADING", "unload": "UNLOADING", "chat": "LIVE", "profile": "PROFILING", "measure": "MEASURING", "levers": "APPLYING", "map": "MAPPING", "export": "EXPORTING", "goal_check": "CHECKING", "goal_align": "ALIGNING", "goal_optimize": "TRAINING", "recalibrate": "CALIBRATING", "trait_edit": "SAVING", "add_trait": "BUILDING", "apply_settings": "APPLYING", "system": "APPLYING", "active": "APPLYING", "exempt": "APPLYING", "rebuild": "REBUILDING", "drop": "APPLYING"}.get(self.busy, (self.busy or "").upper())
			self.header.job = p.get("label") or ""
			if self.busy:
				self.set_status(label, theme.ACCENT)
			else:
				self.set_status("READY" if self.snap.get("loaded") else "IDLE", theme.OK if self.snap.get("loaded") else theme.MUTED)
				self.drawer.progress.setText("")
				self.plan_dispatch()
				if idle:
					self.next_script()
			return
		if kind == "queue":
			self.on_queue(p)
			return
		if kind in ("done", "error"):
			self.plan_finished(kind, p)
			self.last_ok = p.get("name") if kind == "done" and not p.get("stopped") else None
			if p.get("name") == "align" and not self.planned("align"):
				self.lever_table.aligning = False
				self.lever_table.update()
				self.update_mode_hint()
		if kind in ("done", "error") and p.get("name") == "levers" and not self.planned("levers"):
			self.reconcile_levers()
		if kind in ("done", "error") and p.get("name") in ("calibrate", "recalibrate", "levers") and not (self.planned("calibrate") or self.planned("recalibrate")):
			self.calib_pending.clear()
			self.sync_pending()
		if kind == "done" and p.get("stopped") and p.get("id") in self.superseded:
			self.superseded.discard(p.get("id"))
			self.log.add(f"replaced by a newer change: {JOB_LABELS.get(p.get('name'), p.get('name'))} (what it had half done is dropped)")
			return
		if kind == "done" and p.get("stopped"):
			if p.get("requeued"):
				self.log.add("the measure waits: a chat goes first (it runs again right after)")
				return
			self.log.add(f"stopped: {JOB_LABELS.get(p.get('name'), p.get('name'))}" + (" · nothing half-done was kept" if p.get("name") in ("measure", "profile", "map", "levers", "calibrate", "recalibrate") else ""), "warn")
			return
		if kind == "done":
			name, res = p["name"], p.get("result")
			if name == "system":
				self.applied_system = (p.get("params") or {}).get("text", self.applied_system)
			elif name == "preset_load" and isinstance(res, dict):
				self.use_preset(res)
			elif name == "load" and res == "converted":
				self.worker.light("catalog", lambda s: s.catalog())
			elif name == "shot":
				QTimer.singleShot(1500, self._shot)
			elif name == "export" and isinstance(res, dict) and res.get("lora"):
				lo = res["lora"]
				fid = (lo.get("report") or {}).get("fidelity") or {}
				deep = [v for k, v in fid.items() if k.startswith("L") and isinstance(v, dict)]
				self.log.add(f"LoRA exported → {res['dir']} · {lo['gguf']} + {lo['safetensors']} · rank {lo['rank']} on L{lo['layers'][0]}–L{lo['layers'][-1]}" + (f" · approximation: reproduces {deep[-1]['reproduced']:.0%} of the steering at the last layer, {fid['gap_closed']:.0%} of the next-token gap" if deep and "gap_closed" in fid else ""), "ok")
				self.lmstudio_note(res)
				SFX.play("done")
			elif name == "export" and isinstance(res, dict):
				self.log.add(f"exported → {res['baked'] or res['cvec'] or res['dir']}" + (f" · dose {res['manifest']['dose']:g}" if res.get("manifest") else ""), "ok")
				self.lmstudio_note(res)
				SFX.play("done")
			elif name in ("add_trait", "trait_edit") and self.snap.get("loaded") and "baseline" not in self.summaries and not self.planned("profile"):
				self.log.add("the set of levers changed: profiling the model again")
				self.reprofile()
			elif name == "exempt":
				self.lever_table.exempt = set(res or [])
				self.lever_table.update()
			return
		if kind == "error":
			if p.get("name") == "load":
				self.carry = None
			self.log.add(f"{p.get('name') or ''}: {p['text']}", "error")
			self.set_status("ERROR", theme.ERR)
			SFX.play("error")
			if self.goals_dlg is not None and str(p.get("name") or "").startswith("goal_"):
				self.goals_dlg.set_status(f"stopped by an error: {p['text']}", theme.ERR)
			return
		if kind == "progress":
			task, v = p["task"], p["value"]
			self.drawer.progress.setText(f"{task} {v * 100:.0f}%")
			return
		if kind == "plan":
			self.loaded_plan = dict(p.get("plan") or {}, lens_bytes=p.get("lens_bytes", 0))
			return
		if kind == "lens_index":
			self.set_lens_index(p.get("lenses"))
			return
		if kind in ("goals", "lever_dropped", "judge") or kind.startswith("goal_"):
			self.on_goal_event(kind, p)
			return
		if kind in ("loaded", "unloaded"):
			self.calib_pending.clear()
			if kind == "unloaded":
				if self.pending_fit:
					QTimer.singleShot(500, lambda f=self.pending_fit: self.launch_fit(f))
					self.pending_fit = None
				self.apply_pending.clear()
				self.use_unloaded(p)
				self.summaries, self.levers, self.strengths, self.applied, self.ref_odds = {}, {}, {}, {}, None
				self.loaded_plan = None
				self.chat.msgs = []
				self.chat.view.clear()
				self.heat.clear()
				self.goal_scores = {}
			elif not p.get("loaded"):
				self.use_unloaded(p)
			else:
				self.snap = p
				self.goal_defs = p.get("goal_defs") or self.goal_defs
				self.judge_state = p.get("judge") or self.judge_state
				self.summaries = dict(p.get("summaries") or {})
				self.levers = dict(p.get("levers") or {})
				self.inactive = set(p.get("inactive") or [])
				self.applied = dict(p.get("strengths") or {})
				self.targets = dict(p.get("targets") or {})
				self.align_info = {}
				self.update_mode_hint()
				busy_levers = self.planned("levers")
				wanted = {t: self.strengths[t] for t in self.apply_pending if t in self.strengths} if busy_levers else {}
				if not busy_levers:
					self.apply_pending.clear()
				self.strengths = {**self.applied, **wanted}
				self.ref_odds = p.get("ref_odds")
				g = p.get("guard") or {}
				self.lever_table.set_factors(g.get("factors"), g.get("free") or [])
				self.lever_table.set_range(g.get("dose_max") or 1.5)
				self.lever_table.exempt = set(config.load_settings().get("guard_exempt") or [])
				self.lever_table.hidden = list(p.get("hidden") or [])
				self.applied_system = p.get("system_prompt") or ""
				self.chat.system.setText(self.system_intent())
				if p.get("history") and not self.chat.streaming:
					self.chat.show_history(p["history"])
				if self.carry is not None and (self.todo_plan.running() or {}).get("kind") == "load":
					carry, self.carry = self.carry, None
					QTimer.singleShot(0, lambda c=carry: self.apply_carry(c))
			self.refresh_iris()
			self.update_judge()
			if self.goals_dlg is not None:
				self.goals_dlg.set_goals(self.goal_defs)
				self.goals_dlg.set_judge(self.judge_state, config.load_settings()["coherence"].get("judge", ""))
			return
		if kind == "targets":
			self.targets = dict(p.get("targets") or {})
			self.lever_table.targets = dict(self.targets)
			self.lever_table.update()
			return
		if kind == "align_step":
			self.align_info = dict(self.align_info, step=p)
			self.update_mode_hint()
			self.update_sync()
			SFX.play("tick")
			return
		if kind == "aligned":
			self.align_info = dict(p)
			it = self.todo_plan.running()
			if it is not None and it["kind"] == "align":
				it["verdict"] = self.align_verdict(p)
			self.lever_table.aligning = False
			self.lever_table.update()
			self.update_mode_hint()
			self.update_sync()
			return
		if kind == "workspace":
			self.snap["workspace"] = p["workspace"]
			self.snap["band"] = p["band"]
			self.refresh_iris()
			return
		if kind == "profile_prompt":
			self.profiling = p
			return
		if kind == "profile_text":
			self.iris.pulse()
			return
		if kind == "profile_response":
			self.log.add(f"probe {p['index'] + 1}: {p['text'][:140].replace(chr(10), ' ')}")
			return
		if kind == "profile":
			self.summaries[p["tag"]] = p["summary"]
			if p.get("ref_odds") is not None:
				self.ref_odds = p["ref_odds"]
			self.refresh_iris()
			self.update_sync()
			return
		if kind == "lever":
			self.levers[p["id"]] = p
			self.calib_pending.discard(p["id"])
			self.sync_pending()
			self.refresh_iris()
			return
		if kind == "levers":
			if p.get("by") == "align":
				self.strengths.update(p["strengths"])
				for t in [t for t in self.strengths if t not in p["strengths"]]:
					self.strengths[t] = 0.0
			self.applied = dict(p["strengths"])
			self.snap["guard"] = p["guard"]
			self.lever_table.set_factors(p["guard"].get("factors"), p["guard"].get("free") or [])
			self.lever_table.set_range(p["guard"].get("dose_max") or 1.5)
			self.update_judge()
			self.apply_pending = {t for t in self.apply_pending if abs(self.applied.get(t, 0.0) - self.strengths.get(t, 0.0)) > 1e-6}
			self.sync_pending()
			self.refresh_iris()
			return
		if kind == "chat_start":
			self.heat.clear()
			if not p.get("retry") and p.get("text") is not None:
				self.chat.add_user(p["text"])
			self.chat.start_assistant(any(self.applied.values()), self.chat_config(p.get("config")), bool(p.get("retry")))
			return
		if kind == "text":
			self.chat.append_text(p["text"])
			if p["text"].strip():
				self.type_n = (getattr(self, "type_n", 0) + 1) % 7
				SFX.play("type", self.type_n % 4)
			return
		if kind == "frame":
			self.heat.add(p)
			if p["i"] % 10 == 0:
				self.lens_panel.set_info(self.heat.span_text())
			if self.ref_odds is not None:
				odds = np.asarray(p["odds"], dtype=np.float32)
				ref = np.asarray(self.ref_odds, dtype=np.float32)
				if odds.shape == ref.shape:
					self.iris.set_live(odds - ref)
			else:
				self.iris.pulse()
			return
		if kind == "chat_done" and p.get("failed"):
			self.chat.drop_failed()
			return
		if kind == "chat_done":
			self.chat.end_assistant(p.get("stats"))
			SFX.play("eot")
			if self.history_dlg is not None and self.history_dlg.isVisible():
				self.history_dlg.reload()
				self.history_dlg.select(p.get("id"))
			tps = (p.get("stats") or {}).get("tps")
			if tps:
				self.tps.append(tps)
			self.update_header()
			return
		if kind == "chat_reset":
			self.heat.clear()
			return

	def on_goal_event(self, kind, p):
		dlg = self.goals_dlg
		if kind == "judge":
			self.judge_state = p
			self.judge_box.view.set_info(state=dict(p))
			self.update_header()
			if dlg is not None:
				dlg.set_judge(p, config.load_settings()["coherence"].get("judge", ""))
			return
		if kind == "goals":
			self.goal_defs = p["goals"]
			if self.snap.get("loaded"):
				self.snap["goals"] = p["rows"]
				self.refresh_iris()
			if dlg is not None:
				dlg.set_goals(self.goal_defs)
			return
		if kind == "lever_dropped":
			for d in (self.levers, self.strengths, self.applied, self.goal_scores):
				d.pop(p["id"], None)
			self.refresh_iris()
			return
		lid = "goal_" + p.get("goal", "")
		if kind == "goal_row":
			if dlg is not None and p["goal"] == dlg.current:
				dlg.add_row(p["row"])
			return
		if kind == "goal_scan":
			if dlg is not None:
				dlg.set_scan(p)
			return
		if kind == "goal_trial":
			if not p.get("strength"):
				self.goal_scores.setdefault(lid, {})["base"] = p.get("match")
			if dlg is not None and p["goal"] == dlg.current:
				dlg.add_trial(p)
			return
		if kind == "goal_result":
			name = next((g.get("name") for g in self.goal_defs if g.get("id") == p.get("goal")), p.get("goal"))
			self.add_trace({"kind": "goal", "t": time.time(), "context": f"goal check: {name}" + (f" · {p['model']}" if p.get("model") else ""), "match": p.get("match"), "sense": p.get("sense"), "strength": p.get("strength"), "rows": p.get("rows") or []})
			sc = self.goal_scores.setdefault(lid, {})
			sc["now"] = p.get("match")
			if not p.get("strength") and p.get("match") is not None:
				sc["base"] = p["match"]
			self.refresh_iris()
			if dlg is not None:
				dlg.set_result(p)
			return
		if kind == "goal_step":
			self.eva_goal_step(p)
			if dlg is not None:
				dlg.set_step(p)
			return
		if kind == "goal_optimized":
			self.eva_goal_optimized(p)
			self.strengths[p["lever"]] = 1.0
			if (p.get("trained") or {}).get("carrier") == "both":
				self.strengths[GO.lora_key(p["lever"])] = 1.0
			self.keep_planned(p["lever"])
			self.refresh_iris()
			cl = p.get("closed")
			carrier = {"lora": "LoRA", "both": "vector + LoRA"}.get((p.get("trained") or {}).get("carrier"), "vector")
			self.log.add(f"goal trained ({carrier}, L{p['layers'][0]}–L{p['layers'][-1]})" + (f", closes {cl:.0%} of the gap to the character" if cl is not None else ""), "ok")
			if dlg is not None:
				dlg.set_optimized(p)
			return
		if kind == "goal_aligned":
			self.strengths[p["lever"]] = p["strength"]
			self.keep_planned(p["lever"])
			self.refresh_iris()
			if dlg is not None:
				dlg.set_aligned(p)

	def keep_planned(self, lid):
		keys = (lid, GO.lora_key(lid))
		self.applied.update({k: self.strengths[k] for k in keys if k in self.strengths})
		later = [i for i in self.todo_plan.pending() if i["kind"] == "levers"]
		want = later[-1]["params"]["strengths"] if later else {}
		for k in keys:
			if k in want:
				self.strengths[k] = float(want[k])
		self.apply_pending = {t for t in set(self.strengths) | set(self.applied) if abs(self.strengths.get(t, 0.0) - self.applied.get(t, 0.0)) > 1e-6}
		self.sync_pending()

	def start_script(self):
		a = self.args
		self.script = ["started"]
		self.load_model(a.load, "auto" if not a.no_lens else None, a.ctx)
		levers = {}
		for item in (a.lever or "").split(","):
			if "=" in item:
				tid, val = item.split("=", 1)
				levers[tid.strip()] = float(val)
		if levers:
			self.submit_levers(levers)
			if a.measure:
				self.plan_add("measure", "measure the steered model (quick, informative)", lambda s: s.profile("current", quick=True, stop=self.worker.stop))
		if a.chat:
			self.send_chat(a.chat, now=False)
		if a.shot:
			self.plan_add("shot", "screenshot", lambda s: "shot")
		self.plan_apply()

	def next_script(self):
		return

	def _shot(self):
		path = self.args.shot
		self.grab().save(path)
		with open(os.path.splitext(path)[0] + "_log.txt", "w", encoding="utf-8") as f:
			f.write(self.log.toPlainText())
		self.log.add(f"screenshot → {path}")
		if self.args.quit:
			QTimer.singleShot(300, self.close)
