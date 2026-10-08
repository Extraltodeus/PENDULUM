import os
import subprocess
import sys
import time

from PySide6.QtCore import QFile, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout

import config
from hypno import assets as AS
from ui import theme
from ui.widgets import EvaDialog, button, menu_title

ROLE = Qt.ItemDataRole.UserRole


def human(n):
	if n >= 2**30:
		return f"{n / 2**30:.2f} GB"
	if n >= 2**20:
		return f"{n / 2**20:.1f} MB"
	if n >= 1024:
		return f"{n / 1024:.0f} KB"
	return f"{n} B" if n else "—"


def when(t):
	return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else ""


def trashed(path):
	r = QFile.moveToTrash(path)
	return bool(r[0] if isinstance(r, tuple) else r)


class SizeItem(QTreeWidgetItem):
	def __lt__(self, other):
		col = self.treeWidget().sortColumn() if self.treeWidget() else 0
		if col == 1:
			return (self.data(1, ROLE) or 0) < (other.data(1, ROLE) or 0)
		if col == 2:
			return (self.data(2, ROLE) or 0) < (other.data(2, ROLE) or 0)
		return self.text(col).lower() < other.text(col).lower()


class AssetsDialog(EvaDialog):
	def __init__(self, context, log=None, parent=None, llama_dir=None):
		super().__init__(parent)
		self.context = context
		self.log = log or (lambda *a: None)
		self.llama_dir = llama_dir or (lambda: "")
		self.setWindowTitle("Assets")
		self.setMinimumSize(900, 560)
		lay = QVBoxLayout(self)
		lay.setSpacing(6)
		lay.addWidget(self.frameless("ASSETS", "THE FILES PENDULUM WROTE IN THE WORKING FOLDER"))
		top = QHBoxLayout()
		self.search = QLineEdit()
		self.search.setFont(theme.mono(8.5))
		self.search.setPlaceholderText("filter: a model, an export, a goal…")
		self.search.textChanged.connect(self.filter)
		top.addWidget(self.search, 1)
		self.total = QLabel("")
		self.total.setFont(theme.mono(8))
		self.total.setStyleSheet(f"color: {theme.MUTED};")
		top.addWidget(self.total)
		lay.addLayout(top)
		self.tree = QTreeWidget()
		self.tree.setFont(theme.mono(8))
		self.tree.setHeaderLabels(["NAME", "SIZE", "MODIFIED", "NOTE"])
		self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
		self.tree.setSortingEnabled(True)
		self.tree.sortByColumn(1, Qt.SortOrder.DescendingOrder)
		self.tree.setUniformRowHeights(True)
		self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
		self.tree.customContextMenuRequested.connect(self.menu)
		self.tree.itemSelectionChanged.connect(self.selected)
		self.tree.itemDoubleClicked.connect(lambda it, _c: self.reveal([it]) if not it.childCount() else None)
		self.tree.itemClicked.connect(self.clicked)
		self.tree.itemExpanded.connect(lambda it: self.mark(it, True))
		self.tree.itemCollapsed.connect(lambda it: self.mark(it, False))
		self.tree.setRootIsDecorated(False)
		self.tree.setIndentation(16)
		h = self.tree.header()
		h.setStretchLastSection(True)
		h.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
		self.tree.setColumnWidth(0, 380)
		self.tree.setColumnWidth(1, 90)
		self.tree.setColumnWidth(2, 130)
		lay.addWidget(self.tree, 1)
		btns = QHBoxLayout()
		self.note = QLabel("MOVE TO TRASH sends to the Recycle Bin: restore from there if needed · what is in use is skipped")
		self.note.setFont(theme.mono(7.5))
		self.note.setStyleSheet(f"color: {theme.MUTED};")
		self.note.setWordWrap(True)
		btns.addWidget(self.note, 1)
		b_ref = button("REFRESH", "scan the working folder again")
		b_ref.clicked.connect(self.reload)
		self.b_reveal = button("SHOW IN FOLDER", "open the folder of the selected file (double-click does it too)")
		self.b_reveal.clicked.connect(lambda: self.reveal(self.tree.selectedItems()))
		self.b_trash = button("MOVE TO TRASH", "send the selected files and folders to the Recycle Bin", accent=True)
		self.b_trash.clicked.connect(lambda: self.trash(self.tree.selectedItems()))
		for b in (b_ref, self.b_reveal, self.b_trash):
			btns.addWidget(b)
		lay.addLayout(btns)
		self.resize(1180, 720)
		self.reload()

	def restyle(self):
		super().restyle()
		self.reload()

	def reload(self):
		open_keys = {self.tree.topLevelItem(i).data(0, ROLE)["key"] for i in range(self.tree.topLevelItemCount()) if self.tree.topLevelItem(i).isExpanded()} if self.tree.topLevelItemCount() else None
		groups = AS.scan(**self.context())
		self.tree.setSortingEnabled(False)
		self.tree.clear()
		for g in groups:
			top = SizeItem([f"▸ {g['name']}   ({len(g['items'])})", human(g["size"]), "", g["note"]])
			top.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
			top.setData(0, ROLE, {"key": g["key"], "group": True})
			top.setData(1, ROLE, g["size"])
			top.setData(2, ROLE, 0)
			f = top.font(0)
			f.setBold(True)
			top.setFont(0, f)
			top.setForeground(0, theme.qc(theme.ACCENT))
			top.setForeground(3, theme.qc(theme.MUTED))
			top.setToolTip(3, g["note"])
			top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
			for e in g["items"]:
				top.addChild(self.item(e))
			self.tree.addTopLevelItem(top)
			top.setExpanded(open_keys is None or g["key"] in open_keys)
		self.tree.setSortingEnabled(True)
		total = sum(g["size"] for g in groups)
		self.total.setText(f"{config.DATA_DIR.replace(os.sep, '/')}   ·   {human(total)}")
		self.filter(self.search.text())
		self.selected()

	def item(self, e):
		it = SizeItem([("▸ " if e["children"] else "  ") + e["name"] + ("/" if e["dir"] and not e["children"] else ""), human(e["size"]), when(e["mtime"]), " · ".join(x for x in (e["note"], f"in use: {e['in_use']}" if e["in_use"] else "") if x)])
		it.setData(0, ROLE, e)
		it.setData(1, ROLE, e["size"])
		it.setData(2, ROLE, e["mtime"])
		it.setToolTip(0, e["path"].replace(os.sep, "/"))
		it.setToolTip(3, it.text(3))
		it.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
		if e["in_use"]:
			for c in range(4):
				it.setForeground(c, theme.qc(theme.MUTED))
		elif e["note"].startswith("model not found"):
			it.setForeground(3, theme.qc(theme.WARN))
		for c in e["children"]:
			it.addChild(self.item(c))
		return it

	def mark(self, it, open_):
		t = it.text(0)
		if t[:2] in ("▸ ", "▾ "):
			it.setText(0, ("▾ " if open_ else "▸ ") + t[2:])

	def clicked(self, it, col):
		if it.childCount() and col == 0:
			it.setExpanded(not it.isExpanded())

	def filter(self, text):
		words = text.lower().split()
		for i in range(self.tree.topLevelItemCount()):
			top = self.tree.topLevelItem(i)
			shown = 0
			for k in range(top.childCount()):
				ch = top.child(k)
				e = ch.data(0, ROLE)
				hay = (e["name"] + " " + e["note"] + " " + " ".join(c["name"] for c in e["children"])).lower()
				hit = all(w in hay for w in words)
				ch.setHidden(not hit)
				shown += hit
			top.setHidden(bool(words) and not shown)

	def entries(self, items):
		out = []
		for it in items:
			e = it.data(0, ROLE)
			if e and not e.get("group"):
				out.append(e)
		paths = {os.path.normcase(e["path"]) for e in out}
		return [e for e in out if not any(os.path.normcase(e["path"]).startswith(p + os.sep) for p in paths)]

	def selected(self):
		sel = self.entries(self.tree.selectedItems())
		self.b_trash.setEnabled(any(not e["in_use"] for e in sel))
		self.b_reveal.setEnabled(bool(sel))
		if sel:
			free = [e for e in sel if not e["in_use"]]
			self.note.setText(f"{len(sel)} selected · {human(sum(e['size'] for e in free))} can go to the Recycle Bin" + (f" · {len(sel) - len(free)} in use, skipped" if len(free) < len(sel) else ""))
		else:
			self.note.setText("MOVE TO TRASH sends to the Recycle Bin: restore from there if needed · what is in use is skipped")

	def menu(self, pos):
		it = self.tree.itemAt(pos)
		sel = self.entries(self.tree.selectedItems() or ([it] if it else []))
		if not sel:
			return
		m = QMenu(self)
		m.setFont(theme.ui(8))
		menu_title(m, sel[0]["name"].upper() if len(sel) == 1 else f"{len(sel)} ITEMS")
		if len(sel) == 1 and sel[0].get("lora"):
			x = m.addAction("EXPORT THIS LoRA…")
			m.setToolTipsVisible(True)
			x.setToolTip("copy this LoRA to the exports folder with its llama-server launcher and LM Studio override: the model does not need to be loaded")
			x.triggered.connect(lambda: self.export_lora(sel[0]))
		a = m.addAction("SHOW IN FOLDER")
		a.triggered.connect(lambda: self.reveal(self.tree.selectedItems() or [it]))
		t = m.addAction("MOVE TO TRASH")
		t.setEnabled(any(not e["in_use"] for e in sel))
		t.triggered.connect(lambda: self.trash(self.tree.selectedItems() or [it]))
		m.exec(self.tree.viewport().mapToGlobal(pos))

	def export_lora(self, e):
		info = e["lora"]
		goal = info["goal"].lower().replace(" ", "-")
		arch = info.get("arch") or ""
		dlg = QInputDialog(self)
		dlg.setWindowTitle("Export a LoRA")
		dlg.setLabelText(f"export the LoRA of the goal '{info['goal']}'" + (f" ({info['model_name']})" if info["model_name"] else " (its model was not found: the launcher will need its path)") + f"\nstrength {info['scale']:g} (the one set on its lever, editable in the launcher)\n\nfolder name in exports/")
		dlg.setTextValue(f"{arch}_{goal}_lora" if arch else f"{goal}_lora")
		dlg.setStyleSheet(theme.STYLE)
		if not dlg.exec() or not dlg.textValue().strip():
			return
		from hypno.export import export_saved_lora
		try:
			res = export_saved_lora(e["path"], dlg.textValue().strip(), info["model"], info["model_name"], arch, info["goal"], info["scale"], self.llama_dir())
		except OSError as exc:
			self.log(f"LoRA export failed: {exc}", "error")
			return
		self.log(f"LoRA of the goal '{info['goal']}' exported to {res['dir'].replace(os.sep, '/')} (run_llama_server.bat, lmstudio_override.txt)", "ok")
		self.reload()

	def reveal(self, items):
		sel = self.entries(items)
		if not sel:
			return
		path = sel[0]["path"]
		if sys.platform.startswith("win") and os.path.exists(path):
			subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
			return
		QDesktopServices.openUrl(QUrl.fromLocalFile(path if os.path.isdir(path) else os.path.dirname(path)))

	def trash(self, items):
		sel = [e for e in self.entries(items) if not e["in_use"]]
		if not sel:
			return
		size = sum(e["size"] for e in sel)
		names = "\n".join("  " + e["name"] for e in sel[:10]) + (f"\n  … and {len(sel) - 10} more" if len(sel) > 10 else "")
		box = QMessageBox(QMessageBox.Icon.Question, "Move to trash", f"Send {len(sel)} item{'s' if len(sel) != 1 else ''} ({human(size)}) to the Recycle Bin?\n\n{names}\n\nThey can be restored from the Recycle Bin.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, self)
		box.setStyleSheet(theme.STYLE)
		box.setDefaultButton(QMessageBox.StandardButton.Cancel)
		if box.exec() != QMessageBox.StandardButton.Yes:
			return
		done, failed = [], []
		for e in sel:
			ok = [trashed(p) for p in e.get("paths") or [e["path"]] if os.path.exists(p)]
			(done if ok and all(ok) else failed).append(e)
		self.log(f"moved to the Recycle Bin: {len(done)} item{'s' if len(done) != 1 else ''} ({human(sum(e['size'] for e in done))})" + (f" · could not move: {', '.join(e['name'] for e in failed)}" if failed else ""), "warn" if failed else "ok")
		self.reload()
