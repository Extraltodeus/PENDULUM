from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout

from ui import theme
from ui.widgets import DialogTitle, EvaDialog, button

GROUPS = (
	("traits", "PERSONALITY", "the default levers, measured by the lens"),
	("custom", "CUSTOM LEVERS", "your own levers, and the levers made from goals (ADD AS A LEVER)"),
	("goals", "GOALS", "goals placed with TRAIN or ALIGN (GOALS window)"),
)


def group_of(r):
	if r.get("goal"):
		return "goals"
	if r.get("custom") or r.get("lever_only"):
		return "custom"
	return "traits"


class ActiveLeversDialog(EvaDialog):
	def __init__(self, rows, inactive, doses, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Active levers")
		self.setMinimumSize(480, 560)
		self.before = set(inactive)
		lay = QVBoxLayout(self)
		lay.setSpacing(6)
		lay.addWidget(DialogTitle("ACTIVE LEVERS", "THE LEVERS TAKEN INTO ACCOUNT"))
		info = QLabel("Unchecked levers are hidden from LEVERS and left out of every operation: their dose goes back to 0, the alignment and the presets you apply leave them alone. Their traits are still measured by the profile. The choice holds for every model, and presets save it.")
		info.setFont(theme.mono(7.5))
		info.setStyleSheet(f"color: {theme.MUTED};")
		info.setWordWrap(True)
		lay.addWidget(info)
		self.tree = QTreeWidget()
		self.tree.setHeaderHidden(True)
		self.tree.setFont(theme.mono(8.5))
		self.items = {}
		for key, label, tip in GROUPS:
			members = [r for r in rows if not r.get("sub") and group_of(r) == key]
			if not members:
				continue
			top = QTreeWidgetItem([f"{label}   ·   {len(members)}"])
			top.setFlags(top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
			top.setToolTip(0, tip)
			top.setForeground(0, theme.qc(theme.NEON))
			self.tree.addTopLevelItem(top)
			for r in members:
				d = float(doses.get(r["id"], 0.0) or 0.0)
				it = QTreeWidgetItem([r["name"] + (f"   ·   dose {d:+.2f}" if abs(d) > 1e-9 else "")])
				it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
				it.setCheckState(0, Qt.CheckState.Unchecked if r["id"] in self.before else Qt.CheckState.Checked)
				it.setToolTip(0, f"{r['name']}\n+ {r.get('plus', '')}   ↔   − {r.get('minus', '')}" + ("\nunchecked, its dose goes back to 0" if abs(d) > 1e-9 else ""))
				top.addChild(it)
				self.items[r["id"]] = it
			top.setExpanded(True)
		lay.addWidget(self.tree, 1)
		btns = QHBoxLayout()
		for text, tip, on in (("ALL", "check every lever", True), ("NONE", "uncheck every lever", False)):
			b = button(text, tip)
			b.clicked.connect(lambda _c=False, v=on: self.set_all(v))
			btns.addWidget(b)
		btns.addStretch(1)
		cancel = button("CANCEL")
		ok = button("SAVE", "hide and leave out the unchecked levers", accent=True)
		ok.setDefault(True)
		cancel.clicked.connect(self.reject)
		ok.clicked.connect(self.accept)
		btns.addWidget(cancel)
		btns.addWidget(ok)
		lay.addLayout(btns)

	def set_all(self, on):
		for i in range(self.tree.topLevelItemCount()):
			self.tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

	def inactive(self):
		off = {tid for tid, it in self.items.items() if it.checkState(0) != Qt.CheckState.Checked}
		return sorted((self.before - set(self.items)) | off)
