from PySide6.QtWidgets import QVBoxLayout

from ui import theme
from ui.widgets import EvaDialog, ModelDrawer


class ModelsWindow(EvaDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Models")
		self.setStyleSheet(theme.STYLE)
		lay = QVBoxLayout(self)
		lay.setSpacing(8)
		lay.addWidget(self.frameless("MODELS", "pick a model · its VRAM plan and load options · LOAD, or plan loads and unloads in TO DO", "Ctrl+M"))
		self.drawer = ModelDrawer()
		lay.addWidget(self.drawer, 1)
		self.resize(1240, 760)

	def present(self):
		self.show()
		self.raise_()
		self.activateWindow()
		self.drawer.search.setFocus()
