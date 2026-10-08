from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QWidget

BG = "#050506"
PANEL = "#09090b"
PANEL2 = "#0e0e11"
LINE = "#1d1d22"
LINE2 = "#2a2a31"
HANDLE = "#4e4c56"
TEXT = "#e6e3de"
MUTED = "#76747c"
DIM = "#45434a"
ACCENT = "#ff6a1a"
ACCENT2 = "#ffb07a"
COOL = "#8f7dff"
COOL2 = "#c3b8ff"
OK = "#52d68a"
WARN = "#ffb020"
ERR = "#ff4d4d"
ON_ACCENT = "#000000"
NEON = "#ff6a1a"

PHOSPHOR = "#3dff8a"
ALARM = "#ff2a2a"
CYAN = "#30dcff"
AMBER = "#ffb000"
BLUE = "#2f6bff"

ACCENT_RGB = (1.0, 0.42, 0.10)
COOL_RGB = (0.56, 0.49, 1.0)
WHITE_RGB = (0.86, 0.85, 0.83)
ACCENTS = {"ember": ("#ff6a1a", "#ffb07a"), "amber": ("#ffa21a", "#ffd08a"), "crimson": ("#ff3b5c", "#ff9aac"), "cyan": ("#1ac8ff", "#8fe4ff"), "acid": ("#a8f01a", "#d6ff8f"), "blueberry": ("#4c00ff", "#8e47ff"), "nerv": ("#ff7300", "#ffb36b"), "unit-00": ("#3b82ff", "#a3c6ff"), "unit-01": ("#9b4dff", "#c9a2ff"), "unit-02": ("#ff2e3e", "#ff9a5c")}
COOLS = {"nerv": ("#2ec8ff", "#a6e9ff"), "unit-00": ("#ffaa1f", "#ffd894"), "unit-01": ("#45f07a", "#b2ffc6"), "unit-02": ("#24d2ff", "#a3ecff")}
COOL_DEFAULT = ("#8f7dff", "#c3b8ff")
STYLES = {"nerv": "NERV · neon terminal", "plain": "plain · sober"}
STYLE_NAME = "nerv"
ACCENT_NAME = "ember"
PLAIN = {"BG": "#050506", "PANEL": "#09090b", "PANEL2": "#0e0e11", "LINE": "#1d1d22", "LINE2": "#2a2a31", "HANDLE": "#4e4c56", "MUTED": "#76747c", "DIM": "#45434a"}

_mono = None
_cond = None


def mono(size=9, bold=False):
	global _mono
	if _mono is None:
		families = set(QFontDatabase.families())
		_mono = next((f for f in ("JetBrains Mono", "Cascadia Mono", "Cascadia Code", "Consolas", "Courier New") if f in families), "Consolas")
	f = QFont(_mono)
	f.setPointSizeF(size)
	f.setBold(bold)
	f.setStyleHint(QFont.StyleHint.Monospace)
	return f


def cond_family():
	global _cond
	if _cond is None:
		families = set(QFontDatabase.families())
		_cond = next((f for f in ("Bahnschrift", "Arial Narrow") if f in families), "")
	return _cond


def cond(size=9, weight=600, caps=False, px=False):
	fam = cond_family()
	if not fam:
		f = mono(size * 0.9, weight >= 600)
	else:
		f = QFont(fam)
		f.setWeight(QFont.Weight(int(weight)))
		if fam == "Bahnschrift":
			f.setStretch(QFont.Stretch.Condensed)
	if px:
		f.setPixelSize(max(6, int(round(size))))
	elif fam:
		f.setPointSizeF(size)
	if caps:
		f.setCapitalization(QFont.Capitalization.AllUppercase)
	f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
	return f


def nerv():
	return STYLE_NAME == "nerv"


def ui(size=8, bold=False):
	if not nerv():
		return mono(size, bold)
	return cond(size * 1.22, 700 if bold else 600, True)


def tag_font(w, size=8, bold=False):
	w.setProperty("eva_font", [float(size), bool(bold)])
	w.setFont(ui(size, bold))
	return w


def refont(root):
	for w in [root] + root.findChildren(QWidget):
		v = w.property("eva_font")
		if v:
			w.setFont(ui(float(v[0]), bool(v[1])))


def qc(hex_or_rgb, alpha=255):
	c = QColor(hex_or_rgb) if isinstance(hex_or_rgb, str) else QColor.fromRgbF(*hex_or_rgb)
	c.setAlpha(alpha)
	return c


def mix(a, b, t):
	ca, cb = QColor(a), QColor(b)
	t = max(0.0, min(1.0, t))
	return QColor(int(ca.red() + (cb.red() - ca.red()) * t), int(ca.green() + (cb.green() - ca.green()) * t), int(ca.blue() + (cb.blue() - ca.blue()) * t))


def signed_color(v, alpha=255):
	c = QColor(ACCENT) if v >= 0 else QColor(COOL)
	c.setAlpha(alpha)
	return c


def luma(c):
	q = QColor(c)
	return 0.2126 * q.redF() + 0.7152 * q.greenF() + 0.0722 * q.blueF()


def rgba(c, a):
	q = QColor(c)
	return f"rgba({q.red()}, {q.green()}, {q.blue()}, {max(0, min(255, int(a)))})"


def _palette():
	global BG, PANEL, PANEL2, LINE, LINE2, HANDLE, MUTED, DIM
	if not nerv():
		BG, PANEL, PANEL2, LINE, LINE2, HANDLE, MUTED, DIM = (PLAIN[k] for k in ("BG", "PANEL", "PANEL2", "LINE", "LINE2", "HANDLE", "MUTED", "DIM"))
		return
	lift = 0.32 if luma(ACCENT) < 0.2 else 0.0
	BG, PANEL, PANEL2 = "#000000", "#020203", "#09090b"
	LINE = mix("#000000", ACCENT, 0.24 + lift).name()
	LINE2 = mix("#000000", ACCENT, 0.4 + lift).name()
	HANDLE = mix("#000000", ACCENT, 0.62 + lift * 0.5).name()
	MUTED, DIM = "#86828a", "#4d4a52"


def apply_style(name):
	global STYLE_NAME, STYLE
	STYLE_NAME = name if name in STYLES else "nerv"
	_palette()
	STYLE = _style()


def apply_accent(name):
	global ACCENT, ACCENT2, ACCENT_RGB, COOL, COOL2, COOL_RGB, ON_ACCENT, NEON, ACCENT_NAME, STYLE
	ACCENT_NAME = name if name in ACCENTS else "ember"
	ACCENT, ACCENT2 = ACCENTS[ACCENT_NAME]
	COOL, COOL2 = COOLS.get(ACCENT_NAME, COOL_DEFAULT)
	c = QColor(ACCENT)
	ACCENT_RGB = (c.redF(), c.greenF(), c.blueF())
	k = QColor(COOL)
	COOL_RGB = (k.redF(), k.greenF(), k.blueF())
	ON_ACCENT = "#000000" if luma(ACCENT) > 0.3 else "#ffffff"
	NEON = ACCENT if luma(ACCENT) >= 0.22 else ACCENT2
	_palette()
	STYLE = _style()


def _style():
	return _style_nerv() if nerv() else _style_plain()


def _style_nerv():
	glow, hot, sel = rgba(ACCENT, 26), rgba(ACCENT, 60), rgba(ACCENT, 70)
	return f"""
QWidget {{ background: {BG}; color: {TEXT}; }}
QToolTip {{ background: #000000; color: {TEXT}; border: 1px solid {ACCENT}; padding: 5px; }}
QScrollBar:vertical {{ background: {BG}; width: 7px; margin: 0; border: none; border-left: 1px solid {LINE}; }}
QScrollBar::handle:vertical {{ background: {HANDLE}; min-height: 24px; margin: 1px 1px 1px 2px; }}
QScrollBar:horizontal {{ background: {BG}; height: 7px; margin: 0; border: none; border-top: 1px solid {LINE}; }}
QScrollBar::handle:horizontal {{ background: {HANDLE}; min-width: 24px; margin: 2px 1px 1px 1px; }}
QScrollBar::handle:hover {{ background: {ACCENT2}; }}
QScrollBar::handle:pressed {{ background: {ACCENT}; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; border: none; background: none; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: #000000; border: 1px solid {LINE2}; color: {TEXT}; selection-background-color: {ACCENT}; selection-color: {ON_ACCENT}; padding: 4px; }}
QLineEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border: 1px solid {HANDLE}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox QAbstractItemView {{ background: #000000; border: 1px solid {ACCENT}; selection-background-color: {sel}; selection-color: #ffffff; outline: none; }}
QTreeWidget {{ background: {PANEL}; border: 1px solid {LINE2}; outline: none; }}
QTreeWidget::item {{ padding: 2px 0; }}
QTreeWidget::item:selected {{ background: {sel}; color: #ffffff; }}
QTreeWidget::item:hover {{ background: {glow}; }}
QHeaderView {{ background: #000000; border: none; }}
QHeaderView::section {{ background: #000000; color: {ACCENT2}; border: none; border-bottom: 1px solid {ACCENT}; padding: 4px 6px; }}
QListWidget {{ background: {PANEL}; border: none; outline: none; }}
QListWidget::item {{ padding: 6px 4px; border-bottom: 1px solid {LINE}; }}
QListWidget::item:selected {{ background: {glow}; color: {TEXT}; border-left: 2px solid {ACCENT}; }}
QListWidget::item:hover {{ background: {glow}; }}
QPushButton {{ background: {rgba(ACCENT, 14)}; border: 1px solid {LINE2}; color: {ACCENT2}; padding: 4px 12px; }}
QPushButton:hover {{ background: {hot}; border: 1px solid {ACCENT}; color: #ffffff; }}
QPushButton:pressed {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: {ON_ACCENT}; }}
QPushButton:disabled {{ background: transparent; color: {DIM}; border: 1px dashed {LINE}; }}
QPushButton[accent="true"] {{ background: {rgba(ACCENT, 34)}; border: 3px double {ACCENT}; color: #ffffff; padding: 3px 12px; }}
QPushButton[accent="true"]:hover {{ background: {ACCENT}; color: {ON_ACCENT}; }}
QPushButton[accent="true"]:disabled {{ background: transparent; color: {DIM}; border: 3px double {LINE}; }}
QPushButton#win_close:hover {{ background: {ALARM}; border: 1px solid {ALARM}; }}
QPushButton[toggle="true"] {{ color: {MUTED}; }}
QPushButton[toggle="true"]:checked {{ background: {rgba(ACCENT, 46)}; border: 2px solid {ACCENT}; color: #ffffff; padding: 3px 11px; }}
QPushButton[danger="true"]:hover {{ background: {ALARM}; border: 1px solid {ALARM}; color: #ffffff; }}
QPushButton[toggle="true"]:checked:hover {{ background: {hot}; border: 2px solid {ACCENT2}; }}
QMenu {{ background: #000000; border: 1px solid {ACCENT}; padding: 4px 0; }}
QMenu::item {{ padding: 5px 22px 5px 14px; color: {TEXT}; border-left: 3px solid transparent; }}
QMenu::item:selected {{ background: {sel}; color: #ffffff; border-left: 3px solid {ACCENT}; }}
QMenu::item:disabled {{ color: {DIM}; }}
QMenu::separator {{ height: 1px; background: {LINE2}; margin: 4px 8px; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 10px; height: 10px; border: 1px solid {HANDLE}; background: #000000; }}
QCheckBox::indicator:hover {{ border: 1px solid {ACCENT}; }}
QCheckBox::indicator:checked {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT2}, stop:1 {ACCENT}); border: 1px solid {ACCENT2}; }}
QCheckBox, QSlider {{ outline: none; }}
QCheckBox::indicator:focus {{ border: 1px solid {ACCENT2}; }}
QRadioButton {{ spacing: 7px; outline: none; }}
QRadioButton::indicator {{ width: 10px; height: 10px; border: 1px solid {HANDLE}; background: #000000; }}
QRadioButton::indicator:hover {{ border: 1px solid {ACCENT}; }}
QRadioButton::indicator:checked {{ background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 {ACCENT2}, stop:0.55 {ACCENT}, stop:0.62 #000000); border: 1px solid {ACCENT}; }}
QRadioButton::indicator:focus {{ border: 1px solid {ACCENT2}; }}
QRadioButton:disabled, QCheckBox:disabled, QLabel:disabled {{ color: {DIM}; }}
QRadioButton::indicator:disabled, QCheckBox::indicator:disabled {{ background: {BG}; border: 1px dashed {LINE}; }}
QRadioButton::indicator:checked:disabled {{ background: {BG}; border: 1px dashed {LINE}; }}
QCheckBox::indicator:checked:disabled {{ background: {LINE}; border: 1px dashed {LINE2}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QTextBrowser:disabled {{ color: {DIM}; border: 1px dashed {LINE}; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:hover {{ background: {LINE2}; }}
QSlider::groove:horizontal {{ height: 2px; background: {LINE2}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; }}
QSlider::handle:horizontal {{ width: 6px; margin: -6px 0; background: {ACCENT2}; border: 1px solid #000000; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::groove:horizontal:disabled {{ background: {LINE}; }}
QSlider::sub-page:horizontal:disabled {{ background: {LINE2}; }}
QSlider::handle:horizontal:disabled {{ background: {DIM}; border: 1px solid {LINE}; }}
QMessageBox QLabel {{ color: {TEXT}; }}
"""


def _style_plain():
	return f"""
QWidget {{ background: {BG}; color: {TEXT}; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {LINE2}; padding: 4px; }}
QScrollBar:vertical {{ background: {PANEL2}; width: 8px; margin: 0; border: none; }}
QScrollBar::handle:vertical {{ background: {HANDLE}; min-height: 24px; }}
QScrollBar:horizontal {{ background: {PANEL2}; height: 8px; margin: 0; border: none; }}
QScrollBar::handle:horizontal {{ background: {HANDLE}; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {MUTED}; }}
QScrollBar::handle:pressed {{ background: {ACCENT}; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: {PANEL2}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; border: none; background: none; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {{ background: {PANEL2}; border: 1px solid {LINE2}; color: {TEXT}; selection-background-color: {ACCENT}; padding: 4px; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; border: 1px solid {LINE2}; selection-background-color: {LINE2}; }}
QTreeWidget {{ background: {PANEL}; border: 1px solid {LINE}; outline: none; }}
QTreeWidget::item {{ padding: 2px 0; }}
QTreeWidget::item:selected {{ background: {LINE2}; color: {TEXT}; }}
QTreeWidget::item:hover {{ background: {PANEL2}; }}
QHeaderView {{ background: {PANEL2}; border: none; }}
QHeaderView::section {{ background: {PANEL2}; color: {MUTED}; border: none; border-bottom: 1px solid {LINE2}; padding: 4px 6px; }}
QListWidget {{ background: {PANEL}; border: none; outline: none; }}
QListWidget::item {{ padding: 6px 4px; border-bottom: 1px solid {LINE}; }}
QListWidget::item:selected {{ background: {PANEL2}; color: {TEXT}; border-left: 2px solid {ACCENT}; }}
QListWidget::item:hover {{ background: {PANEL2}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {LINE2}; color: {TEXT}; padding: 5px 12px; }}
QPushButton:hover {{ border: 1px solid {ACCENT}; color: {ACCENT2}; }}
QPushButton:pressed {{ background: {LINE}; }}
QPushButton:disabled {{ color: {DIM}; border: 1px solid {LINE}; }}
QPushButton[accent="true"] {{ border: 1px solid {ACCENT}; color: {ACCENT}; }}
QPushButton[accent="true"]:hover {{ background: {ACCENT}; color: {BG}; }}
QPushButton[toggle="true"] {{ color: {MUTED}; }}
QPushButton[toggle="true"]:checked {{ border: 2px solid {ACCENT}; color: {ACCENT}; padding: 4px 11px; }}
QPushButton[danger="true"]:hover {{ border: 1px solid {ERR}; color: {ERR}; }}
QMenu {{ background: {PANEL2}; border: 1px solid {LINE2}; padding: 4px 0; }}
QMenu::item {{ padding: 5px 22px 5px 14px; color: {TEXT}; }}
QMenu::item:selected {{ background: {LINE2}; color: {ACCENT2}; }}
QMenu::item:disabled {{ color: {MUTED}; }}
QMenu::separator {{ height: 1px; background: {LINE2}; margin: 4px 8px; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 11px; height: 11px; border: 1px solid {LINE2}; background: {PANEL2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}
QCheckBox, QSlider {{ outline: none; }}
QCheckBox::indicator:focus {{ border: 1px solid {ACCENT2}; }}
QRadioButton {{ spacing: 6px; outline: none; }}
QRadioButton::indicator {{ width: 10px; height: 10px; border: 1px solid {LINE2}; border-radius: 6px; background: {PANEL2}; }}
QRadioButton::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}
QRadioButton::indicator:focus {{ border: 1px solid {ACCENT2}; }}
QRadioButton:disabled, QCheckBox:disabled, QLabel:disabled {{ color: {DIM}; }}
QRadioButton::indicator:disabled, QCheckBox::indicator:disabled {{ background: {BG}; border: 1px solid {LINE}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{ color: {DIM}; border: 1px solid {LINE}; }}
QSplitter::handle {{ background: {LINE}; }}
QDoubleSpinBox {{ background: {PANEL2}; border: 1px solid {LINE2}; color: {TEXT}; selection-background-color: {ACCENT}; padding: 4px; }}
QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE2}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; }}
QSlider::handle:horizontal {{ width: 10px; margin: -5px 0; background: {TEXT}; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT2}; }}
"""


_palette()
STYLE = _style()
