import argparse
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def streams():
	for name in ("stdout", "stderr"):
		if getattr(sys, name) is None:
			setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def parse():
	parser = argparse.ArgumentParser(description="Pendulum: personality workbench for GGUF models")
	parser.add_argument("--data-dir", default="", help="working folder for this run (profiles, levers, exports, settings, window layout); the one chosen in the settings is neither used nor changed")
	parser.add_argument("--load", default="", help="GGUF to load at startup (runs mapping + baseline profile)")
	parser.add_argument("--ctx", type=int, default=8192)
	parser.add_argument("--no-lens", action="store_true")
	parser.add_argument("--lever", default="", help="levers to apply after loading, e.g. warmth=-1,humor=0.8")
	parser.add_argument("--measure", action="store_true", help="measure the steered profile after applying --lever")
	parser.add_argument("--chat", default="", help="message to send once the model is ready")
	parser.add_argument("--shot", default="", help="save a screenshot of the window at the end of the script")
	parser.add_argument("--quit", action="store_true", help="quit after the screenshot")
	parser.add_argument("--size", default="", help="window size, e.g. 1720x1000")
	return parser.parse_args()


streams()
ARGS = parse()
if ARGS.data_dir:
	os.environ["HYPNOLENS_DATA_DIR"] = os.path.abspath(os.path.expanduser(ARGS.data_dir))

import config

config.setup_env()

from PySide6.QtCore import QLocale, QTimer
from PySide6.QtWidgets import QApplication

from ui import qtlog
from ui.worker import leave


def main():
	args = ARGS
	sys.setswitchinterval(0.001)
	if sys.platform.startswith("win"):
		try:
			ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Pendulum.Workbench")
		except Exception:
			pass
	qtlog.install(os.path.join(config.DATA_DIR, "qt_errors.log"))
	QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
	app = QApplication(sys.argv)
	app.setApplicationName("Pendulum")
	from ui.splash import loading, preload
	splash = loading(app)
	preload(app)
	from ui import wizard

	if wizard.wanted():
		splash.hide()
		wizard.first_start()
		splash.show()
	from ui.window import MainWindow

	win = MainWindow(args)
	win.show()
	splash.finish(win)
	if args.shot and not args.load:
		QTimer.singleShot(4000, win._shot)
	leave(app.exec(), win.worker)


if __name__ == "__main__":
	main()
