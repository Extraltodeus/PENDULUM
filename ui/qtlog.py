import faulthandler
import os
import sys
import time

from PySide6.QtCore import QtMsgType, qInstallMessageHandler

STATE = {"path": None}
LEVELS = {QtMsgType.QtFatalMsg: "FATAL", QtMsgType.QtCriticalMsg: "CRITICAL"}


def handler(mode, ctx, msg):
	if sys.stderr is not None:
		try:
			sys.stderr.write(msg + "\n")
		except Exception:
			pass
	level = LEVELS.get(mode)
	if level is None or not STATE["path"]:
		return
	try:
		os.makedirs(os.path.dirname(STATE["path"]), exist_ok=True)
		with open(STATE["path"], "a", encoding="utf-8") as f:
			f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {level} {msg}\n")
			if mode == QtMsgType.QtFatalMsg:
				f.flush()
				faulthandler.dump_traceback(f, all_threads=True)
	except Exception:
		pass


def install(path):
	STATE["path"] = path
	qInstallMessageHandler(handler)
