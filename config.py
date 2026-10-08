import json
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
WORKDIR_PATH = os.path.join(ROOT, "workdir.json")
DATA_ENV = os.environ.get("PENDULUM_DATA_DIR") or ""
CARRY_FILES = ("settings.json", "ui.json", "goals.json", "custom_traits.json", "trait_overrides.json")


def norm_dir(path):
	path = os.path.expandvars(os.path.expanduser(str(path or "").strip().strip('"')))
	if not path:
		return ""
	path = os.path.abspath(path if os.path.isabs(path) else os.path.join(ROOT, path))
	if os.path.normcase(path) == os.path.normcase(os.path.abspath(DEFAULT_DATA_DIR)):
		return ""
	return path.replace("\\", "/")


def same_dir(a, b):
	return os.path.normcase(os.path.abspath(a or DEFAULT_DATA_DIR)) == os.path.normcase(os.path.abspath(b or DEFAULT_DATA_DIR))


def _read_boot():
	try:
		with open(WORKDIR_PATH, encoding="utf-8") as f:
			data = json.load(f)
		return data if isinstance(data, dict) else {}
	except Exception:
		return {}


def _write_boot(data):
	tmp = WORKDIR_PATH + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=1)
	os.replace(tmp, WORKDIR_PATH)


def load_workdir():
	return "" if DATA_ENV else norm_dir(_read_boot().get("data_dir"))


def save_workdir(path, carry_from=""):
	if DATA_ENV:
		return False
	data = {"data_dir": norm_dir(path)}
	if carry_from and not same_dir(carry_from, data["data_dir"]):
		data["carry_from"] = str(carry_from).replace("\\", "/")
	_write_boot(data)
	return True


def carry_over(src, dst):
	done = []
	if not src or same_dir(src, dst) or not os.path.isdir(src):
		return done
	os.makedirs(dst, exist_ok=True)
	for name in CARRY_FILES:
		a, b = os.path.join(src, name), os.path.join(dst, name)
		if os.path.isfile(a) and not os.path.exists(b):
			shutil.copy2(a, b)
			done.append(name)
	return done


def _data_dir():
	if DATA_ENV:
		return DATA_ENV, "", []
	boot = _read_boot()
	wd = norm_dir(boot.get("data_dir"))
	target = wd or DEFAULT_DATA_DIR
	if wd:
		try:
			os.makedirs(wd, exist_ok=True)
		except OSError as exc:
			return DEFAULT_DATA_DIR, f"working folder {wd} unavailable ({exc}): using {DEFAULT_DATA_DIR.replace(os.sep, '/')} for this run", []
	if not boot.get("carry_from"):
		return target, "", []
	try:
		carried = carry_over(boot["carry_from"], target)
	except OSError:
		carried = []
	try:
		_write_boot({"data_dir": wd})
	except OSError:
		pass
	return target, "", carried


def _paths(data_dir):
	global DATA_DIR, SETTINGS_PATH, LENSES_DIR, PROFILES_DIR, EXPORTS_DIR, CACHE_DIR, CONVERTED_DIR, CUSTOM_TRAITS_PATH, TRAIT_OVERRIDES_PATH, GOALS_PATH
	DATA_DIR = data_dir
	SETTINGS_PATH = os.path.join(data_dir, "settings.json")
	LENSES_DIR = os.path.join(data_dir, "lenses")
	PROFILES_DIR = os.path.join(data_dir, "profiles")
	EXPORTS_DIR = os.path.join(data_dir, "exports")
	CACHE_DIR = os.path.join(data_dir, "cache")
	CONVERTED_DIR = os.path.join(data_dir, "converted")
	CUSTOM_TRAITS_PATH = os.path.join(data_dir, "custom_traits.json")
	TRAIT_OVERRIDES_PATH = os.path.join(data_dir, "trait_overrides.json")
	GOALS_PATH = os.path.join(data_dir, "goals.json")


def use_data_dir(path):
	global DATA_DIR_ERROR, CARRIED
	wd = norm_dir(path)
	target = wd or DEFAULT_DATA_DIR
	os.makedirs(target, exist_ok=True)
	save_workdir(wd)
	_paths(target)
	DATA_DIR_ERROR, CARRIED = "", []
	return target


DATA_DIR, DATA_DIR_ERROR, CARRIED = _data_dir()
TRAITS_PATH = os.path.join(ROOT, "persona", "traits.json")
_paths(DATA_DIR)

DEFAULTS = {
	"llama_dir": "",
	"model_dirs": [],
	"n_ctx": 8192,
	"apply_delay": 3.0,
	"vram_reserve_mb": 900,
	"lens_device": "auto",
	"last_model": "",
	"sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.05, "max_tokens": 400},
	"coherence": {"calibrate": True, "guard": True, "tolerance": 50, "absurdity": 30, "judge": ""},
}


def load_settings():
	data = {}
	if os.path.exists(SETTINGS_PATH):
		try:
			with open(SETTINGS_PATH, encoding="utf-8") as f:
				data = json.load(f)
		except Exception:
			data = {}
	merged = {**DEFAULTS, **data}
	merged["sampling"] = {**DEFAULTS["sampling"], **(data.get("sampling") or {})}
	merged["coherence"] = {**DEFAULTS["coherence"], **(data.get("coherence") or {})}
	return merged


def save_settings(settings):
	os.makedirs(DATA_DIR, exist_ok=True)
	tmp = SETTINGS_PATH + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(settings, f, ensure_ascii=False, indent=1)
	os.replace(tmp, SETTINGS_PATH)


def setup_env():
	os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
	os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
	os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
