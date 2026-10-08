import os
import re
import shlex
import subprocess

MIB = 1024 * 1024
BASE_MARGIN_MIB = 640
HEADLESS_MARGIN_MIB = 256
CREATE_NO_WINDOW = 0x08000000


def display_gpus():
	try:
		import pynvml as nv

		nv.nvmlInit()
	except Exception:
		return None
	try:
		visible = os.environ.get("CUDA_VISIBLE_DEVICES")
		order = [int(x) for x in visible.split(",") if x.strip().isdigit()] if visible else list(range(nv.nvmlDeviceGetCount()))
		out = set()
		for i, dev in enumerate(order):
			h = nv.nvmlDeviceGetHandleByIndex(dev)
			try:
				if nv.nvmlDeviceGetDisplayActive(h) or nv.nvmlDeviceGetDisplayMode(h):
					out.add(i)
			except Exception:
				out.add(i)
		return out
	except Exception:
		return None


def reserve_mibs(n, reserve_mib=BASE_MARGIN_MIB):
	shown = display_gpus()
	return [int(reserve_mib) if shown is None or i in shown else min(HEADLESS_MARGIN_MIB, int(reserve_mib)) for i in range(n)]


LOAD_DEFAULTS = {"split_mode": "layer", "main_gpu": 0, "flash_attn": "auto", "kv_type": "f16", "tensor_split": None, "n_gpu_layers": None}
SPLIT_MODES = ("layer", "row", "none")
FLASH = ("auto", "on", "off")
KV_TYPES = ("f16", "q8_0", "q4_0")


def load_args(options):
	o = {**LOAD_DEFAULTS, **(options or {})}
	args = []
	if o["split_mode"] in SPLIT_MODES and o["split_mode"] != "layer":
		args += ["-sm", o["split_mode"]]
	if int(o["main_gpu"] or 0):
		args += ["-mg", str(int(o["main_gpu"]))]
	if o["flash_attn"] in FLASH and o["flash_attn"] != "auto":
		args += ["-fa", o["flash_attn"]]
	if o["kv_type"] in KV_TYPES and o["kv_type"] != "f16":
		args += ["-ctk", o["kv_type"], "-ctv", o["kv_type"]]
	return args


def _run_fit(llama_dir, path, n_ctx, margins_mib, estimate=False, extra=()):
	exe = os.path.join(llama_dir, "llama-fit-params.exe")
	if not os.path.isfile(exe):
		return None
	args = [exe, "-m", path, "-c", str(int(n_ctx)), "-fitt", ",".join(str(int(m)) for m in margins_mib), "-ub", "512", "-b", "512"]
	if estimate:
		args += ["-fitp", "on"]
	args += list(extra)
	try:
		proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, creationflags=CREATE_NO_WINDOW)
	except Exception:
		return None
	lines = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
	return lines


def _parse_args(line):
	out = {"n_ctx": None, "n_gpu_layers": -1, "tensor_split": None, "overrides": []}
	tokens = shlex.split(line, posix=True)
	i = 0
	while i < len(tokens):
		tok = tokens[i]
		val = tokens[i + 1] if i + 1 < len(tokens) else ""
		if tok in ("-c", "--ctx-size"):
			out["n_ctx"] = int(val)
		elif tok in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
			out["n_gpu_layers"] = int(val) if val not in ("all", "auto") else -1
		elif tok in ("-ts", "--tensor-split"):
			out["tensor_split"] = [float(x) for x in re.split(r"[,/]", val) if x]
		elif tok in ("-ot", "--override-tensor"):
			for entry in val.split(","):
				if "=" in entry:
					pattern, dev = entry.rsplit("=", 1)
					out["overrides"].append((pattern, dev))
		else:
			i += 1
			continue
		i += 2
	return out


def _parse_estimate(lines):
	est = {}
	for line in lines or []:
		parts = line.split()
		if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts[1:]):
			est[parts[0]] = {"model": int(parts[1]) * MIB, "context": int(parts[2]) * MIB, "compute": int(parts[3]) * MIB}
	return est


def fit(llama_dir, path, n_ctx, margins_mib, n_layer=0, extra=()):
	lines = _run_fit(llama_dir, path, n_ctx, margins_mib, extra=extra)
	if not lines:
		return None
	plan = _parse_args(lines[-1])
	plan["estimate"] = _parse_estimate(_run_fit(llama_dir, path, plan["n_ctx"] or n_ctx, margins_mib, estimate=True, extra=extra))
	plan["cpu_overrides"] = [p for p, d in plan["overrides"] if d.upper() == "CPU"]
	host = plan["estimate"].get("Host", {})
	plan["cpu_bytes"] = host.get("model", 0)
	plan["spill"] = bool(plan["cpu_overrides"]) or plan["n_gpu_layers"] not in (-1, None) and plan["n_gpu_layers"] < n_layer
	return plan


def plan_load(llama_dir, path, gpus, n_ctx, n_layer, lens_bytes=0, base_margin_mib=BASE_MARGIN_MIB, options=None):
	plan = _plan(llama_dir, path, gpus, n_ctx, n_layer, lens_bytes, base_margin_mib, load_args(options))
	o = {**LOAD_DEFAULTS, **(options or {})}
	for k in ("split_mode", "main_gpu", "flash_attn", "kv_type"):
		plan[k] = o[k]
	if o["tensor_split"]:
		plan["tensor_split"] = [float(x) for x in o["tensor_split"]]
		plan.setdefault("notes", []).append("tensor split set by hand: " + "/".join(f"{x:g}" for x in plan["tensor_split"]))
	if o["n_gpu_layers"] is not None:
		plan["n_gpu_layers"] = int(o["n_gpu_layers"])
	return plan


def _plan(llama_dir, path, gpus, n_ctx, n_layer, lens_bytes, base_margin_mib, extra):
	n = len(gpus)
	if not n:
		return {"n_ctx": n_ctx, "n_gpu_layers": 0, "tensor_split": None, "overrides": [], "lens_device": "cpu", "spill": True, "estimate": {}, "notes": ["no GPU found: CPU only"]}
	base = reserve_mibs(n, base_margin_mib)
	plain = fit(llama_dir, path, n_ctx, base, n_layer, extra)
	if plain is None:
		return {"n_ctx": n_ctx, "n_gpu_layers": -1, "tensor_split": None, "overrides": [], "lens_device": f"cuda:{max(range(n), key=lambda i: gpus[i]['free'])}", "spill": False, "estimate": {}, "notes": ["llama-fit-params unavailable: default split"]}
	notes = []
	if plain["n_ctx"] and plain["n_ctx"] < n_ctx:
		notes.append(f"context reduced to {plain['n_ctx']} to stay on GPU")
	plain["lens_device"] = "cpu"
	if not lens_bytes:
		plain["lens_device"] = f"cuda:{max(range(n), key=lambda i: gpus[i]['free'])}"
		plain["notes"] = notes
		return plain
	lens_mib = int(lens_bytes / MIB) + 256
	left = {i: gpus[i]["free"] / MIB - sum(plain["estimate"].get(f"CUDA{i}", {}).values()) / MIB - base[i] for i in range(n)}
	roomy = max(range(n), key=lambda i: left[i])
	if plain["estimate"] and not plain["spill"] and left[roomy] >= lens_mib + 768:
		plain["lens_device"] = f"cuda:{roomy}"
		plain["notes"] = notes
		return plain
	order = sorted(range(n), key=lambda i: gpus[i]["free"] - plain["estimate"].get(f"CUDA{i}", {}).get("model", 0), reverse=True)
	for g in order:
		margins = list(base)
		margins[g] += lens_mib
		with_lens = fit(llama_dir, path, plain["n_ctx"] or n_ctx, margins, n_layer, extra)
		if with_lens is None:
			continue
		worse = (with_lens["spill"] and not plain["spill"]) or with_lens["cpu_bytes"] > plain["cpu_bytes"] + 256 * MIB or (with_lens["n_ctx"] or n_ctx) < (plain["n_ctx"] or n_ctx)
		if not worse:
			with_lens["lens_device"] = f"cuda:{g}"
			with_lens["notes"] = notes
			return with_lens
	notes.append("lens kept on CPU: no GPU room without spilling the model")
	plain["notes"] = notes
	return plain
