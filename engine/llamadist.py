import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile

RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8"
SOURCE = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.zip"
PAGE = "https://github.com/ggml-org/llama.cpp/releases"
LAYOUTS = ("", "bin", os.path.join("build", "bin", "Release"), os.path.join("build", "bin"), os.path.join("build", "bin", "RelWithDebInfo"))
SRC = {"dir": ""}


def system():
	if sys.platform.startswith("win"):
		return "win"
	return "mac" if sys.platform == "darwin" else "linux"


def arch():
	return "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"


def lib_file(name):
	return {"win": f"{name}.dll", "mac": f"lib{name}.dylib"}.get(system(), f"lib{name}.so")


def exe(name):
	return name + (".exe" if system() == "win" else "")


def find_bin(root, depth=3):
	if not root or not os.path.isdir(root):
		return ""
	for sub in LAYOUTS:
		d = os.path.join(root, sub)
		if os.path.isfile(os.path.join(d, lib_file("llama"))):
			return os.path.abspath(d)
	if depth <= 0:
		return ""
	try:
		subs = sorted(e.path for e in os.scandir(root) if e.is_dir() and not e.name.startswith("."))
	except OSError:
		return ""
	for d in subs:
		hit = find_bin(d, depth - 1)
		if hit:
			return hit
	return ""


def set_src(path):
	SRC["dir"] = os.path.abspath(path) if path else ""


def is_src(d):
	return bool(d) and os.path.isfile(os.path.join(d, "convert_hf_to_gguf.py")) and os.path.isdir(os.path.join(d, "gguf-py"))


def find_src(*roots):
	for root in (SRC["dir"],) + roots:
		d = os.path.abspath(root) if root else ""
		for _ in range(4):
			if not d:
				break
			for c in (d, os.path.join(d, "src")):
				if is_src(c):
					return c
			up = os.path.dirname(d)
			if up == d:
				break
			d = up
	return ""


def downloaded(root):
	base = os.path.join(root, "llama.cpp")
	try:
		tags = sorted((e.path for e in os.scandir(base) if e.is_dir()), key=os.path.getmtime, reverse=True)
	except OSError:
		return ""
	return next((b for b in (find_bin(t, 2) for t in tags) if b), "")


def guess(*roots):
	from engine.llama import find_llama_dir

	return (find_llama_dir() or next((d for d in (downloaded(r) for r in roots) if d), "")).replace(os.sep, "/")


def gguf_version():
	try:
		import importlib.metadata as md

		return md.version("gguf")
	except Exception:
		return ""


def on_path(name):
	cuda = os.environ.get("CUDA_PATH", "")
	dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d] + ([os.path.join(cuda, "bin"), os.path.join(cuda, "bin", "x64")] if cuda else [])
	return next((os.path.join(d, name) for d in dirs if os.path.isfile(os.path.join(d, name))), "")


def cuda_runtime(bin_dir):
	names = ("cudart64_13.dll", "cudart64_12.dll") if system() == "win" else ("libcudart.so.13", "libcudart.so.12")
	for n in names:
		if bin_dir and os.path.isfile(os.path.join(bin_dir, n)):
			return os.path.join(bin_dir, n)
		hit = on_path(n)
		if hit:
			return hit
	return ""


def status(bin_dir, src_dir=""):
	b = find_bin(bin_dir) if bin_dir else ""
	src = find_src(src_dir, b or bin_dir)
	backends = [k for k, n in (("CUDA", "ggml-cuda"), ("Vulkan", "ggml-vulkan"), ("Metal", "ggml-metal"), ("ROCm", "ggml-hip"), ("SYCL", "ggml-sycl")) if b and os.path.isfile(os.path.join(b, lib_file(n)))]
	return {
		"bin": b,
		"lib": bool(b),
		"fit": bool(b) and os.path.isfile(os.path.join(b, exe("llama-fit-params"))),
		"server": bool(b) and os.path.isfile(os.path.join(b, exe("llama-server"))),
		"backends": backends,
		"cudart": cuda_runtime(b) if "CUDA" in backends else "",
		"src": src,
		"gguf": "local" if src else (gguf_version() and f"pip {gguf_version()}"),
	}


def nvidia():
	try:
		import pynvml

		pynvml.nvmlInit()
		n = pynvml.nvmlDeviceGetCount()
		cc = [pynvml.nvmlDeviceGetCudaComputeCapability(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(n)]
		drv = pynvml.nvmlSystemGetCudaDriverVersion()
		return {"gpus": n, "cuda": (drv // 1000, drv % 1000 // 10), "cc": max(cc) if cc else (0, 0)}
	except Exception:
		return None


def releases(timeout=20):
	req = urllib.request.Request(RELEASES, headers={"Accept": "application/vnd.github+json", "User-Agent": "Pendulum"})
	with urllib.request.urlopen(req, timeout=timeout) as r:
		return json.loads(r.read().decode("utf-8"))


def builds(rels, sysname=None, cpu=None):
	sysname = sysname or system()
	cpu = cpu or arch()
	os_tag = {"win": "win", "linux": "ubuntu", "mac": "macos"}[sysname]
	ext = ".zip" if sysname == "win" else ".tar.gz"
	for rel in rels:
		base = f"llama-{rel['tag_name']}-bin-{os_tag}-"
		names = sorted(a["name"] for a in rel.get("assets") or [] if a["name"].startswith(base) and a["name"].endswith(f"{cpu}{ext}"))
		if names:
			return rel, names
	return None, []


def label(tag, name):
	core = name.split("-bin-", 1)[-1].rsplit(".zip", 1)[0].rsplit(".tar.gz", 1)[0]
	parts = core.split("-")[1:-1]
	if not parts:
		return "Metal" if core.startswith("macos") else "CPU"
	kind = parts[0]
	if kind == "cuda":
		return "CUDA " + (parts[1] if len(parts) > 1 else "")
	return {"vulkan": "Vulkan", "cpu": "CPU", "hip": "ROCm (HIP)", "rocm": "ROCm", "sycl": "SYCL", "openvino": "OpenVINO", "opencl": "OpenCL", "kompute": "Kompute"}.get(kind, kind) + (" " + " ".join(parts[1:]) if len(parts) > 1 else "")


def plan_for(rel, name, why=""):
	tag = rel["tag_name"]
	assets = {a["name"]: a for a in rel.get("assets") or []}
	files = [assets[name]]
	v = _cuda_of(name)
	if v:
		os_tag = name.split("-bin-", 1)[-1].split("-", 1)[0]
		cpu = name.rsplit("-", 1)[-1].split(".", 1)[0]
		rt = [a for n, a in assets.items() if n.startswith("cudart-") and f"-{os_tag}-cuda-" in n and _cuda_of(n) == v and n.rsplit("-", 1)[-1].split(".", 1)[0] == cpu]
		if rt:
			files.append(rt[0])
	return {"tag": tag, "files": [{"name": a["name"], "url": a["browser_download_url"], "size": int(a.get("size") or 0), "sha256": (a.get("digest") or "").split(":", 1)[-1] if (a.get("digest") or "").startswith("sha256:") else ""} for a in files], "source": SOURCE.format(tag=tag), "why": why}


def _cuda_of(name):
	tail = name.split("cuda-", 1)[1] if "cuda-" in name else ""
	parts = tail.split("-", 1)[0].split(".")
	try:
		return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
	except ValueError:
		return None


def pick(rels, sysname=None, cpu=None, gpu="auto"):
	sysname = sysname or system()
	cpu = cpu or arch()
	gpu = nvidia() if gpu == "auto" else gpu
	os_tag = {"win": "win", "linux": "ubuntu", "mac": "macos"}[sysname]
	ext = ".zip" if sysname == "win" else ".tar.gz"
	plain = ("vulkan-", "cpu-") if sysname == "win" else ("vulkan-", "")
	for rel in rels:
		tag = rel["tag_name"]
		assets = {a["name"]: a for a in rel.get("assets") or []}
		base = f"llama-{tag}-bin-{os_tag}-"
		if not any(n.startswith(base) for n in assets):
			continue
		choice, why = None, ""
		if sysname == "mac":
			n = f"{base}{cpu}{ext}"
			if n in assets:
				choice, why = n, "Metal build"
		else:
			if gpu:
				fits = []
				for n in assets:
					if n.startswith(base + "cuda-") and n.endswith(f"-{cpu}{ext}"):
						v = _cuda_of(n)
						if v and v[0] <= gpu["cuda"][0] and (gpu["cc"][0] < 12 or v >= (12, 8)):
							fits.append((v, n))
				if fits:
					v, choice = max(fits)
					why = f"NVIDIA GPU, driver up to CUDA {gpu['cuda'][0]}.{gpu['cuda'][1]}: CUDA {v[0]}.{v[1]} build"
			if choice is None:
				for kind, label in zip(plain, ("Vulkan build (any recent GPU)", "CPU build")):
					n = f"{base}{kind}{cpu}{ext}"
					if n in assets:
						choice, why = n, label + (" · no CUDA build fits this driver and GPU" if gpu else "")
						break
		if choice is None:
			continue
		return plan_for(rel, choice, why)
	return None


class Cancelled(Exception):
	pass


def fetch(url, dest, size=0, sha256="", on_progress=None, cancel=None, chunk=1 << 20):
	tmp = dest + ".part"
	h = hashlib.sha256()
	done = 0
	req = urllib.request.Request(url, headers={"User-Agent": "Pendulum"})
	with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
		total = size or int(r.headers.get("Content-Length") or 0)
		while True:
			if cancel is not None and cancel.is_set():
				f.close()
				os.remove(tmp)
				raise Cancelled()
			b = r.read(chunk)
			if not b:
				break
			f.write(b)
			h.update(b)
			done += len(b)
			if on_progress:
				on_progress(done, total)
	if sha256 and h.hexdigest() != sha256:
		os.remove(tmp)
		raise ValueError(f"{os.path.basename(dest)}: checksum mismatch, the download is corrupted")
	os.replace(tmp, dest)
	return dest


def extract(archive, dest, strip=False):
	os.makedirs(dest, exist_ok=True)
	root = os.path.abspath(dest)
	if archive.endswith(".zip"):
		with zipfile.ZipFile(archive) as z:
			names = z.namelist()
			lead = os.path.commonprefix(names).split("/", 1)[0] + "/" if strip and names else ""
			for n in names:
				rel = n[len(lead):] if lead and n.startswith(lead) else n
				out = os.path.abspath(os.path.join(root, rel))
				if not rel or not out.startswith(root):
					continue
				if n.endswith("/"):
					os.makedirs(out, exist_ok=True)
					continue
				os.makedirs(os.path.dirname(out), exist_ok=True)
				with z.open(n) as src, open(out, "wb") as dst:
					shutil.copyfileobj(src, dst)
		return dest
	with tarfile.open(archive, "r:*") as t:
		t.extractall(root, filter="data")
	return dest


def install(plan, root, with_source=True, on_progress=None, cancel=None, log=None):
	log = log or (lambda m: None)
	home = os.path.join(root, plan["tag"])
	bin_dir = os.path.join(home, "bin")
	os.makedirs(bin_dir, exist_ok=True)
	files = list(plan["files"])
	skip_rt = [f for f in files[1:] if f["name"].startswith("cudart-") and cuda_runtime("")]
	files = [f for f in files if f not in skip_rt]
	if skip_rt:
		log("CUDA runtime already installed: its package is not downloaded")
	jobs = files + ([{"name": f"source-{plan['tag']}.zip", "url": plan["source"], "size": 0, "sha256": ""}] if with_source else [])
	total = sum(f["size"] for f in jobs) or 1
	base = 0
	for f in jobs:
		dest = os.path.join(home, f["name"])
		log(f"downloading {f['name']}" + (f" ({f['size'] / 2**20:.0f} MB)" if f["size"] else ""))
		fetch(f["url"], dest, f["size"], f["sha256"], (lambda d, t, b=base: on_progress((b + d) / max(total, b + t, 1))) if on_progress else None, cancel)
		base += f["size"] or os.path.getsize(dest)
		log(f"extracting {f['name']}")
		if f["name"].startswith("source-"):
			extract(dest, os.path.join(home, "src"), strip=True)
		else:
			extract(dest, bin_dir)
		os.remove(dest)
	b = find_bin(bin_dir)
	return {"bin": b or bin_dir, "src": os.path.join(home, "src") if with_source else "", "tag": plan["tag"]}
