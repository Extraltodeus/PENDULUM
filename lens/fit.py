import json
import os
import re
import subprocess
import sys
import threading
import time

import config

WORKER = os.path.join(config.ROOT, "lens", "fit_worker.py")
CREATE_NO_WINDOW = 0x08000000
GGUF_ARCHS = {"llama", "mistral", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "gemma2", "gemma3", "gemma4", "phi3", "falcon", "stablelm", "starcoder2", "nemotron", "gpt2", "bloom", "deci", "gpt-oss", "lfm2"}


def fits_dir():
	return os.path.join(config.DATA_DIR, "fits")


def _read(path, default=None):
	try:
		with open(path, encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return default


def _write(path, data):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	tmp = path + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False)
	os.replace(tmp, path)


def slug(text):
	return re.sub(r"[^A-Za-z0-9.\-]+", "-", text).strip("-")[:80] or "lens"


def hf_repo(model):
	for b in model.get("base_models") or []:
		m = re.match(r"https?://huggingface\.co/([^/]+/[^/?#]+)", b)
		if m:
			return m.group(1)
	return ""


def corpus(n):
	path = os.path.join(fits_dir(), "corpus_wikitext.json")
	cached = _read(path, []) or []
	if len(cached) >= n:
		return cached[:n]
	from jlens.examples import load_wikitext_prompts

	prompts = load_wikitext_prompts(n)
	_write(path, prompts)
	return prompts


SEC_PER_UNIT = 3.2e-11


def _spp(model, gpu, dim_batch):
	fast = 1.6 if gpu["total"] >= 15 * 2**30 else 1.0
	return SEC_PER_UNIT * int(model.get("n_params") or 0) * int(model.get("n_embd") or 0) / dim_batch / fast


def plan(model, gpus, dtype, devices):
	params = int(model.get("n_params") or 0)
	weights = params * (0.62 if dtype == "nf4" else 2.0)
	sel = [g for i, g in enumerate(gpus) if i in devices]
	need = weights + 3.2 * 2**30
	batch = lambda i: 4 if gpus[i]["total"] >= 15 * 2**30 else 2
	if not sel:
		return {"mode": "none", "note": "select at least one GPU"}
	if len(sel) > 1 and all(need <= g["total"] - 1.0 * 2**30 for g in sel):
		spp = 1.0 / sum(1.0 / _spp(model, gpus[i], batch(i)) for i in devices)
		return {"mode": "data", "devices": [f"cuda:{i}" for i in devices], "dim_batch": [batch(i) for i in devices], "spp": spp, "note": f"one model copy per GPU (~{need / 2**30:.0f} GB each), prompts split between them"}
	if len(sel) == 1 and need <= sel[0]["total"] - 1.0 * 2**30:
		i = devices[0]
		return {"mode": "single", "devices": [f"cuda:{i}"], "dim_batch": [batch(i)], "spp": _spp(model, gpus[i], batch(i)), "note": f"~{need / 2**30:.0f} GB on CUDA{i}"}
	if need <= sum(g["total"] for g in sel) - 1.5 * 2**30 * len(sel):
		return {"mode": "split", "devices": ["auto"], "visible": ",".join(str(i) for i in devices), "dim_batch": [2], "spp": _spp(model, {"total": 0}, 2) * 1.3, "note": f"model split across the GPUs (~{need / 2**30:.0f} GB), slower"}
	return {"mode": "none", "note": f"needs ~{need / 2**30:.0f} GB of VRAM: try nf4 or both GPUs"}


class Fitter:
	def __init__(self, on_state):
		self.on_state = on_state
		self.procs = []
		self.lock = threading.Lock()
		self.state = {"state": "idle"}

	def running(self):
		return self.state.get("state") in ("running", "stopping")

	def _emit(self):
		self.on_state(dict(self.state, workers=[dict(w) for w in self.state.get("workers") or []]))

	def start(self, params):
		with self.lock:
			if self.running():
				raise RuntimeError("a lens fit is already running")
			self.state = {"state": "running", "phase": "corpus", "name": params["name"], "done": 0, "total": params["n_prompts"], "t0": time.time(), "workers": [], "error": None, "eta": None, "params": params}
		threading.Thread(target=self._run, args=(params,), daemon=True).start()
		self._emit()

	def stop(self):
		with self.lock:
			for p in self.procs:
				if p.poll() is None:
					p.terminate()
			if self.state.get("state") == "running":
				self.state["state"] = "stopping"
		self._emit()

	def _python(self):
		exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
		return exe if os.path.exists(exe) else sys.executable

	def _run(self, params):
		try:
			job = os.path.join(fits_dir(), params["name"])
			os.makedirs(job, exist_ok=True)
			_write(os.path.join(job, "params.json"), params)
			prompts = corpus(params["n_prompts"])
			devices = params["devices"]
			if len(devices) == 2:
				cut = int(round(len(prompts) * 0.62))
				slices = [prompts[:cut], prompts[cut:]]
			else:
				slices = [prompts]
			self.state.update(phase="loading", total=len(prompts))
			self._emit()
			env = dict(os.environ)
			if params.get("visible"):
				env["CUDA_VISIBLE_DEVICES"] = params["visible"]
			self.procs = []
			workers = []
			for i, (dev, chunk) in enumerate(zip(devices, slices)):
				sp = os.path.join(job, f"slice{i}.json")
				_write(sp, chunk)
				cmd = [self._python(), "-X", "utf8", WORKER, "--source", params["source"], "--device", dev, "--dtype", params["dtype"], "--prompts", sp, "--checkpoint", os.path.join(job, f"ckpt{i}.pt"), "--out", os.path.join(job, f"part{i}.pt"), "--dim-batch", str(params["dim_batch"][i]), "--max-seq-len", str(params.get("max_seq_len", 128))]
				if params.get("gguf_file"):
					cmd += ["--gguf-file", params["gguf_file"]]
				if params.get("source_layers"):
					cmd += ["--source-layers", ",".join(str(int(l)) for l in params["source_layers"])]
				proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", cwd=config.ROOT, env=env, creationflags=CREATE_NO_WINDOW)
				self.procs.append(proc)
				w = {"device": dev, "done": 0, "total": len(chunk), "state": "loading", "hist": [], "err": ""}
				workers.append(w)
				threading.Thread(target=self._read_out, args=(proc, w), daemon=True).start()
				threading.Thread(target=self._read_err, args=(proc, w), daemon=True).start()
			self.state["workers"] = workers
			self._emit()
			for proc in self.procs:
				proc.wait()
			time.sleep(0.2)
			if self.state.get("state") == "stopping":
				self.state.update(state="stopped", phase="stopped")
				self._emit()
				return
			failed = [w for w, p in zip(workers, self.procs) if p.returncode != 0]
			if failed:
				tail = next((w["err"].strip().splitlines()[-1] for w in failed if w["err"].strip()), "?")
				raise RuntimeError(f"fit worker failed: {tail}")
			self.state.update(phase="merge")
			self._emit()
			self._merge(params, job, len(slices))
		except Exception as exc:
			self.state.update(state="error", phase="error", error=str(exc))
			self._emit()

	def _merge(self, params, job, n_parts):
		from jlens.lens import JacobianLens

		parts = [JacobianLens.load(os.path.join(job, f"part{i}.pt")) for i in range(n_parts)]
		lens = JacobianLens.merge(parts) if len(parts) > 1 else parts[0]
		out_dir = os.path.join(config.LENSES_DIR, params["name"])
		os.makedirs(out_dir, exist_ok=True)
		path = os.path.join(out_dir, f"{params['name']}_jacobian_lens_n{lens.n_prompts}.pt")
		lens.save(path)
		meta = {"name": params["name"], "model_id": params.get("model_id") or params["name"], "source": params["source"], "gguf_file": params.get("gguf_file", ""), "dtype": params["dtype"], "n_prompts": lens.n_prompts, "d_model": lens.d_model, "source_layers": [lens.source_layers[0], lens.source_layers[-1]], "corpus": "Salesforce/wikitext-103-raw-v1", "max_seq_len": params.get("max_seq_len", 128), "devices": params["devices"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "fit_seconds": round(time.time() - self.state["t0"], 1)}
		_write(os.path.join(out_dir, "meta.json"), meta)
		self.state.update(state="done", phase="done", lens_path=path, meta=meta, eta=0, done=self.state.get("total", 0))
		self._emit()

	def _read_err(self, proc, w):
		for line in proc.stderr:
			w["err"] = (w["err"] + line)[-4000:]

	def _read_out(self, proc, w):
		for line in proc.stdout:
			try:
				ev = json.loads(line.strip())
			except Exception:
				continue
			kind = ev.get("event")
			if kind == "loading":
				w["state"] = "loading"
			elif kind == "ready":
				w["state"] = "fitting"
				self.state["phase"] = "fitting"
			elif kind in ("progress", "resume"):
				w["state"] = "fitting"
				w["done"], w["total"] = ev["done"], ev["total"]
				w["hist"] = (w["hist"] + [[w["done"], time.time()]])[-8:]
				self.state["phase"] = "fitting"
			elif kind == "done":
				w["state"] = "done"
				w["done"] = w["total"]
			self._totals()
			self._emit()

	def _totals(self):
		ws = self.state.get("workers") or []
		self.state["done"] = sum(w["done"] for w in ws)
		etas = []
		for w in ws:
			h = w.get("hist") or []
			if w["done"] >= w["total"] or len(h) < 2 or h[-1][0] <= h[0][0]:
				continue
			rate = (h[-1][0] - h[0][0]) / max(h[-1][1] - h[0][1], 1e-3)
			etas.append((w["total"] - w["done"]) / rate)
		self.state["eta"] = max(etas) if etas else None
