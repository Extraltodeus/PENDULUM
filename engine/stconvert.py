import glob
import hashlib
import json
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

from engine.runtime import LoadCancelled

LAYER_RE = re.compile(r"^(.*?)layers\.(\d+)\.")
SHARD_RE = re.compile(r"(?i)-\d{5}-of-\d{5}\.safetensors$")
TENSOR_RE = re.compile(r"^INFO:[\w.-]+:\S+,\s+torch\.\w+ --> \w+, shape")
WRITE_RE = re.compile(r"Writing:\s+(\d+)%")
SCALES = (".scale_weight", ".weight_scale", ".weight_scale_inv")
DROP = SCALES + (".scale_input", ".input_scale", ".comfy_quant", ".weight_scale_2", "rotary_emb.inv_freq")
TOKENIZER = ("tokenizer.json", "tokenizer.model", "vocab.json")
REF_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json", "chat_template.jinja", "chat_template.json")
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
FP8_FORMATS = (None, "float8_e4m3fn", "float8_e5m2")
ARCH = {"qwen3": "qwen3", "qwen3_5": "qwen35", "qwen3_5_text": "qwen35", "qwen3_next": "qwen3next", "qwen2": "qwen2", "qwen2_5_vl": "qwen2vl", "qwen2_vl": "qwen2vl", "llama": "llama", "mistral": "llama", "mistral3": "llama", "gemma2": "gemma2", "gemma3": "gemma3", "gemma3_text": "gemma3", "phi3": "phi3", "olmo2": "olmo2", "glm4": "glm4"}
REFS = (
	("qwen3", 1024, 28, 151936, "Qwen/Qwen3-0.6B"),
	("qwen3", 2048, 28, 151936, "Qwen/Qwen3-1.7B"),
	("qwen3", 2560, 36, 151936, "Qwen/Qwen3-4B"),
	("qwen3", 4096, 36, 151936, "Qwen/Qwen3-8B"),
	("qwen3", 5120, 40, 151936, "Qwen/Qwen3-14B"),
	("qwen3", 5120, 64, 151936, "Qwen/Qwen3-32B"),
	("qwen35", 2560, 32, 248320, "Qwen/Qwen3.5-4B"),
	("qwen2", 3584, 28, 152064, "Qwen/Qwen2.5-VL-7B-Instruct"),
	("llama", 4096, 32, 128256, "unsloth/Llama-3.1-8B-Instruct"),
	("llama", 5120, 40, 131072, "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
	("gemma2", 2304, 26, 256000, "unsloth/gemma-2-2b-it"),
)
_registered = {}


def header(path):
	with open(path, "rb") as f:
		raw = f.read(8)
		if len(raw) < 8:
			raise ValueError("not a safetensors file")
		n = struct.unpack("<Q", raw)[0]
		if n <= 2 or n > 512 * 2**20:
			raise ValueError("not a safetensors file")
		h = json.loads(f.read(n))
	meta = h.pop("__metadata__", None) or {}
	return h, meta


def weight_files(d):
	try:
		names = os.listdir(d)
	except OSError:
		return []
	return sorted(os.path.join(d, f) for f in names if f.startswith("model") and f.endswith(".safetensors"))


def hf_dir(d):
	return os.path.isfile(os.path.join(d, "config.json")) and any(os.path.isfile(os.path.join(d, t)) for t in TOKENIZER) and bool(weight_files(d))


def is_source(path):
	return bool(path) and (path.lower().endswith(".safetensors") or os.path.isdir(path))


def sources(dirs):
	found = []
	for root in dirs:
		if not root or not os.path.isdir(root):
			continue
		for base, _dirs, files in os.walk(root):
			if hf_dir(base):
				found.append(base)
				continue
			if "config.json" in files:
				continue
			for name in files:
				low = name.lower()
				if low.endswith(".safetensors") and not SHARD_RE.search(low) and "lora" not in low:
					found.append(os.path.join(base, name))
	return sorted(set(found), key=lambda p: stem(p).lower())


def stem(path):
	path = os.path.abspath(path).rstrip("\\/")
	parts = path.replace("\\", "/").split("/")
	repo = next((p for p in parts if p.startswith("models--")), None)
	if repo and os.path.isdir(path):
		return repo.split("--")[-1]
	return os.path.basename(path) if os.path.isdir(path) else os.path.splitext(os.path.basename(path))[0]


def signature(path):
	files = weight_files(path) + [os.path.join(path, "config.json")] if os.path.isdir(path) else [path]
	st = [os.stat(f) for f in files]
	return [sum(s.st_size for s in st), int(max(s.st_mtime for s in st))]


def tensors_of(path):
	files = weight_files(path) if os.path.isdir(path) else [path]
	out = {}
	for f in files:
		h, _meta = header(f)
		for k, v in h.items():
			out[k] = dict(v, file=f)
	return out


def prefix_of(names):
	counts = {}
	for n in names:
		m = LAYER_RE.match(n)
		if m:
			counts[m.group(1)] = counts.get(m.group(1), 0) + 1
	return max(counts, key=counts.get) if counts else None


def family(names):
	if any(".linear_attn." in n for n in names):
		return "qwen35"
	if any("pre_feedforward_layernorm" in n for n in names):
		return "gemma3" if any(".q_norm." in n for n in names) else "gemma2"
	if any(".self_attn.q_norm." in n for n in names):
		return "qwen3"
	if any(".self_attn.q_proj.bias" in n for n in names):
		return "qwen2"
	return "llama"


def size_label(n):
	return (f"{n / 1e9:.1f}".rstrip("0").rstrip(".") + "B") if n >= 1e9 else f"{n / 1e6:.0f}M"


def numel(shape):
	n = 1
	for s in shape:
		n *= int(s)
	return n


def converter(llama_dir):
	from engine.llamadist import find_src

	src = find_src(llama_dir or "")
	return os.path.join(src, "convert_hf_to_gguf.py") if src else None


def registered(llama_dir):
	script = converter(llama_dir)
	if script is None:
		return None
	if script in _registered:
		return _registered[script]
	names = set()
	for f in [script] + glob.glob(os.path.join(os.path.dirname(script), "conversion", "*.py")):
		try:
			text = open(f, encoding="utf-8", errors="replace").read()
		except OSError:
			continue
		for m in re.finditer(r"register\((.*?)\)", text, re.S):
			names.update(re.findall(r"[\"']([A-Za-z0-9_]+)[\"']", m.group(1)))
	_registered[script] = names
	return names


def guess_ref(fam, d, n_layer, vocab):
	return next((r for f, h, n, v, r in REFS if f == fam and h == d and n == n_layer and (not v or v == vocab)), "")


def describe(path, llama_dir=""):
	size, mtime = signature(path)
	try:
		t = tensors_of(path)
	except (OSError, ValueError) as exc:
		return {"skip": True, "why": str(exc)}
	names = list(t)
	if any(".lora_" in n or n.endswith(".alpha") for n in names[:200]):
		return {"skip": True, "why": "LoRA"}
	pre = prefix_of(names)
	emb = next((n for n in names if n.endswith("embed_tokens.weight")), None)
	if pre is None or emb is None:
		return {"skip": True, "why": "not a language model"}
	vocab, d = (int(x) for x in t[emb]["shape"][:2])
	n_layer = 1 + max(int(LAYER_RE.match(n).group(2)) for n in names if n.startswith(pre) and LAYER_RE.match(n))
	dtypes = {}
	for v in t.values():
		dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + numel(v["shape"])
	main = max(dtypes, key=dtypes.get)
	fam = family(names)
	entry = {"path": path, "file": stem(path), "name": stem(path), "size": size, "mtime": mtime, "format": "safetensors", "convert": True, "kind": "dir" if os.path.isdir(path) else "file", "n_layer": n_layer, "n_nextn": 0, "n_embd": d, "n_vocab": vocab, "n_ctx_train": 0, "n_expert": 0, "n_params": sum(numel(v["shape"]) for v in t.values()), "quant": "safetensors " + ("FP8" if main.startswith("F8") else main), "weights_bytes": size, "has_chat_template": True, "basename": "", "finetune": "", "base_models": [], "ref": "", "has_head": any(n.endswith("lm_head.weight") for n in names)}
	entry["size_label"] = size_label(entry["n_params"])
	if entry["kind"] == "dir":
		cfg = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
		arch = (cfg.get("architectures") or [""])[0]
		known = registered(llama_dir)
		if known is not None and arch not in known:
			return {"skip": True, "why": f"{arch} cannot be converted by this llama.cpp"}
		tc = cfg.get("text_config") or cfg
		entry.update(arch=ARCH.get(tc.get("model_type") or cfg.get("model_type"), tc.get("model_type") or cfg.get("model_type") or fam), n_ctx_train=int(tc.get("max_position_embeddings") or 0), n_expert=int(tc.get("num_experts") or tc.get("num_local_experts") or 0), hf_arch=arch)
		return entry
	entry.update(arch=fam, ref=guess_ref(fam, d, n_layer, vocab))
	return entry


def out_path(src, out_dir):
	base = stem(src)
	out = os.path.join(out_dir, base + ".gguf")
	rec = record(out)
	if rec and os.path.normcase(os.path.abspath(rec.get("source", ""))) != os.path.normcase(os.path.abspath(src)):
		out = os.path.join(out_dir, f"{base}-{hashlib.sha1(os.path.normcase(os.path.abspath(src)).encode()).hexdigest()[:8]}.gguf")
	return out


def record(out):
	try:
		with open(out + ".json", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return None


def fresh(src, out, outtype=None):
	rec = record(out)
	if not rec or not os.path.isfile(out):
		return False
	try:
		sig = signature(src)
	except OSError:
		return False
	return rec.get("sig") == sig and (outtype is None or rec.get("outtype") == outtype)


def current(src, out, ref=""):
	if not fresh(src, out):
		return False
	return os.path.isdir(src) or not ref or (record(out) or {}).get("ref") == ref


def fetch(repo, name):
	from huggingface_hub import hf_hub_download

	err = None
	for local in (True, False):
		try:
			return hf_hub_download(repo, name, local_files_only=local), None
		except Exception as exc:
			err = exc
	return None, err


def ref_tensors(repo, cfg_path):
	files = weight_files(os.path.dirname(cfg_path))
	if files:
		out = {}
		for f in files:
			h, _meta = header(f)
			out.update({k: v["shape"] for k, v in h.items()})
		return out
	try:
		from huggingface_hub import get_safetensors_metadata

		md = get_safetensors_metadata(repo)
		return {n: list(ti.shape) for fm in md.files_metadata.values() for n, ti in fm.tensors.items()}
	except Exception:
		return None


def comfy_formats(path, names):
	from safetensors import safe_open

	keys = [n for n in names if n.endswith(".comfy_quant")]
	if not keys:
		return set()
	out = set()
	with safe_open(path, "pt") as f:
		for k in keys:
			try:
				out.add(json.loads(bytes(f.get_tensor(k).tolist()).decode("utf-8")).get("format"))
			except Exception:
				out.add("unreadable")
	return out


def expand(s, shape):
	if s.numel() == 1:
		return s.reshape(())
	if len(shape) == 2 and s.numel() == shape[0]:
		return s.reshape(shape[0], 1)
	if len(shape) == 2 and s.dim() == 2:
		bo, bi = -(-shape[0] // s.shape[0]), -(-shape[1] // s.shape[1])
		return s.repeat_interleave(bo, 0)[:shape[0]].repeat_interleave(bi, 1)[:, :shape[1]]
	raise ValueError(f"scale of shape {list(s.shape)} does not fit a weight of shape {list(shape)}")


def mapping(names, ref):
	src = prefix_of(names)
	dst = prefix_of(ref) if ref else "model."
	head = next((n for n in ref or () if n.endswith("lm_head.weight")), "lm_head.weight")
	pairs, odd = {}, []
	if src is None:
		return pairs, odd, [], src, dst, head
	for n in names:
		if n.endswith(DROP) or n == "scaled_fp8":
			continue
		if n.startswith(src):
			t = dst + n[len(src):]
		elif n.endswith("lm_head.weight"):
			t = head
		else:
			continue
		if ref is not None and t not in ref:
			if t != head:
				odd.append(n)
			continue
		pairs[n] = t
	done = set(pairs.values())
	missing = [n for n in ref or () if (n.startswith(dst) or n == head) and n not in done and not n.endswith("inv_freq")]
	return pairs, odd, missing, src, dst, head


def stage(path, repo, tmp, on_progress=None, cancel=None, log=None):
	import torch
	from safetensors import safe_open

	notes = []
	if not repo:
		raise RuntimeError("this file has no config nor tokenizer: right-click it in MODELS → BASE MODEL… and give the Hugging Face repo it was made from (e.g. Qwen/Qwen3-4B)")
	cfg_path, err = fetch(repo, "config.json")
	if cfg_path is None:
		raise RuntimeError(f"cannot get config.json from the Hugging Face repo '{repo}' ({type(err).__name__}): wrong name, gated repo or offline")
	got = []
	for name in REF_FILES:
		p, _err = (cfg_path, None) if name == "config.json" else fetch(repo, name)
		if p:
			shutil.copyfile(p, os.path.join(tmp, name))
			got.append(name)
	if not any(t in got for t in TOKENIZER):
		raise RuntimeError(f"no tokenizer found in '{repo}'")
	h, _meta = header(path)
	names = list(h)
	if any(".absmax" in n or "quant_state" in n for n in names):
		raise RuntimeError("bitsandbytes 4-bit weights cannot be converted: use a BF16 or FP8 version of this file")
	bad = comfy_formats(path, names) - set(FP8_FORMATS)
	if bad:
		raise RuntimeError(f"{', '.join(sorted(str(b) for b in bad))} weights cannot be converted: use a BF16 or FP8 version of this file")
	ref = ref_tensors(repo, cfg_path)
	if ref is None:
		notes.append(f"the tensor list of {repo} is unavailable (offline?): names are mapped without checking")
	pairs, odd, missing, src, dst, head = mapping(names, ref)
	if not pairs:
		raise RuntimeError("no transformer weights found in this file")
	if odd:
		raise RuntimeError(f"this file does not match {repo}: {odd[0]} has no counterpart there (pick the right base model with BASE MODEL…)")
	cfg = json.load(open(os.path.join(tmp, "config.json"), encoding="utf-8"))
	cfg.pop("quantization_config", None)
	tc = cfg.get("text_config")
	untied = not (tc or cfg).get("tie_word_embeddings", cfg.get("tie_word_embeddings", True))
	if head in missing or (ref is None and untied and not any(n.endswith("lm_head.weight") for n in names)):
		missing = [m for m in missing if m != head]
		cfg["tie_word_embeddings"] = True
		if tc:
			tc["tie_word_embeddings"] = True
		if untied:
			notes.append("this file has no output layer (lm_head) and the original model has its own: the input embeddings stand in for it, so next-token predictions (chat, levers, judge) differ from the original model")
	if missing:
		raise RuntimeError(f"this file lacks {len(missing)} weights of {repo} (e.g. {missing[0]}): pick the right base model with BASE MODEL…")
	if ref is not None:
		for n, t in pairs.items():
			shape = list(h[n]["shape"])
			if list(ref[t]) != shape:
				raise RuntimeError(f"{n} is {shape} here but {list(ref[t])} in {repo}: pick the right base model with BASE MODEL…")
	json.dump(cfg, open(os.path.join(tmp, "config.json"), "w", encoding="utf-8"), indent=1)
	plain = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
	items = []
	n_deq = 0
	for n, t in pairs.items():
		dt = h[n]["dtype"]
		if dt in plain:
			items.append((t, dt, h[n]["shape"], n, None))
			continue
		if dt not in ("F8_E4M3", "F8_E5M2", "I8"):
			raise RuntimeError(f"{n} is stored as {dt}: this format cannot be converted")
		base = n[:-len(".weight")] if n.endswith(".weight") else n
		scale = next((base + s for s in SCALES if base + s in h), None)
		if dt == "I8" and scale is None:
			raise RuntimeError(f"{n} is int8 without a scale: this format cannot be converted")
		items.append((t, "BF16", h[n]["shape"], n, scale or ""))
		n_deq += 1
	if n_deq:
		notes.append(f"{n_deq} FP8/int8 weights dequantized to BF16 (scaled: the converted model holds the same values)")
	if src != dst:
		notes.append(f"weight names {src}… read as {dst}…")
	hdr, off = {}, 0
	for t, dt, shape, _n, _s in items:
		size = numel(shape) * BYTES[dt]
		hdr[t] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + size]}
		off += size
	hdr["__metadata__"] = {"format": "pt"}
	raw = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
	raw += b" " * (-len(raw) % 8)
	with safe_open(path, "pt") as f, open(os.path.join(tmp, "model.safetensors"), "wb") as out:
		out.write(struct.pack("<Q", len(raw)))
		out.write(raw)
		for i, (t, dt, shape, n, scale) in enumerate(items):
			if cancel is not None and cancel.is_set():
				raise LoadCancelled("conversion cancelled")
			w = f.get_tensor(n)
			if scale is not None:
				w = w.to(torch.float32)
				if scale:
					w = w * expand(f.get_tensor(scale).to(torch.float32), list(shape))
				w = w.to(torch.bfloat16)
			out.write(w.contiguous().reshape(-1).view(torch.uint8).numpy().data)
			if on_progress and (i % 8 == 0 or i == len(items) - 1):
				on_progress((i + 1) / len(items))
	if log:
		log(f"staged {len(items)} tensors with the config and tokenizer of {repo}")
	return notes


def python_exe():
	exe = sys.executable
	if os.path.basename(exe).lower() == "pythonw.exe":
		alt = os.path.join(os.path.dirname(exe), "python.exe")
		if os.path.isfile(alt):
			return alt
	return exe


def _pump(stream, q):
	while True:
		chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
		if not chunk:
			break
		q.put(chunk.decode("utf-8", errors="replace"))
	q.put(None)


def run(script, d, out, outtype, name, n_tensors, on_progress=None, cancel=None, log=None):
	part = out + ".part"
	cmd = [python_exe(), script, d, "--outfile", part, "--outtype", outtype, "--model-name", name]
	env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
	env.pop("NO_LOCAL_GGUF", None)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, cwd=os.path.dirname(script), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
	q = queue.Queue()
	threading.Thread(target=_pump, args=(p.stdout, q), daemon=True).start()
	tail, seen, buf = [], 0, ""
	try:
		while True:
			if cancel is not None and cancel.is_set():
				p.kill()
				p.wait()
				raise LoadCancelled("conversion cancelled")
			try:
				chunk = q.get(timeout=0.25)
			except queue.Empty:
				continue
			if chunk is None:
				break
			buf += chunk
			lines = re.split(r"[\r\n]", buf)
			buf = lines.pop()
			for line in lines:
				line = line.strip()
				if not line:
					continue
				m = WRITE_RE.search(line)
				if TENSOR_RE.match(line):
					seen += 1
					if on_progress:
						on_progress(0.1 * min(1.0, seen / max(n_tensors, 1)))
				elif m:
					if on_progress:
						on_progress(0.1 + 0.9 * int(m.group(1)) / 100)
				else:
					tail = (tail + [line])[-40:]
					if log and ("ERROR" in line or "Model architecture" in line):
						log(line.split(":", 2)[-1].strip(), "warn" if "ERROR" in line else "info")
		code = p.wait()
	except BaseException:
		if p.poll() is None:
			p.kill()
			p.wait()
		_drop(part)
		raise
	if code != 0 or not os.path.isfile(part):
		_drop(part)
		why = [l for l in tail if "Error" in l or "ERROR" in l] or tail[-3:]
		raise RuntimeError("conversion to GGUF failed: " + " | ".join(why[-3:]))
	os.replace(part, out)


def _drop(path):
	try:
		os.remove(path)
	except OSError:
		pass


def convert(src, out_dir, llama_dir, ref="", outtype="auto", on_progress=None, cancel=None, log=None):
	out = out_path(src, out_dir)
	if current(src, out, ref):
		return out, (record(out) or {}).get("notes") or []
	script = converter(llama_dir)
	if script is None:
		raise RuntimeError("convert_hf_to_gguf.py not found next to the llama.cpp folder: safetensors models need the llama.cpp sources (gguf-py and convert_hf_to_gguf.py, from github.com/ggml-org/llama.cpp)")
	desc = describe(src, llama_dir)
	if desc.get("skip"):
		raise RuntimeError(f"{os.path.basename(src)} cannot be converted: {desc.get('why')}")
	os.makedirs(out_dir, exist_ok=True)
	t0 = time.time()
	n_tensors = len(tensors_of(src))
	prog = on_progress or (lambda v: None)
	notes = []
	repo = ""
	if desc["kind"] == "dir":
		run(script, src, out, outtype, stem(src), n_tensors, prog, cancel, log)
	else:
		repo = ref or desc.get("ref") or ""
		with tempfile.TemporaryDirectory(prefix="stage-", dir=out_dir, ignore_cleanup_errors=True) as tmp:
			notes = stage(src, repo, tmp, lambda v: prog(0.4 * v), cancel, log)
			run(script, tmp, out, outtype, stem(src), n_tensors, lambda v: prog(0.4 + 0.6 * v), cancel, log)
	rec = {"source": src, "sig": signature(src), "ref": repo, "outtype": outtype, "converted": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1), "notes": notes, "converter": script}
	with open(out + ".json", "w", encoding="utf-8") as f:
		json.dump(rec, f, ensure_ascii=False, indent=1)
	return out, notes
