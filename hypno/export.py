import json
import os
import re
import shutil
from datetime import datetime

import numpy as np

import config
from engine.ggufio import gguf_module
from engine.runtime import LORA


def safe_name(name):
	name = re.sub(r"[^\w.\- ]+", "_", (name or "").strip())[:80].strip(" .")
	return name or datetime.now().strftime("hypno_%Y%m%d_%H%M%S")


def unc(path):
	p = os.path.abspath(path)
	drive, rest = os.path.splitdrive(p)
	if len(drive) == 2 and drive[1] == ":":
		return "\\\\localhost\\" + drive[0].upper() + "$" + rest
	return p


def strength(v):
	s = f"{float(v):g}"
	return s if "." in s or "e" in s else s + ".0"


def lmstudio(out_dir, cvec, adapters, lora, dose):
	over = []
	if cvec:
		over.append({"key": "--control-vector-scaled", "value": f"{unc(cvec)}:{strength(dose)}"})
	if adapters:
		over.append({"key": "--lora-scaled", "value": ",".join(f"{unc(os.path.join(out_dir, a['file']))}:{strength(a['scale'] * dose)}" for a in adapters)})
	if lora:
		over.append({"key": "--lora-scaled", "value": f"{unc(os.path.join(out_dir, lora['gguf']))}:1.0"})
	if not over:
		return None
	return {"enabled": True, "disabledParameters": [], "overrideParameters": over, "excludeAllConfig": False}


def write_cvec(path, vectors, arch, llama_dir=""):
	gguf = gguf_module(llama_dir)
	w = gguf.GGUFWriter(path, "controlvector")
	w.add_string("controlvector.model_hint", arch)
	w.add_uint32("controlvector.layer_count", len(vectors))
	for layer in sorted(vectors):
		w.add_tensor(f"direction.{int(layer)}", np.asarray(vectors[layer], dtype=np.float32))
	w.write_header_to_file()
	w.write_kv_data_to_file()
	w.write_tensors_to_file()
	w.close()


def write_lora(path, tensors, arch, alpha, llama_dir="", note=""):
	gguf = gguf_module(llama_dir)
	tmp = path + ".tmp"
	w = gguf.GGUFWriter(tmp, arch)
	w.add_type("adapter")
	w.add_string(gguf.Keys.Adapter.TYPE, "lora")
	w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(alpha))
	if note:
		w.add_string("pendulum.note", note)
	for name, arr in tensors.items():
		w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
	w.write_header_to_file()
	w.write_kv_data_to_file()
	w.write_tensors_to_file()
	w.close()
	os.replace(tmp, path)
	return path


BIAS_ARCHS = ("llama", "mistral3")


def bakeable(arch):
	return arch in BIAS_ARCHS


def bake(src, dst, vectors, llama_dir="", note="", on_progress=None):
	gguf = gguf_module(llama_dir)
	reader = gguf.GGUFReader(src)
	arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
	if not bakeable(arch):
		raise ValueError(f"'{arch}' models have no FFN output bias in llama.cpp: a control vector cannot be baked exactly into them")
	writer = gguf.GGUFWriter(dst, arch)
	for field in reader.fields.values():
		if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
			continue
		vt = field.types[0]
		writer.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
	if note:
		writer.add_string("pendulum.baked_levers", note)
	biases = {f"blk.{int(l)}.ffn_down.bias": np.asarray(v, dtype=np.float32) for l, v in vectors.items()}
	data = []
	for t in reader.tensors:
		if t.name in biases:
			arr = biases.pop(t.name) + np.asarray(t.data, dtype=np.float32).reshape(-1)
			writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, gguf.GGMLQuantizationType.F32)
			data.append(arr)
			continue
		writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
		data.append(t.data)
	for name, arr in sorted(biases.items()):
		writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, gguf.GGMLQuantizationType.F32)
		data.append(arr)
	writer.write_header_to_file()
	writer.write_kv_data_to_file()
	writer.write_ti_data_to_file()
	total = max(1, sum(int(d.nbytes) for d in data))
	done = 0
	for d in data:
		writer.write_tensor_data(d, tensor_endianess=reader.endianess)
		done += int(d.nbytes)
		if on_progress:
			on_progress(done / total)
	writer.close()
	return dst


APPROX = "a lever adds a fixed vector to the activations; a LoRA changes weights, so it can only approximate it: rank 1 per layer on the MLP output weight, fitted on sample prompts (closed form), fidelity measured on unseen prompts"


def export(session, name, dose=1.0, kind="cvec", on_progress=None, style="hf", prefix=None):
	baked = kind == "baked"
	name = safe_name(name)
	out_dir = os.path.join(config.EXPORTS_DIR, name)
	os.makedirs(out_dir, exist_ok=True)
	info = session.engine.info
	active = session.active_vectors()
	loras = dict(active.get(LORA) or {})
	vectors = {l: v for l, v in active.items() if l != LORA}
	if not vectors and not loras:
		raise ValueError("no active lever: move at least one slider before exporting")
	dose = float(dose)
	cvec = os.path.join(out_dir, f"{name}.cvec.gguf") if vectors and kind != "lora" else None
	if cvec:
		write_cvec(cvec, vectors, info["arch"], session.engine.lib.dir)
	adapters = []
	for i, (src, scale) in enumerate(sorted(loras.items()) if kind != "lora" else []):
		dst = os.path.join(out_dir, f"{name}.{i + 1}.lora.gguf")
		shutil.copyfile(src, dst)
		adapters.append({"file": os.path.basename(dst), "scale": float(scale)})
	lora = export_lora(session, out_dir, name, dose, style, prefix, on_progress) if kind == "lora" else None
	manifest = {
		"name": name,
		"created": datetime.now().isoformat(timespec="seconds"),
		"model": info["path"],
		"model_name": info["name"],
		"arch": info["arch"],
		"lens": session.lens.name if session.lens else None,
		"band": session.band,
		"strengths": {k: v for k, v in session.strengths.items() if v},
		"guard_scale": session.guard_scale,
		"blame": dict(getattr(session, "blame", {}) or {}),
		"dose": dose,
		"coherence_kl": session.coherence_kl,
		"control_vector": os.path.basename(cvec) if cvec else None,
		"lora_adapters": adapters,
		"lora": lora,
		"levers": {tid: {"layers": lv["layers"], "gain": lv["gain"], "kind": lv.get("kind", "persona"), "carrier": lv.get("carrier", "vector"), "reads_as": lv.get("reads_as", [])} for tid, lv in session.levers.items() if session.strengths.get(tid)},
		"profile_before": session.profile_values("baseline"),
		"profile_after": session.profile_values("current"),
		"system_prompt": session.system_prompt,
	}
	model = info["path"]
	if baked:
		if not vectors:
			raise ValueError("nothing to bake: the active levers are LoRA adapters only")
		model = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(info['path']))[0]}.{name}.baked.gguf")
		bake(info["path"], model, {l: np.asarray(v, dtype=np.float32) * dose for l, v in vectors.items()}, session.engine.lib.dir, json.dumps(manifest["strengths"]), on_progress)
		manifest["baked_model"] = model
	with open(os.path.join(out_dir, "pendulum.json"), "w", encoding="utf-8") as f:
		json.dump(manifest, f, ensure_ascii=False, indent=1)
	flags = []
	if cvec and not baked:
		flags.append(f'--control-vector-scaled "{os.path.basename(cvec)}:{dose:g}"')
	if adapters:
		flags.append("--lora-scaled " + '"' + ",".join(f"{a['file']}:{a['scale'] * dose:g}" for a in adapters) + '"')
	if lora:
		flags.append(f'--lora-scaled "{lora["gguf"]}:1"')
	notes = ([f"levers baked into the model (dose {dose:g})"] if baked else []) + ([f"levers converted to a LoRA (dose {dose:g} included, approximation: see pendulum.json)"] if lora else [])
	launcher(out_dir, session.engine.lib.dir, model, flags, notes, info["n_ctx"])
	lm = lmstudio(out_dir, cvec if not baked else None, adapters, lora, dose)
	text = lmstudio_file(out_dir, lm)
	return {"dir": out_dir, "cvec": cvec, "manifest": manifest, "baked": model if baked else None, "adapters": adapters, "lora": lora, "lmstudio": lm, "lmstudio_text": text}


def launcher(out_dir, llama_dir, model, flags, notes, n_ctx):
	server = os.path.join(llama_dir, "llama-server.exe") if llama_dir else "llama-server.exe"
	lines = ["@echo off", 'cd /d "%~dp0"'] + [f"rem {n}" for n in notes]
	if flags:
		lines.append("rem after each file, the number past the colon is its strength: edit it to make the levers weaker or stronger")
	lines.append(f'"{server}" -m "{model}" ' + " ".join(flags + ["-ngl 999", f"-c {n_ctx}", "--port 8080"]))
	with open(os.path.join(out_dir, "run_llama_server.bat"), "w", encoding="utf-8", newline="\r\n") as f:
		f.write("\n".join(lines) + "\n")


def lmstudio_file(out_dir, lm):
	if not lm:
		return ""
	text = "```json\n" + json.dumps(lm, ensure_ascii=False, indent=2) + "\n```\n"
	with open(os.path.join(out_dir, "lmstudio_override.txt"), "w", encoding="utf-8") as f:
		f.write(text)
	return text


def export_saved_lora(src, name, model="", model_name="", arch="", goal="", scale=1.0, llama_dir="", n_ctx=None):
	name = safe_name(name)
	out_dir = os.path.join(config.EXPORTS_DIR, name)
	os.makedirs(out_dir, exist_ok=True)
	dst = os.path.join(out_dir, f"{name}.lora.gguf")
	shutil.copyfile(src, dst)
	adapters = [{"file": os.path.basename(dst), "scale": float(scale)}]
	manifest = {"name": name, "created": datetime.now().isoformat(timespec="seconds"), "model": model or None, "model_name": model_name or None, "arch": arch or None, "goal": goal or None, "source": src, "lora_adapters": adapters, "note": "LoRA trained in GOALS (TRAIN), copied as it is (exported without loading the model)"}
	with open(os.path.join(out_dir, "pendulum.json"), "w", encoding="utf-8") as f:
		json.dump(manifest, f, ensure_ascii=False, indent=1)
	notes = [f"LoRA of the goal '{goal}' as trained in GOALS (TRAIN)"] + ([] if model else ["its model was not found in your model folders: replace MODEL.gguf with its path"])
	launcher(out_dir, llama_dir, model or "MODEL.gguf", [f'--lora-scaled "{adapters[0]["file"]}:{float(scale):g}"'], notes, n_ctx or config.DEFAULTS["n_ctx"])
	lm = lmstudio(out_dir, None, adapters, None, 1.0)
	return {"dir": out_dir, "manifest": manifest, "adapters": adapters, "lmstudio": lm, "lmstudio_text": lmstudio_file(out_dir, lm)}


def export_lora(session, out_dir, name, dose, style="hf", prefix=None, on_progress=None):
	from hypno import lorafit

	info = session.engine.info
	style = style if style in lorafit.STYLES else "hf"
	prefix = lorafit.STYLES[style][0] if prefix is None else prefix
	modules, rank, steer, report, held = lorafit.build(session, dose, on_progress)
	tensors = {}
	for (l, g), (a, b) in sorted(modules.items()):
		tensors[f"blk.{l}.{g}.weight.lora_a"] = a
		tensors[f"blk.{l}.{g}.weight.lora_b"] = b
	strengths = {k: v for k, v in session.strengths.items() if v}
	gguf_path = os.path.join(out_dir, f"{name}.lora.gguf")
	write_lora(gguf_path, tensors, info["arch"], float(rank), session.engine.lib.dir, note=json.dumps({"levers": strengths, "dose": dose}))
	if held:
		if on_progress:
			on_progress(0.85)
		report["fidelity"] = lorafit.measure(session, gguf_path, steer, held)
	meta = {"pendulum": APPROX, "model": info["name"], "levers": strengths, "dose": dose, "rank": rank, "fidelity": report.get("fidelity") or {}}
	if style == "peft":
		pdir = os.path.join(out_dir, f"{name}.peft")
		os.makedirs(pdir, exist_ok=True)
		st = lorafit.write_safetensors(os.path.join(pdir, "adapter_model.safetensors"), modules, style, prefix, rank, meta)
		with open(os.path.join(pdir, "adapter_config.json"), "w", encoding="utf-8") as f:
			json.dump(lorafit.peft_config(modules, rank, info["name"]), f, indent=1)
	else:
		st = lorafit.write_safetensors(os.path.join(out_dir, f"{name}.safetensors"), modules, style, prefix, rank, meta)
	if on_progress:
		on_progress(1.0)
	return {"gguf": os.path.basename(gguf_path), "safetensors": os.path.relpath(st, out_dir).replace(os.sep, "/"), "style": style, "prefix": prefix, "example_key": lorafit.example_key(style, prefix), "rank": rank, "layers": sorted({l for l, _g in modules}), "approximation": APPROX, "report": report}
