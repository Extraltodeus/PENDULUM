import gc
import hashlib
import json
import os
import re
import time

import numpy as np
import torch

import config
from engine.ggufio import summary as gguf_summary
from engine.gpus import plan_load, reserve_mibs
from engine.llama import find_llama_dir, same_dir
from engine.llamadist import find_bin, lib_file
from engine.runtime import LORA, Engine
from engine import stconvert, torchload
from hypno import catalog
from hypno import history as HI
from hypno import presets as PS
from hypno.export import write_lora
from lens.lens import Lens, build_index, resolve_path
from lens.match import best, lens_vram, score as lens_score
from lens.readout import Readout, selfcheck
from lens.vocab import masks
from lens.workspace import analyze, detect_band
from engine.mtmd import Vision, pick_mmproj
from persona import imagewords as IW
from persona import levers as LV
from persona.coherence import CALIB_TARGET, EXTRA_DEFAULTS as COH_EXTRA, GUARD_MODES, LEGACY_MODES, JUDGE_MIN_AUC, Coherence, Judge, Stopped, check, judge_key, question_list, threshold
from persona import goals as GO
from persona import optimize as OPT
from persona import profile as PR
from persona import target as TG
from persona.traits import TraitScorer, build_lexicons, builtin_definitions, edited_fields, hidden_definitions, load_definitions, load_overrides, make_trait, load_custom, save_custom, save_overrides, signature

def _key(path):
	st = os.stat(path)
	stem = re.sub(r"[^\w.\-]+", "_", os.path.splitext(os.path.basename(path))[0])[:80]
	return f"{stem}_{st.st_size}"


def _jsonable(x):
	if isinstance(x, dict):
		return {str(k): _jsonable(v) for k, v in x.items() if not isinstance(v, torch.Tensor)}
	if isinstance(x, (list, tuple)):
		return [_jsonable(v) for v in x]
	if isinstance(x, np.ndarray):
		return x.tolist()
	if isinstance(x, (np.floating, np.integer)):
		return x.item()
	return x


PROFILE_MODES = ("write", "read", "read_write")
READ_GAIN_DEFAULT = 1.6
MIX_DRIFT = 0.5


class Job:
	def __init__(self, emit, total):
		self.emit = emit
		self.total = max(float(total), 1.0)
		self.done = 0.0
		self.w = 0.0

	def step(self, w):
		self.done += self.w
		self.w = float(w)
		self.total = max(self.total, self.done + self.w)
		self.emit("job_progress", {"done": self.done, "total": self.total, "w": self.w})

	def finish(self):
		self.done, self.w = self.total, 0.0
		self.emit("job_progress", {"done": self.total, "total": self.total, "w": 0.0})


class Ticks:
	def __init__(self, emit, coh, measures, n=None):
		self.emit = emit
		self.coh = coh
		self.done = 0
		self.value = 0.0
		self.total = max(1, coh.ticks(measures, n))
		coh.on_tick = self.tick
		emit("progress", {"task": "guard", "value": 0.0})

	def tick(self):
		self.done += 1
		if self.done >= self.total:
			self.total = self.done + self.coh.ticks(1)
		self.value = max(self.value, self.done / self.total)
		self.emit("progress", {"task": "guard", "value": self.value})


class Session:
	def __init__(self, emit=None):
		config.setup_env()
		self.settings = config.load_settings()
		self.emit = emit or (lambda kind, payload: None)
		self.engine = None
		self.info = None
		self.plan = None
		self.lens = None
		self.lens_index = None
		self.readout = None
		self.strings = None
		self.word = None
		self.english = None
		self.defs = load_definitions()
		self.lexicons = None
		self.scorer = None
		self.axis_best = None
		self.workspace = None
		self.band = None
		self.reference = None
		self.summaries = {}
		self.levers = {}
		self.strengths = {}
		self.guard_scale = 1.0
		self.coherence_kl = 0.0
		self.blame = {}
		self.guard_free = []
		self.current_vectors = {}
		self.image_cache = {}
		self.probe = None
		self.coh = None
		self.judge = None
		self.judge_engine = None
		self.judge_state = {"state": "off"}
		self.goals = GO.load()
		self.goal_results = {}
		self.torch_net = None
		self.parked = False
		self.coherence_info = None
		self.guard_basis = None
		self.judged = {}
		self.history = []
		self.system_prompt = ""
		self.sampling = dict(self.settings["sampling"])
		self.stop_event = None
		self.soft = False
		self.trace_ctx = ""
		self.targets = {}
		self.jcols = {}
		self.read_gains = {}
		self.read_mix = None
		self.read_missing = set()
		self.lever_source = ""

	def log(self, text, level="info"):
		self.emit("log", {"text": text, "level": level, "t": time.time()})

	def llama_dir(self):
		d = find_llama_dir(self.settings.get("llama_dir", ""))
		if d and not same_dir(d, self.settings.get("llama_dir", "")):
			self.set_setting("llama_dir", d.replace("\\", "/"))
		return d

	def set_setting(self, key, value):
		self.settings = {**self.settings, key: value}
		config.save_settings(self.settings)

	def ensure_engine(self):
		if self.engine is None:
			d = self.llama_dir()
			if not d:
				raise RuntimeError("llama.cpp DLLs not found: set the llama.cpp folder in the settings")
			self.engine = Engine(d)
			self.log(f"llama.cpp ready ({d})")
		return self.engine

	def gpus(self):
		return self.engine.lib.gpus() if self.engine else []

	def model_dirs(self):
		return list(self.settings.get("model_dirs") or []) + catalog.default_dirs()

	def catalog(self, refresh_lenses=False):
		self.ensure_engine()
		models = catalog.scan(self.model_dirs(), self.engine.lib.dir, self.settings.get("st_refs") or {})
		choices = self.settings.get("mmproj") or {}
		for m in models:
			if m.get("path") and not m.get("convert"):
				m["mmproj"] = pick_mmproj(m["path"], choices)
				m["mmproj_choice"] = choices.get(m["path"], "")
		self.lens_index = self.index_lenses(online=not os.environ.get("HF_HUB_OFFLINE"), refresh=refresh_lenses)
		return {"models": models, "lenses": self.lens_index, "llama_dir": self.engine.lib.dir, "margin": int(self.settings.get("vram_reserve_mb", 900))}

	def update_settings(self, values, light=False):
		before = (self.model_dirs(), list(self.settings.get("lens_repos") or []), self.coh_settings())
		self.settings = {**self.settings, **values}
		self.sampling = dict(self.settings["sampling"])
		config.save_settings(self.settings)
		want = self.settings.get("llama_dir") or ""
		if want and not find_bin(want, 1):
			self.log(f"no {lib_file('llama')} in {want}: the llama.cpp folder found automatically is used instead", "warn")
		if self.engine is not None and not same_dir(find_llama_dir(want), self.engine.lib.dir):
			self.log("llama.cpp folder changed: restart Pendulum to use it", "warn")
		self.log("settings saved")
		loaded = self.engine is not None and self.engine.loaded
		stale = self.judge_engine is not None and self.judge_engine.loaded and (self.judge_engine.info or {}).get("path") != self.coh_settings()["judge"]
		res = {"rescan": before[0] != self.model_dirs(), "relens": before[1] != list(self.settings.get("lens_repos") or []), "coherence": before[2] != self.coh_settings(), "stale": bool(not loaded and stale)}
		if not light:
			self.apply_settings()
		return res

	def apply_settings(self):
		if self.engine is not None and self.engine.loaded:
			self.ensure_judge()
			if any(self.strengths.values()):
				self.apply_levers()
		elif self.judge_engine is not None and self.judge_engine.loaded and (self.judge_engine.info or {}).get("path") != self.coh_settings()["judge"]:
			self.judge_engine.unload()
			self.judge = None
			if self.coh is not None:
				self.coh.judge = None
			self.log("judge unloaded" + (": the new one loads at the first coherence check" if self.coh_settings()["judge"] else ""))
			self.set_judge_state(state="off")

	def index_lenses(self, online=True, refresh=False):
		return build_index(online=online, refresh=refresh, extra=self.settings.get("extra_lenses") or [], repos=self.settings.get("lens_repos") or None)

	def pick_lens(self, s, lens_entry):
		if lens_entry == "auto":
			if self.lens_index is None:
				self.lens_index = self.index_lenses(online=False)
			return best(self.lens_index, s)
		return lens_entry if isinstance(lens_entry, dict) else None

	def plan_for(self, path, n_ctx=None, lens_entry="auto"):
		self.ensure_engine()
		s = gguf_summary(path, self.llama_dir())
		entry = self.pick_lens(s, lens_entry)
		n_ctx = int(n_ctx or self.settings.get("n_ctx", 8192))
		lb = lens_vram(s, entry)
		plan = plan_load(self.engine.lib.dir, path, self.engine.lib.gpus(), n_ctx, s["n_layer"], lens_bytes=lb, base_margin_mib=int(self.settings.get("vram_reserve_mb", 900)), options=self.load_options(path))
		return {"summary": s, "lens": entry, "plan": plan, "lens_bytes": lb}

	def load_options(self, path):
		per = (self.settings.get("load_models") or {}).get(os.path.basename(path)) or {}
		return {**(self.settings.get("load") or {}), **per}

	def set_load_options(self, path, options, per_model=True):
		allm = dict(self.settings.get("load_models") or {})
		if per_model:
			allm[os.path.basename(path)] = dict(options)
			self.settings = {**self.settings, "load_models": allm}
		else:
			allm.pop(os.path.basename(path), None)
			self.settings = {**self.settings, "load": dict(options), "load_models": allm}
		config.save_settings(self.settings)

	def unload(self):
		self.free_torch()
		self.parked = False
		if self.readout is not None:
			self.readout.free()
		self.readout = None
		self.lens = None
		self.probe = None
		self.scorer = None
		self.lexicons = None
		if self.engine is not None and self.engine.loaded:
			self.engine.unload()
		self.info = None
		self.workspace = self.band = self.reference = None
		self.summaries = {}
		self.levers = {}
		self.strengths = {}
		self.current_vectors = {}
		self.guard_scale, self.coherence_kl = 1.0, 0.0
		self.blame = {}
		self.guard_free = []
		self.coh = None
		self.coherence_info = None
		self.guard_basis = None
		self.judged = {}
		self.history = []
		self.targets = {}
		self.jcols = {}
		self.read_gains = {}
		self.read_mix = None
		self.read_missing = set()
		torch.cuda.empty_cache()
		self.emit("unloaded", self.snapshot())

	def set_st_ref(self, path, repo):
		refs = dict(self.settings.get("st_refs") or {})
		if repo:
			refs[path] = repo
		else:
			refs.pop(path, None)
		self.set_setting("st_refs", refs)
		self.log(f"base model of {os.path.basename(path)}: {repo or 'automatic'}")
		return self.catalog()

	def convert_source(self, path, cancel=None):
		ref = (self.settings.get("st_refs") or {}).get(path, "")
		outtype = self.settings.get("convert_outtype") or "auto"
		out = stconvert.out_path(path, config.CONVERTED_DIR)
		if stconvert.current(path, out, ref):
			return out
		self.log(f"converting {stconvert.stem(path)} to GGUF ({'16-bit' if outtype == 'auto' else outtype}, once: kept in {config.CONVERTED_DIR.replace(os.sep, '/')})")
		self.emit("progress", {"task": "convert", "value": 0.0})
		t0 = time.time()
		out, notes = stconvert.convert(path, config.CONVERTED_DIR, self.llama_dir(), ref=ref, outtype=outtype, on_progress=lambda v: self.emit("progress", {"task": "convert", "value": v}), cancel=cancel, log=self.log)
		for note in notes:
			self.log(note, "warn" if "differ" in note else "info")
		self.log(f"converted in {time.time() - t0:.0f} s → {os.path.basename(out)} ({os.path.getsize(out) / 2**30:.2f} GB)", "ok")
		return out

	def load(self, path, lens_entry="auto", n_ctx=None, on_progress=None, cancel=None, keep=True):
		self.unload()
		if stconvert.is_source(path):
			path = self.convert_source(path, cancel)
		self.log(f"planning {os.path.basename(path)}")
		self.emit("progress", {"task": "plan", "value": 0.0})
		p = self.plan_for(path, n_ctx, lens_entry)
		plan, entry = p["plan"], p["lens"]
		for note in plan.get("notes") or []:
			self.log(note, "warn")
		if plan.get("spill"):
			self.log(f"model does not fit in VRAM: {len(plan.get('cpu_overrides') or [])} tensor groups / layers left on CPU — expect a large slowdown", "warn")
		self.emit("plan", _jsonable({"plan": plan, "summary": p["summary"], "lens": entry, "lens_bytes": p["lens_bytes"]}))
		self.log(f"loading weights (ngl={plan.get('n_gpu_layers')}, split={plan.get('tensor_split') or 'auto'}, ctx={plan.get('n_ctx')})")
		self.info = self.engine.load(path, plan, on_progress=on_progress, cancel=cancel)
		self.plan = plan
		self.log(f"{self.info['name']} · {self.info['arch']} · {self.info['n_layer']} layers · d={self.info['n_embd']} · {self.info['quant']} · {self.info['load_seconds']} s")
		self.strings = self.engine.vocab_strings()
		self.word, self.english = masks(self.strings)
		self.lens = None
		if entry:
			self.log(f"lens {entry['base']} ({entry.get('n_prompts') or '?'} prompts, {entry['source']})")
			if not lens_score(entry, p["summary"]):
				self.log(f"the lens '{entry['base']}' only has the same shape as this model: it was fitted on another model, so readouts, the band and the profile may be wrong (train a lens for this model or use the logit lens)", "warn")
			try:
				path_lens = entry.get("path") if entry.get("path") and os.path.exists(entry["path"]) else None
				if path_lens is None:
					self.log(f"downloading the lens ({entry.get('size', 0) / 2**30:.2f} GB) from {entry['repo']}…")
					self.emit("progress", {"task": "lens download", "value": 0.0})
				if path_lens is None:
					path_lens = resolve_path(entry, on_progress=lambda v: self.emit("progress", {"task": "lens download", "value": v}))
					self.lens_index = self.index_lenses(online=False)
					self.emit("lens_index", {"lenses": self.lens_index})
				self.lens = Lens.load(path_lens, entry["base"])
			except Exception as exc:
				self.log(f"lens unavailable ({exc}): using the logit lens", "warn")
			if self.lens is not None and self.lens.d_model != self.info["n_embd"]:
				self.log(f"lens d_model {self.lens.d_model} != model {self.info['n_embd']}: using the logit lens", "warn")
				self.lens = None
			if self.lens is not None and self.lens.layers and self.lens.layers[-1] >= self.engine.n_layer:
				self.log(f"lens has layers up to L{self.lens.layers[-1]} but the model has {self.engine.n_layer}: using the logit lens", "warn")
				self.lens = None
		else:
			self.log("no Jacobian lens selected for this model: logit lens (J = I)", "warn")
		if self.lens is None:
			self.lens = Lens.logit_lens(self.info["n_embd"])
		device = plan.get("lens_device") or "cpu"
		self.readout = Readout(self.engine.gguf, self.lens, device, self.engine.n_layer, vocab_mask=self.word)
		chk = selfcheck(self.engine, self.readout)
		self.log(f"unembed self-check cos={chk['cos']:.5f} top1={'ok' if chk['top1'] else 'MISMATCH'}{' (γ+1)' if chk['gamma_offset'] else ''}", "info" if chk["cos"] > 0.99 else "warn")
		self.lexicons = build_lexicons(self.engine, self.readout, self.defs, self.english)
		self.scorer = TraitScorer(self.readout, self.defs, self.lexicons)
		self._axis_best()
		self.warm_up()
		self.set_setting("last_model", path)
		self.restore(keep)
		self.emit("loaded", self.snapshot())
		return self.info

	def _axis_best(self):
		axes = self.scorer.axis_scores(self.lexicons, self.english)
		stack = torch.stack([axes[t] for t in self.scorer.ids])
		best = stack.abs().argmax(0)
		val = stack.gather(0, best[None])[0]
		self.axis_best = (best.to(torch.int16).cpu().numpy(), val.float().cpu().numpy())

	def trait_defs(self):
		if self.scorer is None:
			return [d for d in self.defs if d.get("seeds_plus") and d.get("seeds_minus")]
		return [d for d in self.defs if d["id"] in self.scorer.ids]

	def trait_rows(self):
		lex = self.lexicons or {}
		return [{"id": d["id"], "name": d["name"], "plus": d["plus"], "minus": d["minus"], "custom": d.get("custom", False), "edited": d.get("edited", False), "seeds_plus": d.get("seeds_plus", []), "seeds_minus": d.get("seeds_minus", []), "desc_plus": d.get("desc_plus", ""), "desc_minus": d.get("desc_minus", ""), "images_plus": IW.image_paths(d, "plus"), "images_minus": IW.image_paths(d, "minus"), "poles_plus": IW.pole_paths(d, "plus"), "poles_minus": IW.pole_paths(d, "minus"), "lexicon": {p: lex[d["id"]][p]["strs"][:24] for p in ("plus", "minus")} if d["id"] in lex else {}} for d in self.trait_defs()]

	def extra_rows(self):
		measured = {d["id"] for d in self.trait_defs()}
		return [{"id": d["id"], "name": d["name"], "plus": d["plus"], "minus": d["minus"], "custom": True, "lever_only": True, "from_goal": d.get("from_goal", ""), "no_lexicon": bool(d.get("seeds_plus") and d.get("seeds_minus")), "seeds_plus": d.get("seeds_plus", []), "seeds_minus": d.get("seeds_minus", []), "desc_plus": d.get("desc_plus", ""), "desc_minus": d.get("desc_minus", ""), "images_plus": IW.image_paths(d, "plus"), "images_minus": IW.image_paths(d, "minus"), "poles_plus": IW.pole_paths(d, "plus"), "poles_minus": IW.pole_paths(d, "minus"), "lexicon": {}} for d in self.defs if d.get("custom") and d["id"] not in measured]

	def snapshot(self):
		if not self.engine or not self.engine.loaded:
			return _jsonable({"loaded": False, "traits": self.trait_rows(), "hidden": [{"id": d["id"], "name": d["name"], "plus": d["plus"], "minus": d["minus"]} for d in hidden_definitions()], "goals": self.goal_rows(), "extra_levers": self.extra_rows(), "inactive": sorted(self.inactive()), "goal_defs": self.goals, "judge": self.judge_state})
		return _jsonable({
			"loaded": True,
			"info": self.info,
			"lens": {"name": self.lens.name, "identity": self.lens.identity, "n_prompts": self.lens.n_prompts, "device": str(self.readout.device)},
			"hidden": [{"id": d["id"], "name": d["name"], "plus": d["plus"], "minus": d["minus"]} for d in hidden_definitions()],
			"traits": self.trait_rows(),
			"workspace": self.workspace,
			"band": self.band,
			"summaries": self.summaries,
			"levers": {tid: {"layers": lv["layers"], "gain": lv["gain"], "gen_mult": lv.get("gen_mult"), "reads_as": lv.get("reads_as", []), "reads_as_minus": lv.get("reads_as_minus", []), "kl": lv.get("kl"), "kind": lv.get("kind", "persona"), "closed": lv.get("closed"), "carrier": lv.get("carrier"), "rank": (lv.get("lora") or {}).get("rank")} for tid, lv in self.levers.items()},
			"strengths": self.strengths,
			"targets": self.targets,
			"guard": self.guard_info(),
			"system_prompt": self.system_prompt,
			"history": self.history,
			"ref_odds": self.reference["odds"] if self.reference else None,
			"goals": self.goal_rows(),
			"extra_levers": self.extra_rows(),
			"inactive": sorted(self.inactive()),
			"goal_defs": self.goals,
			"judge": self.judge_state,
		})

	def state_dir(self):
		return os.path.join(config.PROFILES_DIR, _key(self.info["path"]))

	def persist(self):
		if not self.info:
			return
		d = self.state_dir()
		os.makedirs(d, exist_ok=True)
		state = {"version": LV.VERSION, "history": self.history[-40:], "lens": self.lens.name, "workspace": self.workspace, "band": self.band, "reference": {k: v for k, v in (self.reference or {}).items() if k != "pmean"}, "summaries": self.summaries, "strengths": self.strengths, "system_prompt": self.system_prompt, "targets": self.targets, "jcols": self.jcols, "read_gains": self.read_gains, "read_mix": self.read_mix, "judged": {t: [float(v[0]), float(v[1])] for t, v in self.judged.items()}, "levers": {tid: {k: v for k, v in lv.items() if k not in ("dirs",)} for tid, lv in self.levers.items()}}
		tmp = os.path.join(d, "state.json.tmp")
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump(_jsonable(state), f, ensure_ascii=False, default=lambda o: None)
		os.replace(tmp, os.path.join(d, "state.json"))
		arrays = {f"{tid}__{l}": lv["dirs"][l] for tid, lv in self.levers.items() for l in lv["layers"] if l in (lv.get("dirs") or {})}
		if arrays:
			np.savez_compressed(os.path.join(d, "levers.tmp.npz"), **arrays)
			os.replace(os.path.join(d, "levers.tmp.npz"), os.path.join(d, "levers.npz"))
		if self.reference is not None and isinstance(self.reference.get("pmean"), torch.Tensor):
			torch.save(self.reference["pmean"].half().cpu(), os.path.join(d, "ref_pmean.pt"))

	def restore(self, keep=True):
		path = os.path.join(self.state_dir(), "state.json")
		if not os.path.exists(path):
			return False
		try:
			with open(path, encoding="utf-8") as f:
				state = json.load(f)
			if state.get("lens") != self.lens.name:
				return False
			ws = state.get("workspace")
			if ws and len(ws.get("layers") or []) != self.engine.n_layer:
				self.log("saved analysis was made with a different layer count: re-mapping", "warn")
				return False
			if state.get("version") != LV.VERSION:
				state["levers"], state["strengths"] = {}, {}
			self.workspace = state.get("workspace")
			self.band = detect_band(self.workspace) if self.workspace else state.get("band")
			self.reference = state.get("reference") or None
			pm = os.path.join(self.state_dir(), "ref_pmean.pt")
			if self.reference is not None and os.path.exists(pm):
				self.reference["pmean"] = torch.load(pm, weights_only=True).float().to(self.readout.device)
			self.summaries = state.get("summaries") or {}
			ids = list(self.scorer.ids)
			if any([t.get("id") for t in s.get("traits") or []] != ids for s in self.summaries.values()) or (self.reference is not None and len(self.reference.get("odds") or []) != len(ids)):
				self.keep_answers()
				self.log("the saved profile was made with another set of traits: it will be measured again", "warn")
				self.reference, self.summaries = None, {}
			self.system_prompt = state.get("system_prompt", "") if keep else ""
			self.history = state.get("history") or []
			npz = os.path.join(self.state_dir(), "levers.npz")
			arrays = {}
			if os.path.exists(npz):
				with np.load(npz) as z:
					arrays = {k: z[k] for k in z.files}
			self.levers = {}
			known = {d["id"] for d in self.defs} | {GO.lever_id(g) for g in self.goals}
			for tid, lv in (state.get("levers") or {}).items():
				if tid not in known:
					continue
				sig = signature(self.trait_by_id(tid) or {})
				if lv.get("sig", sig) != sig:
					continue
				dirs = {int(l): arrays[f"{tid}__{l}"] for l in lv["layers"] if f"{tid}__{l}" in arrays}
				lora = lv.get("lora")
				if lora and not os.path.exists(os.path.join(self.state_dir(), lora["file"])):
					continue
				if len(dirs) == len(lv["layers"]) or (lora and not dirs and lv.get("carrier") == "lora"):
					lv["norm"] = {int(k): v for k, v in lv["norm"].items()}
					lv["sep"] = {int(k): v for k, v in lv.get("sep", {}).items()}
					lv["dirs"] = dirs
					self.levers[tid] = lv
			self.strengths = {k: float(v) for k, v in (state.get("strengths") or {}).items() if GO.base_id(k) in self.levers} if keep else {}
			self.targets = {k: float(v) for k, v in (state.get("targets") or {}).items() if k in ids} if keep else {}
			self.jcols = {k: c for k, c in (state.get("jcols") or {}).items() if k in self.levers and c.get("ids") == ids}
			self.read_gains = {k: g for k, g in (state.get("read_gains") or {}).items() if k in self.levers and isinstance(g, dict)}
			mix = state.get("read_mix")
			self.read_mix = mix if isinstance(mix, dict) and all(k in self.levers for k in mix.get("ids") or []) else None
			self.judged = {k: (float(v[0]), float(v[1])) for k, v in (state.get("judged") or {}).items() if k in self.levers and isinstance(v, list) and len(v) == 2} if keep else {}
			self.log(f"restored saved analysis for this model ({len(self.levers)} calibrated levers)" + ("" if keep else " · its last lever settings are left aside: the current ones are carried over"))
			if any(self.strengths.values()):
				self.emit("loaded", self.snapshot())
				self.apply_levers()
			return True
		except Exception as exc:
			self.log(f"saved state ignored: {exc}", "warn")
			return False

	def analyze_workspace(self, stop=None):
		self.log("mapping the workspace band…")
		t = time.time()
		ws = analyze(self.engine, self.readout, on_progress=lambda p: self.emit("progress", {"task": "workspace", "value": p}), stop=stop)
		check(stop)
		self.workspace = ws
		self.band = ws["band"]
		self.log(f"workspace band L{self.band[0]}–L{self.band[1]} of {self.engine.n_layer} ({time.time() - t:.1f} s)")
		self.emit("workspace", _jsonable({"workspace": ws, "band": self.band}))
		self.persist()
		return ws

	def ensure_band(self, stop=None):
		if self.band is None:
			self.analyze_workspace(stop)
		return self.band

	def ensure_reference(self, stop=None):
		if self.reference is None or "pmean" not in self.reference:
			self.ensure_band(stop)
			ref = PR.reference(self.engine, self.readout, self.scorer, self.band, self.word, stop=stop)
			check(stop)
			self.reference = ref
		return self.reference

	def profile(self, tag="baseline", quick=False, stop=None):
		if tag in ("baseline", "baseline_quick") and any(self.strengths.values()):
			keep = self.strengths
			self.strengths = {}
			try:
				return self.profile(tag, quick, stop)
			finally:
				self.strengths = keep
				self.resteer()
		self.ensure_reference(stop)
		if quick and tag != "baseline_quick" and "baseline_quick" not in self.summaries:
			keep = self.strengths
			self.strengths = {}
			try:
				self.log("measuring a quick baseline first, so the steered model is compared on the same prompts…")
				self.profile("baseline_quick", True, stop)
			finally:
				self.strengths = keep
				self.resteer()
		prompts = [PR.BATTERY[i] for i in PR.QUICK] if quick else PR.BATTERY
		vecs = self.active_vectors()
		base = None if tag in ("baseline", "baseline_quick") else self.read_base(prompts, quick)
		again = self.saved_answers(tag, prompts) if tag in ("baseline", "baseline_quick") else None
		t = time.time()
		if again is not None:
			return self.reread_profile(tag, prompts, again, quick, stop, t)
		try:
			factor = self.read_factor(self.strengths, stop, wait=False) if base else 1.0
			waiting = sorted(k for k in self.read_missing if self.strengths.get(k))
			if base and waiting:
				self.log("READ + WRITE: no reading scale yet for " + ", ".join(f"'{(self.trait_by_id(k) or {}).get('name', k)}'" for k in waiting) + f": the average of this model's levers is used meanwhile (×{self.read_fill():.2f}), its own is measured when the TO DO list is idle")
				self.emit("read_missing", {"ids": waiting})
			self.log(f"profiling personality ({len(prompts)} prompts{', steered' if vecs else ''}" + (", reading the baseline answers" if base else "") + (f", reading scaled ×{factor:.2f}" if base and abs(factor - 1.0) > 0.005 else "") + ")…")
			prof = self.run_profile(prompts, vecs, base, lambda k, p: self.emit("profile_" + k, p), stop, factor)
		finally:
			self.resteer()
		check(stop)
		return self.store_profile(tag, prof, quick, self.profile_mode() if base else "write", t)

	def store_profile(self, tag, prof, quick, mode, t):
		summ = PR.summarize(prof, self.reference, self.band, self.scorer.ids, self.lexicons, self.word, self.strings)
		summ["responses"] = prof["responses"]
		summ["quick"] = quick
		summ["system"] = self.system_prompt or ""
		summ["mode"] = mode
		best_t, best_v = self.axis_best
		for c in summ["concepts"]:
			c["trait"] = int(best_t[c["id"]])
			c["tv"] = float(best_v[c["id"]])
		self.translate(summ["concepts"])
		summ["strengths"] = dict(self.strengths)
		self.summaries[tag] = _jsonable(summ)
		self.log(f"profile '{tag}' done ({time.time() - t:.1f} s)")
		self.emit("profile", {"tag": tag, "summary": self.summaries[tag], "ref_odds": self.ref_odds()})
		self.persist()
		return summ

	def keep_answers(self):
		self.reread = {tag: {"system": s.get("system", "") or "", "responses": [r for r in s.get("responses") or [] if not r.get("read")]} for tag, s in (self.summaries or {}).items() if tag in ("baseline", "baseline_quick") and s.get("responses")}

	def saved_answers(self, tag, prompts):
		kept = (getattr(self, "reread", None) or {}).get(tag)
		if not kept or kept["system"] != (self.system_prompt or ""):
			return None
		texts = {r["prompt"]: r for r in kept["responses"]}
		if any(p not in texts for p in prompts):
			return None
		return [texts[p] for p in prompts]

	def reread_profile(self, tag, prompts, answers, quick, stop, t):
		self.log(f"the set of levers changed: reading this model's {len(prompts)} saved baseline answers again with the new levers (no new generation)…")
		self.engine.set_steering(None)
		try:
			prof = PR.forced(self.engine, self.readout, self.scorer, self.band, self.word, prompts, [r["text"] for r in answers], system=self.system_prompt or None, on_event=lambda k, p: self.emit("profile_" + k, p), stop=stop)
		finally:
			self.resteer()
		check(stop)
		prof["responses"] = [dict(r) for r in answers]
		self.reread.pop(tag, None)
		return self.store_profile(tag, prof, quick, "reread", t)

	def profile_mode(self):
		m = self.settings.get("profile_mode")
		if m in PROFILE_MODES:
			return m
		return "read_write" if self.settings.get("profile_read", True) else "write"

	def base_answers(self, prompts, quick):
		base = self.summaries.get("baseline_quick" if quick else "baseline")
		if not base or len(base.get("delta") or []) != len(self.scorer.ids):
			return None
		texts = {r["prompt"]: r["text"] for r in base.get("responses") or [] if not r.get("read")}
		if any(p not in texts for p in prompts):
			return None
		return {"summary": base, "answers": [texts[p] for p in prompts]}

	def read_base(self, prompts, quick):
		mode = self.profile_mode()
		if mode == "write" or not getattr(self.lens, "jacobians", None):
			return None
		base = self.base_answers(prompts, quick)
		if base is None or (mode == "read_write" and (base["summary"].get("system", self.system_prompt) or "") != (self.system_prompt or "")):
			return None
		return base

	def plain_read(self, base, prompts, stop=None):
		summ = base["summary"]
		system = summ.get("system", self.system_prompt) or None
		kept = getattr(self, "read_plain", None)
		if kept and kept[0] is summ and kept[1] == (system, tuple(prompts)):
			return kept[2]
		self.engine.set_steering(None)
		plain = PR.forced(self.engine, self.readout, self.scorer, self.band, self.word, prompts, base["answers"], system=system, stop=stop)
		check(stop)
		self.read_plain = (summ, (system, tuple(prompts)), plain)
		return plain

	def run_profile(self, prompts, vecs, base, on_event=None, stop=None, factor=1.0):
		args = (self.engine, self.readout, self.scorer, self.band, self.word)
		if base is None:
			self.engine.set_steering(vecs or None)
			return PR.run(*args, prompts=prompts, system=self.system_prompt or None, on_event=on_event, stop=stop)
		plain = self.plain_read(base, prompts, stop)
		self.engine.set_steering(vecs or None)
		now = PR.forced(*args, prompts, base["answers"], system=self.system_prompt or None, on_event=on_event, stop=stop)
		check(stop)
		return PR.shifted(now, plain, base["summary"], self.reference, factor)

	def gain_known(self, k):
		g = self.read_gains.get(k)
		return float(g["gain"]) if g and k in self.levers and g.get("sig") == self.jsig(k)[:-1] else None

	def read_fill(self):
		known = [g for g in (self.gain_known(k) for k in list(self.read_gains)) if g is not None]
		return float(np.mean(known)) if known else READ_GAIN_DEFAULT

	def mix_key(self, ids):
		return sorted(ids), [self.jsig(k)[:-1] for k in sorted(ids)]

	def read_factor(self, doses, stop=None, wait=True):
		if self.profile_mode() != "read_write":
			return 1.0
		w = {k: abs(float(v)) for k, v in (doses or {}).items() if v and k in self.levers}
		if not w:
			return 1.0
		gains = {}
		for k in w:
			g = self.gain_known(k)
			if g is None and wait:
				g = self.read_gain(k, stop)
			if g is None:
				self.read_missing.add(k)
				g = self.read_fill()
			gains[k] = g
		f = float(sum(w[k] * gains[k] for k in w) / sum(w.values()))
		mix = self.read_mix or {}
		if len(w) > 1 and [mix.get("ids"), mix.get("sigs")] == list(self.mix_key(w)):
			f *= float(mix.get("fix", 1.0))
		return f

	def measure_read_gains(self, ids, stop=None):
		done = []
		for k in ids:
			if k in self.levers and self.gain_known(k) is None:
				self.read_gain(k, stop)
				done.append(k)
		cur = self.summaries.get("current")
		if done and cur and cur.get("mode") == "read_write" and any(self.strengths.get(k) for k in done):
			self.profile("current", bool(cur.get("quick")), stop)
		return done

	def mix_round(self, stop=None):
		cur = self.summaries.get("current") or {}
		if self.profile_mode() != "read_write" or cur.get("mode") != "read_write" or "baseline_quick" not in self.summaries:
			return False
		w = [k for k, v in self.strengths.items() if v and k in self.levers]
		if len(w) < 2:
			return False
		ids, sigs = self.mix_key(w)
		mix = self.read_mix or {}
		same = [mix.get("ids"), mix.get("sigs")] == [ids, sigs]
		if same and max(abs(float(self.strengths[k]) - float((mix.get("doses") or {}).get(k, 0.0))) for k in ids) <= MIX_DRIFT:
			return False
		names = [(self.trait_by_id(k) or {}).get("name", k) for k in ids]
		self.log(f"READ + WRITE: the mix {' + '.join(names)} is written once, to scale its reading (levers add up a little more when written together)…")
		try:
			self.engine.set_steering(self.active_vectors() or None)
			wrote = PR.run(self.engine, self.readout, self.scorer, self.band, self.word, prompts=[PR.BATTERY[i] for i in PR.QUICK], system=self.system_prompt or None, stop=stop)
			check(stop)
		finally:
			self.resteer()
		base = self.trait_scores("baseline_quick")
		ws = PR.band_scores(wrote["speak"], self.reference, self.band) - base
		rs = self.trait_scores("current") - base
		rr = float(np.dot(rs, rs))
		fix = float(np.clip(np.dot(rs, ws) / rr, 0.6, 1.8)) if rr > 1e-8 else 1.0
		corr = float(np.corrcoef(rs, ws)[0, 1]) if rr > 1e-8 and np.std(ws) > 1e-9 else None
		total = (float(mix.get("fix", 1.0)) if same else 1.0) * fix
		self.read_mix = {"ids": ids, "sigs": sigs, "doses": {k: float(self.strengths[k]) for k in ids}, "fix": round(total, 4), "corr": None if corr is None else round(corr, 3)}
		self.log(f"  the mix reads ×{total:.2f} on top of its levers' own scales" + (f" (read and written agree at r = {corr:.2f})" if corr is not None else ""), "ok" if corr is None or corr >= 0.8 else "warn")
		self.trace("read_gain", lever=" + ".join(names), mix=True, doses={n: float(self.strengths[k]) for n, k in zip(names, ids)}, gain=round(fix, 4), total=round(total, 4), corr=corr, written={(self.trait_by_id(t) or {}).get("name", t): round(float(ws[i]), 3) for i, t in enumerate(self.scorer.ids)}, read={(self.trait_by_id(t) or {}).get("name", t): round(float(rs[i]), 3) for i, t in enumerate(self.scorer.ids)})
		self.persist()
		return True

	def read_gain(self, k, stop=None):
		sig = self.jsig(k)[:-1]
		g = self.read_gains.get(k)
		if g and g.get("sig") == sig:
			return float(g["gain"])
		prompts = [PR.BATTERY[i] for i in PR.QUICK]
		if "baseline_quick" not in self.summaries:
			self.profile("baseline_quick", True, stop)
		base = self.base_answers(prompts, True)
		if base is None:
			return 1.0
		cs = self.coh_settings()
		use_gen = max(0.0, min(1.0, float(cs["individual"]) / 100.0)) if cs["calibrate"] else 0.0
		v = LV.combine(self.levers, {k: TG.PROBE}, 1.0, use_gen)
		summ = base["summary"]
		system = summ.get("system", self.system_prompt) or None
		args = (self.engine, self.readout, self.scorer, self.band, self.word)
		name = (self.trait_by_id(k) or {}).get("name", k)
		self.log(f"READ + WRITE: '{name}' alone at {TG.PROBE:+.2f}, written once and read, to scale its reading…")
		try:
			plain = self.plain_read(base, prompts, stop)
			self.engine.set_steering(v)
			wrote = PR.run(*args, prompts=prompts, system=system, stop=stop)
			check(stop)
			self.engine.set_steering(v)
			now = PR.forced(*args, prompts, base["answers"], system=system, stop=stop)
			check(stop)
		finally:
			self.resteer()
		ws = PR.band_scores(wrote["speak"], self.reference, self.band) - self.trait_scores("baseline_quick")
		rs = PR.band_scores(now["speak"], self.reference, self.band) - PR.band_scores(plain["speak"], self.reference, self.band)
		rr = float(np.dot(rs, rs))
		gain = float(np.clip(np.dot(rs, ws) / rr, 0.5, 4.0)) if rr > 1e-8 else 1.0
		corr = float(np.corrcoef(rs, ws)[0, 1]) if rr > 1e-8 and np.std(ws) > 1e-9 else None
		self.read_gains[k] = {"sig": sig, "gain": round(gain, 4), "corr": None if corr is None else round(corr, 3)}
		self.read_missing.discard(k)
		self.log(f"'{name}': its reading is scaled ×{gain:.2f}" + (f" (read and written agree at r = {corr:.2f})" if corr is not None else ""), "ok" if corr is None or corr >= 0.8 else "warn")
		self.trace("read_gain", lever=name, dose=TG.PROBE, gain=round(gain, 4), corr=corr, written={(self.trait_by_id(t) or {}).get("name", t): round(float(ws[i]), 3) for i, t in enumerate(self.scorer.ids)}, read={(self.trait_by_id(t) or {}).get("name", t): round(float(rs[i]), 3) for i, t in enumerate(self.scorer.ids)})
		self.persist()
		return gain

	@torch.no_grad()
	def translate(self, concepts):
		todo = [c for c in concepts if not c["str"].strip().isascii()]
		if not todo:
			return
		W = self.readout.W
		ids = torch.tensor([c["id"] for c in todo], device=W.device)
		q = W[ids].float()
		q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
		if getattr(self.readout, "wnorm", None) is None:
			self.readout.wnorm = torch.cat([W[i:i + 32768].float().norm(dim=-1) for i in range(0, W.shape[0], 32768)]).clamp_min(1e-6)
		sims = (W @ q.T.to(W.dtype)).float() / self.readout.wnorm[:, None]
		sims = sims.masked_fill(~self.english.to(W.device)[:, None], -1.0)
		best = sims.argmax(0).tolist()
		for c, b in zip(todo, best):
			c["en"] = self.strings[b].strip()

	def ref_odds(self):
		return _jsonable(self.reference["odds"]) if self.reference else None

	def profile_values(self, tag):
		s = self.summaries.get(tag)
		return {t["id"]: t["value"] for t in s["traits"]} if s else None

	def ensure_lever(self, tid, stop=None):
		if tid in self.levers:
			return self.levers[tid]
		trait = self.trait_by_id(tid)
		if trait is None:
			raise RuntimeError(f"unknown lever '{tid}'")
		self.ensure_band(stop)
		self.log(f"extracting the '{trait['name']}' lever ({trait['plus']} ↔ {trait['minus']})…")
		t = time.time()
		self.engine.set_steering(None)
		progress = lambda p: self.emit("progress", {"task": f"lever:{tid}", "value": p})
		if IW.has_images(trait):
			ex = self.image_extract(trait, stop, progress)
		else:
			ex = LV.extract(self.engine, trait, self.band, stop=stop, on_progress=progress, prompts=trait.get("prompts"))
		if ex is None:
			return None
		layers = LV.pick_layers(ex["sep"], self.band)
		lever = {"layers": layers, "dirs": {l: ex["dirs"][l] for l in layers}, "norm": {l: ex["norm"][l] for l in layers}, "sep": {l: ex["sep"][l] for l in ex["sep"]}, "samples": ex["samples"], "sig": signature(trait)}
		center = layers[len(layers) // 2]
		lever["reads_as"] = LV.reads_as(self.readout, ex["dirs"][center], center, ex["norm"][center], self.strings)
		lever["reads_as_minus"] = LV.reads_as(self.readout, -ex["dirs"][center], center, ex["norm"][center], self.strings)
		if self.probe is None:
			self.probe = LV.KLProbe(self.engine, self.readout)
		lever["gain"], lever["kl"] = LV.calibrate(self.probe, lever)
		self.levers[tid] = lever
		self.judged.pop(tid, None)
		if self.coh_settings()["calibrate"]:
			self.calibrate_gen(tid, stop)
		self.resteer()
		self.log(f"'{trait['name']}' lever: L{layers[0]}–L{layers[-1]}, gain {LV.effective_gain(lever, self.coh_settings()['calibrate']):.3f} · reads as: {' '.join(s.strip() for s in lever['reads_as'][:6])} ({time.time() - t:.1f} s)")
		self.emit("lever", _jsonable({"id": tid, "layers": layers, "gain": lever["gain"], "gen_mult": lever.get("gen_mult"), "reads_as": lever["reads_as"], "reads_as_minus": lever["reads_as_minus"], "sep": lever["sep"]}))
		self.persist()
		return lever

	def coh_settings(self):
		cs = {**config.DEFAULTS["coherence"], **COH_EXTRA, **(self.settings.get("coherence") or {})}
		cs["guard_mode"] = LEGACY_MODES.get(cs["guard_mode"], cs["guard_mode"])
		return cs

	def set_judge_state(self, **state):
		if state == self.judge_state:
			return
		self.judge_state = state
		self.emit("judge", dict(state))

	def judge_tests(self):
		path = os.path.join(config.CACHE_DIR, "judge_selftest.json")
		try:
			with open(path, encoding="utf-8") as f:
				d = json.load(f)
			return d if isinstance(d, dict) else {}
		except (OSError, ValueError):
			return {}

	def save_judge_test(self, key, cal):
		d = self.judge_tests()
		d[key] = _jsonable(cal)
		os.makedirs(config.CACHE_DIR, exist_ok=True)
		path = os.path.join(config.CACHE_DIR, "judge_selftest.json")
		with open(path + ".tmp", "w", encoding="utf-8") as f:
			json.dump(d, f, ensure_ascii=False)
		os.replace(path + ".tmp", path)

	def ensure_judge(self):
		path = self.coh_settings()["judge"]
		if not path or not os.path.isfile(path):
			if self.judge_engine is not None and self.judge_engine.loaded:
				self.judge_engine.unload()
				self.log("judge unloaded")
			self.judge = None
			if self.coh is not None:
				self.coh.judge = None
			self.set_judge_state(state="off")
			return None
		cs = self.coh_settings()
		custom = cs["judge_prompt"] if "{reply}" in (cs["judge_prompt"] or "") else ""
		if self.judge is not None and self.judge_engine.info and self.judge_engine.info["path"] == path and getattr(self.judge, "custom", "") == custom:
			self.judge.temperature = max(0.05, float(cs["judge_temperature"] or 1.0))
			return self.judge
		self.judge = None
		if self.coh is not None:
			self.coh.judge = None
		if self.judge_engine is not None and self.judge_engine.loaded:
			self.judge_engine.unload()
			self.log("previous judge unloaded")
		name = os.path.basename(path)
		if path == getattr(self, "judge_failed", None):
			self.set_judge_state(state="rejected", name=name, path=path)
			return None
		key = judge_key(path, cs["judge_temperature"], custom)
		known = self.judge_tests().get(key)
		if isinstance(known, dict) and known.get("auc") is not None and float(known["auc"]) < JUDGE_MIN_AUC:
			self.judge_failed = path
			self.log(f"judge {name} rejected by an earlier self-test (AUC {float(known['auc']):.2f} < {JUDGE_MIN_AUC:.2f}, same file and settings): using the surprise measure", "warn")
			self.set_judge_state(state="rejected", name=name, path=path, auc=float(known["auc"]))
			return None
		self.ensure_engine()
		if self.judge_engine is None:
			self.judge_engine = Engine(self.llama_dir(), lib=self.engine.lib)
		gpus = self.engine.lib.gpus()
		need = os.path.getsize(path) + 700 * 2**20
		best = max(range(len(gpus)), key=lambda i: gpus[i]["free"]) if gpus else -1
		if best < 0 or gpus[best]["free"] < need:
			self.log(f"not enough free VRAM for the judge {name} ({need / 2**30:.1f} GB needed): using the surprise measure", "warn")
			self.set_judge_state(state="no_vram", name=name, path=path, need=need)
			return None
		split = [0.0] * len(gpus)
		split[best] = 1.0
		self.log(f"loading the judge {name} on CUDA{best}…")
		self.set_judge_state(state="loading", name=name, path=path, gpu=best)
		try:
			self.judge_engine.load(path, {"n_gpu_layers": -1, "tensor_split": split, "n_ctx": 3072})
			judge = Judge(self.judge_engine, cs["judge_temperature"], custom)
			judge.custom = custom
			cached = judge.restore(known)
			auc = judge.auc if cached else judge.selftest()
			if not cached:
				self.save_judge_test(key, judge.calibration())
		except Exception as exc:
			self.log(f"judge unavailable ({exc}): using the surprise measure", "warn")
			self.set_judge_state(state="error", name=name, path=path, error=str(exc))
			return None
		self.trace("judge", context=f"judge self-test: {name}" + (" (kept from an earlier test)" if cached else ""), name=name, auc=auc, variant=judge.variant, anchors=list(judge.anchors), cap=judge.absurd_cap, report=judge.report, accepted=auc >= JUDGE_MIN_AUC, minimum=JUDGE_MIN_AUC)
		if auc < JUDGE_MIN_AUC:
			self.judge_engine.unload()
			self.judge_failed = path
			self.log(f"judge rejected: it cannot tell coherent replies from word salad (self-test AUC {auc:.2f} < {JUDGE_MIN_AUC:.2f}{', from an earlier test' if cached else ''}): using the surprise measure", "warn")
			self.set_judge_state(state="rejected", name=name, path=path, auc=auc)
			return None
		self.judge = judge
		self.log(f"judge ready: {judge.name} · self-test AUC {auc:.2f} with the '{judge.variant}' question · scale {judge.anchors[0]:.2f}→{judge.anchors[1]:.2f}" + (" (self-test kept from an earlier load: same judge file and settings)" if cached else ""))
		if judge.absurd_cap < 1.0:
			self.log(f"this judge rates word salad as grammatical absurdity: absurdity tolerance capped at {judge.absurd_cap:.0%} with it", "warn")
		self.set_judge_state(state="ready", name=name, path=path, gpu=best, auc=round(float(auc), 3), variant=judge.variant, cap=judge.absurd_cap, size=os.path.getsize(path))
		if self.coh is not None:
			self.coh.judge = self.judge
		return self.judge

	def coherence(self, mode="guard"):
		if self.coh is None or self.coh.engine is not self.engine:
			self.coh = Coherence(self.engine)
		cs = self.coh_settings()
		self.coh.absurd = float(cs["absurdity"]) / 100.0
		self.coh.configure(question_list(cs["test_prompts"], cs.get(f"{mode}_questions") or 0), cs["test_tokens"], cs["test_system"], cs["test_samples"])
		self.coh.on_live = lambda d: self.emit("judge_live", _jsonable(d))
		self.coh.judge = self.ensure_judge()
		self.coh.stop = self.stop_event
		self.coh.on_measure = self.trace_measure
		self.coh.on_start = self.trace_start
		self.coh.on_tick = None
		return self.coh

	def trace(self, kind, **data):
		self.emit("trace", _jsonable(dict(data, kind=kind, t=time.time(), context=data.get("context", self.trace_ctx))))

	def close_measure(self):
		if getattr(self, "measure_id", None):
			self.trace("measure", id=self.measure_id, stopped=True, note="stopped before the end")
			self.measure_id = None

	def trace_start(self, d):
		self.close_measure()
		self.measure_id = f"m{time.time_ns()}"
		self.trace("measure", id=self.measure_id, running=True, **d)

	def trace_measure(self, d):
		self.trace("measure", id=getattr(self, "measure_id", None), **d)
		self.measure_id = None

	def lever_state(self):
		return {"strengths": dict(self.strengths), "guard": self.guard_info(), "layers": self.engine.steering_layers if self.engine is not None else []}

	def after_stop(self):
		self.soft = False
		self.close_measure()
		if self.engine is None or not self.engine.loaded:
			return
		self.resteer()
		self.emit("levers", self.lever_state())

	def calibrate_gen(self, tid, stop=None):
		lever = self.levers[tid]
		name = (self.trait_by_id(tid) or {}).get("name", tid)
		coh = self.coherence("calib")
		self.trace_ctx = f"calibrating '{name}' on real generations"
		self.log(f"calibrating '{name}' on real generations ({'judge' if coh.judge else 'surprise'})…")
		m, last = LV.calibrate_gen(coh, lever, CALIB_TARGET, steps=max(2, int(self.coh_settings()["calib_steps"])), on_step=lambda i, n, mult, r: self.emit("progress", {"task": f"lever:{tid}", "value": i / n}))
		lever["gen_mult"] = float(m)
		lever["gen_by"] = last["by"] if last else "surprise"
		self.judged.pop(tid, None)
		self.log(f"'{name}': strength 1 = {m:.2f}× the old scale (coherence target {CALIB_TARGET:.0%})")
		self.trace("calibration", lever=name, mult=float(m), target=CALIB_TARGET, by=lever["gen_by"])
		self.trace_ctx = ""
		self.engine.set_steering(None)
		self.emit("lever", _jsonable({"id": tid, "layers": lever["layers"], "gain": lever["gain"], "gen_mult": lever["gen_mult"], "reads_as": lever.get("reads_as", []), "reads_as_minus": lever.get("reads_as_minus", []), "sep": lever.get("sep", {})}))

	def calibrate_all(self, ids, stop=None):
		for tid in ids:
			if stop is not None and stop.is_set():
				break
			if self.ensure_lever(tid, stop) is None:
				break
			if self.coh_settings()["calibrate"] and "gen_mult" not in self.levers[tid]:
				self.calibrate_gen(tid, stop)
		self.persist()
		if any(self.strengths.values()):
			self.apply_levers()

	def inactive(self):
		return set(self.settings.get("inactive_levers") or [])

	def set_active(self, inactive):
		off = set(inactive)
		self.set_setting("inactive_levers", sorted(off))
		zeroed = [k for k, v in self.strengths.items() if v and GO.base_id(k) in off]
		for k in zeroed:
			self.strengths[k] = 0.0
		for k in [k for k in self.targets if k in off]:
			self.targets.pop(k)
		loaded = self.engine is not None and self.engine.loaded
		if zeroed and loaded:
			self.apply_levers()
		elif loaded:
			self.persist()
		self.log(f"active levers saved: {len(off)} inactive" + (" · back to 0: " + ", ".join(sorted({(self.trait_by_id(GO.base_id(k)) or {}).get('name', k) for k in zeroed})) if zeroed else ""), "ok")
		self.emit("loaded", self.snapshot())
		return sorted(off)

	def set_strengths(self, strengths, stop=None):
		off = self.inactive()
		wanted = {k: (0.0 if GO.base_id(k) in off else max(-TG.HARD, min(TG.HARD, float(v)))) for k, v in strengths.items() if not k.endswith(GO.LORA) or GO.base_id(k) in self.levers}
		unplaced = [k for k, v in wanted.items() if v and GO.is_goal(k) and GO.base_id(k) not in self.levers]
		if unplaced:
			self.log("not placed on this model: " + ", ".join((self.trait_by_id(GO.base_id(k)) or {}).get("name", k) for k in unplaced) + " · TRAIN or ALIGN it in GOALS first (kept at 0 here)", "warn")
			for k in unplaced:
				wanted[k] = 0.0
		fresh = False
		for tid, v in wanted.items():
			if v and tid not in self.levers and not tid.endswith(GO.LORA):
				if self.ensure_lever(tid, stop) is None:
					self.emit("levers", self.lever_state())
					return
				fresh = True
		if self.coh_settings()["calibrate"]:
			for tid, v in wanted.items():
				if v and tid in self.levers and "gen_mult" not in self.levers[tid]:
					self.calibrate_gen(tid, stop)
					fresh = True
		if not fresh and all(abs(self.strengths.get(k, 0.0) - v) < 1e-6 for k, v in wanted.items()):
			self.emit("levers", self.lever_state())
			return
		prev = dict(self.strengths)
		self.strengths.update(wanted)
		try:
			self.apply_levers()
		except Stopped:
			self.strengths = prev
			raise

	def image_extract(self, trait, stop=None, on_progress=None):
		mm = self.mmproj()
		if not mm:
			raise RuntimeError(f"'{trait['name']}' is made from images: this model has no mmproj*.gguf next to it to see them")
		plus, minus = IW.image_paths(trait, "plus"), IW.image_paths(trait, "minus")
		layers = IW.extract_layers(self.band, self.engine.n_layer)
		saved = IW.load_reads(self.image_reads_path(trait["id"]))
		where = {p: (pole, os.path.basename(p)) for pole, paths in (("plus", plus), ("minus", minus)) for p in paths}
		missing = [p for p in plus + minus if not os.path.isfile(p) and not self.saved_read(saved, where[p], p, layers)]
		if missing:
			raise RuntimeError(f"'{trait['name']}': {len(missing)} image(s) missing in {os.path.dirname(missing[0])}, never read on this model")
		vis, hits = [None], [0]

		def read(path):
			agg = self.image_cache.get(self.image_key(path, layers)) if os.path.isfile(path) else None
			if agg is None:
				agg = self.saved_read(saved, where[path], path, layers)
				if agg is not None and os.path.isfile(path):
					self.keep_image(path, layers, agg)
			if agg is not None:
				hits[0] += 1
				return agg
			if vis[0] is None:
				vis[0] = Vision(self.engine, mm)
			caps, spans = vis[0].read(path, IW.PROMPT, layers)
			agg = IW.aggregate(caps, IW.rows_of(spans, "after"), layers)
			self.keep_image(path, layers, agg)
			return agg

		try:
			poles = set(IW.pole_paths(trait, "plus") + IW.pole_paths(trait, "minus"))
			ex = IW.extract(read, self.engine, plus, minus, self.band, stop, on_progress, {p: IW.POLE_WEIGHT for p in poles})
		finally:
			if vis[0] is not None:
				vis[0].close()
		self.save_image_reads(trait, layers, saved)
		n = len(plus) + len(minus)
		self.log(f"'{trait['name']}' from images: " + (f"{n - hits[0]} read through {os.path.basename(mm)}" if n > hits[0] else "nothing to read again") + (f", {hits[0]} reused from earlier readings" if hits[0] else ""))
		return ex

	def image_reads_path(self, tid):
		return os.path.join(self.state_dir(), "image_reads", f"{tid}.npz")

	def saved_read(self, saved, ref, path, layers):
		e = saved.get(ref)
		if e is None or e["layers"] != list(layers):
			return None
		if os.path.isfile(path):
			st = os.stat(path)
			if st.st_size != e["size"] or st.st_mtime_ns != e["mtime"]:
				return None
		return e["agg"]

	def save_image_reads(self, trait, layers, saved=None):
		saved = saved if saved is not None else IW.load_reads(self.image_reads_path(trait["id"]))
		entries = []
		for pole in ("plus", "minus"):
			for path in IW.image_paths(trait, pole):
				name = os.path.basename(path)
				if os.path.isfile(path):
					agg = self.image_cache.get(self.image_key(path, layers))
					st = os.stat(path)
					if agg is not None:
						entries.append({"pole": pole, "name": name, "size": st.st_size, "mtime": st.st_mtime_ns, "agg": agg})
						continue
				e = saved.get((pole, name))
				if e is not None and e["layers"] == list(layers):
					entries.append({k: e[k] for k in ("pole", "name", "size", "mtime")} | {"agg": e["agg"]})
		try:
			IW.save_reads(self.image_reads_path(trait["id"]), entries)
		except OSError as exc:
			self.log(f"'{trait['name']}': its image readings could not be saved ({exc})", "warn")

	def image_key(self, path, layers):
		st = os.stat(path)
		return ((self.info or {}).get("path", ""), os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns, tuple(layers))

	def keep_image(self, path, layers, agg):
		if len(self.image_cache) >= 256:
			self.image_cache.clear()
		self.image_cache[self.image_key(path, layers)] = agg

	def mmproj(self):
		return pick_mmproj((self.info or {}).get("path", ""), self.settings.get("mmproj")) if self.engine is not None and self.engine.loaded else ""

	def set_mmproj(self, path, choice):
		choices = dict(self.settings.get("mmproj") or {})
		if choice:
			choices[path] = choice
		else:
			choices.pop(path, None)
		self.set_setting("mmproj", choices)
		self.log(f"vision of {os.path.basename(path)}: " + ({"": "automatic (an mmproj*.gguf next to it)", "none": "off"}.get(choice) or os.path.basename(choice)))
		return self.catalog()

	def image_words(self, plus, minus, where="after", stop=None):
		if self.engine is None or not self.engine.loaded or self.readout is None:
			raise RuntimeError("load a model first")
		mm = self.mmproj()
		if not mm:
			raise RuntimeError("no mmproj*.gguf next to the model: this model cannot see images")
		if not plus and not minus:
			raise RuntimeError("add images to at least one side")
		lo, hi = self.band or (0, self.engine.n_layer - 1)
		layers = list(range(lo, hi + 1))
		wide = IW.extract_layers((lo, hi), self.engine.n_layer)
		mask = self.english.to(self.readout.device)
		total = len(plus) + len(minus) + 1
		done = 0
		self.engine.set_steering(None)
		self.log(f"reading {len(plus)} + {len(minus)} images through {os.path.basename(mm)} (lens on L{lo}–L{hi}, {where} tokens)")
		vis = Vision(self.engine, mm)
		profiles, tops = {}, {}
		try:
			for pole, paths in (("plus", plus), ("minus", minus)):
				acc = None
				for path in paths:
					check(stop)
					caps, spans = vis.read(path, IW.PROMPT, sorted(set(layers) | set(wide)))
					self.keep_image(path, wide, IW.aggregate(caps, IW.rows_of(spans, "after"), wide))
					if where == "both":
						p = (IW.profile(self.readout, caps, IW.rows_of(spans, "image"), layers) + IW.profile(self.readout, caps, IW.rows_of(spans, "after"), layers)) / 2
					else:
						p = IW.profile(self.readout, caps, IW.rows_of(spans, where), layers)
					tops.setdefault(pole, []).append({"image": os.path.basename(path), "tokens": sum(n for k, _a, n in spans if k == "image"), "top": IW.top_tokens(p, self.strings, mask)})
					acc = p if acc is None else acc + p
					done += 1
					self.emit("progress", {"task": "image_words", "value": done / total})
				profiles[pole] = acc / len(paths) if acc is not None else None
			if profiles["plus"] is None or profiles["minus"] is None:
				check(stop)
				toks = self.engine.chat_tokens([{"role": "user", "content": IW.PROMPT}])
				self.engine.reset()
				self.engine.set_capture(layers)
				try:
					self.engine.decode(toks)
					caps = self.engine.pop_capture()
				finally:
					self.engine.set_capture(None)
					self.engine.reset()
				neutral = IW.profile(self.readout, caps, list(range(len(toks))), layers)
				profiles = {k: (v if v is not None else neutral) for k, v in profiles.items()}
		finally:
			vis.close()
			self.resteer()
		words = IW.contrast(profiles["plus"], profiles["minus"], self.strings, mask)
		words = {k: (v if (plus if k == "plus" else minus) else []) for k, v in words.items()}
		self.log(f"image words · + {', '.join(words['plus'][:8])} · − {', '.join(words['minus'][:8])}", "ok")
		res = {"plus": words["plus"], "minus": words["minus"], "where": where, "per_image": tops, "mmproj": os.path.basename(mm)}
		self.emit("image_words", res)
		return res

	def resteer(self):
		if self.engine is not None and self.engine.loaded:
			self.engine.set_steering(self.current_vectors or None)

	def active_vectors(self):
		return self.current_vectors if any(self.strengths.values()) else {}

	def apply_levers(self, how="auto"):
		keep = (self.current_vectors, self.guard_scale, self.coherence_kl, self.coherence_info, self.blame, self.guard_free, self.guard_basis, dict(self.judged))
		self.soft = True
		try:
			self._apply_levers(how)
		except Stopped:
			self.current_vectors, self.guard_scale, self.coherence_kl, self.coherence_info, self.blame, self.guard_free, self.guard_basis, self.judged = keep
			raise
		finally:
			self.soft = False

	def judge_auto(self):
		return not getattr(self, "defer_judge", False)

	def deferred(self, fn):
		self.defer_judge = True
		try:
			return fn(self)
		finally:
			self.defer_judge = False

	def dose_max(self):
		return max(0.5, min(5.0, float(self.settings.get("dose_max") or TG.DOSE)))

	def use_gen(self):
		cs = self.coh_settings()
		return max(0.0, min(1.0, float(cs["individual"]) / 100.0)) if cs["calibrate"] else 0.0

	def split_levers(self):
		cs = self.coh_settings()
		off = self.inactive()
		active = {t: s for t, s in self.strengths.items() if s and t in self.levers and GO.base_id(t) not in off}
		exempt = set(self.settings.get("guard_exempt") or [])
		free = {t: s for t, s in active.items() if t in exempt or (not cs["guard_optimized"] and self.levers[t].get("kind") == "optimized")}
		return active, free, {t: s for t, s in active.items() if t not in free}

	def unjudged(self, guarded):
		return {t: s for t, s in guarded.items() if t not in self.judged or abs(self.judged[t][0] - s) > 1e-6}

	def guard_key(self, coh, cs, use_gen):
		return (coh.judge.name if coh.judge else "", coh.absurd, tuple(coh.prompts), coh.n_tokens, coh.system, coh.samples, cs["tolerance"], cs["guard_mode"], use_gen)

	def lever_sig(self, tid):
		lv = self.levers.get(tid) or {}
		return (id(lv), lv.get("gain"), lv.get("gen_mult"), tuple(lv.get("layers") or ()))

	def lowered(self, basis, guarded, key):
		if not basis or basis["key"] != key:
			return False
		old = basis["strengths"]
		for t, v in guarded.items():
			o = old.get(t, 0.0)
			if t not in old or basis["sigs"].get(t) != self.lever_sig(t) or abs(v) > abs(o) + 1e-9 or v * o < 0:
				return False
		return True

	def lever_names(self, strengths):
		return ", ".join(f"{(self.trait_by_id(t) or {}).get('name', t)} {v:+.2f}" for t, v in sorted(strengths.items()))

	def _apply_levers(self, how="auto"):
		cs = self.coh_settings()
		if how == "auto":
			how = "full" if self.judge_auto() else "manual"
		active, free, guarded = self.split_levers()
		self.guard_free = sorted(free)
		use_gen = self.use_gen()
		self.current_vectors = {}
		self.coherence_kl = 0.0
		if active and self.probe is None:
			self.probe = LV.KLProbe(self.engine, self.readout)
		if how == "full" and not cs["guard"]:
			self.exact(guarded, use_gen)
		elif how == "full" and cs["guard_mode"] == "individual":
			self.guard_changed(cs, guarded, use_gen, "INDIVIDUAL")
		elif how == "full":
			self.guard_full(cs, guarded, use_gen)
		elif how == "judge":
			self.guard_changed(cs, guarded, use_gen)
		elif how == "kl":
			self.guard_kl(guarded, use_gen)
		else:
			self.as_set(guarded, use_gen)
		if free:
			merged = dict(self.current_vectors)
			for l, v in LV.combine(self.levers, free, 1.0, use_gen).items():
				merged[l] = merged[l] + v if l in merged else v
			self.current_vectors = merged
			self.log("applied as set, outside the coherence guard: " + ", ".join((self.trait_by_id(t) or {}).get("name", t) for t in free))
		loras = {}
		exempt = set(self.settings.get("guard_exempt") or [])
		for t, lv in self.levers.items():
			s = self.lora_dose(t)
			if not lv.get("lora") or not s:
				continue
			path = self.lora_path(t)
			out = t in exempt or (not cs["guard_optimized"] and lv.get("kind") == "optimized")
			loras[path] = loras.get(path, 0.0) + s * (1.0 if out else self.guard_scale * self.blame.get(t, 1.0))
		if loras:
			self.current_vectors = {**self.current_vectors, LORA: loras}
			if self.probe is None:
				self.probe = LV.KLProbe(self.engine, self.readout)
		if active or loras:
			self.coherence_kl = self.probe.kl(self.current_vectors)
		self.resteer()
		self.emit("levers", {"strengths": dict(self.strengths), "guard": self.guard_info(), "layers": self.engine.steering_layers, "by": self.lever_source})
		self.persist()

	def guard_full(self, cs, guarded, use_gen):
		basis = self.guard_basis
		self.coherence_info = None
		self.guard_basis = None
		self.blame = {}
		self.guard_scale = 1.0
		if not guarded:
			self.judged = {}
			return
		self.current_vectors, self.guard_scale, self.coherence_kl = LV.guard(self.probe, self.levers, guarded, use_gen=use_gen)
		if cs["guard"]:
			coh = self.coherence()
			limit = threshold(cs["tolerance"])
			key = self.guard_key(coh, cs, use_gen)
			self.trace_ctx = "lever change: " + self.lever_names(guarded)
			if self.guard_scale >= 0.999 and self.lowered(basis, guarded, key):
				self.coherence_info = dict(basis["info"], kept=True)
				self.guard_basis = basis
				self.log(f"levers only lowered from a mix that passed the coherence check ({basis['info']['coherence']:.0%}): no new check needed")
			else:
				self.log(f"checking coherence on real generations ({'judge' if coh.judge else 'surprise'}, threshold {limit:.0%}, absurdity tolerance {coh.absurd:.0%})…")
				self.current_vectors, self.guard_scale, res, factors = self.run_guard(cs, coh, guarded, self.guard_scale, limit, use_gen)
				self.blame = {k: f for k, f in factors.items() if f < 0.999}
				self.coherence_info = {"coherence": res["coherence"], "by": res["by"], "threshold": limit, "surprise": res.get("surprise"), "first": res.get("first"), "drop": res.get("drop")}
				if res.get("first") is not None:
					self.log(f"coherence {res['first']:.0%} at the strengths you set: below the {limit:.0%} threshold, so the guard stepped in", "warn")
				self.log(f"coherence {res['coherence']:.0%} ({res['by']})" + (f" · levers scaled ×{self.guard_scale:.2f}" if self.guard_scale < 0.999 else ""), "info" if res["coherence"] >= limit else "warn")
				if res["coherence"] >= limit and self.guard_scale >= 0.999 and not self.blame:
					self.guard_basis = {"strengths": dict(guarded), "sigs": {t: self.lever_sig(t) for t in guarded}, "key": key, "info": {k: v for k, v in self.coherence_info.items() if k != "first"}}
			self.trace("guard", first=self.coherence_info.get("first"), coherence=self.coherence_info["coherence"], threshold=limit, scale=self.guard_scale, blame={(self.trait_by_id(k) or {}).get("name", k): f for k, f in self.blame.items()}, by=self.coherence_info.get("by"), kept=bool(self.coherence_info.get("kept")), mode=cs["guard_mode"], drop=self.coherence_info.get("drop"))
			self.trace_ctx = ""
		self.judged = {t: (float(s), float(self.guard_scale * self.blame.get(t, 1.0))) for t, s in guarded.items()}

	def run_guard(self, cs, coh, moved, scale, limit, use_gen, base=None):
		mode = cs["guard_mode"] if cs["guard_mode"] in GUARD_MODES else "uniform"
		steps, rounds = max(1, int(cs["guard_steps"])), max(1, int(cs["blame_rounds"]))
		name = lambda k: (self.trait_by_id(k) or {}).get("name", k)
		on_blame = lambda i, n, k, f, r: self.log(f"  worst lever: '{name(k)}' hurts the most, halved (×{f:.2f}) → coherence {r['coherence']:.0%}")
		on_gen = lambda i, n, s, r: self.log(f"  scaling ×{s:.2f} → coherence {r['coherence']:.0%}")
		if mode == "quick":
			n = max(1, int(cs["quick_questions"])) * len(moved)
			Ticks(self.emit, coh, 1, n)
			on_quick = lambda r, ok: self.log(f"  quick check ({n} repl{'ies' if n != 1 else 'y'}): coherence {r['coherence']:.0%}" + (", validated" if ok else f" below the threshold, checked again in full ({cs['quick_fallback']})"), "info" if ok else "warn")
			return LV.guard_quick(coh, self.levers, moved, scale, limit, n, use_gen, cs["quick_fallback"], steps, rounds, on_gen, on_blame, base, on_quick)
		if mode == "read":
			ok, bad = float(cs["read_pass"]), float(cs["read_fail"])
			Ticks(self.emit, coh, steps + 1)
			say = {"pass": (f"≤ {ok:.2f}: validated by reading, nothing generated", "info"), "fail": (f"≥ {bad:.2f}: too far, scaled down by reading", "warn"), "judge": (f"between {ok:.2f} and {bad:.2f}: the judge decides ({cs['read_fallback']})", "info")}
			on_read = lambda s, d, what: self.log(f"  read check" + (f" ×{s:.2f}" if s < 0.999 else "") + f": the levers lower the model's own replies by {d:.2f} per token " + say[what][0], say[what][1])
			return LV.guard_read(coh, self.levers, moved, scale, limit, use_gen, ok, bad, cs["read_fallback"], steps, rounds, on_gen, on_blame, base, on_read)
		if mode == "individual" and len(moved) > 1:
			Ticks(self.emit, coh, 1 + rounds * len(moved))
			on_order = lambda i, order, score: self.log(f"  round {i}: replies closest to " + ", ".join(f"'{name(k)}'" + (f" ({score[k]:.2f})" if k in score else "") for k in order) + ": halved in that order until one passes")
			return LV.guard_blame(coh, self.levers, moved, scale, limit, use_gen, rounds=rounds, on_step=on_blame, base=base, steps=steps, on_order=on_order)
		Ticks(self.emit, coh, steps + 1)
		vecs, scale, res = LV.guard_gen(coh, self.levers, moved, scale, limit, use_gen, steps=steps, on_step=on_gen, base=base)
		return vecs, scale, res, {}

	def guard_kl(self, guarded, use_gen):
		self.coherence_info = None
		self.blame = {}
		self.guard_scale = 1.0
		if guarded:
			self.current_vectors, self.guard_scale, self.coherence_kl = LV.guard(self.probe, self.levers, guarded, use_gen=use_gen)

	def exact(self, guarded, use_gen):
		self.guard_scale = 1.0
		self.blame = {}
		self.coherence_info = None
		self.guard_basis = None
		self.current_vectors = LV.combine(self.levers, guarded, 1.0, use_gen)

	def as_set(self, guarded, use_gen):
		moved = self.unjudged(guarded)
		self.judged = {t: v for t, v in self.judged.items() if t in guarded and t not in moved}
		self.guard_scale = 1.0
		self.blame = {t: self.judged[t][1] for t in guarded if t not in moved and self.judged[t][1] < 0.999}
		if moved:
			self.coherence_info = None
			self.guard_basis = None
		self.current_vectors = LV.combine(self.levers, {t: s * self.blame.get(t, 1.0) for t, s in guarded.items()}, 1.0, use_gen)

	def guard_changed(self, cs, guarded, use_gen, tag="JUDGE"):
		self.coherence_info = None
		self.guard_basis = None
		self.guard_scale = 1.0
		self.blame = {}
		if not guarded:
			self.judged = {}
			if tag == "JUDGE":
				self.log("JUDGE: no lever under the coherence guard is set")
			return
		moved = self.unjudged(guarded) or dict(guarded)
		factors = {t: self.judged[t][1] for t in guarded if t not in moved}
		fixed = {t: s * factors[t] for t, s in guarded.items() if t not in moved}
		base = LV.combine(self.levers, fixed, 1.0, use_gen)
		names = self.lever_names(moved)
		vecs, scale, _kl = LV.guard(self.probe, self.levers, moved, use_gen=use_gen, base=base)
		if scale < 0.999:
			self.log(f"{tag} · KL guard: {names} scaled ×{scale:.2f} (the next-token distribution drifts too far on a neutral text)", "warn")
		parts = {t: scale for t in moved}
		if cs["guard"]:
			coh = self.coherence()
			limit = threshold(cs["tolerance"])
			self.trace_ctx = f"{tag}: " + names
			self.log(f"{tag}: checking {names} on real generations ({'judge' if coh.judge else 'surprise'}, threshold {limit:.0%}, absurdity tolerance {coh.absurd:.0%})" + (f", the {len(fixed)} other lever{'s' if len(fixed) > 1 else ''} kept as judged before" if fixed else "") + "…")
			vecs, scale, res, fx = self.run_guard(cs, coh, moved, scale, limit, use_gen, base=base)
			parts = {t: scale * fx.get(t, 1.0) for t in moved}
			self.coherence_info = {"coherence": res["coherence"], "by": res["by"], "threshold": limit, "surprise": res.get("surprise"), "first": res.get("first"), "drop": res.get("drop")}
			if res.get("first") is not None:
				self.log(f"coherence {res['first']:.0%} at the strengths you set: below the {limit:.0%} threshold, so the judge lowered them", "warn")
			cut = {t: f for t, f in parts.items() if f < 0.999}
			self.log(f"{tag}: coherence {res['coherence']:.0%} ({res['by']})" + (" · " + ", ".join(f"{(self.trait_by_id(t) or {}).get('name', t)} ×{f:.2f}" for t, f in sorted(cut.items())) if cut else " · kept as set"), "info" if res["coherence"] >= limit else "warn")
			self.trace("guard", first=self.coherence_info.get("first"), coherence=self.coherence_info["coherence"], threshold=limit, scale=min(parts.values()), blame={(self.trait_by_id(k) or {}).get("name", k): f for k, f in cut.items()}, by=self.coherence_info.get("by"), kept=False, mode=cs["guard_mode"], drop=self.coherence_info.get("drop"))
			self.trace_ctx = ""
		factors.update(parts)
		self.blame = {t: f for t, f in factors.items() if f < 0.999}
		self.current_vectors = LV.plus(vecs, base)
		self.judged = {t: (float(s), float(factors.get(t, 1.0))) for t, s in guarded.items()}

	def judge_levers(self, stop=None):
		self.need_model()
		_active, _free, guarded = self.split_levers()
		moved = self.unjudged(guarded)
		self.apply_levers("judge")
		return {"judged": sorted(moved or guarded)}

	def jsig(self, tid):
		lv = self.levers.get(tid) or {}
		cs = self.coh_settings()
		use_gen = max(0.0, min(1.0, float(cs["individual"]) / 100.0)) if cs["calibrate"] else 0.0
		return [round(float(lv.get("gain", 0.0)), 6), round(float(lv.get("gen_mult") or 1.0), 6), [int(l) for l in lv.get("layers") or []], str(lv.get("sig", "")), round(use_gen, 3), self.system_prompt]

	def trait_scores(self, tag):
		by = {t["id"]: float(t.get("score", 0.0)) for t in (self.summaries.get(tag) or {}).get("traits") or []}
		return np.array([by.get(t, 0.0) for t in self.scorer.ids], dtype=np.float64)

	def applied_doses(self, ids):
		free = set(self.guard_free or [])
		return np.array([float(self.strengths.get(t, 0.0)) * (1.0 if t in free else self.guard_scale) * float(self.blame.get(t, 1.0)) for t in ids], dtype=np.float64)

	def scores_with(self, vectors, stop=None, doses=None):
		self.ensure_reference(stop)
		prompts = [PR.BATTERY[i] for i in PR.QUICK]
		base = self.read_base(prompts, True)
		try:
			factor = self.read_factor(doses, stop) if base else 1.0
			prof = self.run_profile(prompts, vectors, base, lambda k, p: self.emit("progress", {"task": "align", "value": (p["index"] + 0.5) / max(p["total"], 1)}) if k == "prompt" else None, stop, factor)
		finally:
			self.resteer()
		check(stop)
		summ = PR.summarize(prof, self.reference, self.band, self.scorer.ids, self.lexicons, self.word, self.strings)
		by = {t["id"]: float(t["score"]) for t in summ["traits"]}
		return np.array([by.get(t, 0.0) for t in self.scorer.ids], dtype=np.float64)

	def jcol_fresh(self, k, ids):
		c = self.jcols.get(k)
		return bool(c) and c.get("sig") == self.jsig(k) and c.get("ids") == ids

	def jcol(self, k, base, stop=None):
		ids = list(self.scorer.ids)
		if self.jcol_fresh(k, ids):
			return np.asarray(self.jcols[k]["col"], dtype=np.float64)
		cs = self.coh_settings()
		use_gen = max(0.0, min(1.0, float(cs["individual"]) / 100.0)) if cs["calibrate"] else 0.0
		name = (self.trait_by_id(k) or {}).get("name", k)
		self.trace_ctx = f"alignment: what '{name}' moves on its own"
		self.log(f"measuring what '{name}' moves on its own (alone at {TG.PROBE:+.2f}, {len(PR.QUICK)} prompts)…")
		sc = self.scores_with(LV.combine(self.levers, {k: TG.PROBE}, 1.0, use_gen), stop, {k: TG.PROBE})
		col = (sc - base) / TG.PROBE
		self.jcols[k] = {"sig": self.jsig(k), "ids": ids, "col": [round(float(x), 5) for x in col]}
		moved = sorted(((abs(TG.to_value(base[i] + col[i] * TG.PROBE) - TG.to_value(base[i])), ids[i]) for i in range(len(ids))), reverse=True)
		self.trace("matrix", lever=name, dose=TG.PROBE, moves={(self.trait_by_id(t) or {}).get("name", t): round(float(TG.to_value(base[ids.index(t)] + col[ids.index(t)] * TG.PROBE) - TG.to_value(base[ids.index(t)])), 3) for _d, t in moved})
		self.trace_ctx = ""
		return col

	def same_strengths(self, other):
		keys = {k for k, v in self.strengths.items() if v} | {k for k, v in (other or {}).items() if v}
		return all(abs(float(self.strengths.get(k, 0.0)) - float((other or {}).get(k, 0.0))) < 1e-6 for k in keys)

	def align_targets(self, targets, stop=None):
		self.need_model()
		ids = list(self.scorer.ids)
		self.targets = {t: max(-TG.LIMIT, min(TG.LIMIT, float(v))) for t, v in (targets or {}).items() if t in ids}
		self.persist()
		self.emit("targets", {"targets": dict(self.targets)})
		if not self.targets:
			self.log("no target left: the levers keep their doses")
			return {"targets": {}}
		K = sorted(self.targets, key=ids.index)
		names = {k: (self.trait_by_id(k) or {}).get("name", k) for k in K}
		auto = self.judge_auto()
		how = "kl" if auto else "manual"
		cap = self.dose_max()
		cal = self.coh_settings()["calibrate"]
		cur = self.summaries.get("current")
		reuse = cur is not None and cur.get("quick") and self.same_strengths(cur.get("strengths"))
		step = 1.2 if auto else 1.0
		prep = sum(4 if k not in self.levers else (3 if cal and "gen_mult" not in self.levers[k] else 0) for k in K) + ("baseline_quick" not in self.summaries) + sum(1 for k in K if not self.jcol_fresh(k, ids)) + (0 if reuse else 1)
		job = Job(self.emit, prep + 3 * step + (1.5 if auto else 0.0))
		for k in K:
			if k not in self.levers:
				job.step(4)
				if self.ensure_lever(k, stop) is None:
					check(stop)
					raise RuntimeError(f"the '{names[k]}' lever could not be built")
			if cal and "gen_mult" not in self.levers[k]:
				job.step(3)
				self.calibrate_gen(k, stop)
		if "baseline_quick" not in self.summaries:
			job.step(1)
			self.profile("baseline_quick", True, stop)
		base = self.trait_scores("baseline_quick")
		cols = []
		for k in K:
			if not self.jcol_fresh(k, ids):
				job.step(1)
			cols.append(self.jcol(k, base, stop))
		J = np.stack(cols, axis=1)
		self.persist()
		rows = [ids.index(k) for k in K]
		cur = self.summaries.get("current")
		if cur is not None and cur.get("quick") and self.same_strengths(cur.get("strengths")):
			m = self.trait_scores("current")
		else:
			job.step(1)
			self.profile("current", True, stop)
			m = self.trait_scores("current")
		if self.mix_round(stop):
			self.profile("current", True, stop)
			m = self.trait_scores("current")
		a = self.applied_doses(K)
		tv = np.array([self.targets[k] for k in K])
		J0 = J.copy()
		self.log("aligning " + ", ".join(f"{names[k]} → {self.targets[k]:+.2f}" for k in K) + f" (up to {TG.ROUNDS} rounds of a quick measure, doses within ±{cap:.2f}" + (", the judge checks the result once at the end" if auto else ", applied as set: JUDGE checks them when you ask") + ")…")
		why = ""
		lost = np.zeros(len(K), dtype=bool)
		caps = np.full(len(K), cap)
		seen = []
		ran = 0
		self.lever_source = "align"
		try:
			for r in range(TG.ROUNDS + 1):
				gap = TG.gaps(m, tv, rows)
				now = tv - gap
				seen.append({"round": r, "gap": gap, "doses": {k: float(self.strengths.get(k, 0.0)) for k in K}})
				score = [float(np.sum(h["gap"][~lost] ** 2)) for h in seen]
				worst = float(np.max(np.abs(gap[~lost]))) if (~lost).any() else 0.0
				self.trace("align", context=f"alignment round {r}", round=r, gap=float(np.max(np.abs(gap))), rows=[{"name": names[k], "target": self.targets[k], "now": float(now[j]), "dose": float(self.strengths.get(k, 0.0)), "applied": float(a[j]), "out": bool(lost[j])} for j, k in enumerate(K)])
				self.emit("align_step", {"round": r, "rounds": TG.ROUNDS, "gap": float(np.max(np.abs(gap))), "now": {k: float(now[j]) for j, k in enumerate(K)}})
				if worst <= TG.TOL:
					why = "reached" if not lost.any() else "limit"
					break
				if r == TG.ROUNDS:
					why = "rounds"
					break
				if r >= 2 and score[-1] > min(score[:-2]) + 1e-4 and score[-2] > min(score[:-2]) + 1e-4:
					why = "noisy"
					break
				want, out = TG.plan(J, m, tv, a, rows, caps, cap)
				lost |= out
				asked = np.array([float(self.strengths.get(k, 0.0)) for k in K])
				if float(np.max(np.abs(want - asked))) < 0.01:
					why = "stuck" if not lost.any() else "limit"
					break
				self.log(f"  round {r + 1}: " + ", ".join(f"{names[k]} {now[j]:+.2f}→{self.targets[k]:+.2f}, dose {want[j]:+.2f}" + ((" (max)" if caps[j] >= cap - 1e-6 else " (guard limit)") if out[j] else "") for j, k in enumerate(K)))
				job.step(step)
				ran += 1
				self.strengths.update({k: round(float(v), 3) for k, v in zip(K, want)})
				self.apply_levers(how)
				a2 = self.applied_doses(K)
				self.profile("current", True, stop)
				if self.mix_round(stop):
					self.profile("current", True, stop)
				m2 = self.trait_scores("current")
				J = TG.secant(J, a2 - a, m2 - m, J0, rows)
				cut = np.abs(a2) < np.abs(want) - 0.02
				if cut.any():
					caps = np.where(cut, np.minimum(caps, np.maximum(np.abs(a2), 0.05)), caps)
					self.log("  the KL guard cut the doses: " + ", ".join(f"{names[k]} {want[j]:+.2f}→{a2[j]:+.2f}" for j, k in enumerate(K) if cut[j]) + " · they stay within that from now on")
				a, m = a2, m2
			fit = lambda g: float(np.sum(g[~lost] ** 2) + 0.1 * np.sum(g[lost] ** 2))
			gap = TG.gaps(m, tv, rows)
			best = min(seen, key=lambda h: fit(h["gap"]))
			if fit(gap) > fit(best["gap"]) + 1e-3:
				self.log(f"  the closest result was round {best['round']}: its doses are put back")
				job.step(step)
				self.strengths.update({k: round(v, 3) for k, v in best["doses"].items()})
				self.apply_levers(how)
				a = self.applied_doses(K)
				self.profile("current", True, stop)
				m = self.trait_scores("current")
		finally:
			self.lever_source = ""
		if auto and ran:
			job.step(1.5)
			before = self.applied_doses(K)
			self.apply_levers("full")
			a = self.applied_doses(K)
			if float(np.max(np.abs(a - before))) > 0.02:
				self.log("  the judge cut the doses found: measuring again")
				job.step(1)
				self.profile("current", True, stop)
				m = self.trait_scores("current")
		job.finish()
		now = TG.to_value(m[rows])
		report = []
		for j, k in enumerate(K):
			dose = float(self.strengths.get(k, 0.0))
			guarded = abs(dose) - abs(float(a[j])) > 0.05 or (bool(lost[j]) and caps[j] < cap - 1e-6)
			capped = not guarded and (abs(dose) >= cap - 1e-3 or bool(lost[j]))
			report.append({"id": k, "name": names[k], "target": self.targets[k], "now": float(now[j]), "dose": dose, "applied": float(a[j]), "ok": abs(self.targets[k] - float(now[j])) <= TG.TOL, "capped": capped, "guarded": guarded})
		miss = [x for x in report if not x["ok"]]
		text = ", ".join(f"{x['name']} {x['now']:+.2f} (target {x['target']:+.2f}, dose {x['dose']:+.2f})" for x in report)
		if not miss:
			self.log(f"targets reached: {text}", "ok")
		else:
			cause = {"stuck": "the levers no longer move the measure", "rounds": "ALIGN again can get closer", "noisy": "two rounds without progress: the measure is noisy for these levers on this model"}.get(why, "")
			self.log(f"targets partly reached: {text}" + "".join(f" · {x['name']}: " + (f"out of reach even at the maximum dose (±{cap:.2f}, settings)" if x["capped"] else ("held back by the coherence guard" if x["guarded"] else "not reached yet")) for x in miss) + (f" ({cause})" if cause else "") + ("" if ran else " · no dose changed"), "warn")
		self.persist()
		res = {"targets": dict(self.targets), "report": report, "why": why, "rounds": ran, "dose_max": cap}
		self.emit("aligned", _jsonable(res))
		return res

	def lora_dose(self, tid):
		lv = self.levers.get(tid) or {}
		if lv.get("carrier") == "both" and GO.lora_key(tid) in self.strengths:
			return float(self.strengths[GO.lora_key(tid)])
		return float(self.strengths.get(tid, 0.0))

	def set_exempt(self, tid, on):
		ex = set(self.settings.get("guard_exempt") or [])
		if on:
			ex.add(tid)
		else:
			ex.discard(tid)
		self.set_setting("guard_exempt", sorted(ex))
		name = (self.trait_by_id(tid) or {}).get("name", tid)
		self.log(f"'{name}': " + ("the coherence guard now ignores this lever (judge overridden)" if on else "back under the coherence guard"))
		if self.engine is not None and self.engine.loaded and self.strengths.get(tid):
			self.apply_levers("manual" if on else "auto")
		return sorted(ex)

	def lora_path(self, tid):
		return os.path.join(self.state_dir(), self.levers[tid]["lora"]["file"])

	def guard_info(self):
		_active, _free, guarded = self.split_levers()
		return {"scale": self.guard_scale, "kl": self.coherence_kl, "factors": dict(self.blame), "free": list(self.guard_free), "auto": self.judge_auto(), "unjudged": sorted(self.unjudged(guarded)), "dose_max": self.dose_max(), **(self.coherence_info or {})}

	def recalibrate_levers(self, ids=None, stop=None):
		ids = [t for t in (ids or list(self.levers)) if t in self.levers and self.levers[t].get("kind") != "optimized"]
		if not ids:
			self.log("no lever to recalibrate yet")
			return
		if not self.coh_settings()["calibrate"]:
			self.log("calibration on generations is off in the settings: nothing to recalibrate", "warn")
			return
		for tid in ids:
			self.levers[tid].pop("gen_mult", None)
			self.levers[tid].pop("gen_by", None)
		self.log(f"recalibrating {len(ids)} lever(s) with the current judge…")
		self.calibrate_all(ids, stop)

	def add_trait(self, name, seeds_plus, seeds_minus, plus="", minus="", desc_plus="", desc_minus="", images_plus=None, images_minus=None, poles_plus=None, poles_minus=None):
		t = make_trait(name, seeds_plus, seeds_minus, plus, minus, taken={d["id"] for d in builtin_definitions()})
		for k, v in (("desc_plus", desc_plus), ("desc_minus", desc_minus)):
			if (v or "").strip():
				t[k] = v.strip()
		self.set_images(t, {"images_plus": images_plus, "images_minus": images_minus, "poles_plus": poles_plus, "poles_minus": poles_minus})
		self.levers.pop(t["id"], None)
		self.strengths.pop(t["id"], None)
		custom = [c for c in load_custom() if c["id"] != t["id"]] + [t]
		save_custom(custom)
		self.reload_traits()
		return t

	def goal_as_lever(self, gid):
		res = self.goal_lever_def(gid)
		if res["move"]:
			self.goal_lever_move(gid)
		return res

	def goal_lever_def(self, gid):
		goal = self.goal(gid)
		lid = GO.lever_id(goal)
		custom = load_custom()
		if any(c.get("from_goal") == gid for c in custom):
			raise RuntimeError(f"the goal '{goal['name']}' already has a lever")
		src = GO.as_trait(goal)
		t = make_trait(goal["name"], [], [], src["plus"], src["minus"], taken={d["id"] for d in self.defs} | {d["id"] for d in builtin_definitions()} | {lid})
		t.update(system_plus=src["system_plus"], system_minus=src["system_minus"], prompts=list(src["prompts"]), from_goal=gid)
		save_custom(custom + [t])
		self.defs = load_definitions()
		lv = self.levers.get(lid)
		move = lv is not None and lv.get("kind") != "optimized"
		self.log(f"goal '{goal['name']}': its character is now also a regular lever" + (" (the persona direction it has on this model moves to it next)" if move else " (built the first time you move it)"), "ok")
		return {"lever": t["id"], "goal": gid, "name": goal["name"], "move": move, "extra_levers": self.extra_rows()}

	def goal_lever_move(self, gid):
		goal = self.goal(gid)
		lid = GO.lever_id(goal)
		t = next((d for d in self.defs if d.get("from_goal") == gid), None)
		lv = self.levers.get(lid)
		if t is None or lv is None or lv.get("kind") == "optimized" or t["id"] in self.levers:
			return None
		self.levers[t["id"]] = dict(self.levers.pop(lid), sig=signature(t))
		v = self.strengths.pop(lid, 0.0)
		if v:
			self.strengths[t["id"]] = v
		self.jcols.pop(lid, None)
		self.read_gains.pop(lid, None)
		self.emit("lever_dropped", {"id": lid})
		if self.engine is not None and self.engine.loaded:
			self.persist()
		self.log(f"goal '{goal['name']}': the persona direction it had on this model moved to its lever (the goal is active again only through TRAIN or ALIGN)", "ok")
		self.emit("loaded", self.snapshot())
		return t["id"]

	def reload_traits(self):
		self.defs = load_definitions()
		if self.engine and self.engine.loaded:
			self.lexicons = build_lexicons(self.engine, self.readout, self.defs, self.english)
			self.scorer = TraitScorer(self.readout, self.defs, self.lexicons)
			self._axis_best()
			self.keep_answers()
			self.reference = None
			self.summaries = {}
			self.read_plain = None
			self.persist()
		self.emit("loaded", self.snapshot())

	def update_trait(self, tid, values):
		custom = load_custom()
		t = next((c for c in custom if c["id"] == tid), None)
		if t is None:
			base = next((d for d in builtin_definitions() if d["id"] == tid), None)
			if base is None:
				raise RuntimeError(f"unknown lever '{tid}'")
			ov = load_overrides()
			e = edited_fields(base, {**(ov["edited"].get(tid) or {}), **values})
			if e:
				ov["edited"][tid] = e
			else:
				ov["edited"].pop(tid, None)
			save_overrides(ov)
			t = {**base, **e}
		else:
			img = bool(values.get("images_plus") or values.get("images_minus"))
			for k in ("name", "plus", "minus", "desc_plus", "desc_minus"):
				if values.get(k) is not None and (k == "name" or not img):
					t[k] = str(values[k]).strip()
			for k in ("seeds_plus", "seeds_minus"):
				if values.get(k) is not None and not img:
					t[k] = [s.strip() for s in values[k] if s.strip()]
			self.set_images(t, values)
			save_custom(custom)
		had = self.strengths.get(tid, 0.0)
		self.drop_lever(tid)
		self.reload_traits()
		self.log(f"lever '{t['name']}' edited" + (": its strength was reset, set it again to rebuild the lever" if had else ""))
		return t

	def duplicate_trait(self, tid, values=None):
		src = next((d for d in self.defs if d["id"] == tid), None)
		if src is None:
			raise RuntimeError(f"unknown lever '{tid}'")
		v = {k: src.get(k) for k in ("name", "plus", "minus", "seeds_plus", "seeds_minus", "desc_plus", "desc_minus")}
		v.update({k: x for k, x in (values or {}).items() if k in v and x})
		name = (v["name"] if values and values.get("name") else f"{src['name']} copy").strip()
		t = make_trait(name, v.get("seeds_plus") or [], v.get("seeds_minus") or [], v.get("plus") or "", v.get("minus") or "", taken={d["id"] for d in self.defs} | {d["id"] for d in builtin_definitions()})
		for k in ("desc_plus", "desc_minus"):
			if v.get(k):
				t[k] = v[k]
		for k in ("system_plus", "system_minus", "prompts"):
			if src.get(k):
				t[k] = list(src[k]) if isinstance(src[k], list) else src[k]
		self.set_images(t, values if values and ("images_plus" in values or "images_minus" in values) else {**{f"images_{p}": IW.image_paths(src, p) for p in ("plus", "minus")}, **{f"poles_{p}": IW.pole_paths(src, p) for p in ("plus", "minus")}})
		save_custom([c for c in load_custom() if c["id"] != t["id"]] + [t])
		self.reload_traits()
		self.log(f"lever '{src['name']}' duplicated as '{t['name']}'")
		return t

	def set_images(self, t, values):
		touched = False
		for pole in ("plus", "minus"):
			k = f"images_{pole}"
			if values.get(k) is None:
				continue
			touched = True
			pairs = IW.store_images(t["id"], pole, values[k])
			marked = {os.path.normcase(os.path.abspath(p)) for p in values.get(f"poles_{pole}") or []}
			names = [n for _s, n in pairs]
			poles = [n for s, n in pairs if os.path.normcase(s) in marked]
			for key, val in ((k, names), (f"poles_{pole}", poles)):
				if val:
					t[key] = val
				else:
					t.pop(key, None)
		if touched and IW.has_images(t):
			self.image_seeds(t)

	def image_seeds(self, t):
		for k in ("desc_plus", "desc_minus", "system_plus", "system_minus", "prompts"):
			t.pop(k, None)
		words = {"plus": [], "minus": []}
		if not all(os.path.isfile(p) for pole in ("plus", "minus") for p in IW.image_paths(t, pole)):
			self.log(f"'{t['name']}': some of its images are gone, its words are kept as they were", "warn")
			return
		if self.mmproj() and self.readout is not None:
			try:
				words = self.image_words(IW.image_paths(t, "plus"), IW.image_paths(t, "minus"), "after")
				self.save_image_reads(t, IW.extract_layers(self.band or (0, self.engine.n_layer - 1), self.engine.n_layer))
			except Exception as exc:
				self.log(f"'{t['name']}': the words of its images could not be read ({exc}): it steers without being measured", "warn")
		else:
			self.log(f"'{t['name']}': no model that sees is loaded, its images give no words yet: it steers without being measured", "warn")
		t["seeds_plus"], t["seeds_minus"] = list(words.get("plus") or []), list(words.get("minus") or [])
		t["plus"] = t["seeds_plus"][0] if t["seeds_plus"] else t["name"]
		t["minus"] = t["seeds_minus"][0] if t["seeds_minus"] else "not " + t["name"]

	def delete_trait(self, tid):
		custom = load_custom()
		if any(c["id"] == tid for c in custom):
			save_custom([c for c in custom if c["id"] != tid])
			self.drop_lever(tid)
			self.reload_traits()
			self.log(f"lever '{tid}' deleted")
			return
		d = next((d for d in self.defs if d["id"] == tid), None)
		if d is None:
			raise RuntimeError(f"unknown lever '{tid}'")
		ov = load_overrides()
		if tid not in ov["hidden"]:
			ov["hidden"].append(tid)
		save_overrides(ov)
		self.drop_lever(tid)
		self.reload_traits()
		self.log(f"default lever '{d['name']}' removed (right-click any lever → RESTORE A REMOVED LEVER brings it back)")

	def restore_trait(self, tid):
		ov = load_overrides()
		edited, hidden = tid in ov["edited"], tid in ov["hidden"]
		if not edited and not hidden:
			return
		ov["edited"].pop(tid, None)
		ov["hidden"] = [h for h in ov["hidden"] if h != tid]
		save_overrides(ov)
		if edited:
			self.drop_lever(tid)
		self.reload_traits()
		name = next((d["name"] for d in self.defs if d["id"] == tid), tid)
		self.log(f"default lever '{name}' restored" + (" as shipped (its edits are dropped)" if edited else ""))

	def save_preset(self, name, strengths, system="", targets=None, mode="dose"):
		ids = set(strengths)
		targets = {t: float(v) for t, v in (self.targets if targets is None else targets).items()}
		custom = {c["id"]: c for c in load_custom()}
		goals = [g for g in self.goals if GO.lever_id(g) in ids]
		data = {"version": 1, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "model": os.path.basename((self.info or {}).get("path", "")), "mode": mode if mode in ("dose", "target") else "dose", "strengths": {t: float(v) for t, v in strengths.items() if abs(float(v)) > 1e-9}, "targets": targets, "system": system or "", "traits": [custom[t] for t in sorted(ids | set(targets)) if t in custom], "goals": goals, "exempt": [t for t in (self.settings.get("guard_exempt") or []) if t in ids], "inactive": sorted(self.inactive())}
		path = PS.save(data)
		self.log(f"preset '{name}' saved ({data['mode'].upper()} mode): {len(data['strengths'])} lever(s)" + (f", {len(targets)} target(s)" if targets else "") + f", {len(goals)} goal(s)" + (", system prompt" if system else "") + f" → {path}", "ok")
		return {"path": path, "name": name}

	def prepare_preset(self, data):
		have = {d["id"] for d in self.defs}
		custom = load_custom()
		mine = {c["id"] for c in custom}
		added = [t for t in data.get("traits") or [] if t.get("id") and t["id"] not in have and t["id"] not in mine]
		if added:
			save_custom(custom + added)
			self.reload_traits()
		g_added, g_kept = [], []
		for g in data.get("goals") or []:
			old = next((x for x in self.goals if x["id"] == g.get("id")), None)
			if old is None:
				self.goals.append(g)
				g_added.append(g["name"])
			elif any(old.get(k) != g.get(k) for k in GO.KEYS + ("custom",)):
				g_kept.append(g["name"])
		if g_added:
			GO.save(self.goals)
			self.emit_goals()
		ex = set(self.settings.get("guard_exempt") or [])
		if set(data.get("exempt") or []) - ex:
			self.set_setting("guard_exempt", sorted(ex | set(data.get("exempt") or [])))
		known = {d["id"] for d in self.defs} | {GO.lever_id(g) for g in self.goals}
		wanted = {t: float(v) for t, v in (data.get("strengths") or {}).items()}
		missing = sorted(t for t in wanted if GO.base_id(t) not in known)
		ids = {d["id"] for d in self.trait_defs()}
		targets = {t: float(v) for t, v in (data.get("targets") or {}).items() if t in ids} if "targets" in data else None
		return {"name": data.get("name", "?"), "mode": data.get("mode") if data.get("mode") in ("dose", "target") else None, "strengths": {t: v for t, v in wanted.items() if GO.base_id(t) in known}, "targets": targets, "system": data.get("system") or "", "added": [t["name"] for t in added], "goals_added": g_added, "goals_kept": g_kept, "missing": missing, "exempt": sorted(set(data.get("exempt") or []) - ex), "inactive": list(data.get("inactive") or []) if "inactive" in data else None}

	def trait_by_id(self, tid):
		d = next((d for d in self.defs if d["id"] == tid), None)
		if d is not None:
			return d
		g = next((g for g in self.goals if GO.lever_id(g) == tid), None)
		return GO.as_trait(g) if g else None

	def goal(self, gid):
		g = next((g for g in self.goals if g["id"] == gid), None)
		if g is None:
			raise RuntimeError(f"unknown goal '{gid}'")
		return g

	def goal_rows(self):
		return [{"id": GO.lever_id(g), "goal": g["id"], "name": g["name"], "plus": g["name"], "minus": "default assistant", "custom": True, "lexicon": {}} for g in self.goals]

	def emit_goals(self):
		self.emit("goals", {"goals": self.goals, "rows": self.goal_rows()})

	def drop_lever(self, tid):
		had = bool(self.strengths.get(tid)) or bool(self.strengths.get(GO.lora_key(tid)))
		self.levers.pop(tid, None)
		self.judged.pop(tid, None)
		self.strengths.pop(tid, None)
		self.strengths.pop(GO.lora_key(tid), None)
		self.emit("lever_dropped", {"id": tid})
		if self.engine is None or not self.engine.loaded:
			return
		if had:
			self.apply_levers()
		else:
			self.persist()

	def save_goal(self, values):
		g = GO.make(values.get("name"), values.get("who"), values.get("want"), values.get("me"), values.get("id") or "", values.get("custom"))
		old = next((x for x in self.goals if x["id"] == g["id"]), None)
		self.goals = [g if x["id"] == g["id"] else x for x in self.goals] if old else self.goals + [g]
		GO.save(self.goals)
		if old and any(old.get(k) != g.get(k) for k in GO.KEYS):
			self.goal_results.pop(g["id"], None)
			self.drop_lever(GO.lever_id(g))
		self.emit_goals()
		return g

	def delete_goal(self, gid):
		g = self.goal(gid)
		self.goals = [x for x in self.goals if x["id"] != gid]
		GO.save(self.goals)
		self.goal_results.pop(gid, None)
		self.drop_lever(GO.lever_id(g))
		self.emit_goals()

	def store_goal(self, values):
		g = GO.make(values.get("name"), values.get("who"), values.get("want"), values.get("me"), values.get("id") or "", values.get("custom"))
		old = next((x for x in self.goals if x["id"] == g["id"]), None)
		self.goals = [g if x["id"] == g["id"] else x for x in self.goals] if old else self.goals + [g]
		GO.save(self.goals)
		drop = None
		if old and any(old.get(k) != g.get(k) for k in GO.KEYS):
			self.goal_results.pop(g["id"], None)
			drop = GO.lever_id(g) if GO.lever_id(g) in self.levers else None
		self.emit_goals()
		self.log(f"goal '{g['name']}' saved" + (": its lever was made for the old text, it is dropped at the next step of TO DO" if drop else ""))
		return {"goal": g, "drop": drop, "name": g["name"]}

	def forget_goal(self, gid):
		g = self.goal(gid)
		self.goals = [x for x in self.goals if x["id"] != gid]
		GO.save(self.goals)
		self.goal_results.pop(gid, None)
		self.emit_goals()
		lid = GO.lever_id(g)
		self.log(f"goal '{g['name']}' deleted")
		return {"drop": lid if lid in self.levers else None, "name": g["name"]}

	def need_model(self):
		if not self.engine or not self.engine.loaded:
			raise RuntimeError("load a model first")

	def check_goal(self, gid, stop=None, compare=None):
		self.need_model()
		goal = self.goal(gid)
		judge = self.ensure_judge()
		compare = bool(self.settings.get("goal_compare", False)) if compare is None else bool(compare)
		vectors = self.active_vectors()
		n = len(list(GO.items(goal, GO.LANGS)))
		self.log(f"goal '{goal['name']}' on {self.info['name']}: asking {n} questions" + (" with the active levers" if self.active_vectors() else "") + (f" · judge {judge.name}" if judge else " · no judge: answers only"))
		rows = []

		def on_row(r):
			r["model"] = self.info["name"]
			rows.append(r)
			self.emit("goal_row", {"goal": gid, "row": r})
			self.emit("progress", {"task": f"goal:{gid}", "value": len(rows) / max(n, 1)})

		self.resteer()
		reader = self.goal_reader(goal, stop) if GO.read_settings(self.settings)["pull"] else None
		res = GO.check(self.engine, judge, goal, self.system_prompt, stop=stop, on_row=on_row, vectors=vectors if compare and vectors else None, compare_base=compare and bool(vectors), reader=reader)
		self.resteer()
		return self.goal_result(goal, res) if res is not None else None

	def goal_reads(self, goal, stop=None):
		key = (self.info["path"], json.dumps([goal.get(k) for k in GO.KEYS] + [goal.get("custom") or []], ensure_ascii=False))
		cache = getattr(self, "pull_cache", None)
		if cache and cache[0] == key:
			return cache[1]
		sysp = GO.system(goal)
		out = {}
		try:
			for _k, _l, q, t in GO.items(goal, GO.LANGS):
				check(stop)
				toks = self.engine.tokenize(t, parse_special=False)
				out[(q, t)] = (toks, OPT.logprob(self.engine, q, toks), OPT.logprob(self.engine, q, toks, system=sysp))
		finally:
			self.resteer()
		self.pull_cache = (key, out)
		return out

	def goal_reader(self, goal, stop=None):
		reads = self.goal_reads(goal, stop)

		def read(q, t):
			toks, bare, twin = reads.get((q, t)) or (self.engine.tokenize(t, parse_special=False), 0.0, 0.0)
			try:
				lp = OPT.logprob(self.engine, q, toks, system=self.system_prompt, vectors=self.active_vectors())
			finally:
				self.resteer()
			return float(lp - bare), float(twin - bare)
		return read

	def goal_pull(self, goal, stop=None):
		read = self.goal_reader(goal, stop)
		pr = [read(q, t) for _k, _l, q, t in GO.items(goal, GO.LANGS)]
		return GO.pull_share(sum(p for p, _t in pr), sum(t for _p, t in pr)), float(np.mean([p for p, _t in pr])) if pr else 0.0

	def goal_result(self, goal, res, **extra):
		lid = GO.lever_id(goal)
		res = dict(res, goal=goal["id"], model=self.info["name"], strength=self.strengths.get(lid, 0.0), scale=self.guard_scale, judge=self.judge.name if self.judge else None, levers={k: v for k, v in self.strengths.items() if v and k != lid}, **extra)
		self.goal_results[goal["id"]] = res
		self.emit("goal_result", _jsonable(res))
		if res.get("match") is not None:
			self.log(f"goal '{goal['name']}': match {res['match']:.0%} · sense {res['sense']:.0%}" + (f" · {res['nonsense']} nonsense answer(s)" if res["nonsense"] else "") + (f" · goal lever {res['strength']:+.2f}" if res["strength"] else ""), "info" if res["match"] >= 0.5 else "warn")
		return res

	def align_goal(self, gid, stop=None):
		self.need_model()
		goal = self.goal(gid)
		judge = self.ensure_judge()
		rs = GO.read_settings(self.settings)
		mode = rs["align"] if judge is not None or rs["align"] != "judge" else "judge"
		if judge is None and mode == "judge":
			raise RuntimeError("aligning needs a judge to score the answers: pick one in the settings (⚙ · coherence), or ALIGN BY READ in the GOALS options")
		if judge is None and mode == "read_judge":
			mode = "read"
		lid = GO.lever_id(goal)
		start = self.strengths.get(lid, 0.0)
		if self.ensure_lever(lid, stop) is None:
			return None
		if self.coh_settings()["calibrate"] and "gen_mult" not in self.levers[lid]:
			self.calibrate_gen(lid, stop)
		if mode != "judge":
			return self.align_goal_read(goal, lid, start, judge, mode, rs, stop)
		steps = (0.0,) + GO.GRID
		trials = []
		for i, s in enumerate(steps):
			if stop is not None and stop.is_set():
				break
			self.strengths[lid] = s
			self.apply_levers()
			self.log(f"goal '{goal['name']}' on {self.info['name']}: trying the goal lever at {s:+.2f}" + (f" (guard ×{self.guard_scale:.2f})" if s and self.guard_scale < 0.999 else "") + "…")
			res = GO.check(self.engine, judge, goal, self.system_prompt, stop=stop)
			if res is None:
				break
			res.update(strength=s, scale=self.guard_scale, model=self.info["name"])
			for r in res.get("rows") or []:
				r["model"] = self.info["name"]
			trials.append(res)
			self.emit("goal_trial", _jsonable({"goal": gid, **res}))
			self.emit("progress", {"task": f"goal:{gid}", "value": (i + 1) / len(steps)})
			if s and not GO.sensible(res):
				break
		stopped = stop is not None and stop.is_set()
		best = None if stopped else GO.pick(trials)
		final = best["strength"] if best else start
		self.strengths[lid] = final
		self.apply_levers()
		summary = [{"strength": t["strength"], "scale": t["scale"], "match": t["match"], "sense": t["sense"], "nonsense": t["nonsense"]} for t in trials]
		self.emit("goal_aligned", {"goal": gid, "lever": lid, "strength": final, "found": best is not None, "stopped": stopped, "trials": _jsonable(summary)})
		if stopped:
			self.log(f"goal '{goal['name']}': alignment stopped, goal lever back to {final:+.2f}", "warn")
		elif best is None:
			self.log(f"goal '{goal['name']}': no strength kept the answers sensible, goal lever back to {final:+.2f}", "warn")
		else:
			base = trials[0]["match"] if trials and trials[0]["match"] is not None else None
			self.log(f"goal '{goal['name']}': goal lever {final:+.2f} → match {best['match']:.0%}" + (f" (without it {base:.0%})" if base is not None else "") + f" · sense {best['sense']:.0%}", "ok" if best["match"] >= 0.5 else "warn")
		if best is not None and best["scale"] == self.guard_scale:
			return self.goal_result(goal, best, trials=summary)
		return self.check_goal(gid, stop) if not stopped else None

	def align_goal_read(self, goal, lid, start, judge, mode, rs, stop=None):
		gid = goal["id"]
		cs = self.coh_settings()
		ok, bad = float(cs["read_pass"]), float(cs["read_fail"])
		reach = float(rs["reach"])
		halvings = max(1, int(rs["halvings"]))
		verify = int(rs["verify"]) if mode == "read_judge" else 0
		self.log(f"goal '{goal['name']}' on {self.info['name']}: seeking the strength of the goal lever by reading: from 1, doubled until it pulls {reach:.0%} of the character's pull, pulls less than before or the replies break (reading drop ≥ {bad:.2f}), then {halvings} step{'s' if halvings != 1 else ''} between the last two" + (" · then the judge checks the best ones" if mode == "read_judge" else ""))
		scan = []
		total = GO.SEEK_DOUBLINGS + halvings + verify
		stopped = False
		coh = self.coherence()

		def probe(s):
			check(stop)
			self.strengths[lid] = s
			self.apply_levers("manual")
			pull, raw = self.goal_pull(goal, stop)
			drop = coh.read_drop(self.active_vectors(), ok, bad)
			scan.append({"strength": round(s, 4), "pull": pull, "pull_raw": raw, "drop": drop, "broken": drop >= bad})
			self.emit("goal_scan", _jsonable({"goal": gid, "scan": scan, "done": False}))
			self.emit("progress", {"task": f"goal:{gid}", "value": min(1.0, len(scan) / total)})
			return scan[-1]

		seek, how = 0.0, ""
		try:
			self.goal_reads(goal, stop)
			seek, how = GO.seek(probe, reach, halvings)
		except Stopped:
			stopped = True
			seek = 0.0
		self.emit("goal_scan", _jsonable({"goal": gid, "scan": scan, "done": True}))
		self.trace("goal_scan", context=f"goal alignment by reading: {goal['name']} · {self.info['name']}", goal=goal["name"], scan=scan, mode=mode)
		trials = []
		best = None
		if mode == "read_judge" and seek > 0:
			for j, s in enumerate(GO.verify_strengths(seek, verify)):
				if stop is not None and stop.is_set():
					stopped = True
					break
				self.strengths[lid] = s
				self.apply_levers()
				self.log(f"goal '{goal['name']}' on {self.info['name']}: the judge checks the goal lever at {s:+.2f}" + (f" (guard ×{self.guard_scale:.2f})" if self.guard_scale < 0.999 else "") + "…")
				res = GO.check(self.engine, judge, goal, self.system_prompt, stop=stop, reader=self.goal_reader(goal, stop))
				self.resteer()
				if res is None:
					stopped = True
					break
				res.update(strength=s, scale=self.guard_scale, model=self.info["name"])
				for r in res.get("rows") or []:
					r["model"] = self.info["name"]
				trials.append(res)
				self.emit("goal_trial", _jsonable({"goal": gid, **res}))
				self.emit("progress", {"task": f"goal:{gid}", "value": min(1.0, (len(scan) + j + 1) / total)})
			best = None if stopped else GO.pick(trials)
			chosen = best["strength"] if best else 0.0
		else:
			chosen = seek
		found = chosen > 0 and not stopped
		final = chosen if found else start
		lv = self.levers.get(lid) or {}
		fold = found and not lv.get("lora") and abs(chosen - 1.0) > GO.FOLD_MIN
		if fold:
			lv["gain"] = float(lv["gain"]) * chosen
			final = 1.0
			self.emit("lever", _jsonable({"id": lid, "layers": lv["layers"], "gain": lv["gain"], "gen_mult": lv.get("gen_mult"), "reads_as": lv.get("reads_as", []), "reads_as_minus": lv.get("reads_as_minus", []), "sep": {}, "kind": lv.get("kind", "persona"), "carrier": lv.get("carrier"), "closed": lv.get("closed"), "rank": None}))
		self.strengths[lid] = final
		self.apply_levers()
		if fold:
			self.persist()
		summary = [{"strength": t["strength"], "scale": t["scale"], "match": t["match"], "sense": t["sense"], "nonsense": t["nonsense"], "pull": t.get("pull")} for t in trials]
		self.emit("goal_aligned", {"goal": gid, "lever": lid, "strength": final, "chosen": chosen, "fold": chosen if fold else None, "found": found, "stopped": stopped, "trials": _jsonable(summary), "scan": _jsonable(scan), "mode": mode})
		if stopped:
			self.log(f"goal '{goal['name']}': alignment stopped, goal lever back to {final:+.2f}", "warn")
			return None
		top_pull = max((r["pull"] for r in scan if r["pull"] is not None and not r["broken"]), default=None)
		if not found:
			self.log(f"goal '{goal['name']}': " + ("no strength kept the answers sensible" if mode == "read_judge" and seek > 0 else "the goal lever does not pull the model toward the targets before the replies break") + f", goal lever back to {final:+.2f}", "warn")
		else:
			at = next((r for r in scan if abs(r["strength"] - chosen) < 1e-6), None)
			self.log(f"goal '{goal['name']}': goal lever ×{chosen:.2f}" + {"reach": " (smallest strength that reaches REACH)", "past": " (strongest pull before it fades or breaks)", "rising": " (the pull was still rising at the last strength tried)"}.get(how, "") + (f" → pull {at['pull']:.0%} of the character's (reading drop {at['drop']:.2f})" if at and at["pull"] is not None else "") + (f", most {top_pull:.0%}" if top_pull is not None else "") + (f" · match {best['match']:.0%} · sense {best['sense']:.0%}" if best else "") + (f" · folded into the lever: dose 1 is now this strength" if fold else ""), "ok")
		if best is not None and best["scale"] == self.guard_scale and not fold:
			return self.goal_result(goal, best, trials=summary, scan=scan)
		return self.check_goal(gid, stop)

	def free_torch(self):
		if self.torch_net is None:
			return
		self.torch_net = None
		gc.collect()
		torch.cuda.empty_cache()

	def torch_model(self, first=1, cfg=None):
		path = self.info["path"]
		if self.judge_engine is not None and self.judge_engine.loaded:
			self.log("unloading the judge while training (it reloads on the next check)")
			self.judge_engine.unload()
			self.judge = None
			if self.coh is not None:
				self.coh.judge = None
			self.set_judge_state(state="off", name=os.path.basename(self.coh_settings()["judge"] or ""), reason="unloaded for training")
		need = self.train_need(first, cfg)
		if self.torch_net is not None and self.torch_net[0] == (path, need["mode"]):
			return self.torch_net[1]
		self.free_torch()
		free = self.train_free()
		plan = torchload.train_plan(need, free)
		ideal = torchload.train_plan(need, self.train_room())
		if self.slower(plan, ideal) and self.engine.loaded:
			self.log("setting the llama.cpp copy of the model aside while training (reloaded right after)", "warn")
			self.engine.unload()
			self.parked = True
			gc.collect()
			torch.cuda.empty_cache()
			free = self.train_free()
			plan = torchload.train_plan(need, free)
		if self.slower(plan, ideal) and self.readout is not None:
			moved = self.readout.park()
			if moved:
				self.log(f"moving the lens readout to the CPU while training ({moved / 2**30:.1f} GB)", "warn")
				gc.collect()
				torch.cuda.empty_cache()
				free = self.train_free()
				plan = torchload.train_plan(need, free)
		if plan is None:
			self.unpark()
			raise RuntimeError(self.train_short(need, free))
		self.log(f"loading a trainable copy of the model in torch ({(plan['weights'] - plan['ram']) / 2**30:.1f} GB " + ("of frozen weights kept quantized, unpacked on the fly" if plan["quant"] else "bf16") + " on " + " + ".join(f"CUDA{i}" for i, v in free.items() if v > 0) + (f", its first {plan['cpu_layers']} layers in RAM ({plan['ram'] / 2**30:.1f} GB, before the trained ones)" if plan["cpu_layers"] else "") + (", gradient checkpointing" if plan["checkpointing"] else "") + (f"; steps about {plan['slower']:.1f} times slower than a full bf16 copy" if plan["slower"] > 1.05 else "") + ")…")
		t = time.time()
		model = torchload.load(path, max_memory=plan["room"], llama_dir=self.llama_dir(), train=plan)
		k = getattr(model, "cpu_layers", 0)
		if k > plan["cpu_layers"]:
			self.log(f"{k} layers kept in RAM to fit the GPUs", "warn")
		if plan["checkpointing"] and not getattr(model, "checkpointing", False):
			self.log("gradient checkpointing unavailable for this model: training with the memory it needs", "warn")
		if plan["quant"]:
			count, saved, moved = (tuple(getattr(model, "quantized", ())) + (0, 0, 0))[:3]
			self.log(f"{count} weight matrices kept quantized ({saved / 2**30:.1f} GB saved" + (f", {moved} of them reordered on the fly" if moved else "") + ")")
		self.log(f"trainable copy ready ({time.time() - t:.0f} s)")
		self.torch_net = ((path, need["mode"]), model)
		return model

	def slower(self, plan, ideal):
		return ideal is not None and (plan is None or ideal["cost"] < plan["cost"] * 0.98)

	def train_need(self, first, cfg=None):
		return torchload.need(self.engine.gguf, self.engine.n_layer, int(self.info["n_embd"]), int(first), {**OPT.DEFAULTS, **(cfg or {})})

	def reserves(self):
		n = torch.cuda.device_count() if torch.cuda.is_available() else 0
		return [m * 2**20 for m in reserve_mibs(n, int(self.settings.get("vram_reserve_mb", 900)))]

	def train_free(self):
		if not torch.cuda.is_available():
			return {}
		res = self.reserves()
		return {i: max(0, torch.cuda.mem_get_info(i)[0] - res[i]) for i in range(torch.cuda.device_count())}

	def train_room(self):
		free = self.train_free()
		est = (self.plan or {}).get("estimate") or {}
		if self.engine is not None and self.engine.loaded:
			for i in free:
				free[i] += sum((est.get(f"CUDA{i}") or {}).values())
		if self.readout is not None and self.readout.device.type == "cuda" and not getattr(self.readout, "parked", None):
			i = self.readout.device.index or 0
			if i in free:
				free[i] += self.readout.nbytes()
		if self.judge_engine is not None and self.judge_engine.loaded and self.judge_engine.info:
			i = int(self.judge_state.get("gpu", 0) or 0)
			if i in free:
				free[i] += os.path.getsize(self.judge_engine.info["path"]) + 700 * 2**20
		return free

	def train_short(self, need, free):
		have = sum(free.values())
		least, quant = torchload.least(need, len(free))
		return f"not enough VRAM to train on this model: it needs at least {least / 2**30:.1f} GB on the GPUs ({'quantized' if quant else 'bf16'} weights {need['qweights' if quant else 'weights'] / 2**30:.1f} GB, with the {need['first']} layers before the trained ones kept in RAM and gradient checkpointing), {have / 2**30:.1f} GB can be used (" + ", ".join(f"CUDA{i} {v / 2**30:.1f}" for i, v in free.items()) + ", after the VRAM reserve of the display GPUs): free some VRAM or use a smaller model"

	def unpark(self):
		if self.readout is not None:
			self.readout.unpark()
		if not getattr(self, "parked", False):
			return
		self.parked = False
		self.log("reloading the llama.cpp copy of the model…")
		self.engine.load(self.info["path"], self.plan, n_ctx=self.info.get("n_ctx") or 8192)
		self.resteer()

	def optimize_goal(self, gid, options=None, stop=None):
		self.need_model()
		goal = self.goal(gid)
		arch = self.engine.gguf.arch
		if not torchload.supported(arch):
			raise RuntimeError(f"training a goal lever needs a trainable copy of the model, and '{arch}' GGUF files cannot be loaded in torch yet: use ALIGN")
		experts = int(self.engine.gguf.arch_get("expert_count", 0) or 0)
		if experts:
			raise RuntimeError(f"training a goal lever is not available yet for mixture-of-experts models ({arch}, {experts} experts): use ALIGN")
		cfg = {**OPT.DEFAULTS, **(self.settings.get("optimize") or {}), **(options or {})}
		if cfg["carrier"] not in OPT.CARRIERS:
			cfg["carrier"] = OPT.DEFAULTS["carrier"]
		if cfg.get("layers") not in OPT.LAYER_MODES:
			cfg["layers"] = OPT.LAYER_DEFAULT[cfg["carrier"]]
		lid = GO.lever_id(goal)
		trait = GO.as_trait(goal)
		self.ensure_band(stop)
		layers = OPT.window(self.band, self.engine.n_layer, cfg["layers"], int(cfg["width"]))
		need = self.train_need(layers[0], cfg)
		room = self.train_room()
		if torchload.train_plan(need, {i: v + 2**29 for i, v in room.items()}) is None:
			raise RuntimeError(self.train_short(need, room))
		pick_on = int(cfg.get("draws") or 0) > int(cfg["samples"]) and bool(self.coh_settings()["judge"])
		key = self.teach_key(goal, cfg)
		pairs, pick = self.cached_pairs(goal, key, pick_on)
		if pairs:
			self.log(f"goal '{goal['name']}': reusing the {len(pairs)} teaching answers of the previous training (same goal, model and answer options)")
			kept = [p["score"] for p in pairs if p.get("key")]
		else:
			self.engine.set_steering(None)
			variants = {c["q"]: GO.rephrase(self.engine, c["q"], stop=stop) for c in goal.get("custom") or []}
			if variants:
				self.log(f"goal '{goal['name']}': the model rephrased the custom questions ({sum(len(v) for v in variants.values())} variants) to teach the facts more broadly")
			items = GO.teach_items(goal, variants)
			judge = self.ensure_judge() if pick_on else None
			pick = GO.picker(judge) if judge is not None else None
			self.engine.set_steering(None)
			self.log(f"goal '{goal['name']}': the character answers {len(items)} prompts to teach the goal lever" + (f" (the judge keeps its best {cfg['samples']} of {cfg['draws']} answers to each identity question)" if pick else "") + "…")
			pairs = OPT.teacher(self.engine, items, GO.system(goal), cfg, stop, on_progress=lambda v: self.emit("progress", {"task": f"goal:{gid}", "value": 0.2 * v}), pick=pick)
			if not pairs:
				self.resteer()
				return None
			kept = [p["score"] for p in pairs if p.get("key")]
			if pick and kept:
				self.log(f"goal '{goal['name']}': the kept teaching answers match the goal at {float(np.mean(kept)):.0%} (judge)")
			self.keep_pairs(goal, key, pick is not None, pairs)
		t = time.time()
		prompt_ids = {p["prompt"]: self.engine.chat_tokens(OPT.msgs(p["prompt"])) for p in pairs}
		kl_ids = self.engine.text_tokens(LV.KL_TEXT)[:160]
		what = {"vector": "goal vector", "lora": f"goal LoRA (rank {cfg['rank']})", "both": f"goal vector + LoRA (rank {cfg['rank']})"}[cfg["carrier"]]
		try:
			model = self.torch_model(layers[0], cfg)
			self.log(f"goal '{goal['name']}': training the {what} on L{layers[0]}–L{layers[-1]} ({cfg['steps']} steps, preservation ×{cfg['lam']:g})…")

			def on_step(i, n, ce, kl):
				self.emit("progress", {"task": f"goal:{gid}", "value": 0.25 + 0.65 * (i + 1) / n})
				if i % 10 == 0 or i == n - 1:
					self.emit("goal_step", {"goal": gid, "step": i + 1, "steps": n, "ce": ce, "kl": kl})

			out = OPT.distill(model, pairs, lambda p: prompt_ids[p], kl_ids, layers, cfg, stop=stop, on_step=on_step)
		finally:
			model = None
			if not cfg.get("keep_torch") or self.parked:
				self.free_torch()
			self.unpark()
		if out is None:
			self.resteer()
			return None
		res, hist = out
		lever = self.trained_lever(lid, res, layers, cfg, signature(trait), goal["name"])
		self.levers[lid] = lever
		full = LV.unit_vectors(lever)
		if lever.get("lora"):
			full[LORA] = {self.lora_path(lid): 1.0}
		if self.probe is None:
			self.probe = LV.KLProbe(self.engine, self.readout)
		lever["kl"] = self.probe.kl(full)
		lever["closed"] = OPT.closed(self.engine, pairs, full)
		lever["trained"] = {"steps": cfg["steps"], "lam": cfg["lam"], "layers": cfg["layers"], "carrier": cfg["carrier"], "rank": cfg["rank"] if cfg["carrier"] != "vector" else None, "ce": hist[-1][0], "kl_train": hist[-1][1], "seconds": round(time.time() - t, 1), "pairs": len(pairs), "teacher_match": float(np.mean(kept)) if pick and kept else None}
		self.strengths[lid] = 1.0
		if lever.get("carrier") == "both":
			self.strengths[GO.lora_key(lid)] = 1.0
		else:
			self.strengths.pop(GO.lora_key(lid), None)
		self.emit("lever", _jsonable({"id": lid, "layers": lever["layers"], "gain": lever["gain"], "gen_mult": lever["gen_mult"], "reads_as": lever["reads_as"], "reads_as_minus": lever["reads_as_minus"], "sep": {}, "kind": "optimized", "carrier": lever["carrier"], "closed": lever["closed"], "rank": (lever.get("lora") or {}).get("rank")}))
		closed = lever["closed"]
		if (closed is not None and closed < 0.15) or lever["kl"] > 0.5:
			self.log(f"goal '{goal['name']}': the {what} did not take" + (f", it closes only {closed:.0%} of the gap" if closed is not None and closed < 0.15 else "") + (f", it drifts on neutral text (KL {lever['kl']:.2f})" if lever["kl"] > 0.5 else "") + (": the LoRA carrier learns this kind of goal better" if cfg["carrier"] == "vector" else ": try more steps or more preservation"), "warn")
		self.log(f"goal '{goal['name']}': {what} trained in {time.time() - t:.0f} s" + (f", closes {closed:.0%} of the gap to the character" if closed is not None else "") + f" · KL on neutral text {lever['kl']:.2f}" + (f" · reads as: {' '.join(s.strip() for s in lever['reads_as'][:6])}" if lever["reads_as"] else ""), "ok")
		self.emit("goal_optimized", _jsonable({"goal": gid, "lever": lid, "closed": closed, "kl": lever["kl"], "layers": layers, "trained": lever["trained"]}))
		self.apply_levers()
		return self.check_goal(gid, stop)

	def teach_key(self, goal, cfg):
		data = {"goal": {k: goal.get(k) for k in GO.KEYS + ("custom",)}, "system": GO.system(goal), "cfg": {k: cfg.get(k) for k in ("samples", "draws", "max_tokens", "topk")}, "judge": os.path.basename(self.coh_settings()["judge"] or "")}
		return hashlib.sha1(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

	def teach_path(self, goal):
		return os.path.join(self.state_dir(), f"teach_{goal['id']}.pt")

	def cached_pairs(self, goal, key, pick_on):
		path = self.teach_path(goal)
		if not os.path.isfile(path):
			return None, None
		try:
			data = torch.load(path, weights_only=True)
		except Exception as exc:
			self.log(f"teaching answers of the previous training unreadable ({exc}): the character answers again", "warn")
			return None, None
		if data.get("key") != key or (pick_on and not data.get("picked")):
			return None, None
		return data.get("pairs") or None, bool(data.get("picked"))

	def keep_pairs(self, goal, key, picked, pairs):
		try:
			os.makedirs(self.state_dir(), exist_ok=True)
			tmp = self.teach_path(goal) + ".tmp"
			torch.save({"key": key, "picked": bool(picked), "pairs": pairs}, tmp)
			os.replace(tmp, self.teach_path(goal))
		except Exception as exc:
			self.log(f"teaching answers not kept for the next training: {exc}", "warn")

	def trained_lever(self, lid, res, layers, cfg, sig, name=""):
		vec = res.get("vector")
		if vec:
			lever = OPT.as_lever(vec, sig)
			center = lever["layers"][len(lever["layers"]) // 2]
			lever["reads_as"] = LV.reads_as(self.readout, lever["dirs"][center], center, lever["norm"][center], self.strings)
			lever["reads_as_minus"] = LV.reads_as(self.readout, -lever["dirs"][center], center, lever["norm"][center], self.strings)
		else:
			lever = {"layers": list(layers), "dirs": {}, "norm": {}, "gain": 1.0, "gen_mult": 1.0, "gen_by": "optimized", "kind": "optimized", "sig": sig, "sep": {}, "reads_as": [], "reads_as_minus": []}
		lever["carrier"] = cfg["carrier"]
		lora = res.get("lora")
		if lora:
			fname = f"{lid}.lora.gguf"
			path = os.path.join(self.state_dir(), fname)
			os.makedirs(self.state_dir(), exist_ok=True)
			self.engine.drop_lora(path)
			write_lora(path, OPT.lora_tensors(lora["weights"]), self.info["arch"], lora["alpha"], self.llama_dir(), note=f"Pendulum goal '{name}'")
			lever["lora"] = {"file": fname, "rank": lora["rank"], "targets": sorted({t for _l, t in lora["weights"]}), "layers": sorted({int(l) for l, _t in lora["weights"]})}
		return lever

	def live_layers(self):
		L = self.engine.n_layer
		return sorted(set(range(0, L, 1 if L <= 40 else 2)) | {L - 1})

	def warm_up(self):
		layers = self.live_layers()
		self.readout.frame(np.zeros((len(layers), self.info["n_embd"]), dtype=np.float32), layers, self.scorer)

	def chat(self, text, stop=None, live=True, retry=False):
		if not retry and self.history and self.history[-1]["role"] == "user":
			self.history.pop()
		turn = sum(1 for m in self.history if m["role"] == "user")
		self.history.append({"role": "user", "content": text})
		head = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
		room = self.engine.n_ctx - int(self.sampling.get("max_tokens") or 512) - 8
		past = list(self.history)
		toks = self.engine.chat_tokens(head + past)
		while len(toks) > room and len(past) > 1:
			past = past[2:] if len(past) > 2 else past[1:]
			toks = self.engine.chat_tokens(head + past)
		if len(past) < len(self.history):
			self.log(f"the dialogue is longer than the context ({self.engine.n_ctx} tokens): its {len(self.history) - len(past)} oldest messages are left out of this reply (CLEAR starts a new dialogue)", "warn")
		L = self.engine.n_layer
		layers = self.live_layers()
		expand = np.array([min(range(len(layers)), key=lambda j: abs(layers[j] - l)) for l in range(L)])
		ids = self.scorer.ids
		best_t, best_v = self.axis_best

		def on_step(i, tok, caps):
			if not live:
				return
			H = np.stack([caps[l][-1] for l in layers])
			odds, ti, tp = self.readout.frame(H, layers, self.scorer)
			top = ti.cpu().numpy()[expand]
			prob = tp.cpu().numpy()[expand]
			odds = odds.cpu().numpy()[:, expand]
			self.emit("frame", {"i": i, "tok": self.strings[tok], "odds": odds.round(3).tolist(), "top": [[self.strings[x] for x in row] for row in top], "p": prob.round(4).tolist(), "trait": [[int(best_t[x]) for x in row] for row in top], "tv": [[float(best_v[x]) for x in row] for row in top], "ids": ids})

		seed = int(np.random.default_rng().integers(1, 2**31 - 1))
		self.resteer()
		cfg = self.chat_config(seed)
		gid = f"{time.strftime('%Y%m%d%H%M%S')}-{seed % 100000:05d}"
		self.emit("chat_start", {"prompt_tokens": len(toks), "config": dict(cfg, id=gid), "retry": bool(retry), "text": text})
		try:
			res = self.engine.generate(toks, self.sampling, stop=stop, on_text=lambda t: self.emit("text", {"text": t}), on_step=on_step, capture_layers=layers if live else None, seed=seed)
		except Exception:
			self.emit("chat_done", {"text": "", "stats": {}, "id": gid, "failed": True})
			raise
		self.history.append({"role": "assistant", "content": res["text"]})
		try:
			HI.append({"id": gid, "t": time.strftime("%Y-%m-%d %H:%M:%S"), "model": cfg["model"], "prompt": text, "turn": turn, "reply": res["text"], "config": cfg, "stats": res["stats"], "retry": bool(retry), "stopped": bool(stop is not None and stop.is_set())})
		except OSError as exc:
			self.log(f"history not saved: {exc}", "warn")
		self.emit("chat_done", {"text": res["text"], "stats": res["stats"], "id": gid})
		self.persist()
		return res

	def chat_config(self, seed):
		active = {t: float(v) for t, v in self.strengths.items() if v}
		return _jsonable({"strengths": active, "targets": dict(self.targets), "guard": self.guard_info(), "system": self.system_prompt, "sampling": dict(self.sampling), "model": os.path.basename((self.info or {}).get("path", "")) or (self.info or {}).get("name", ""), "seed": seed, "kinds": {t: {"kind": (self.levers.get(t) or {}).get("kind", "persona"), "carrier": (self.levers.get(t) or {}).get("carrier")} for t in active}})

	def retry(self, stop=None, live=True):
		before = list(self.history)
		while self.history and self.history[-1]["role"] == "assistant":
			self.history.pop()
		if not self.history or self.history[-1]["role"] != "user":
			self.history = before
			raise RuntimeError("nothing to retry: send a message first")
		try:
			return self.chat(self.history.pop()["content"], stop, live, retry=True)
		except Exception:
			self.history = before
			raise

	def choose_reply(self, text):
		if self.history and self.history[-1]["role"] == "assistant" and self.history[-1]["content"] != text:
			self.history[-1]["content"] = text
			self.persist()

	def rewind_chat(self):
		while self.history:
			if self.history.pop()["role"] == "user":
				break
		self.persist()

	def reset_chat(self):
		self.history = []
		self.emit("chat_reset", {})
		self.persist()
