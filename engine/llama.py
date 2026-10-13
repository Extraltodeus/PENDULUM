import ctypes
import os
import shutil

import numpy as np

from engine.llamadist import exe, find_bin, lib_file, system

c_int32 = ctypes.c_int32
c_int64 = ctypes.c_int64
c_uint32 = ctypes.c_uint32
c_uint64 = ctypes.c_uint64
c_size_t = ctypes.c_size_t
c_float = ctypes.c_float
c_bool = ctypes.c_bool
c_char_p = ctypes.c_char_p
c_void_p = ctypes.c_void_p
c_int8 = ctypes.c_int8

LOG_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, c_char_p, c_void_p)
PROGRESS_CB = ctypes.CFUNCTYPE(c_bool, c_float, c_void_p)
EVAL_CB = ctypes.CFUNCTYPE(c_bool, c_void_p, c_bool, c_void_p)
ABORT_CB = ctypes.CFUNCTYPE(c_bool, c_void_p)

SPLIT_NONE = 0
SPLIT_LAYER = 1
SPLIT_ROW = 2
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
DEV_CPU = 0
DEV_GPU = 1


class ModelParams(ctypes.Structure):
	_fields_ = [
		("devices", c_void_p),
		("tensor_buft_overrides", c_void_p),
		("n_gpu_layers", c_int32),
		("split_mode", c_int32),
		("load_mode", c_int32),
		("main_gpu", c_int32),
		("tensor_split", ctypes.POINTER(c_float)),
		("progress_callback", c_void_p),
		("progress_callback_user_data", c_void_p),
		("kv_overrides", c_void_p),
		("vocab_only", c_bool),
		("check_tensors", c_bool),
		("use_extra_bufts", c_bool),
		("no_host", c_bool),
		("no_alloc", c_bool),
		("load_mtp", c_bool),
		("_tail", ctypes.c_uint8 * 128),
	]


class ContextParams(ctypes.Structure):
	_fields_ = [
		("n_ctx", c_uint32),
		("n_batch", c_uint32),
		("n_ubatch", c_uint32),
		("n_seq_max", c_uint32),
		("n_rs_seq", c_uint32),
		("n_outputs_max", c_uint32),
		("n_outputs_max_per_seq", c_uint32),
		("n_threads", c_int32),
		("n_threads_batch", c_int32),
		("ctx_type", c_int32),
		("rope_scaling_type", c_int32),
		("pooling_type", c_int32),
		("attention_type", c_int32),
		("flash_attn_type", c_int32),
		("rope_freq_base", c_float),
		("rope_freq_scale", c_float),
		("yarn_ext_factor", c_float),
		("yarn_attn_factor", c_float),
		("yarn_beta_fast", c_float),
		("yarn_beta_slow", c_float),
		("yarn_orig_ctx", c_uint32),
		("defrag_thold", c_float),
		("cb_eval", c_void_p),
		("cb_eval_user_data", c_void_p),
		("type_k", c_int32),
		("type_v", c_int32),
		("abort_callback", c_void_p),
		("abort_callback_data", c_void_p),
		("embeddings", c_bool),
		("offload_kqv", c_bool),
		("no_perf", c_bool),
		("op_offload", c_bool),
		("swa_full", c_bool),
		("kv_unified", c_bool),
		("samplers", c_void_p),
		("n_samplers", c_size_t),
		("ctx_other", c_void_p),
		("_tail", ctypes.c_uint8 * 128),
	]


class Batch(ctypes.Structure):
	_fields_ = [
		("n_tokens", c_int32),
		("token", ctypes.POINTER(c_int32)),
		("embd", ctypes.POINTER(c_float)),
		("pos", ctypes.POINTER(c_int32)),
		("n_seq_id", ctypes.POINTER(c_int32)),
		("seq_id", ctypes.POINTER(ctypes.POINTER(c_int32))),
		("logits", ctypes.POINTER(c_int8)),
	]


SIGNATURES = {
	"llama_backend_init": (None,),
	"llama_log_set": (None, c_void_p, c_void_p),
	"llama_max_devices": (c_size_t,),
	"llama_model_default_params": (ModelParams,),
	"llama_context_default_params": (ContextParams,),
	"llama_model_load_from_file": (c_void_p, c_char_p, ModelParams),
	"llama_model_free": (None, c_void_p),
	"llama_init_from_model": (c_void_p, c_void_p, ContextParams),
	"llama_free": (None, c_void_p),
	"llama_model_n_embd": (c_int32, c_void_p),
	"llama_model_n_layer": (c_int32, c_void_p),
	"llama_model_n_layer_nextn": (c_int32, c_void_p),
	"llama_model_get_vocab": (c_void_p, c_void_p),
	"llama_model_meta_val_str": (c_int32, c_void_p, c_char_p, c_char_p, c_size_t),
	"llama_model_desc": (c_int32, c_void_p, c_char_p, c_size_t),
	"llama_model_size": (c_uint64, c_void_p),
	"llama_model_n_params": (c_uint64, c_void_p),
	"llama_model_chat_template": (c_char_p, c_void_p, c_char_p),
	"llama_n_ctx": (c_uint32, c_void_p),
	"llama_get_memory": (c_void_p, c_void_p),
	"llama_memory_clear": (None, c_void_p, c_bool),
	"llama_memory_seq_rm": (c_bool, c_void_p, c_int32, c_int32, c_int32),
	"llama_batch_init": (Batch, c_int32, c_int32, c_int32),
	"llama_batch_free": (None, Batch),
	"llama_decode": (c_int32, c_void_p, Batch),
	"llama_synchronize": (None, c_void_p),
	"llama_get_logits_ith": (ctypes.POINTER(c_float), c_void_p, c_int32),
	"llama_vocab_n_tokens": (c_int32, c_void_p),
	"llama_vocab_is_eog": (c_bool, c_void_p, c_int32),
	"llama_vocab_bos": (c_int32, c_void_p),
	"llama_vocab_eos": (c_int32, c_void_p),
	"llama_vocab_get_add_bos": (c_bool, c_void_p),
	"llama_tokenize": (c_int32, c_void_p, c_char_p, c_int32, ctypes.POINTER(c_int32), c_int32, c_bool, c_bool),
	"llama_token_to_piece": (c_int32, c_void_p, c_int32, c_char_p, c_int32, c_int32, c_bool),
	"llama_set_adapter_cvec": (c_int32, c_void_p, ctypes.POINTER(c_float), c_size_t, c_int32, c_int32, c_int32),
	"llama_adapter_lora_init": (c_void_p, c_void_p, c_char_p),
	"llama_set_adapters_lora": (c_int32, c_void_p, ctypes.POINTER(c_void_p), c_size_t, ctypes.POINTER(c_float)),
	"ggml_backend_load_all_from_path": (None, c_char_p),
	"ggml_backend_dev_count": (c_size_t,),
	"ggml_backend_dev_get": (c_void_p, c_size_t),
	"ggml_backend_dev_name": (c_char_p, c_void_p),
	"ggml_backend_dev_description": (c_char_p, c_void_p),
	"ggml_backend_dev_memory": (None, c_void_p, ctypes.POINTER(c_size_t), ctypes.POINTER(c_size_t)),
	"ggml_backend_dev_type": (ctypes.c_int, c_void_p),
	"ggml_backend_dev_buffer_type": (c_void_p, c_void_p),
	"ggml_backend_buft_name": (c_char_p, c_void_p),
	"ggml_get_name": (c_char_p, c_void_p),
	"ggml_nbytes": (c_size_t, c_void_p),
	"ggml_nrows": (c_int64, c_void_p),
	"ggml_backend_tensor_get": (None, c_void_p, c_void_p, c_size_t, c_size_t),
}


def find_llama_dir(hint=""):
	usual = ("D:/llama.cpp", "C:/llama.cpp") if system() == "win" else ("/opt/llama.cpp", os.path.expanduser("~/llama.cpp"), "/usr/local/lib", "/usr/lib")
	for cand in (hint, os.environ.get("PENDULUM_LLAMA", "")) + usual + (os.path.dirname(shutil.which(exe("llama-server")) or ""),):
		found = find_bin(cand, 1) if cand else ""
		if found:
			return found
	return ""


def same_dir(a, b):
	return bool(a) and bool(b) and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class Lib:
	def __init__(self, llama_dir):
		self.dir = os.path.abspath(llama_dir)
		self._dll_dirs = []
		mode = ctypes.DEFAULT_MODE
		if system() == "win":
			self._dll_dirs.append(os.add_dll_directory(self.dir))
			for name in ("cudart64_12.dll", "cudart64_13.dll"):
				found = shutil.which(name)
				if found:
					self._dll_dirs.append(os.add_dll_directory(os.path.dirname(found)))
		else:
			mode = ctypes.RTLD_GLOBAL
		self._libs = [ctypes.CDLL(os.path.join(self.dir, lib_file(n)), mode=mode) for n in ("ggml-base", "ggml", "llama")]
		for name, sig in SIGNATURES.items():
			fn = self._lookup(name)
			fn.restype = sig[0]
			fn.argtypes = list(sig[1:])
			setattr(self, name, fn)
		self.logs = []
		self._log_cb = LOG_CB(self._on_log)
		self.llama_log_set(ctypes.cast(self._log_cb, c_void_p), None)
		self.ggml_backend_load_all_from_path(self.dir.encode())
		self.llama_backend_init()
		self.check_layout()

	def _lookup(self, name):
		for lib in reversed(self._libs):
			try:
				return getattr(lib, name)
			except AttributeError:
				continue
		raise AttributeError(f"symbol not found in llama.cpp DLLs: {name}")

	def _on_log(self, level, text, _ud):
		try:
			self.logs.append((int(level), (text or b"").decode("utf-8", "replace")))
			del self.logs[:-400]
		except Exception:
			pass

	def check_layout(self):
		mp = self.llama_model_default_params()
		cp = self.llama_context_default_params()
		problems = []
		if mp.split_mode != SPLIT_LAYER or mp.main_gpu != 0 or mp.n_gpu_layers not in (-1, 999) or not mp.use_extra_bufts:
			problems.append("llama_model_params")
		if (cp.n_batch, cp.n_ubatch, cp.n_seq_max) != (2048, 512, 1) or abs(cp.yarn_ext_factor + 1.0) > 1e-6 or cp.type_k != GGML_TYPE_F16 or not cp.offload_kqv:
			problems.append("llama_context_params")
		if problems:
			raise RuntimeError(f"llama.cpp ABI mismatch ({', '.join(problems)}): the DLLs in {self.dir} do not match the bindings")

	def devices(self):
		out = []
		for i in range(self.ggml_backend_dev_count()):
			dev = self.ggml_backend_dev_get(i)
			free, total = c_size_t(0), c_size_t(0)
			self.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
			out.append({
				"index": i,
				"name": (self.ggml_backend_dev_name(dev) or b"").decode(),
				"description": (self.ggml_backend_dev_description(dev) or b"").decode(),
				"type": int(self.ggml_backend_dev_type(dev)),
				"free": int(free.value),
				"total": int(total.value),
			})
		return out

	def gpus(self):
		return [d for d in self.devices() if d["type"] == DEV_GPU]

	def tensor_to_numpy(self, t, n_embd):
		n = self.ggml_nbytes(t)
		buf = np.empty(n // 4, dtype=np.float32)
		self.ggml_backend_tensor_get(t, buf.ctypes.data, 0, n)
		return buf.reshape(-1, n_embd)


class BuftOverride(ctypes.Structure):
	_fields_ = [("pattern", c_char_p), ("buft", c_void_p)]
