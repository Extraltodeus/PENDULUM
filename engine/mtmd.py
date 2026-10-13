import ctypes
import os
import re
from ctypes import POINTER, byref, c_bool, c_char_p, c_int, c_int32, c_size_t, c_ubyte, c_void_p

from engine.llamadist import lib_file, system

CHUNK_TEXT, CHUNK_IMAGE = 0, 1
PREFER = ("f16", "bf16", "f32", "q8")


class Params(ctypes.Structure):
	_fields_ = [("use_gpu", c_bool), ("device", c_void_p), ("print_timings", c_bool), ("n_threads", c_int), ("image_marker", c_char_p), ("media_marker", c_char_p), ("flash_attn_type", c_int), ("warmup", c_bool), ("image_min_tokens", c_int), ("image_max_tokens", c_int), ("cb_eval", c_void_p), ("cb_eval_user_data", c_void_p), ("batch_max_tokens", c_int32), ("progress_callback", c_void_p), ("progress_callback_user_data", c_void_p), ("_pad", c_ubyte * 256)]


class InputText(ctypes.Structure):
	_fields_ = [("text", c_char_p), ("text_len", c_size_t), ("add_special", c_bool), ("parse_special", c_bool)]


class BitmapWrapper(ctypes.Structure):
	_fields_ = [("bitmap", c_void_p), ("video_ctx", c_void_p)]


SIGNATURES = {
	"mtmd_context_params_default": (Params,),
	"mtmd_default_marker": (c_char_p,),
	"mtmd_init_from_file": (c_void_p, c_char_p, c_void_p, Params),
	"mtmd_free": (None, c_void_p),
	"mtmd_support_vision": (c_bool, c_void_p),
	"mtmd_helper_bitmap_init_from_file": (BitmapWrapper, c_void_p, c_char_p, c_bool),
	"mtmd_bitmap_free": (None, c_void_p),
	"mtmd_input_chunks_init": (c_void_p,),
	"mtmd_input_chunks_size": (c_size_t, c_void_p),
	"mtmd_input_chunks_get": (c_void_p, c_void_p, c_size_t),
	"mtmd_input_chunks_free": (None, c_void_p),
	"mtmd_input_chunk_get_type": (c_int, c_void_p),
	"mtmd_input_chunk_get_n_tokens": (c_size_t, c_void_p),
	"mtmd_tokenize": (c_int32, c_void_p, c_void_p, POINTER(InputText), POINTER(c_void_p), c_size_t),
	"mtmd_helper_eval_chunks": (c_int32, c_void_p, c_void_p, c_void_p, c_int32, c_int32, c_int32, c_bool, POINTER(c_int32)),
}


def find_mmproj(model_path):
	if not model_path:
		return ""
	folder = os.path.dirname(os.path.abspath(model_path))
	try:
		names = [n for n in os.listdir(folder) if "mmproj" in n.lower() and n.lower().endswith(".gguf")]
	except OSError:
		return ""
	if not names:
		return ""
	rank = lambda n: next((i for i, k in enumerate(PREFER) if re.search(rf"(^|[-_.]){k}([-_.]|$)", n.lower())), len(PREFER))
	return os.path.join(folder, sorted(names, key=lambda n: (rank(n), n))[0])


class Vision:
	def __init__(self, engine, mmproj):
		self.engine = engine
		self.lib = ctypes.CDLL(os.path.join(engine.lib.dir, lib_file("mtmd")), mode=ctypes.DEFAULT_MODE if system() == "win" else ctypes.RTLD_GLOBAL)
		for name, sig in SIGNATURES.items():
			fn = getattr(self.lib, name)
			fn.restype = sig[0]
			fn.argtypes = list(sig[1:])
		p = self.lib.mtmd_context_params_default()
		p.use_gpu = True
		p.print_timings = False
		p.warmup = False
		self.ctx = self.lib.mtmd_init_from_file(os.path.abspath(mmproj).encode("utf-8"), engine.model, p)
		if not self.ctx:
			raise RuntimeError(f"could not load the vision projector {os.path.basename(mmproj)}: {engine._last_error()}")
		if not self.lib.mtmd_support_vision(self.ctx):
			self.close()
			raise RuntimeError(f"{os.path.basename(mmproj)} has no vision encoder")
		self.marker = self.lib.mtmd_default_marker().decode("utf-8")
		self.path = mmproj

	def close(self):
		if self.ctx:
			self.lib.mtmd_free(self.ctx)
			self.ctx = None

	def read(self, image, prompt, layers):
		e = self.engine
		bw = self.lib.mtmd_helper_bitmap_init_from_file(self.ctx, os.path.abspath(image).encode("utf-8"), False)
		if not bw.bitmap:
			raise RuntimeError(f"could not read the image {os.path.basename(image)}")
		chunks = self.lib.mtmd_input_chunks_init()
		try:
			text = e.template.render([{"role": "user", "content": self.marker + "\n" + prompt}], add_generation_prompt=True)
			raw = text.encode("utf-8")
			it = InputText(raw, len(raw), True, True)
			bitmaps = (c_void_p * 1)(bw.bitmap)
			rc = self.lib.mtmd_tokenize(self.ctx, chunks, byref(it), bitmaps, 1)
			if rc != 0:
				raise RuntimeError(f"mtmd_tokenize failed ({rc})")
			spans, at = [], 0
			for i in range(self.lib.mtmd_input_chunks_size(chunks)):
				ch = self.lib.mtmd_input_chunks_get(chunks, i)
				n = int(self.lib.mtmd_input_chunk_get_n_tokens(ch))
				spans.append(("image" if self.lib.mtmd_input_chunk_get_type(ch) == CHUNK_IMAGE else "text", at, n))
				at += n
			e.reset()
			e.set_capture(layers)
			new = c_int32(0)
			try:
				rc = self.lib.mtmd_helper_eval_chunks(self.ctx, e.ctx, chunks, 0, 0, e.n_batch, True, byref(new))
				caps = e.pop_capture()
			finally:
				e.set_capture(None)
				e.reset()
			if rc != 0:
				raise RuntimeError(f"decoding the image failed ({rc}): {e._last_error()}")
			return caps, spans
		finally:
			self.lib.mtmd_input_chunks_free(chunks)
			self.lib.mtmd_bitmap_free(bw.bitmap)
