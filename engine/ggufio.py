import math
import mmap
import os
import struct
import sys

import numpy as np
import torch

from engine.dequant import dequant_rows

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
NP_TYPES = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32, 7: np.bool_, 10: np.uint64, 11: np.int64, 12: np.float64}
T_STRING = 8
T_ARRAY = 9
KEEP_ITEMS = 256

_gguf = None


def gguf_module(llama_dir=""):
	global _gguf
	if _gguf is not None:
		return _gguf
	from engine.llamadist import find_src

	src = find_src(llama_dir) if llama_dir else ""
	local = os.path.join(src, "gguf-py") if src else ""
	if local and os.path.isdir(local) and local not in sys.path:
		sys.path.insert(0, local)
	import gguf
	_gguf = gguf
	return gguf


class TensorInfo:
	def __init__(self, name, dims, ttype, offset, qname, block_size, type_size):
		self.name = name
		self.dims = dims
		self.ttype = ttype
		self.offset = offset
		self.qname = qname
		self.block_size = block_size
		self.type_size = type_size
		self.n_per_row = int(dims[0])
		self.rows = int(math.prod(dims[1:])) if len(dims) > 1 else 1
		self.row_bytes = self.n_per_row // block_size * type_size
		self.nbytes = self.rows * self.row_bytes


class GGUFFile:
	def __init__(self, path, llama_dir=""):
		self.path = path
		self._file = open(path, "rb")
		self.mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
		if self.mm[:4] != b"GGUF":
			raise ValueError(f"not a GGUF file: {path}")
		self.version, n_tensors, n_kv = struct.unpack_from("<IQQ", self.mm, 4)
		off = 24
		self.fields = {}
		for _ in range(n_kv):
			key, off = self._str(off)
			vtype, = struct.unpack_from("<I", self.mm, off)
			value, off = self._value(vtype, off + 4)
			self.fields[key] = value
		gguf = gguf_module(llama_dir)
		infos = []
		for _ in range(n_tensors):
			name, off = self._str(off)
			nd, = struct.unpack_from("<I", self.mm, off)
			dims = struct.unpack_from(f"<{nd}Q", self.mm, off + 4)
			off += 4 + 8 * nd
			ttype, toff = struct.unpack_from("<IQ", self.mm, off)
			off += 12
			infos.append((name, dims, ttype, toff))
		align = int(self.fields.get("general.alignment", 32) or 32)
		data_start = (off + align - 1) // align * align
		self.tensors = {}
		for name, dims, ttype, toff in infos:
			qtype = gguf.GGMLQuantizationType(ttype)
			block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
			self.tensors[name] = TensorInfo(name, dims, ttype, data_start + toff, qtype.name, block_size, type_size)
		self.arch = self.fields.get("general.architecture", "") or ""

	def close(self):
		try:
			self.mm.close()
			self._file.close()
		except Exception:
			pass

	def _str(self, off):
		n, = struct.unpack_from("<Q", self.mm, off)
		return bytes(self.mm[off + 8:off + 8 + n]).decode("utf-8", "replace"), off + 8 + n

	def _value(self, vtype, off):
		if vtype == T_STRING:
			return self._str(off)
		if vtype != T_ARRAY:
			fmt = SCALARS[vtype]
			return struct.unpack_from(fmt, self.mm, off)[0], off + struct.calcsize(fmt)
		itype, count = struct.unpack_from("<IQ", self.mm, off)
		off += 12
		if itype == T_STRING:
			keep = count <= KEEP_ITEMS
			items = []
			for _ in range(count):
				n, = struct.unpack_from("<Q", self.mm, off)
				if keep:
					items.append(bytes(self.mm[off + 8:off + 8 + n]).decode("utf-8", "replace"))
				off += 8 + n
			return (items if keep else ("<strings>", count)), off
		if itype == T_ARRAY:
			items = []
			for _ in range(count):
				sub, off = self._value(T_ARRAY, off)
				items.append(sub)
			return items, off
		dt = np.dtype(NP_TYPES[itype])
		arr = np.frombuffer(self.mm, dtype=dt, count=count, offset=off)
		return (arr.tolist() if count <= KEEP_ITEMS else arr), off + dt.itemsize * count

	def get(self, key, default=None):
		value = self.fields.get(key)
		return default if value is None else value

	def arch_get(self, key, default=None):
		return self.get(f"{self.arch}.{key}", default)

	def has(self, name):
		return name in self.tensors

	def raw(self, name):
		t = self.tensors[name]
		return np.frombuffer(self.mm, dtype=np.uint8, count=t.nbytes, offset=t.offset).reshape(t.rows, t.row_bytes)

	def dequant(self, name, device="cpu", dtype=torch.float32, rows=None):
		t = self.tensors[name]
		raw = self.raw(name)
		if rows is not None:
			raw = raw[np.asarray(rows, dtype=np.int64)]
		gguf = gguf_module()

		def fallback(part):
			return gguf.quants.dequantize(part, gguf.GGMLQuantizationType(t.ttype)).astype(np.float32)

		return dequant_rows(raw, t.qname, t.block_size, t.type_size, t.n_per_row, device=device, dtype=dtype, numpy_fallback=fallback)

	def unembed_name(self):
		return "output.weight" if self.has("output.weight") else "token_embd.weight"

	def main_quant(self):
		sizes = {}
		for t in self.tensors.values():
			sizes[t.qname] = sizes.get(t.qname, 0) + t.nbytes
		return max(sizes, key=sizes.get) if sizes else "?"

	def tensor_bytes(self):
		return sum(t.nbytes for t in self.tensors.values())


def summary(path, llama_dir=""):
	g = GGUFFile(path, llama_dir)
	try:
		n_base = int(g.get("general.base_model.count", 0) or 0)
		return {
			"path": path,
			"file": os.path.basename(path),
			"size": os.path.getsize(path),
			"mtime": os.path.getmtime(path),
			"arch": g.arch,
			"name": g.get("general.name", "") or "",
			"basename": g.get("general.basename", "") or "",
			"size_label": g.get("general.size_label", "") or "",
			"finetune": g.get("general.finetune", "") or "",
			"base_models": [s for i in range(n_base) for s in (g.get(f"general.base_model.{i}.repo_url", ""), g.get(f"general.base_model.{i}.name", "")) if s],
			"n_layer": int(g.arch_get("block_count", 0) or 0),
			"n_nextn": int(g.arch_get("nextn_predict_layers", 0) or 0),
			"n_params": sum(t.rows * t.n_per_row for t in g.tensors.values()),
			"n_embd": int(g.arch_get("embedding_length", 0) or 0),
			"n_ctx_train": int(g.arch_get("context_length", 0) or 0),
			"n_expert": int(g.arch_get("expert_count", 0) or 0),
			"quant": g.main_quant(),
			"weights_bytes": g.tensor_bytes(),
			"has_chat_template": bool(g.get("tokenizer.chat_template")),
		}
	finally:
		g.close()
