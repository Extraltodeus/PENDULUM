import re

import numpy as np
import torch

from engine.dequant import KERNELS, dequant_rows

DENSE = ("F32", "F16", "BF16")
BLOCK = re.compile(r"blk\.(\d+)\.")
LAYER = re.compile(r"\.layers\.(\d+)\.")
CHUNK = 2**24
PROBE = 8
HEADS = ("output.weight",)
EMBEDS = ("token_embd.weight",)


def quantizable(t):
	return len(t.dims) == 2 and t.qname in KERNELS and t.qname not in DENSE


def unpack(raw, qname, type_size, dtype):
	return KERNELS[qname](raw.reshape(-1, type_size)).reshape(raw.shape[0], -1).to(dtype)


def device_of(mod):
	return next(t for t in (*mod.parameters(), *mod.buffers())).device


class _Linear(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, mod):
		ctx.mod = mod
		out = x.new_empty(*x.shape[:-1], mod.out_features)
		for s, w in mod.chunks(x.dtype):
			out[..., s:s + w.shape[0]] = x @ w.T
		return out

	@staticmethod
	def backward(ctx, g):
		gx = g.new_zeros(*g.shape[:-1], ctx.mod.in_features, dtype=torch.float32)
		for s, w in ctx.mod.chunks(g.dtype):
			gx.add_(g[..., s:s + w.shape[0]] @ w)
		return gx.to(g.dtype), None


class QuantLinear(torch.nn.Module):
	def __init__(self, raw, qname, type_size, in_features, bias=None, rows=None, cols=None):
		super().__init__()
		self.raw = raw if isinstance(raw, torch.nn.Parameter) else torch.nn.Parameter(raw, requires_grad=False)
		self.qname = qname
		self.type_size = type_size
		self.out_features = raw.shape[0]
		self.in_features = in_features
		self.bias = bias
		self.register_buffer("rows", rows)
		self.register_buffer("cols", cols)

	def chunks(self, dtype=torch.bfloat16):
		step = max(1, CHUNK // self.in_features)
		for s in range(0, self.out_features, step):
			yield s, unpack(self.raw[s:s + step], self.qname, self.type_size, dtype)

	def dense(self, dtype=torch.bfloat16):
		w = torch.empty((self.out_features, self.in_features), dtype=dtype, device=self.raw.device)
		for s, part in self.chunks(dtype):
			w[s:s + part.shape[0]] = part
		if self.rows is not None:
			w = w.index_select(0, self.rows)
		if self.cols is not None:
			out = torch.empty_like(w)
			out[:, self.cols] = w
			w = out
		return w

	def forward(self, x):
		if self.cols is not None:
			x = x.index_select(-1, self.cols)
		y = _Linear.apply(x, self)
		if self.rows is not None:
			y = y.index_select(-1, self.rows)
		return y if self.bias is None else y + self.bias.to(y.dtype)

	def extra_repr(self):
		return f"in_features={self.in_features}, out_features={self.out_features}, {self.qname}" + (", rows permuted" if self.rows is not None else "") + (", columns permuted" if self.cols is not None else "")


class QuantEmbedding(torch.nn.Module):
	def __init__(self, raw, qname, type_size, dim, dtype=torch.bfloat16, scale=None, padding_idx=None):
		super().__init__()
		self.raw = raw if isinstance(raw, torch.nn.Parameter) else torch.nn.Parameter(raw, requires_grad=False)
		self.qname = qname
		self.type_size = type_size
		self.num_embeddings = raw.shape[0]
		self.embedding_dim = dim
		self.dtype = dtype
		self.padding_idx = padding_idx
		self.register_buffer("scale", None if scale is None else torch.as_tensor(scale, dtype=torch.float32).reshape(()))

	def forward(self, ids):
		flat = ids.reshape(-1).to(self.raw.device)
		out = unpack(self.raw.index_select(0, flat), self.qname, self.type_size, self.dtype).reshape(*ids.shape, self.embedding_dim)
		return out if self.scale is None else out * self.scale.to(out.dtype)

	def extra_repr(self):
		return f"{self.num_embeddings}, {self.embedding_dim}, {self.qname}"


def raw_tensor(g, name):
	return torch.from_numpy(np.array(g.raw(name), copy=True))


def _rows(t, raw, s, e, work, dtype):
	return dequant_rows(raw[s:e], t.qname, t.block_size, t.type_size, t.n_per_row, device=work, dtype=dtype, chunk_rows=max(1, CHUNK // t.n_per_row), work=work)


def _same(g, name, w, work):
	t = g.tensors[name]
	raw = g.raw(name)
	wd = w.detach()
	if not torch.equal(_rows(t, raw, 0, PROBE, work, wd.dtype), wd[:PROBE].to(work)):
		return False
	step = max(1, CHUNK // t.n_per_row)
	return all(torch.equal(_rows(t, raw, s, s + step, work, wd.dtype), wd[s:s + step].to(work)) for s in range(0, t.rows, step))


def _keys(m, r):
	return (m.double() @ r).cpu()


def _order(a, b):
	ia, ib = torch.argsort(a), torch.argsort(b)
	if not torch.equal(a[ia], b[ib]):
		return None
	out = torch.empty_like(ia)
	out[ia] = ib
	return out


def _permuted(g, name, w, work):
	t = g.tensors[name]
	wd = w.detach().to(work)
	full = _rows(t, g.raw(name), 0, t.rows, work, wd.dtype)
	gen = torch.Generator(device="cpu").manual_seed(0)
	r = torch.randn(wd.shape[1], generator=gen, dtype=torch.float64).to(work)
	rows = _order(_keys(wd, r), _keys(full, r))
	if rows is not None and torch.equal(full.index_select(0, rows.to(work)), wd):
		return rows, None
	c = torch.randn(wd.shape[0], generator=gen, dtype=torch.float64).to(work)
	cols = _order(_keys(wd.T, c), _keys(full.T, c))
	if cols is not None and torch.equal(full.index_select(1, cols.to(work)), wd):
		return None, torch.argsort(cols)
	return False


def _swap(model, mname, new):
	parent, _dot, leaf = mname.rpartition(".")
	setattr(model.get_submodule(parent) if parent else model, leaf, new)


def _embedding(model, g, work):
	emb = model.get_input_embeddings()
	name = next((n for n in EMBEDS if g.has(n) and quantizable(g.tensors[n])), None)
	if name is None or type(emb) is QuantEmbedding or not isinstance(getattr(emb, "weight", None), torch.Tensor) or emb.weight.is_meta:
		return None, 0
	if tuple(emb.weight.shape) != (g.tensors[name].rows, g.tensors[name].n_per_row) or not _same(g, name, emb.weight, work):
		return None, 0
	t = g.tensors[name]
	scale = getattr(emb, "embed_scale", None)
	q = QuantEmbedding(raw_tensor(g, name), t.qname, t.type_size, t.n_per_row, emb.weight.dtype, float(scale) if scale is not None else None, emb.padding_idx)
	saved = emb.weight.numel() * emb.weight.element_size() - q.raw.numel()
	mname = next(n for n, m in model.named_modules() if m is emb)
	return (mname, emb, q), saved


def _head(model, g, work, emb_swap):
	head = model.get_output_embeddings()
	if type(head) is not torch.nn.Linear or head.bias is not None:
		return None, 0
	mname = next(n for n, m in model.named_modules() if m is head)
	if emb_swap is not None and head.weight is emb_swap[1].weight:
		q = emb_swap[2]
		return (mname, QuantLinear(q.raw, q.qname, q.type_size, q.embedding_dim)), 0
	name = next((n for n in HEADS if g.has(n) and quantizable(g.tensors[n])), None)
	if name is None or tuple(head.weight.shape) != (g.tensors[name].rows, g.tensors[name].n_per_row) or not _same(g, name, head.weight, work):
		return None, 0
	t = g.tensors[name]
	q = QuantLinear(raw_tensor(g, name), t.qname, t.type_size, t.n_per_row)
	return (mname, q), head.weight.numel() * head.weight.element_size() - q.raw.numel()


def quantize(model, g, work="cpu"):
	groups = {}
	for name, t in g.tensors.items():
		m = BLOCK.match(name)
		if quantizable(t) and m:
			groups.setdefault((int(m.group(1)), int(t.dims[0]), int(t.dims[1])), []).append(name)
	used, saved, moved = set(), 0, 0
	emb, s_emb = _embedding(model, g, work)
	head, s_head = _head(model, g, work, emb)
	skip = {id(m) for m in (model.get_input_embeddings(), model.get_output_embeddings()) if m is not None}
	for mname, mod in list(model.named_modules()):
		if type(mod) is not torch.nn.Linear or id(mod) in skip or mod.weight.is_meta:
			continue
		m = LAYER.search("." + mname)
		if not m:
			continue
		out_f, in_f = mod.weight.shape
		cands = [c for c in groups.get((int(m.group(1)), in_f, out_f), []) if c not in used]
		hit = next(((c, None, None) for c in cands if _same(g, c, mod.weight, work)), None)
		if hit is None:
			for c in cands:
				perm = _permuted(g, c, mod.weight, work)
				if perm:
					hit = (c, perm[0], perm[1])
					moved += 1
					break
		if hit is None:
			continue
		cand, rows, cols = hit
		t = g.tensors[cand]
		bias = None if mod.bias is None else torch.nn.Parameter(mod.bias.detach().clone(), requires_grad=False)
		_swap(model, mname, QuantLinear(raw_tensor(g, cand), t.qname, t.type_size, in_f, bias, rows, cols))
		used.add(cand)
		saved += mod.weight.numel() * mod.weight.element_size() - t.nbytes
	for swap in (emb, head):
		if swap is not None:
			_swap(model, swap[0], swap[-1])
			used.add(swap[0])
	if str(work).startswith("cuda"):
		torch.cuda.empty_cache()
	return len(used), saved + s_emb + s_head, moved
