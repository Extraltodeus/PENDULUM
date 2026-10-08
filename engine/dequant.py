import numpy as np
import torch

QK_K = 256
KVALUES_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

_consts = {}


def _c(values, dtype, shape, device):
	key = (values, dtype, shape, str(device))
	t = _consts.get(key)
	if t is None:
		t = torch.tensor(values, dtype=dtype, device=device).view(shape)
		_consts[key] = t
	return t


def _split(blocks, *sizes):
	out, start = [], 0
	for s in sizes:
		out.append(blocks[:, start:start + s])
		start += s
	out.append(blocks[:, start:])
	return out


def _f16(x):
	return x.contiguous().view(torch.float16).to(torch.float32)


def _nib(qs, n, group):
	return (qs.reshape(n, -1, 1, group) >> _c((0, 4), torch.uint8, (1, 1, 2, 1), qs.device)) & 0x0F


def _q8_0(b):
	d, x = _split(b, 2)
	return x.contiguous().view(torch.int8).float() * _f16(d)


def _q4_0(b):
	n = b.shape[0]
	d, qs = _split(b, 2)
	return _f16(d) * (_nib(qs, n, 16).reshape(n, -1).to(torch.int8) - 8).float()


def _q4_1(b):
	n = b.shape[0]
	d, m, qs = _split(b, 2, 2)
	return _f16(d) * _nib(qs, n, 16).reshape(n, -1).float() + _f16(m)


def _qh32(qh, n):
	return ((qh.contiguous().view(torch.int32).reshape(n, 1) >> _c(tuple(range(32)), torch.int32, (1, 32), qh.device)) & 1).to(torch.uint8)


def _q5_0(b):
	n = b.shape[0]
	d, qh, qs = _split(b, 2, 4)
	q = (_nib(qs, n, 16).reshape(n, -1) | (_qh32(qh, n) << 4)).to(torch.int8) - 16
	return _f16(d) * q.float()


def _q5_1(b):
	n = b.shape[0]
	d, m, qh, qs = _split(b, 2, 2, 4)
	q = _nib(qs, n, 16).reshape(n, -1) | (_qh32(qh, n) << 4)
	return _f16(d) * q.float() + _f16(m)


def _q2_k(b):
	n = b.shape[0]
	scales, qs, d, dmin = _split(b, QK_K // 16, QK_K // 4, 2)[:4]
	dl = (_f16(d) * (scales & 0xF).float()).reshape(n, 16, 1)
	ml = (_f16(dmin) * (scales >> 4).float()).reshape(n, 16, 1)
	q = ((qs.reshape(n, -1, 1, 32) >> _c((0, 2, 4, 6), torch.uint8, (1, 1, 4, 1), b.device)) & 3).reshape(n, 16, 16).float()
	return (dl * q - ml).reshape(n, -1)


def _q3_k(b):
	n = b.shape[0]
	hmask, qs, scales, d = _split(b, QK_K // 8, QK_K // 4, 12)
	dev = b.device
	lsc = (scales[:, :8].reshape(n, 1, 8) >> _c((0, 4), torch.uint8, (1, 2, 1), dev)).reshape(n, 16)
	hsc = (scales[:, 8:12].reshape(n, 1, 4) >> _c((0, 2, 4, 6), torch.uint8, (1, 4, 1), dev)).reshape(n, 16)
	sc = (((lsc & 0x0F) | ((hsc & 0x03) << 4)).to(torch.int8) - 32).float()
	dl = (_f16(d) * sc).reshape(n, 16, 1)
	ql = ((qs.reshape(n, -1, 1, 32) >> _c((0, 2, 4, 6), torch.uint8, (1, 1, 4, 1), dev)).reshape(n, 16, 16)) & 3
	qh = ((hmask.reshape(n, -1, 1, 32) >> _c(tuple(range(8)), torch.uint8, (1, 1, 8, 1), dev)).reshape(n, 16, 16)) & 1
	qh = qh ^ 1
	q = (ql.to(torch.int8) - (qh << 2).to(torch.int8)).float()
	return (dl * q).reshape(n, QK_K)


def _scale_min(scales, n):
	s = scales.reshape(n, 3, 4)
	d, m, md = s[:, 0], s[:, 1], s[:, 2]
	sc = torch.cat([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
	mn = torch.cat([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], dim=-1)
	return sc.float(), mn.float()


def _q4_k(b):
	n = b.shape[0]
	d, dmin, scales, qs = _split(b, 2, 2, 12)
	sc, mn = _scale_min(scales, n)
	dd = (_f16(d) * sc).reshape(n, -1, 1)
	dm = (_f16(dmin) * mn).reshape(n, -1, 1)
	q = _nib(qs, n, 32).reshape(n, -1, 32).float()
	return (dd * q - dm).reshape(n, QK_K)


def _q5_k(b):
	n = b.shape[0]
	d, dmin, scales, qh, qs = _split(b, 2, 2, 12, QK_K // 8)
	sc, mn = _scale_min(scales, n)
	dd = (_f16(d) * sc).reshape(n, -1, 1)
	dm = (_f16(dmin) * mn).reshape(n, -1, 1)
	ql = _nib(qs, n, 32).reshape(n, -1, 32)
	h = ((qh.reshape(n, -1, 1, 32) >> _c(tuple(range(8)), torch.uint8, (1, 1, 8, 1), b.device)) & 1).reshape(n, -1, 32)
	return (dd * (ql | (h << 4)).float() - dm).reshape(n, QK_K)


def _q6_k(b):
	n = b.shape[0]
	ql, qh, scales, d = _split(b, QK_K // 2, QK_K // 4, QK_K // 16)
	dd = (_f16(d) * scales.contiguous().view(torch.int8).float()).reshape(n, QK_K // 16, 1)
	lo = _nib(ql, n, 64).reshape(n, -1, 32)
	hi = ((qh.reshape(n, -1, 1, 32) >> _c((0, 2, 4, 6), torch.uint8, (1, 1, 4, 1), b.device)) & 3).reshape(n, -1, 32)
	q = ((lo | (hi << 4)).to(torch.int8) - 32).reshape(n, QK_K // 16, -1).float()
	return (dd * q).reshape(n, QK_K)


def _iq4_nl(b):
	n = b.shape[0]
	d, qs = _split(b, 2)
	idx = _nib(qs, n, 16).reshape(n, -1).long()
	return _f16(d) * _c(KVALUES_IQ4NL, torch.float32, (16,), b.device)[idx]


def _iq4_xs(b):
	n = b.shape[0]
	d, sh, sl, qs = _split(b, 2, 2, QK_K // 64)
	dev = b.device
	sh = sh.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
	sl = ((sl.reshape(n, -1, 1) >> _c((0, 4), torch.uint8, (1, 1, 2), dev)).reshape(n, -1) & 0x0F).to(torch.int32)
	sh = ((sh.reshape(n, 1, 1) >> _c(tuple(range(0, 16, 2)), torch.int32, (1, -1, 1), dev)).reshape(n, -1) & 0x03)
	scales = ((sl | (sh << 4)) - 32).float()
	dl = (_f16(d) * scales).reshape(n, -1, 1)
	idx = ((qs.reshape(n, -1, 1, 16) >> _c((0, 4), torch.uint8, (1, 1, 2, 1), dev)).reshape(n, -1, 32) & 0x0F).long()
	return (dl * _c(KVALUES_IQ4NL, torch.float32, (16,), dev)[idx]).reshape(n, -1)


def _bf16(b):
	return b.contiguous().view(torch.bfloat16).float()


def _f16raw(b):
	return b.contiguous().view(torch.float16).float()


def _f32(b):
	return b.contiguous().view(torch.float32)


KERNELS = {
	"F32": _f32, "F16": _f16raw, "BF16": _bf16,
	"Q4_0": _q4_0, "Q4_1": _q4_1, "Q5_0": _q5_0, "Q5_1": _q5_1, "Q8_0": _q8_0,
	"Q2_K": _q2_k, "Q3_K": _q3_k, "Q4_K": _q4_k, "Q5_K": _q5_k, "Q6_K": _q6_k,
	"IQ4_NL": _iq4_nl, "IQ4_XS": _iq4_xs,
}


def dequant_rows(raw, qname, block_size, type_size, n_per_row, device="cpu", dtype=torch.float32, chunk_rows=8192, numpy_fallback=None, work=None):
	rows = raw.shape[0]
	out = torch.empty((rows, n_per_row), dtype=dtype, device=device)
	kernel = KERNELS.get(qname)
	for start in range(0, rows, chunk_rows):
		part = np.array(raw[start:start + chunk_rows], copy=True)
		if kernel is None:
			vals = torch.from_numpy(numpy_fallback(part)).reshape(part.shape[0], n_per_row)
			out[start:start + part.shape[0]] = vals.to(device=device, dtype=dtype)
			continue
		blocks = torch.from_numpy(part).to(work or device).reshape(-1, type_size)
		vals = kernel(blocks).reshape(part.shape[0], n_per_row)
		out[start:start + part.shape[0]] = vals.to(device=device, dtype=dtype)
	return out
