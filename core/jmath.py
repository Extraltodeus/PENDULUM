import torch

GAMMA_EPS = 1e-6
INV_COND_EPS = 0.2


def unit(v, dim=-1):
	return v / v.norm(dim=dim, keepdim=True).clamp_min(1e-8)


def transport(rows, J):
	if J is None:
		return rows
	return rows @ J.to(rows.device, rows.dtype)


def jlens_vectors(w_rows, J):
	return unit(transport(unit(w_rows.float()), J))


def fused_direction(w_rows, J):
	return unit(transport(w_rows.float().mean(0, keepdim=True), J))[0]


def orthonormal_basis(rows, tol=1e-3):
	_u, s, vh = torch.linalg.svd(rows, full_matrices=False)
	if s.numel() == 0 or float(s[0]) <= 0:
		return rows[:0]
	return vh[:s.numel()][s > tol * s[0]]


def mean_direction(rows):
	return unit(rows.mean(0))


def effective_coeffs(mode, factor, g):
	if mode != "scale":
		return -min(g, 1.0), g * factor
	alpha = g * (factor - 1.0)
	if factor < 1.0:
		alpha = max(alpha, min(factor, 0.0) - 1.0)
	return alpha, 0.0


def compose_left(U, V, W, Vf):
	if U is None:
		return W, Vf
	return torch.cat([U, W], dim=1), torch.cat([V, Vf + V @ (U.T @ Vf)], dim=1)


def compress_uv(U, V, tol=1e-5):
	Qu, Ru = torch.linalg.qr(U)
	Qv, Rv = torch.linalg.qr(V)
	Us, S, Vh = torch.linalg.svd(Ru @ Rv.T)
	keep = S > tol * S.max().clamp_min(1e-12)
	return Qu @ (Us[:, keep] * S[keep]), Qv @ Vh.T[:, keep]


def cumulative(factors, n_layers):
	if not factors:
		return {}
	out = {}
	U = V = None
	for layer in range(min(factors), n_layers):
		if factors.get(layer):
			for w, v in factors[layer]:
				U, V = compose_left(U, V, w, v)
			U, V = compress_uv(U, V)
		if U is not None:
			out[layer + 1] = (U, V)
	return out


def gamma_pair(gamma, U, V):
	safe = torch.where(gamma.abs() < GAMMA_EPS, torch.full_like(gamma, GAMMA_EPS), gamma)
	return gamma.unsqueeze(1) * U, V / safe.unsqueeze(1)


def apply_read(W, Ug, Vg):
	B = W @ Ug
	return W + B @ Vg.T, B, Vg.T.contiguous()


def inverse_uv(U, V):
	small = torch.eye(U.shape[1]) + V.T @ U
	svals = torch.linalg.svdvals(small)
	regularized = bool(svals.min() < INV_COND_EPS * max(1.0, float(svals.max())))
	inv = torch.linalg.pinv(small, rtol=INV_COND_EPS) if regularized else torch.linalg.inv(small)
	return U @ inv, V, regularized


def apply_write(W, U_inv, V):
	A = V.T @ W
	return W - U_inv @ A, (-U_inv).contiguous(), A
