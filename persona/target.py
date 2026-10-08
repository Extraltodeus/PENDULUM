import numpy as np

SCALE = 2.5
CLIP = 0.97
DOSE = 1.5
HARD = 10.0
PROBE = 0.8
TOL = 0.08
ROUNDS = 4
LIMIT = 0.95
TRUST = 1.0
ETA = 0.5
FLOOR = 0.3


def to_score(v):
	return SCALE * np.arctanh(np.clip(np.asarray(v, dtype=np.float64), -CLIP, CLIP))


def to_value(s):
	return np.tanh(np.asarray(s, dtype=np.float64) / SCALE)


def solve(J, err, s0, lo=-DOSE, hi=DOSE, lam=0.02, iters=80):
	J = np.asarray(J, dtype=np.float64)
	s0 = np.asarray(s0, dtype=np.float64)
	n = J.shape[1]
	lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (n,))
	hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), (n,))
	A = J.T @ J + lam * np.eye(n)
	b = J.T @ np.asarray(err, dtype=np.float64)
	d = np.clip(s0 + np.linalg.solve(A, b), lo, hi) - s0
	for _ in range(iters):
		for k in range(n):
			r = b[k] - A[k] @ d + A[k, k] * d[k]
			d[k] = np.clip(s0[k] + r / A[k, k], lo[k], hi[k]) - s0[k]
	return d


def plan(J, m, targets, a, rows, caps=None, dose=DOSE):
	a = np.asarray(a, dtype=np.float64)
	caps = np.full(len(a), dose) if caps is None else np.minimum(np.asarray(caps, dtype=np.float64), dose)
	now = to_value(np.asarray(m, dtype=np.float64)[rows])
	err = np.asarray(targets, dtype=np.float64) - now
	slope = np.maximum(1.0 - now ** 2, 0.05) / SCALE
	Jv = slope[:, None] * np.asarray(J, dtype=np.float64)[rows]
	lo = np.minimum(np.maximum(-caps, a - TRUST), caps)
	hi = np.maximum(np.minimum(caps, a + TRUST), -caps)
	keep = np.ones(len(a), dtype=bool)
	new = a.copy()
	for _ in range(len(a)):
		d = solve(Jv[keep], err[keep], a, lo, hi)
		new = a + d
		miss = err - Jv @ d
		stuck = [j for j in range(len(a)) if keep[j] and abs(new[j]) >= caps[j] - 1e-6 and abs(miss[j]) > TOL]
		if not stuck:
			break
		for j in stuck:
			keep[j] = False
			lo[j] = hi[j] = new[j]
	return new, ~keep


def secant(J, da, dm, J0, rows):
	da = np.asarray(da, dtype=np.float64)
	n2 = float(da @ da)
	if n2 < 1e-4:
		return J
	J2 = J + ETA * np.outer(np.asarray(dm, dtype=np.float64) - J @ da, da) / n2
	for j, r in enumerate(rows):
		d0 = float(J0[r, j])
		if abs(d0) > 1e-6 and (np.sign(J2[r, j]) != np.sign(d0) or abs(J2[r, j]) < FLOOR * abs(d0)):
			J2[r, j] = FLOOR * d0
	return J2


def gaps(m, targets, rows):
	return np.asarray(targets, dtype=np.float64) - to_value(np.asarray(m, dtype=np.float64)[rows])
