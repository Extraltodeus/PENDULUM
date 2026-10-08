import numpy as np
import torch


class Readout:
	def __init__(self, gguf, lens, device, n_layer, vocab_mask=None):
		self.device = torch.device(device)
		self.lens = lens
		self.n_layer = n_layer
		self.eps = float(gguf.arch_get("attention.layer_norm_rms_epsilon", 1e-6) or 1e-6)
		cap = gguf.arch_get("final_logit_softcapping")
		self.softcap = float(cap) if cap else None
		dtype = torch.float16 if self.device.type == "cuda" else torch.float32
		self.W = gguf.dequant(gguf.unembed_name(), device=self.device, dtype=dtype)
		self.gamma = gguf.dequant("output_norm.weight", device=self.device).flatten().float()
		self.n_vocab, self.d = self.W.shape
		layers = [l for l in sorted(lens.jacobians) if l < n_layer]
		self.Jstack = torch.stack([lens.jacobians[l].to(dtype) for l in layers]).to(self.device) if layers else None
		self.J = {l: self.Jstack[i] for i, l in enumerate(layers)}
		self.jidx = {l: i for i, l in enumerate(layers)}
		self.mask = vocab_mask.to(self.device) if vocab_mask is not None else None
		self.negmask = torch.zeros(self.n_vocab, device=self.device, dtype=torch.float32).masked_fill(~self.mask, float("-inf")) if self.mask is not None else None
		self.negmask16 = self.negmask.to(dtype) if self.negmask is not None else None
		self._frame_plans = {}
		self.dtype = dtype

	def nbytes(self):
		return self.W.numel() * self.W.element_size() + (self.Jstack.numel() * self.Jstack.element_size() if self.Jstack is not None else 0)

	def park(self):
		if self.device.type != "cuda" or getattr(self, "parked", None):
			return 0
		names = [n for n in ("W", "Jstack", "gamma", "mask", "negmask", "negmask16", "wnorm") if isinstance(getattr(self, n, None), torch.Tensor)]
		size = sum(getattr(self, n).numel() * getattr(self, n).element_size() for n in names)
		for n in names:
			setattr(self, n, getattr(self, n).cpu())
		self.J = {l: self.Jstack[i] for l, i in self.jidx.items()} if self.Jstack is not None else {}
		self.parked = names
		self._frame_plans = {}
		torch.cuda.empty_cache()
		return size

	def unpark(self):
		names = getattr(self, "parked", None)
		if not names:
			return
		for n in names:
			setattr(self, n, getattr(self, n).to(self.device))
		self.J = {l: self.Jstack[i] for l, i in self.jidx.items()} if self.Jstack is not None else {}
		self.parked = None
		self._frame_plans = {}

	def free(self):
		self.W = self.J = self.Jstack = self.gamma = self.mask = None
		if self.device.type == "cuda":
			torch.cuda.empty_cache()

	def transport(self, h, layer):
		J = self.J.get(layer)
		if J is None:
			return h
		return h @ J.T.to(h.dtype)

	def final(self, x):
		x = x.float()
		y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.gamma
		logits = (y.to(self.dtype) @ self.W.T).float()
		if self.softcap:
			logits = self.softcap * torch.tanh(logits / self.softcap)
		return logits

	def logits(self, h, layer):
		h = torch.as_tensor(h, device=self.device).to(self.dtype)
		return self.final(self.transport(h, layer))

	def logprobs(self, h, layer):
		return torch.log_softmax(self.logits(h, layer), dim=-1)

	@torch.no_grad()
	def layers_logprobs(self, hs, layers):
		h = torch.as_tensor(hs, device=self.device).to(self.dtype)
		rows = []
		for i, l in enumerate(layers):
			J = self.J.get(l)
			rows.append(h[i] @ J.T if J is not None else h[i])
		return torch.log_softmax(self.final(torch.stack(rows)), dim=-1)

	@torch.no_grad()
	def frame(self, hs, layers, scorer, k=3):
		h = torch.as_tensor(hs, device=self.device).to(self.dtype)
		key = tuple(layers)
		plan = self._frame_plans.get(key)
		if plan is None:
			have = [i for i, l in enumerate(layers) if l in self.jidx]
			js = [self.jidx[layers[i]] for i in have]
			plan = (torch.tensor(have, device=self.device, dtype=torch.long), self.Jstack[torch.tensor(js, device=self.device)] if js else None) if have else (None, None)
			self._frame_plans = {key: plan}
		t = h
		if plan[0] is not None:
			t = h.clone()
			t[plan[0]] = torch.bmm(plan[1], h[plan[0]][:, :, None])[:, :, 0]
		x = t.float()
		y = (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.gamma).to(self.dtype)
		logits = y @ self.W.T
		if self.softcap:
			logits = (self.softcap * torch.tanh(logits.float() / self.softcap)).to(self.dtype)
		lse = torch.logsumexp(logits, -1, keepdim=True)
		sel = logits[:, scorer.all].float() - lse.float()
		odds, _sal = scorer.score_selected(sel)
		masked = logits + self.negmask16 if self.negmask is not None else logits
		tv, ti = masked.topk(k, dim=-1)
		return odds, ti, (tv.float() - lse.float()).exp()

	def direction(self, token_ids, layer, weights=None):
		rows = self.W[torch.as_tensor(token_ids, device=self.device)].float()
		rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)
		J = self.J.get(layer)
		v = rows @ J.float() if J is not None else rows
		v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
		w = torch.ones(len(token_ids), device=self.device) if weights is None else torch.as_tensor(weights, device=self.device, dtype=torch.float32)
		return (w[:, None] * v).sum(0) / w.sum().clamp_min(1e-8)

	def topk(self, h, layer, k=8, masked=True):
		lp = self.logprobs(h, layer)
		raw_v, raw_i = lp.topk(k, dim=-1)
		if masked and self.mask is not None:
			mv, mi = lp.masked_fill(~self.mask, float("-inf")).topk(k, dim=-1)
			return mi.cpu().numpy(), mv.exp().cpu().numpy()
		return raw_i.cpu().numpy(), raw_v.exp().cpu().numpy()

	def ranks(self, lp, token_ids):
		sel = lp[:, token_ids]
		return (lp[:, None, :] > sel[:, :, None]).sum(-1) if lp.shape[0] * len(token_ids) * lp.shape[1] < 2e8 else torch.stack([(lp > sel[:, i:i + 1]).sum(-1) for i in range(len(token_ids))], dim=1)


def selfcheck(engine, readout, text="The quick brown fox jumps over the lazy dog because"):
	tokens = engine.text_tokens(text)
	last = engine.n_layer - 1
	engine.reset()
	engine.set_capture([last])
	engine.decode(tokens, logits="last")
	ref = engine.last_logits()
	h = engine.pop_capture()[last][-1:]
	engine.set_capture(None)
	ours = readout.final(torch.as_tensor(h, device=readout.device)).cpu().numpy()[0]
	cos = float(np.dot(ours, ref) / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-12))
	top_ok = int(np.argmax(ours)) == int(np.argmax(ref))
	if cos < 0.99:
		readout.gamma = readout.gamma + 1.0
		alt = readout.final(torch.as_tensor(h, device=readout.device)).cpu().numpy()[0]
		cos_alt = float(np.dot(alt, ref) / (np.linalg.norm(alt) * np.linalg.norm(ref) + 1e-12))
		if cos_alt > cos:
			return {"cos": cos_alt, "top1": int(np.argmax(alt)) == int(np.argmax(ref)), "gamma_offset": 1.0}
		readout.gamma = readout.gamma - 1.0
	return {"cos": cos, "top1": top_ok, "gamma_offset": 0.0}
