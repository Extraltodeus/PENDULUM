import numpy as np
import torch

CORPUS = [
	"The river begins as a thin stream high in the mountains, where snow melts slowly through the spring. By the time it reaches the valley, it has gathered the water of a dozen smaller creeks and turned wide and brown. Farmers along its banks have depended on it for centuries, and the old stone mills that still stand near the bridges show how much of the local economy once turned with its current.",
	"To make a simple bread, mix flour, water, salt and a little yeast in a large bowl. Knead the dough for about ten minutes until it becomes smooth and elastic, then let it rest in a warm place until it doubles in size. Shape it into a loaf, let it rise again, and bake it in a hot oven until the crust is golden and the bottom sounds hollow when tapped.",
	"When she opened the door, the house was completely silent. The lights were off, the curtains drawn, and a single cup of tea sat cooling on the kitchen table. She called her brother's name twice, then walked slowly toward the stairs, listening for footsteps. Somewhere above her, a window had been left open, and the wind was moving the pages of a book.",
	"The committee met on Tuesday to review the budget for the coming year. Several members argued that the funds reserved for road repairs should be increased, while others insisted that the library and the public swimming pool deserved priority. After two hours of discussion, they agreed to postpone the final vote until the engineers had delivered their report.",
	"A function in a program is a named block of code that performs a specific task. It can receive inputs, called parameters, and it can return a result to the part of the program that called it. Splitting a large program into small functions makes it easier to read, to test and to change, because each piece can be understood on its own.",
	"Honey bees communicate the location of flowers through a movement known as the waggle dance. The direction of the dance on the vertical surface of the comb indicates the angle between the sun and the food source, while the duration of the waggling part tells the other workers how far they will have to fly. Researchers decoded this behavior in the twentieth century.",
	"Dear Martin, thank you for your letter and for the photographs of the new garden. The roses look wonderful, and I can hardly believe how quickly the young apple tree has grown. We are hoping to visit at the end of the summer, if the trains are running normally by then. Please give my regards to your mother and tell her that the recipe worked perfectly.",
	"The first computers filled entire rooms and consumed enormous amounts of electricity. Engineers programmed them by connecting cables and flipping switches, and a single calculation could take hours. Over the following decades, transistors and then integrated circuits made machines smaller, faster and cheaper, until a device more powerful than all of them fit easily in a pocket.",
]

SKIP = 6


def _smooth(x, k=3):
	if len(x) < k:
		return np.asarray(x, dtype=np.float64)
	pad = np.pad(np.asarray(x, dtype=np.float64), (k // 2, k // 2), mode="edge")
	return np.convolve(pad, np.ones(k) / k, mode="valid")


def _unit(x):
	x = np.asarray(x, dtype=np.float64)
	lo, hi = np.nanmin(x), np.nanmax(x)
	return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _participation(h):
	h = h - h.mean(0, keepdim=True)
	g = h @ h.T
	ev = torch.linalg.eigvalsh(g.double()).clamp_min(0)
	return float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12))


def analyze(engine, readout, corpus=CORPUS, max_tokens=200, on_progress=None, stop=None):
	layers = list(range(engine.n_layer))
	L = len(layers)
	agree1 = np.zeros(L)
	agree5 = np.zeros(L)
	kurt = np.zeros(L)
	auto = np.zeros(L)
	auto_null = np.zeros(L)
	norms = np.zeros(L)
	count = 0
	samples = {l: [] for l in layers}
	rng = np.random.default_rng(0)
	for ti, text in enumerate(corpus):
		if stop is not None and stop.is_set():
			break
		toks = engine.text_tokens(text)[:max_tokens]
		final = {}
		engine.reset()
		engine.set_capture(layers)
		engine.decode(toks, logits="all", on_logits=lambda i, lg: final.__setitem__(i, int(np.argmax(lg))))
		caps = engine.pop_capture()
		engine.set_capture(None)
		pos = list(range(SKIP, len(toks) - 1))
		target = torch.tensor([final[p] for p in pos], device=readout.device)
		n = len(pos)
		for li, l in enumerate(layers):
			h = torch.as_tensor(caps[l][pos], device=readout.device)
			lg = readout.logits(h, l)
			top5 = lg.topk(5, dim=-1).indices
			agree1[li] += float((top5[:, 0] == target).sum())
			agree5[li] += float((top5 == target[:, None]).any(-1).sum())
			z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True).clamp_min(1e-6)
			kurt[li] += float((z.pow(4).mean(-1) - 3.0).sum())
			top1 = top5[:, 0].cpu().numpy()
			auto[li] += float((top1[1:] == top1[:-1]).sum())
			perm = rng.permutation(top1)
			auto_null[li] += float((perm[1:] == perm[:-1]).sum())
			norms[li] += float(h.float().norm(dim=-1).sum())
			pick = rng.choice(n, size=min(n, 48), replace=False)
			samples[l].append(h[pick].float().cpu())
		count += n
		if on_progress:
			on_progress((ti + 1) / len(corpus))
	pr = np.array([_participation(torch.cat(samples[l]).to(readout.device)) / min(engine.n_embd, sum(len(s) for s in samples[l])) for l in layers])
	m = max(count, 1)
	pairs = max(count - len(corpus), 1)
	metrics = {
		"layers": layers,
		"agree_top1": (agree1 / m).tolist(),
		"agree_top5": (agree5 / m).tolist(),
		"kurtosis": (kurt / m).tolist(),
		"autocorr": ((auto - auto_null) / pairs).tolist(),
		"eff_dim": pr.tolist(),
		"norm": (norms / m).tolist(),
		"n_positions": count,
	}
	metrics["band"] = detect_band(metrics)
	return metrics


def detect_band(metrics):
	L = len(metrics["layers"])
	auto = _smooth(metrics["autocorr"])
	auto5 = _smooth(metrics["autocorr"], 5)
	a5 = _smooth(metrics["agree_top5"], 1)
	peak = int(np.argmax(auto))
	top = max(float(auto[peak]), 1e-6)
	top5 = max(float(auto5.max()), 1e-6)
	floor = int(round(0.2 * L))
	onset = next((i for i in range(floor, max(floor, peak) + 1) if auto5[i] >= 0.3 * top5 and auto5[min(i + 1, L - 1)] >= 0.3 * top5), int(round(0.35 * L)))
	tail = [i for i in range(peak, L) if auto[i] >= 0.5 * top]
	end = max(tail) if tail else int(round(0.9 * L))
	motor = next((i for i in range(onset + 1, L) if a5[i] >= 0.8), L - 1)
	end = min(end, motor - 1, L - 2)
	if end - onset < max(3, L // 6):
		onset, end = int(round(0.35 * L)), int(round(0.9 * L)) - 1
	return [int(onset), int(end)]
