import math
import time

import moderngl
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from ui import eva, evahud, theme

R_PUPIL = 0.62
R_L0 = 0.86
R_LN = 2.28
R_V0 = 2.62
R_V1 = 3.95
STRIDE = 11

VS_POINTS = """
#version 330
uniform mat4 mvp;
uniform float t;
uniform float pulse;
uniform float px;
uniform sampler2D act;
uniform int K;
uniform int L;
uniform vec3 c_plus;
uniform vec3 c_minus;
uniform int hover;
in vec3 pos;
in vec3 col;
in float size;
in vec4 prm;
out vec3 v_col;
void main() {
	vec3 p = pos;
	float kind = prm.w;
	float ph = prm.z;
	float bright = 1.0;
	vec3 c = col;
	if (kind > 3.5 && kind < 4.5) {
		float r = length(p.xz);
		float a = atan(p.z, p.x) + t * (0.35 - 0.28 * r / 0.62) + ph * 0.2;
		p.xz = vec2(cos(a), sin(a)) * r;
		bright = 0.55 + 0.45 * sin(t * 1.3 + r * 9.0 - ph * 3.0);
	}
	if (kind > 4.5) {
		float a = atan(p.z, p.x) + t * 0.012;
		float r = length(p.xz);
		p.xz = vec2(cos(a), sin(a)) * r;
	}
	if (kind > 0.5 && kind < 1.5) {
		float v = texture(act, vec2((prm.x * float(L - 1) + 0.5) / float(L), (prm.y + 0.5) / float(K))).r;
		float m = clamp(abs(v), 0.0, 1.6);
		c = (v >= 0.0 ? c_plus : c_minus) * (0.12 + m);
		bright = 1.0;
		if (int(prm.y + 0.5) == hover) { bright = 1.8; }
	}
	if (kind > 1.5 && kind < 2.5) {
		bright = 0.75 + 0.35 * sin(t * 2.0 - prm.x * 10.0 + ph * 6.28);
		if (int(prm.y + 0.5) == hover) { bright *= 1.9; }
	}
	p.y += 0.012 * sin(t * 0.8 + ph * 6.2831);
	float boost = 0.0;
	if (pulse >= 0.0 && kind < 2.5) {
		float d = prm.x - pulse;
		boost = exp(-d * d * 180.0) * 1.8;
	}
	v_col = c * (bright + boost);
	gl_Position = mvp * vec4(p, 1.0);
	gl_PointSize = clamp(size * px * (1.0 + 0.5 * boost) / gl_Position.w, 1.0, 26.0);
}
"""

FS_POINTS = """
#version 330
in vec3 v_col;
out vec4 frag;
void main() {
	vec2 d = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(d, d);
	if (r2 > 1.0) discard;
	float g = exp(-r2 * 3.2);
	frag = vec4(v_col * g, g);
}
"""

VS_LINES = """
#version 330
uniform mat4 mvp;
in vec3 pos;
in vec4 col;
out vec4 v_col;
void main() {
	v_col = col;
	gl_Position = mvp * vec4(pos, 1.0);
}
"""

FS_LINES = """
#version 330
in vec4 v_col;
out vec4 frag;
void main() {
	frag = vec4(v_col.rgb * v_col.a, v_col.a);
}
"""


def perspective(fovy, aspect, near, far):
	f = 1.0 / math.tan(math.radians(fovy) / 2)
	return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], dtype=np.float32)


def look_at(eye, target, up=(0, 1, 0)):
	eye, target, up = np.array(eye, dtype=np.float32), np.array(target, dtype=np.float32), np.array(up, dtype=np.float32)
	f = target - eye
	f /= np.linalg.norm(f)
	s = np.cross(f, up)
	s /= np.linalg.norm(s)
	u = np.cross(s, f)
	m = np.eye(4, dtype=np.float32)
	m[0, :3], m[1, :3], m[2, :3] = s, u, -f
	m[:3, 3] = -m[:3, :3] @ eye
	return m


def layer_radius(l, L):
	return R_L0 + (R_LN - R_L0) * (l / max(L - 1, 1))


def value_radius(v):
	return R_V0 + (R_V1 - R_V0) * (max(-1.0, min(1.0, v)) + 1) / 2


def demo_state():
	rng = np.random.default_rng(3)
	names = ["Warmth", "Sarcasm", "Altruism", "Emotion", "Creativity", "Vulgarity", "Humor", "Confidence", "Optimism", "Drama"]
	L = 32
	act = np.zeros((len(names), L))
	for k in range(len(names)):
		c = rng.uniform(12, 26)
		act[k] = rng.normal(0, 0.3) + np.exp(-((np.arange(L) - c) ** 2) / 18) * rng.uniform(-2.2, 2.2)
	vals = np.tanh(act[:, 10:28].mean(1))
	return {"L": L, "band": [10, 28], "traits": [{"id": n.lower(), "name": n, "value": float(v), "base": float(v) * 0.6, "plus": "", "minus": ""} for n, v in zip(names, vals)], "act": act.tolist(), "concepts": [], "demo": True}


FLIP = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)


BUTTONS = {"left": Qt.MouseButton.LeftButton, "right": Qt.MouseButton.RightButton, "middle": Qt.MouseButton.MiddleButton}
NAMES = {v: k for k, v in BUTTONS.items()}
CLICKS = {"none": None, "middle click": ("middle", 1), "middle double-click": ("middle", 2), "right click": ("right", 1), "right double-click": ("right", 2), "left double-click": ("left", 2)}
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))
NO_AXIS = 0.005


def similar_slots(a):
	a = np.asarray(a, dtype=np.float64)
	K = len(a)
	if K < 4:
		return list(range(K))
	u = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
	sim = u @ u.T
	best, best_s = list(range(K)), -1e18
	for s in range(K):
		path, left = [s], set(range(K)) - {s}
		while left:
			j = max(sorted(left), key=lambda j: sim[path[-1], j])
			path.append(j)
			left.discard(j)
		improved = True
		while improved:
			improved = False
			for i in range(1, K - 1):
				for j in range(i + 1, K):
					a0, b0, c0, d0 = path[i - 1], path[i], path[j], path[(j + 1) % K]
					if sim[a0, c0] + sim[b0, d0] > sim[a0, b0] + sim[c0, d0] + 1e-12:
						path[i:j + 1] = path[i:j + 1][::-1]
						improved = True
		score = sum(sim[path[i], path[(i + 1) % K]] for i in range(K))
		if score > best_s + 1e-12:
			best, best_s = path, score
	r = best.index(0)
	path = best[r:] + best[:r]
	if path[-1] < path[1]:
		path = [path[0]] + path[1:][::-1]
	slot = [0] * K
	for pos, k in enumerate(path):
		slot[k] = pos
	return slot


def help_text(orbit="left", pan="right", center="none", reset="none"):
	keys = lambda key, click: f"({key}" + (f" · {click}" if click and click != "none" else "") + ")"
	return "\n".join([
		"PERSONA IRIS",
		"",
		f"{orbit + '-drag':<17}orbit and tilt",
		f"{pan + '-drag':<17}move the view",
		"wheel            zoom (toward the cursor once the view was moved)",
		f"◎                re-center, keep zoom and tilt   {keys('C', center)}",
		f"⌖                reset tilt, zoom and position   {keys('Home', reset)}",
		"⟳                stop / start the rotation   (R)",
		"◇                concept labels: auto → all → off",
		"(mouse buttons and clicks: settings ⚙ → NAVIGATION)",
		"",
		"hover a trait    its poles and what its lever reads as,",
		"                 its row in LEVERS and its layers in WORKSPACE",
		"hover a concept  its card (lift, probability, closest axis)",
		"left-click       pin the card   (Esc or × to close)",
		"",
		"rings = layers · orange ring = workspace band",
		"polygon = measured profile (grey = baseline)",
		"clusters = where each trait lights up, layer by layer",
	])


class IrisView(QWidget):
	traitHovered = Signal(int)
	traitClicked = Signal(int)
	traitPinned = Signal(int)
	rotateToggled = Signal(bool)
	conceptsModeChanged = Signal(str)
	conceptHovered = Signal(int)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setMinimumSize(420, 320)
		self.setMouseTracking(True)
		self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
		self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
		self.rotate = True
		self.t_last = time.perf_counter()
		self.chips = {}
		self.chip_hover = None
		self.pinned = -1
		self.plate = ""
		self.peek = None
		self.card_close = None
		self.card_rect = None
		self.cmode = "auto"
		self.chover = -1
		self.cpinned = -1
		self.bhover = ""
		self.bpinned = ""
		self.hits = []
		self.hud_hits = []
		self.card_tips = []
		self.concept_pts = []
		self.pan = [0.0, 0.0]
		self.pan_drag = None
		self.goal = {}
		self.press = None
		self.orbit_name = "left"
		self.pan_name = "right"
		self.center_click = "none"
		self.reset_click = "none"
		self.click_map = {}
		self.spread = True
		self.order_mode = "profile"
		self.slot = []
		self._auto_key = None
		self._auto = []
		self.ctx = None
		self.gl_error = ""
		self.fb_size = None
		self.frame = None
		self.ov_img = None
		self.ov_t = 0.0
		self.ov_dirty = True
		self.state = demo_state()
		self.dirty = True
		self.t0 = time.perf_counter()
		self.az = -0.55
		self.el = 0.52
		self.zoom = 1.0
		self.spin = 0.035
		self.drag = None
		self.hover = -1
		self.pulse_t = -10.0
		self.live = None
		self.live_level = 0.0
		self.labels = []
		self.caption = ""
		self.hud = None
		self.max_concepts = 6
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.update)
		self.timer.start(16)
		self.mvp = np.eye(4, dtype=np.float32)

	def set_fps(self, fps):
		self.timer.start(max(8, int(1000 / max(fps, 1))))

	def set_state(self, state):
		self.state = state
		self.slot = self._slots()
		self._auto_key = None
		self.dirty = self.ov_dirty = True
		if self.pinned >= len(state.get("traits") or []):
			self.pinned = -1

	def _slots(self):
		K = len(self.state.get("traits") or [])
		act = self.state.get("order_act")
		if act is None:
			act = self.state.get("act")
		if self.order_mode != "similar" or K < 4 or act is None:
			return list(range(K))
		a = np.asarray(act, dtype=np.float64)
		if a.ndim != 2 or a.shape[0] != K or not np.isfinite(a).all():
			return list(range(K))
		lo, hi = self.state.get("band") or [0, a.shape[1] - 1]
		if 0 <= lo <= hi < a.shape[1]:
			a = a[:, lo:hi + 1]
		return similar_slots(a)

	def set_order(self, mode):
		mode = mode if mode in ("profile", "similar") else "profile"
		if mode == self.order_mode:
			return
		self.order_mode = mode
		self.slot = self._slots()
		self._auto_key = None
		self.dirty = self.ov_dirty = True

	def arm(self, k, K):
		s = self.slot[k] if 0 <= k < len(self.slot) and len(self.slot) == K else k
		return 2 * math.pi * s / max(K, 1) - math.pi / 2

	def axis_of(self, c):
		k = int(c.get("trait", -1))
		return k if 0 <= k < len(self.state.get("traits") or []) and abs(float(c.get("tv", 0.0) or 0.0)) >= NO_AXIS else -1

	def set_spread(self, on):
		if bool(on) != self.spread:
			self.spread = bool(on)
			self._auto_key = None
			self.ov_dirty = True

	def set_clicks(self, center, reset):
		self.center_click = center if center in CLICKS else "none"
		self.reset_click = reset if reset in CLICKS and reset != self.center_click else "none"
		self.click_map = {CLICKS[n]: action for n, action in ((self.center_click, "center"), (self.reset_click, "reset")) if CLICKS.get(n)}

	def set_live(self, odds_delta):
		self.live = np.asarray(odds_delta, dtype=np.float32)
		self.live_level = 1.0
		self.pulse_t = time.perf_counter()

	def pulse(self):
		self.pulse_t = time.perf_counter()

	def _init_gl(self):
		self.ctx = moderngl.create_standalone_context(require=330)
		self.prog = self.ctx.program(vertex_shader=VS_POINTS, fragment_shader=FS_POINTS)
		self.lprog = self.ctx.program(vertex_shader=VS_LINES, fragment_shader=FS_LINES)
		self.tex = None
		self.vbo = self.vao = self.lvbo = self.lvao = None
		self.msaa = self.fb = None

	def _framebuffers(self, w, h):
		if self.fb_size == (w, h):
			return
		for fb in (self.msaa, self.fb):
			if fb is not None:
				for a in fb.color_attachments:
					a.release()
				fb.release()
		samples = min(4, int(self.ctx.max_samples))
		self.msaa = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((w, h), 4, samples=samples)]) if samples > 1 else None
		self.fb = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((w, h), 4)])
		self.buf = bytearray(w * h * 4)
		self.fb_size = (w, h)

	def _build(self):
		st = self.state
		L = int(st["L"])
		K = max(1, len(st["traits"]))
		lo, hi = st.get("band") or [int(L * 0.33), int(L * 0.9)]
		rng = np.random.default_rng(7)
		pts = []

		def add(pos, col, size, prm):
			n = len(pos)
			arr = np.zeros((n, STRIDE), dtype=np.float32)
			arr[:, 0:3] = pos
			arr[:, 3:6] = col
			arr[:, 6] = size
			arr[:, 7:11] = prm
			pts.append(arr)

		for l in range(L):
			n = 520 + 12 * l
			a = rng.uniform(0, 2 * np.pi, n)
			r = layer_radius(l, L) + rng.normal(0, 0.012, n)
			y = rng.normal(0, 0.012, n)
			inband = lo <= l <= hi
			base = np.array(theme.WHITE_RGB) * (0.22 if inband else 0.085)
			if inband:
				base = base * 0.8 + np.array(theme.ACCENT_RGB) * 0.04
			add(np.stack([np.cos(a) * r, y, np.sin(a) * r], 1), base, rng.uniform(1.4, 2.6, n), np.stack([np.full(n, l / max(L - 1, 1)), np.full(n, -1.0), rng.uniform(0, 1, n), np.zeros(n)], 1))
		for k in range(K):
			th = self.arm(k, K)
			for l in range(L):
				n = 34
				a = th + rng.normal(0, 0.035, n)
				r = layer_radius(l, L) + rng.normal(0, 0.016, n)
				add(np.stack([np.cos(a) * r, rng.normal(0, 0.01, n), np.sin(a) * r], 1), np.ones(3), rng.uniform(2.0, 3.4, n), np.stack([np.full(n, l / max(L - 1, 1)), np.full(n, float(k)), rng.uniform(0, 1, n), np.ones(n)], 1))
		for k, tr in enumerate(st["traits"]):
			th = self.arm(k, K)
			v = float(tr.get("value", 0.0))
			color = np.array(theme.ACCENT_RGB if v >= 0 else theme.COOL_RGB)
			r0, r1 = R_LN + 0.08, value_radius(v)
			self._dendrite(add, rng, th, r0, r1, color, k, 1.0, depth=0)
		n = 2600
		rr = R_PUPIL * np.sqrt(rng.uniform(0.02, 1, n))
		arm = rng.integers(0, 3, n)
		a = arm * (2 * np.pi / 3) + rr * 7.5 + rng.normal(0, 0.18, n)
		add(np.stack([np.cos(a) * rr, rng.normal(0, 0.006, n), np.sin(a) * rr], 1), np.array(theme.WHITE_RGB) * 0.28, rng.uniform(1.0, 2.2, n), np.stack([np.zeros(n), np.full(n, -1.0), rng.uniform(0, 1, n), np.full(n, 4.0)], 1))
		n = 1800
		a = rng.uniform(0, 2 * np.pi, n)
		rr = rng.uniform(4.2, 9.0, n)
		add(np.stack([np.cos(a) * rr, rng.normal(0, 0.9, n), np.sin(a) * rr], 1), np.array(theme.WHITE_RGB) * rng.uniform(0.03, 0.09, n)[:, None], rng.uniform(1.0, 2.0, n), np.stack([np.full(n, -5.0), np.full(n, -1.0), rng.uniform(0, 1, n), np.full(n, 5.0)], 1))
		concepts = st.get("concepts") or []
		lifts = [float(c.get("lift", 0.0)) for c in concepts]
		lmin, lmax = (min(lifts), max(lifts)) if lifts else (0.0, 1.0)
		free = 0
		for c in concepts:
			k = self.axis_of(c)
			strength = (float(c.get("lift", 0.0)) - lmin) / max(lmax - lmin, 1e-6)
			align = min(1.0, abs(float(c.get("tv", 0.0) or 0.0)) * 3.0)
			if k >= 0:
				a = self.arm(k, K) + rng.normal(0, 0.1 + 0.32 * (1.0 - align))
			else:
				a = -math.pi / 2 + math.pi / K + GOLDEN * free + rng.normal(0, 0.04)
				free += 1
			r = R_V1 + 0.45 + rng.uniform(0, 1.9) * (1.15 - 0.5 * strength)
			y = float(rng.normal(0.05, 0.14))
			n = 10 + int(20 * strength)
			spread = 0.025 + 0.02 * strength
			add(np.stack([np.cos(a) * r + rng.normal(0, spread, n), y + rng.normal(0, spread, n), np.sin(a) * r + rng.normal(0, spread, n)], 1), np.array(theme.ACCENT_RGB) * (0.3 + 0.35 * strength) + 0.08, rng.uniform(1.3, 2.4 + 1.0 * strength, n), np.stack([np.full(n, -3.0), np.full(n, float(k)), rng.uniform(0, 1, n), np.full(n, 3.0)], 1))
			c["_pos"] = (float(np.cos(a) * r), y, float(np.sin(a) * r))
		data = np.concatenate(pts).astype("f4")
		if self.vbo is not None:
			self.vbo.release()
			self.vao.release()
		self.vbo = self.ctx.buffer(data.tobytes())
		self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f 1f 4f", "pos", "col", "size", "prm")])
		self.n_points = len(data)
		self._build_lines(L, K, lo, hi)
		if self.tex is not None:
			self.tex.release()
		self.tex = self.ctx.texture((L, K), 1, dtype="f4")
		self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
		self.tex.repeat_x = self.tex.repeat_y = False
		self.K, self.L = K, L
		self._auto_key = None
		self.dirty = False

	def _dendrite(self, add, rng, th, r0, r1, color, k, strength, depth):
		steps = max(12, int(abs(r1 - r0) * 120))
		rs = np.linspace(r0, r1, steps)
		wander = np.cumsum(rng.normal(0, 0.006 * (1 + depth), steps))
		lift = np.cumsum(rng.normal(0, 0.004, steps))
		a = th + wander
		pos = np.stack([np.cos(a) * rs, lift, np.sin(a) * rs], 1)
		frac = np.linspace(0, 1, steps)
		n_per = 5 if depth == 0 else 3
		cloud = pos.repeat(n_per, 0) + rng.normal(0, 0.006 + 0.004 * depth, (steps * n_per, 3))
		fr = frac.repeat(n_per)
		fade = (0.95 - 0.35 * fr) * strength
		add(cloud, color[None, :] * fade[:, None] * (0.8 if depth == 0 else 0.5), rng.uniform(1.6, 3.2, len(cloud)) * (1.0 if depth == 0 else 0.8), np.stack([fr, np.full(len(cloud), float(k)), rng.uniform(0, 1, len(cloud)), np.full(len(cloud), 2.0)], 1))
		if depth >= 2:
			return
		for _ in range(int(rng.integers(3, 7 - depth * 2))):
			i = int(rng.integers(steps // 5, steps))
			rb = rs[i]
			span = (0.18 + 0.35 * rng.random()) / (1 + depth)
			side = rng.choice([-1, 1])
			self._dendrite(add, rng, float(a[i]) + side * 0.02, rb, rb + span * np.sign(r1 - r0 if r1 != r0 else 1), color, k, strength * 0.7, depth + 1)

	def _build_lines(self, L, K, lo, hi):
		segs = []

		def circle(r, y, col, n=240, dash=False):
			a = np.linspace(0, 2 * np.pi, n + 1)
			p = np.stack([np.cos(a) * r, np.full(n + 1, y), np.sin(a) * r], 1)
			for i in range(n):
				if dash and i % 3 == 2:
					continue
				segs.append((p[i], p[i + 1], col))

		for v, alpha in ((-1.0, 0.16), (-0.5, 0.07), (0.0, 0.2), (0.5, 0.07), (1.0, 0.16)):
			circle(value_radius(v), 0.0, (*theme.WHITE_RGB, alpha), dash=v not in (-1.0, 0.0, 1.0))
		circle(layer_radius(lo, L) - 0.03, 0.0, (*theme.ACCENT_RGB, 0.22), 200)
		circle(layer_radius(hi, L) + 0.03, 0.0, (*theme.ACCENT_RGB, 0.22), 200)
		circle(R_PUPIL + 0.08, 0.0, (*theme.WHITE_RGB, 0.12), 160)
		for r, tilt, alpha in ((5.2, 0.0, 0.08), (6.4, 0.18, 0.06), (7.8, -0.12, 0.05)):
			a = np.linspace(0, 2 * np.pi, 361)
			p = np.stack([np.cos(a) * r, np.sin(a) * r * math.sin(tilt), np.sin(a) * r * math.cos(tilt)], 1)
			for i in range(360):
				segs.append((p[i], p[i + 1], (*theme.WHITE_RGB, alpha)))
		for k in range(K):
			th = self.arm(k, K)
			d = np.array([math.cos(th), 0.0, math.sin(th)])
			segs.append((d * R_LN, d * (R_V1 + 0.12), (*theme.WHITE_RGB, 0.13 if k != self.focus() else 0.5)))
		for key, col in (("base", (*theme.WHITE_RGB, 0.38)), ("value", (*theme.ACCENT_RGB, 0.95))):
			vals = [float(t.get(key, 0.0) or 0.0) for t in self.state["traits"]]
			if not vals or (key == "base" and all(t.get("base") is None for t in self.state["traits"])):
				continue
			ps = []
			for k in sorted(range(len(vals)), key=lambda k: self.arm(k, K)):
				th = self.arm(k, K)
				r = value_radius(vals[k])
				ps.append(np.array([math.cos(th) * r, 0.02, math.sin(th) * r]))
			for i in range(len(ps)):
				segs.append((ps[i], ps[(i + 1) % len(ps)], col))
		data = np.zeros((len(segs) * 2, 7), dtype="f4")
		for i, (a, b, c) in enumerate(segs):
			data[2 * i, :3], data[2 * i + 1, :3] = a, b
			data[2 * i, 3:], data[2 * i + 1, 3:] = c, c
		if self.lvbo is not None:
			self.lvbo.release()
			self.lvao.release()
		self.lvbo = self.ctx.buffer(data.tobytes())
		self.lvao = self.ctx.vertex_array(self.lprog, [(self.lvbo, "3f 4f", "pos", "col")])

	def _act_texture(self):
		act = np.asarray(self.state.get("act") or np.zeros((self.K, self.L)), dtype=np.float32)
		if act.shape != (self.K, self.L):
			act = np.zeros((self.K, self.L), dtype=np.float32)
		act = act / 2.2
		if self.live is not None and self.live.shape == act.shape and self.live_level > 0.01:
			act = act * (1 - 0.6 * self.live_level) + self.live / 1.6 * self.live_level
			self.live_level *= 0.965
		return np.ascontiguousarray(act, dtype="f4")

	def _advance(self):
		now = time.perf_counter()
		dt = min(now - self.t_last, 0.1)
		self.t_last = now
		if self.rotate and self.drag is None:
			self.az += self.spin * dt
		if self.goal:
			a = min(1.0, dt * 9.0)
			done = True
			for key, value in self.goal.items():
				cur = getattr(self, key)
				if key == "pan":
					nxt = [c + (v - c) * a for c, v in zip(cur, value)]
					far = any(abs(v - n) > 1e-3 for v, n in zip(value, nxt))
				else:
					nxt = cur + (value - cur) * a
					far = abs(value - nxt) > 1e-3
				setattr(self, key, nxt if far else value)
				done = done and not far
			self.ov_dirty = True
			if done:
				self.goal = {}

	def center_view(self):
		self.goal = {"pan": [0.0, 0.0]}

	def set_rotate(self, on):
		self.rotate = bool(on)
		self.update()

	def focus(self):
		return self.hover if self.hover >= 0 else self.pinned

	def set_peek(self, card):
		self.peek = card

	def pin(self, k):
		self.pinned = k if 0 <= k < len(self.state.get("traits") or []) else -1
		if self.pinned >= 0:
			self.cpinned = -1
			self.bpinned = ""
		self.lines_dirty = self.ov_dirty = True
		self.traitPinned.emit(self.pinned)

	def pin_concept(self, ci):
		self.cpinned = ci if 0 <= ci < len(self.state.get("concepts") or []) else -1
		if self.cpinned >= 0:
			self.bpinned = ""
			if self.pinned >= 0:
				self.pin(-1)
		self.ov_dirty = True

	def pin_band(self, key):
		self.bpinned = key if key in ("onset", "motor") else ""
		if self.bpinned:
			self.cpinned = -1
			if self.pinned >= 0:
				self.pin(-1)
		self.ov_dirty = True

	def set_bhover(self, key):
		if key != self.bhover:
			self.bhover = key
			self.ov_dirty = True

	def band_card(self, key):
		st = self.state
		L = int(st.get("L") or 0)
		lo, hi = st.get("band") or [0, 0]
		ws = st.get("ws") or {}
		ac, a5 = ws.get("autocorr") or [], ws.get("agree_top5") or []
		at = lambda arr, l, f: format(arr[l], f) if 0 <= l < len(arr) else "—"
		if key == "onset":
			return {"title": f"WORKSPACE ONSET · L{lo}", "rows": [
				("meaning", "from this layer on, the lens reads abstract content", theme.TEXT),
				("", "that persists from token to token: the model's workspace", theme.TEXT),
				("measured by", f"autocorrelation of the lens top-1 vs a shuffled order: {at(ac, lo, '+.3f')}", theme.MUTED),
				("before", f"L0–L{max(lo - 1, 0)}: token-level processing, noisy readings", theme.MUTED),
				("band", f"L{lo}–L{hi} of {L} layers", theme.ACCENT2),
				("levers", "persona levers act in the first 65% of the band", theme.ACCENT2),
			]}
		return {"title": f"MOTOR REGIME · after L{hi}", "rows": [
			("meaning", f"from L{hi + 1} on, the lens agrees with the final output:", theme.TEXT),
			("", "the model is no longer thinking, it is choosing its words", theme.TEXT),
			("measured by", f"top-5 agreement with the real output: {at(a5, hi, '.2f')} at L{hi} → {at(a5, hi + 1, '.2f')} at L{hi + 1}", theme.MUTED),
			("band", f"L{lo}–L{hi} of {L} layers: where the persona lives", theme.ACCENT2),
		]}

	def _hit_at(self, pos):
		return next(((kind, key) for r, kind, key in reversed(getattr(self, "hits", [])) if r.contains(pos)), None)

	def set_chover(self, ci):
		if ci != self.chover:
			self.chover = ci
			self.ov_dirty = True
			self.conceptHovered.emit(ci)

	def set_cmode(self, mode):
		self.cmode = mode if mode in ("auto", "all", "off") else "auto"
		self.ov_dirty = True

	def _auto_order(self, concepts):
		key = (len(concepts), self.max_concepts, self.spread)
		if self._auto_key == key:
			return self._auto
		order = list(range(len(concepts)))
		if self.spread:
			m = max(1, self.max_concepts)
			slices = [[] for _ in range(m)]
			for ci, c in enumerate(concepts):
				pos = c.get("_pos")
				if pos:
					a = (math.atan2(pos[2], pos[0]) + math.pi / 2) % (2 * math.pi)
					slices[min(m - 1, int(a / (2 * math.pi) * m))].append(ci)
			order, depth = [], 0
			while True:
				layer = [s[depth] for s in slices if depth < len(s)]
				if not layer:
					break
				order += sorted(layer)
				depth += 1
		self._auto_key, self._auto = key, order
		return order

	def concept_card(self, ci):
		concepts = self.state.get("concepts") or []
		c = concepts[ci]
		traits = self.state.get("traits") or []
		k, tv = self.axis_of(c), float(c.get("tv", 0.0) or 0.0)
		rows = []
		if c.get("en"):
			rows.append(("english", c["en"], theme.TEXT))
		rows.append(("lift", f"×{math.exp(c.get('lift', 0.0)):.1f}   more likely while the model speaks than on neutral text", theme.ACCENT2))
		rows.append(("probability", f"{c.get('p', 0.0):.4f}   mean lens reading in the workspace band, while speaking", theme.TEXT))
		if 0 <= k < len(traits):
			t = traits[k]
			rows.append(("closest axis", f"{t['name']}   {tv:+.2f}, toward '{(t.get('plus') if tv >= 0 else t.get('minus')) or '?'}'", theme.TEXT))
		else:
			rows.append(("closest axis", "none: no trait axis reads this token", theme.MUTED))
		rows.append(("rank", f"#{ci + 1} of {len(concepts)} signature concepts of this profile", theme.MUTED))
		rows.append(("token", repr(c["str"]), theme.DIM))
		return {"title": (c["str"].strip() or repr(c["str"])).upper(), "rows": rows}

	def reset_view(self):
		self.goal = {"el": 0.52, "zoom": 1.0, "pan": [0.0, 0.0]}

	def _camera(self, w, h):
		az = self.az
		aspect = w / max(h, 1)
		dpr = self.devicePixelRatioF()
		top = min((118.0 if self.hud.get("side") else 92.0) * dpr, h * 0.3) if self.hud else 0.0
		fit_h = (R_V1 + 1.4 + 230.0 / max(w / dpr, 300) * 4.0) / (math.tan(math.radians(18.0)) * aspect * 0.9)
		fit_v = (R_V1 + 0.9) * (0.35 + math.sin(self.el)) / (math.tan(math.radians(18.0)) * 0.95) * h / max(h - top, 1)
		dist = max(fit_h, fit_v, 7.0) * self.zoom
		self.dist = dist
		tx, tz = self.pan
		eye = (tx + dist * math.cos(self.el) * math.cos(az), dist * math.sin(self.el), tz + dist * math.cos(self.el) * math.sin(az))
		proj = perspective(36.0, aspect, 0.1, 120.0)
		shift = np.eye(4, dtype=np.float32)
		shift[1, 3] = -top / max(h, 1)
		view = look_at(eye, (tx, -0.35, tz))
		return shift @ proj @ view

	def _render(self):
		if self.ctx is None:
			self._init_gl()
		if self.dirty:
			self._build()
		elif getattr(self, "lines_dirty", False):
			lo, hi = self.state.get("band") or [int(self.L * 0.33), int(self.L * 0.9)]
			self._build_lines(self.L, self.K, lo, hi)
		self.lines_dirty = False
		dpr = self.devicePixelRatioF()
		w, h = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
		self._framebuffers(w, h)
		self._advance()
		target = self.msaa or self.fb
		target.use()
		self.ctx.viewport = (0, 0, w, h)
		self.ctx.clear(0.012, 0.012, 0.016, 1.0)
		self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
		self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
		self.ctx.blend_func = moderngl.ONE, moderngl.ONE
		mvp = self._camera(w, h)
		self.mvp = mvp
		m = (FLIP @ mvp).T.astype("f4").tobytes()
		self.lprog["mvp"].write(m)
		self.lvao.render(moderngl.LINES)
		self.tex.write(self._act_texture().tobytes())
		self.tex.use(0)
		now = time.perf_counter()
		pulse = (now - self.pulse_t) / 0.55
		self.prog["mvp"].write(m)
		self.prog["t"].value = now - self.t0
		self.prog["pulse"].value = pulse if pulse <= 1.25 else -1.0
		self.prog["px"].value = 11.0 * dpr * min(w, h) / 900.0
		self.prog["act"].value = 0
		self.prog["K"].value = self.K
		self.prog["L"].value = self.L
		self.prog["c_plus"].value = theme.ACCENT_RGB
		self.prog["c_minus"].value = theme.COOL_RGB
		self.prog["hover"].value = self.focus()
		self.vao.render(moderngl.POINTS)
		if self.msaa is not None:
			self.ctx.copy_framebuffer(self.fb, self.msaa)
		self.fb.read_into(self.buf, components=4, alignment=1)
		img = QImage(self.buf, w, h, w * 4, QImage.Format.Format_RGBX8888)
		img.setDevicePixelRatio(dpr)
		self.frame = img

	def paintEvent(self, e):
		if not self.gl_error:
			try:
				self._render()
			except Exception as exc:
				self.gl_error = str(exc)
		p = QPainter(self)
		try:
			if self.frame is not None and not self.gl_error:
				p.drawImage(0, 0, self.frame)
			else:
				p.fillRect(self.rect(), theme.qc(theme.PANEL))
				p.setPen(theme.qc(theme.ERR))
				p.setFont(theme.mono(8))
				p.drawText(self.rect().adjusted(12, 12, -12, -12), Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, f"3D view unavailable: {self.gl_error}")
			p.drawImage(0, 0, self._labels_image())
			p.setRenderHint(QPainter.RenderHint.Antialiasing)
			if eva.on() and self.plate:
				evahud.plate_title(self, p)
			if self.hud:
				self._hud(p, self.hud)
			self._chips(p)
			n_c = len(self.state.get("concepts") or [])
			if self.peek:
				self._draw_card(p, self.peek["title"], self.peek["rows"], False)
			elif 0 <= self.chover < n_c and self.chover != self.cpinned:
				card = self.concept_card(self.chover)
				self._draw_card(p, card["title"], card["rows"], False)
			elif self.bhover and self.bhover != self.bpinned:
				card = self.band_card(self.bhover)
				self._draw_card(p, card["title"], card["rows"], False)
			elif 0 <= self.cpinned < n_c:
				card = self.concept_card(self.cpinned)
				self._draw_card(p, card["title"], card["rows"], True)
			elif self.bpinned:
				card = self.band_card(self.bpinned)
				self._draw_card(p, card["title"], card["rows"], True)
			elif self.pinned >= 0:
				self._card(p, self.state["traits"][self.pinned])
		finally:
			p.end()

	def _words(self, items, n):
		out, seen = [], set()
		for s in items or []:
			w = s.strip()
			if w and w.lower() not in seen:
				seen.add(w.lower())
				out.append(w)
		return out[:n]

	def _card(self, p, tr):
		fmt = lambda v: "—" if v is None else f"{v:+.2f}"
		rows = [("", f"+ {tr.get('plus') or '?'}   ↔   − {tr.get('minus') or '?'}", theme.TEXT)]
		vb, vn = tr.get("v_base"), tr.get("v_now")
		meas = f"base {fmt(vb)}" + (f"   now {fmt(vn)}   ({'▲' if vn - (vb or 0) >= 0 else '▼'}{abs(vn - (vb or 0)):.2f})" if vn is not None else "")
		rows.append(("measured", meas if vb is not None or vn is not None else "not profiled yet", theme.TEXT))
		band = self.state.get("band")
		if tr.get("peaks"):
			rows.append(("lights up", " · ".join(f"L{l}" for l in tr["peaks"]) + (f"   (band L{band[0]}–L{band[1]})" if band else ""), theme.TEXT))
		if tr.get("layers"):
			applied = tr.get("applied") or 0.0
			setting = tr.get("setting") or 0.0
			rows.append(("lever", f"L{tr['layers'][0]}–L{tr['layers'][-1]} · gain {tr.get('gain') or 0:.2f} · setting {setting:+.2f}" + (f" → applied {applied:+.2f}" if abs(applied - setting) > 1e-3 else ""), theme.TEXT))
			if tr.get("reads"):
				rows.append(("reads as +", " · ".join(self._words(tr["reads"], 7)), theme.ACCENT2))
			if tr.get("reads_minus"):
				rows.append(("reads as −", " · ".join(self._words(tr["reads_minus"], 7)), theme.COOL2))
		else:
			rows.append(("lever", "not calibrated yet (move its slider or CALIBRATE)", theme.MUTED))
		if tr.get("lex_plus"):
			rows.append(("measured with +", ", ".join(self._words(tr["lex_plus"], 8)), theme.MUTED))
		if tr.get("lex_minus"):
			rows.append(("measured with −", ", ".join(self._words(tr["lex_minus"], 8)), theme.MUTED))
		if tr.get("concepts"):
			rows.append(("model concepts", " · ".join(tr["concepts"][:6]), theme.MUTED))
		self._draw_card(p, tr["name"].upper(), rows, True)

	def _draw_card(self, p, title, rows, pinned):
		if eva.on():
			evahud.card(self, p, title, rows, pinned)
			return
		p.setFont(theme.mono(7.5))
		fm = p.fontMetrics()
		lw = max(fm.horizontalAdvance(k) for k, _v, _c in rows) + 14
		p.setFont(theme.mono(9, True))
		tw = p.fontMetrics().horizontalAdvance(title) + 60
		p.setFont(theme.mono(7.5))
		width = min(max(max(fm.horizontalAdvance(v) for _k, v, _c in rows) + lw + 28, tw), self.width() * 0.62)
		lh = 15
		height = 34 + lh * len(rows) + 8
		x, y = self.card_pos(width, height)
		r = QRectF(x, y, width, height)
		if pinned:
			if self.card_rect is None or abs(self.card_rect.width() - width) > 0.5 or abs(self.card_rect.height() - height) > 0.5:
				self.ov_dirty = True
			self.card_rect = r
		p.setPen(QPen(theme.qc(theme.LINE2), 1))
		p.setBrush(theme.qc(theme.PANEL2, 238))
		p.drawRect(r)
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(theme.qc(theme.ACCENT if pinned else theme.COOL))
		p.drawRect(QRectF(x, y, width, 2))
		p.setFont(theme.mono(9, True))
		p.setPen(theme.qc(theme.TEXT))
		p.drawText(QPointF(x + 12, y + 22), title)
		if pinned:
			self.card_close = QRectF(x + width - 26, y + 6, 20, 20)
			p.setPen(QPen(theme.qc(theme.ACCENT if self.chip_hover == "close" else theme.MUTED), 1.2))
			c = self.card_close.center()
			p.drawLine(QPointF(c.x() - 4, c.y() - 4), QPointF(c.x() + 4, c.y() + 4))
			p.drawLine(QPointF(c.x() - 4, c.y() + 4), QPointF(c.x() + 4, c.y() - 4))
		p.setFont(theme.mono(7.5))
		yy = y + 34 + 10
		for k, v, col in rows:
			p.setPen(theme.qc(theme.DIM))
			p.drawText(QPointF(x + 12, yy), k)
			p.setPen(theme.qc(col))
			p.drawText(QPointF(x + 12 + lw, yy), fm.elidedText(v, Qt.TextElideMode.ElideRight, int(width - lw - 24)))
			yy += lh

	def card_pos(self, width, height, margin=12.0):
		x, y = margin, self.height() - height - margin
		chips = list((getattr(self, "chips", None) or {}).values())
		if chips and x + width > min(r.left() for r in chips) - 8 and y + height > min(r.top() for r in chips) - 6:
			y = min(r.top() for r in chips) - 8 - height
		return x, y

	def _chips(self, p):
		if eva.on():
			evahud.chips(self, p)
			return
		p.setFont(theme.mono(7.5))
		fm = p.fontMetrics()
		items = [("help", "?"), ("reset", "⌖ RESET"), ("center", "◎ CENTER"), ("rotate", "⟳ ROTATE" if self.rotate else "❚❚ ROTATE"), ("concepts", f"◇ CONCEPTS {self.cmode.upper()}")]
		x = self.width() - 10.0
		y = self.height() - 30.0
		self.chips = {}
		for key, text in items:
			w = fm.horizontalAdvance(text) + 16
			x -= w
			r = QRectF(x, y, w, 20)
			self.chips[key] = r
			hot = key == self.chip_hover
			on = key == "rotate" and self.rotate
			p.setPen(QPen(theme.qc(theme.ACCENT if hot else (theme.ACCENT2 if on else theme.LINE2), 220 if hot else 160), 1))
			p.setBrush(theme.qc(theme.PANEL2, 200))
			p.drawRect(r)
			p.setPen(theme.qc(theme.ACCENT2 if hot or on else theme.MUTED))
			p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
			x -= 6

	def _chip_at(self, pos):
		if (self.pinned >= 0 or self.cpinned >= 0 or self.bpinned) and self.card_close is not None and self.card_close.contains(pos):
			return "close"
		return next((k for k, r in self.chips.items() if r.contains(pos)), None)

	def unpin_all(self):
		if self.cpinned >= 0:
			self.pin_concept(-1)
		if self.bpinned:
			self.pin_band("")
		if self.pinned >= 0:
			self.pin(-1)

	def _concept_at(self, pos):
		best, bd = -1, 13.0
		for ci, pt in self.concept_pts:
			d = math.hypot(pt.x() - pos.x(), pt.y() - pos.y())
			if d < bd:
				best, bd = ci, d
		return best

	def help_tip(self):
		text = help_text(self.orbit_name, self.pan_name, self.center_click, self.reset_click)
		return evahud.help_html(text) if eva.on() else text

	def show_help(self):
		r = self.chips.get("help")
		at = self.mapToGlobal(r.topLeft().toPoint() if r else self.rect().center())
		QToolTip.showText(at, self.help_tip(), self, self.rect(), 30000)

	def keyPressEvent(self, e):
		if e.key() == Qt.Key.Key_R and not e.modifiers():
			self.set_rotate(not self.rotate)
			self.rotateToggled.emit(self.rotate)
		elif e.key() == Qt.Key.Key_Home:
			self.reset_view()
		elif e.key() == Qt.Key.Key_C and not e.modifiers():
			self.center_view()
		elif e.key() == Qt.Key.Key_Escape and (self.pinned >= 0 or self.cpinned >= 0 or self.bpinned):
			self.unpin_all()
		else:
			super().keyPressEvent(e)

	def _labels_image(self):
		dpr = self.devicePixelRatioF()
		size = (max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
		now = time.perf_counter()
		if self.ov_img is None or (self.ov_img.width(), self.ov_img.height()) != size:
			self.ov_img = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
			self.ov_dirty = True
		if self.ov_dirty or self.drag is not None or now - self.ov_t > 1 / 15:
			self.ov_img.setDevicePixelRatio(dpr)
			self.ov_img.fill(Qt.GlobalColor.transparent)
			q = QPainter(self.ov_img)
			try:
				self._overlay(q)
			finally:
				q.end()
			self.ov_t, self.ov_dirty = now, False
		return self.ov_img

	def project(self, p):
		v = self.mvp @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
		if v[3] <= 0.01:
			return None
		return QPointF((v[0] / v[3] * 0.5 + 0.5) * self.width(), (1 - (v[1] / v[3] * 0.5 + 0.5)) * self.height())

	def _overlay(self, p):
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		st = self.state
		K = max(1, len(st["traits"]))
		self.labels = []
		center = self.project((0.0, 0.0, 0.0))
		cx = center.x() if center is not None else self.width() / 2
		self.rects = [QRectF(0, 0, self.width(), evahud.hud_bottom(self, self.hud) if eva.on() else (126 if self.hud.get("side") else 100))] if self.hud else []
		if (self.pinned >= 0 or self.cpinned >= 0 or self.bpinned) and self.card_rect is not None:
			self.rects.append(QRectF(self.card_rect))
		self.hits = []
		nerv = eva.on()
		p.setFont(theme.mono(8.5))
		fm = p.fontMetrics()
		for k, tr in enumerate(st["traits"]):
			th = self.arm(k, K)
			anchor = self.project((math.cos(th) * (R_V1 + 0.18), 0.0, math.sin(th) * (R_V1 + 0.18)))
			tip = self.project((math.cos(th) * value_radius(tr.get("value", 0.0)), 0.02, math.sin(th) * value_radius(tr.get("value", 0.0))))
			if anchor is None:
				continue
			self.labels.append((k, anchor))
			v = float(tr.get("value", 0.0))
			hot = k == self.focus()
			left = anchor.x() < cx
			if nerv:
				evahud.trait_label(self, p, k, tr, anchor, tip, left, hot)
				continue
			name = tr["name"].upper()
			val = f"{v:+.2f}"
			base = tr.get("base")
			delta = f" {'▲' if v - base >= 0 else '▼'}{abs(v - base):.2f}" if base is not None and abs(v - base) >= 0.01 else ""
			lever = tr.get("lever")
			lev = f"  ◆{lever:+.2f}" if lever else ""
			text_w = fm.horizontalAdvance(name + "  " + val + delta + lev)
			x = anchor.x() - text_w - 14 if left else anchor.x() + 14
			y = anchor.y() + 4
			self.rects.append(QRectF(x - 2, y - 11, text_w + 4, 15 + (28 if k == self.hover else 0)))
			self.hits.append((QRectF(x - 4, y - 12, text_w + 8, 17), "trait", k))
			p.setPen(QPen(theme.qc(theme.TEXT if hot else theme.MUTED, 110 if not hot else 220), 1))
			p.drawLine(anchor, QPointF(anchor.x() + (-10 if left else 10), anchor.y()))
			if tip is not None and hot:
				p.setPen(QPen(theme.qc(theme.ACCENT, 160), 1, Qt.PenStyle.DotLine))
				p.drawLine(anchor, tip)
			p.setPen(theme.qc(theme.TEXT if hot else "#b9b6b0"))
			p.drawText(QPointF(x, y), name)
			xx = x + fm.horizontalAdvance(name + "  ")
			p.setPen(theme.signed_color(v))
			p.drawText(QPointF(xx, y), val)
			xx += fm.horizontalAdvance(val)
			if delta:
				p.setPen(theme.qc(theme.MUTED))
				p.drawText(QPointF(xx, y), delta)
				xx += fm.horizontalAdvance(delta)
			if lev:
				p.setPen(theme.qc(theme.ACCENT2))
				p.drawText(QPointF(xx, y), lev)
			if k == self.hover:
				sub = f"+ {tr.get('plus', '')}   − {tr.get('minus', '')}"
				p.setPen(theme.qc(theme.MUTED))
				p.drawText(QPointF(x, y + 14), sub)
				reads = tr.get("reads")
				if reads:
					p.setPen(theme.qc(theme.ACCENT2, 200))
					p.drawText(QPointF(x, y + 28), "reads as: " + " ".join(s.strip() for s in reads[:6]))
		L = int(st["L"])
		lo, hi = st.get("band") or [0, 0]
		p.setFont(evahud.label_font("band") if nerv else theme.mono(7.5))
		fm = p.fontMetrics()
		for l, tag, a, dy in ((lo, "WORKSPACE ONSET", math.pi * 0.5, 22), (hi, "MOTOR REGIME", math.pi * 0.5, 40)):
			pt = self.project((math.cos(a) * layer_radius(l, L), 0.0, math.sin(a) * layer_radius(l, L)))
			if pt is None:
				continue
			text = f"L{l} {tag}"
			tw = fm.horizontalAdvance(text)
			end = QPointF(pt.x() - 70, pt.y() + dy)
			r = QRectF(end.x() - tw - 6, end.y() - 8, tw + 4, 13)
			for _ in range(8):
				hit = next((q for q in self.rects if q.intersects(r)), None)
				if hit is None:
					break
				shift = hit.bottom() - r.top() + 2
				r.translate(0, shift)
				end.setY(end.y() + shift)
			self.rects.append(r)
			key = "onset" if tag == "WORKSPACE ONSET" else "motor"
			self.hits.append((QRectF(r).adjusted(-3, -2, 3, 2), "band", key))
			hot = key in (self.bhover, self.bpinned)
			p.setPen(QPen(theme.qc(theme.ACCENT, 230 if hot else 120), 1.2 if hot else 1))
			p.drawLine(pt, end)
			p.setPen(theme.qc(theme.TEXT if hot else theme.ACCENT2, 255 if hot else 200))
			p.drawText(QPointF(r.left() + 2, end.y() + 4), text)
		concepts = st.get("concepts") or []
		pts = []
		for ci, c in enumerate(concepts):
			pos = c.get("_pos")
			if not pos:
				continue
			v = self.mvp @ np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)
			if v[3] <= 0.01:
				continue
			pts.append((ci, QPointF((v[0] / v[3] * 0.5 + 0.5) * self.width(), (1 - (v[1] / v[3] * 0.5 + 0.5)) * self.height()), float(v[3])))
		self.concept_pts = [(ci, pt) for ci, pt, _d in pts]
		d0 = min((d for _c, _p, d in pts), default=0.0)
		d1 = max((d for _c, _p, d in pts), default=1.0)
		focus = {self.chover, self.cpinned} - {-1}
		shown = 0
		rank = {ci: r for r, ci in enumerate(self._auto_order(concepts))} if self.cmode == "auto" else {}
		for ci, pt, depth in sorted(pts, key=lambda x: (x[0] not in focus, rank.get(x[0], x[0] + len(concepts)) if rank else x[0])):
			c = concepts[ci]
			hot = ci in focus
			if not hot and (self.cmode == "off" or (self.cmode == "auto" and shown >= self.max_concepts)):
				continue
			p.setFont(evahud.label_font("concept", hot) if nerv else theme.mono(8, hot))
			fm = p.fontMetrics()
			label = c["str"].strip() + (f" · {c['en']}" if c.get("en") else "")
			end = QPointF(pt.x() + 16, pt.y() - 12)
			r = QRectF(end.x() + 1, end.y() - 9, fm.horizontalAdvance(label) + 4, 14)
			if not hot and self.cmode == "all":
				for dy in (0, -15, 15, -30, 30, -45, 45):
					q = r.translated(0, dy)
					if not any(o.intersects(q) for o in self.rects):
						r, end = q, QPointF(end.x(), end.y() + dy)
						break
			if not hot and (not self.rect().contains(r.toRect()) or any(q.intersects(r) for q in self.rects)):
				continue
			self.rects.append(r)
			self.hits.append((QRectF(r).adjusted(-2, -2, 2, 2), "concept", ci))
			shown += 0 if hot else 1
			fade = 1.0 if hot or self.cmode != "all" else 1.0 - 0.7 * (depth - d0) / max(d1 - d0, 1e-6)
			if hot:
				p.setPen(QPen(theme.qc(theme.ACCENT, 230), 1.2))
				p.setBrush(Qt.BrushStyle.NoBrush)
				p.drawEllipse(pt, 7.0, 7.0)
				p.drawLine(pt, end)
				p.setPen(theme.qc(theme.TEXT))
			else:
				p.setPen(QPen(theme.qc(theme.TEXT, int(70 * fade)), 1))
				p.drawLine(pt, end)
				p.setPen(theme.qc(theme.TEXT, int(190 * fade)))
			p.drawText(QPointF(end.x() + 3, end.y() + 3), label)
		if self.caption:
			p.setFont(evahud.label_font("caption") if nerv else theme.mono(9))
			p.setPen(theme.qc(theme.MUTED))
			w = p.fontMetrics().horizontalAdvance(self.caption)
			p.drawText(QPointF((self.width() - w) / 2, self.height() - 18), self.caption)

	def _hud(self, p, hud):
		if eva.on():
			evahud.hud(self, p, hud)
			return
		now = time.perf_counter()
		W = self.width()
		stages = hud.get("stages") or []
		n = len(stages)
		span = min(560.0, W - 80.0)
		x0 = (W - span) / 2
		y = 26.0
		step = span / max(n - 1, 1)
		p.setFont(theme.mono(7.2, True))
		fm = p.fontMetrics()
		for i, (label, state, sub) in enumerate(stages):
			x = x0 + i * step
			if i < n - 1:
				nxt = stages[i + 1][1]
				done = state == "done" and nxt in ("done", "active")
				p.setPen(QPen(theme.qc(theme.ACCENT if done else theme.LINE2, 170 if done else 255), 1))
				p.drawLine(QPointF(x + 7, y), QPointF(x + step - 7, y))
				if state == "done" and nxt == "active":
					f = (now * 0.9) % 1.0
					p.setPen(Qt.PenStyle.NoPen)
					p.setBrush(theme.qc(theme.ACCENT2, 220))
					p.drawEllipse(QPointF(x + 7 + f * (step - 14), y), 1.8, 1.8)
			if state == "done":
				p.setPen(Qt.PenStyle.NoPen)
				p.setBrush(theme.qc(theme.ACCENT))
				p.drawEllipse(QPointF(x, y), 4.0, 4.0)
			elif state == "active":
				a = 0.5 + 0.5 * math.sin(now * 5.0)
				p.setPen(QPen(theme.qc(theme.ACCENT2, int(120 + 135 * a)), 1.3))
				p.setBrush(Qt.BrushStyle.NoBrush)
				p.drawEllipse(QPointF(x, y), 5.5 + 2.5 * a, 5.5 + 2.5 * a)
				p.setPen(Qt.PenStyle.NoPen)
				p.setBrush(theme.qc(theme.ACCENT2))
				p.drawEllipse(QPointF(x, y), 3.0, 3.0)
			else:
				p.setPen(QPen(theme.qc(theme.DIM), 1))
				p.setBrush(Qt.BrushStyle.NoBrush)
				p.drawEllipse(QPointF(x, y), 3.5, 3.5)
			p.setPen(theme.qc(theme.TEXT if state == "active" else (theme.ACCENT2 if state == "done" else theme.DIM)))
			p.setFont(theme.mono(7.2, True))
			p.drawText(QPointF(x - fm.horizontalAdvance(label) / 2, y + 20), label)
			if sub:
				p.setFont(theme.mono(6.8))
				sw = p.fontMetrics().horizontalAdvance(sub)
				p.setPen(theme.qc(theme.MUTED if state != "idle" else theme.DIM))
				p.drawText(QPointF(x - sw / 2, y + 32), sub)
		title = hud.get("title")
		if not title:
			return
		y2 = y + 54
		p.setFont(theme.mono(8.5, True))
		fm = p.fontMetrics()
		busy = bool(hud.get("busy"))
		elapsed = time.time() - hud["t0"] if hud.get("t0") else None
		clock = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}" if elapsed is not None else ""
		queued = hud.get("queued") or 0
		right = clock + (f"  ·  +{queued} queued" if queued else "")
		detail = hud.get("detail") or ""
		p.setFont(theme.mono(7.5))
		fm2 = p.fontMetrics()
		tw = fm.horizontalAdvance(title)
		avail = span - tw - fm2.horizontalAdvance(right) - 30
		detail = fm2.elidedText(detail, Qt.TextElideMode.ElideRight, int(max(avail, 40)))
		p.setFont(theme.mono(8.5, True))
		p.setPen(theme.qc(theme.ACCENT if busy else theme.MUTED))
		p.drawText(QPointF(x0, y2), title)
		p.setFont(theme.mono(7.5))
		p.setPen(theme.qc(theme.TEXT, 200) if busy else theme.qc(theme.DIM))
		p.drawText(QPointF(x0 + tw + 12, y2), detail)
		p.setPen(theme.qc(theme.MUTED))
		p.drawText(QPointF(x0 + span - fm2.horizontalAdvance(right), y2), right)
		side = hud.get("side")
		if side:
			ys = y2 + 24
			p.setFont(theme.mono(7.5, True))
			p.setPen(theme.qc(theme.COOL2))
			p.drawText(QPointF(x0, ys), side["title"])
			p.setFont(theme.mono(7.5))
			p.setPen(theme.qc(theme.TEXT, 180))
			p.drawText(QPointF(x0 + p.fontMetrics().horizontalAdvance(side["title"]) + 16, ys), side["detail"])
			p.setPen(Qt.PenStyle.NoPen)
			p.setBrush(theme.qc(theme.LINE2))
			p.drawRect(QRectF(x0, ys + 6, span, 1.5))
			p.setBrush(theme.qc(theme.COOL, 220))
			p.drawRect(QRectF(x0, ys + 6, span * max(0.0, min(1.0, side.get("progress") or 0.0)), 1.5))
		if not busy:
			return
		yb = y2 + 8
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(theme.qc(theme.LINE2))
		p.drawRect(QRectF(x0, yb, span, 2))
		prog = hud.get("progress")
		if prog is not None:
			p.setBrush(theme.qc(theme.ACCENT, 230))
			p.drawRect(QRectF(x0, yb, span * max(0.0, min(1.0, prog)), 2))
			p.setBrush(theme.qc(theme.ACCENT2))
			p.drawEllipse(QPointF(x0 + span * max(0.0, min(1.0, prog)), yb + 1), 2.2, 2.2)
		else:
			f = (now * 0.45) % 1.0
			w = span * 0.18
			xs = x0 - w + f * (span + w)
			a, b = max(x0, xs), min(x0 + span, xs + w)
			if b > a:
				p.setBrush(theme.qc(theme.ACCENT, 200))
				p.drawRect(QRectF(a, yb, b - a, 2))

	def set_buttons(self, orbit, pan):
		self.orbit_name = orbit if orbit in BUTTONS else "left"
		self.pan_name = pan if pan in BUTTONS and pan != self.orbit_name else ("right" if self.orbit_name != "right" else "left")

	def _chip_action(self, chip):
		if chip == "rotate":
			self.set_rotate(not self.rotate)
			self.rotateToggled.emit(self.rotate)
		elif chip == "reset":
			self.reset_view()
		elif chip == "center":
			self.center_view()
		elif chip == "help":
			self.show_help()
		elif chip == "concepts":
			self.set_cmode({"auto": "all", "all": "off", "off": "auto"}[self.cmode])
			self.conceptsModeChanged.emit(self.cmode)
		elif chip == "close":
			self.unpin_all()

	def _click(self):
		if self.chover >= 0:
			self.pin_concept(-1 if self.cpinned == self.chover else self.chover)
		elif self.bhover:
			self.pin_band("" if self.bpinned == self.bhover else self.bhover)
		elif self.hover >= 0:
			self.pin(-1 if self.pinned == self.hover else self.hover)
			self.traitClicked.emit(self.hover)

	def mousePressEvent(self, e):
		b = e.button()
		if b == Qt.MouseButton.LeftButton:
			chip = self._chip_at(e.position())
			if chip:
				self._chip_action(chip)
				return
		self.press = (e.position(), b, 1)
		if b == BUTTONS[self.orbit_name]:
			self.goal = {}
			self.drag = (e.position(), self.az, self.el, False)
		elif b == BUTTONS[self.pan_name]:
			self.goal = {}
			self.pan_drag = (e.position(), list(self.pan))
			self.setCursor(Qt.CursorShape.ClosedHandCursor)

	def mouseMoveEvent(self, e):
		if self.pan_drag is not None:
			start, (px, pz) = self.pan_drag
			dx = e.position().x() - start.x()
			dy = e.position().y() - start.y()
			k = getattr(self, "dist", 10.0) * 0.0016
			sa, ca = math.sin(self.az), math.cos(self.az)
			nx = px - sa * dx * k - ca * dy * k
			nz = pz + ca * dx * k - sa * dy * k
			r = math.hypot(nx, nz)
			if r > 9.0:
				nx, nz = nx * 9.0 / r, nz * 9.0 / r
			self.pan = [float(nx), float(nz)]
			self.ov_dirty = True
			return
		if self.drag is not None:
			start, az, el, _moved = self.drag
			dx = e.position().x() - start.x()
			dy = e.position().y() - start.y()
			self.az = az + dx * 0.006
			self.el = max(0.08, min(1.35, el + dy * 0.005))
			self.drag = (start, az, el, _moved or abs(dx) + abs(dy) > 4)
			return
		chip = self._chip_at(e.position())
		if chip != self.chip_hover:
			self.chip_hover = chip
			if chip:
				self.setCursor(Qt.CursorShape.PointingHandCursor)
			else:
				self.unsetCursor()
		if chip:
			modes = [("auto", "AUTO", "a few labels, never overlapping"), ("all", "ALL", "every concept, spread out, faded with depth"), ("off", "OFF", "no labels: hover a cloud to see it")]
			concepts_tip = "\n".join(["concept labels   (click to cycle)"] + [f"{'▸' if key == self.cmode else ' '} {name:<5} {desc}" for key, name, desc in modes] + ["", "hover a concept cloud: its card · click: pin it"])
			also = lambda click: f" · {click}" if click != "none" else ""
			self.setToolTip({"rotate": ("stop" if self.rotate else "start") + " the rotation   (R)", "reset": f"reset the view: tilt, zoom and position   (Home{also(self.reset_click)})", "center": f"re-center the view on the iris,\nkeeping zoom and tilt   (C{also(self.center_click)})", "help": self.help_tip(), "close": "close the card   (Esc)", "concepts": concepts_tip}[chip])
			return
		if self.over_card(e.position()):
			self.set_chover(-1)
			self.set_bhover("")
			if self.hover != -1:
				self.hover = -1
				self.lines_dirty = self.ov_dirty = True
				self.traitHovered.emit(-1)
			self.setToolTip(self._tip_at(e.position()))
			return
		hit = self._hit_at(e.position())
		ci = hit[1] if hit and hit[0] == "concept" else (self._concept_at(e.position()) if hit is None else -1)
		self.set_chover(ci)
		self.set_bhover(hit[1] if hit and hit[0] == "band" else "")
		best, bd = (hit[1] if hit and hit[0] == "trait" else -1), 70.0
		if ci < 0 and hit is None:
			for k, pt in self.labels:
				d = math.hypot(pt.x() - e.position().x(), pt.y() - e.position().y())
				if d < bd:
					best, bd = k, d
		if best != self.hover:
			self.hover = best
			self.lines_dirty = self.ov_dirty = True
			self.traitHovered.emit(best)
		tr = self.state["traits"][best] if 0 <= best < len(self.state["traits"]) else None
		self.setToolTip(f"{tr['name']}: {tr.get('plus', '')} ↔ {tr.get('minus', '')}" if tr else self._tip_at(e.position()))

	def over_card(self, pos):
		pinned = self.pinned >= 0 or self.cpinned >= 0 or self.bpinned
		return bool(pinned) and self.card_rect is not None and self.card_rect.adjusted(-2, -2, 2, 2).contains(pos)

	def _tip_at(self, pos):
		if not eva.on():
			return ""
		pinned = self.pinned >= 0 or self.cpinned >= 0 or self.bpinned
		return next((text for r, text in (self.card_tips if pinned else []) + self.hud_hits if r.contains(pos)), "")

	def mouseReleaseEvent(self, e):
		b = e.button()
		press = self.press
		moved = press is not None and (e.position() - press[0]).manhattanLength() > 4
		if self.pan_drag is not None and b == BUTTONS[self.pan_name]:
			self.pan_drag = None
			self.unsetCursor()
		if self.drag is not None and b == BUTTONS[self.orbit_name]:
			self.drag = None
		if press is None or press[1] != b:
			return
		self.press = None
		if moved:
			return
		action = self.click_map.get((NAMES.get(b), press[2]))
		if action == "center":
			self.center_view()
		elif action == "reset":
			self.reset_view()
		elif b == Qt.MouseButton.LeftButton:
			self._click()

	def mouseDoubleClickEvent(self, e):
		self.mousePressEvent(e)
		if self.press is not None and self.press[1] == e.button():
			self.press = (self.press[0], self.press[1], 2)

	def _ground_point(self, pos):
		try:
			inv = np.linalg.inv(self.mvp.astype(np.float64))
		except np.linalg.LinAlgError:
			return None
		x = pos.x() / max(self.width(), 1) * 2.0 - 1.0
		y = 1.0 - pos.y() / max(self.height(), 1) * 2.0
		a = inv @ np.array([x, y, -1.0, 1.0])
		b = inv @ np.array([x, y, 1.0, 1.0])
		if abs(a[3]) < 1e-9 or abs(b[3]) < 1e-9:
			return None
		a, b = a[:3] / a[3], b[:3] / b[3]
		if abs(b[1] - a[1]) < 1e-9:
			return None
		t = (-0.35 - a[1]) / (b[1] - a[1])
		if not 0.0 <= t <= 1.0:
			return None
		p = a + (b - a) * t
		return p if math.hypot(p[0], p[2]) < 14.0 else None

	def wheelEvent(self, e):
		self.goal = {}
		old = self.zoom
		new = max(0.08, min(3.0, old * (0.88 if e.angleDelta().y() > 0 else 1 / 0.88)))
		if new == old:
			return
		pivot = self._ground_point(e.position()) if math.hypot(*self.pan) > 0.05 else None
		self.zoom = new
		if pivot is not None:
			s = new / old
			nx = pivot[0] + (self.pan[0] - pivot[0]) * s
			nz = pivot[2] + (self.pan[1] - pivot[2]) * s
			r = math.hypot(nx, nz)
			if r > 9.0:
				nx, nz = nx * 9.0 / r, nz * 9.0 / r
			self.pan = [float(nx), float(nz)]
		self.ov_dirty = True

	def leaveEvent(self, e):
		if self.chip_hover:
			self.chip_hover = None
			self.unsetCursor()
		self.set_chover(-1)
		if self.hover != -1:
			self.hover = -1
			self.lines_dirty = self.ov_dirty = True
			self.traitHovered.emit(-1)
