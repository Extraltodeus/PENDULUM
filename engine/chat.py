import json
from datetime import datetime

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

FALLBACK_TEMPLATE = (
	"{% for message in messages %}"
	"{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
	"{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}"
	"{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}"
	"{% endif %}{% endfor %}"
	"{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)


def _raise(message):
	raise jinja2.exceptions.TemplateError(message)


def _tojson(x, indent=None, ensure_ascii=False, separators=None, sort_keys=False):
	return json.dumps(x, indent=indent, ensure_ascii=ensure_ascii, separators=separators, sort_keys=sort_keys)


def _env():
	env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
	env.filters["tojson"] = _tojson
	env.globals["raise_exception"] = _raise
	env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
	return env


class ChatTemplate:
	def __init__(self, source, bos="", eos=""):
		self.source = source or FALLBACK_TEMPLATE
		self.fallback = not source
		self.bos = bos
		self.eos = eos
		self._tmpl = _env().from_string(self.source)

	def render(self, messages, add_generation_prompt=True, enable_thinking=False):
		kwargs = {"messages": messages, "add_generation_prompt": add_generation_prompt, "bos_token": self.bos, "eos_token": self.eos, "enable_thinking": enable_thinking, "tools": None, "documents": None}
		try:
			return self._tmpl.render(**kwargs)
		except jinja2.exceptions.TemplateError:
			if not any(m["role"] == "system" for m in messages):
				raise
			merged = [dict(m) for m in messages if m["role"] != "system"]
			sys_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
			if merged and merged[0]["role"] == "user":
				merged[0]["content"] = f"{sys_text}\n\n{merged[0]['content']}"
			kwargs["messages"] = merged
			return self._tmpl.render(**kwargs)
