import argparse
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config

config.setup_env()

import jlens
import torch
import transformers


def say(event, **data):
	print(json.dumps({"event": event, **data}), flush=True)


class Progress(logging.Handler):
	def emit(self, record):
		if not record.args:
			return
		if record.msg.startswith("  prompt"):
			say("progress", done=record.args[0], total=record.args[1], seconds=record.args[4])
		elif record.msg.startswith("  resuming"):
			say("resume", done=record.args[0], total=record.args[1])


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--source", required=True)
	ap.add_argument("--gguf-file", default="")
	ap.add_argument("--device", default="cuda:0")
	ap.add_argument("--dtype", default="bf16")
	ap.add_argument("--prompts", required=True)
	ap.add_argument("--checkpoint", required=True)
	ap.add_argument("--out", required=True)
	ap.add_argument("--dim-batch", type=int, default=4)
	ap.add_argument("--max-seq-len", type=int, default=128)
	ap.add_argument("--source-layers", default="")
	a = ap.parse_args()
	logging.basicConfig(level=logging.INFO, handlers=[Progress()])
	with open(a.prompts, encoding="utf-8") as f:
		prompts = json.load(f)
	kw = {"dtype": torch.bfloat16, "device_map": "auto" if a.device == "auto" else {"": a.device}}
	if a.dtype == "nf4":
		kw["quantization_config"] = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
	tk = {}
	if a.gguf_file:
		kw["gguf_file"] = tk["gguf_file"] = a.gguf_file
	say("loading", device=a.device)
	model = transformers.AutoModelForCausalLM.from_pretrained(a.source, **kw)
	tokenizer = transformers.AutoTokenizer.from_pretrained(a.source, **tk)
	lm = jlens.from_hf(model, tokenizer)
	say("ready", n_layers=lm.n_layers, d_model=lm.d_model)
	every = max(1, round((lm.n_layers - 1) * lm.d_model ** 2 * 4 / 150e6))
	sources = [int(x) for x in a.source_layers.split(",") if x.strip()] or None
	lens = jlens.fit(lm, prompts, source_layers=sources, dim_batch=a.dim_batch, max_seq_len=a.max_seq_len, checkpoint_path=a.checkpoint, checkpoint_every=every)
	lens.save(a.out)
	say("done", n_prompts=lens.n_prompts, d_model=lens.d_model, n_layers=lm.n_layers)


main()
