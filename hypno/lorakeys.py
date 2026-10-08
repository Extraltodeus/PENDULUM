NORMED_FFN = ("gemma", "gemma2", "gemma3", "gemma3n", "gemma4", "olmo2", "cohere2", "glm4")
STYLES = {
	"hf": ("model.", "Hugging Face / transformers names"),
	"peft": ("base_model.model.model.", "PEFT adapter folder (adapter_model.safetensors + adapter_config.json)"),
	"comfyui": ("text_encoders.transformer.model.", "ComfyUI text encoder (LoraLoader: strength_clip)"),
	"kohya": ("lora_te_", "kohya style, single text encoder"),
}
MODULES = {"ffn_down": "mlp.down_proj", "ffn_up": "mlp.up_proj", "ffn_gate": "mlp.gate_proj", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj"}


def key_of(style, prefix, layer, module, part):
	if style == "kohya":
		return f"{prefix}{f'layers.{layer}.{module}'.replace('.', '_')}.{'lora_down' if part == 'A' else 'lora_up'}.weight"
	return f"{prefix}layers.{layer}.{module}.lora_{part}.weight"


def example_key(style, prefix, layer=16):
	return key_of(style, prefix, layer, "mlp.down_proj", "A")


def arch_ok(arch, n_expert=0):
	if arch in NORMED_FFN:
		return False, f"'{arch}' models normalize the MLP output before adding it to the residual stream: a LoRA there cannot reproduce a lever"
	if int(n_expert or 0) > 0:
		return False, f"'{arch}' is a mixture of experts: there is no single MLP output weight to carry the lever"
	return True, ""
