"""Adapters and action-head execution on a shared Qwen3-VL prefix."""
from __future__ import annotations
import torch
from PIL import Image, ImageDraw
from oxygen_models.common.cache import PrefixState, _snapshot_cache, init_language_from_prefix, language_step

class RuntimeAdapter:
    def __init__(self, interface):
        self.vlm = interface
        self.device = interface.model.device



def example():
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((22, 66, 86, 132), fill=(190, 35, 35))
    draw.rectangle((134, 62, 207, 145), fill=(35, 70, 180))
    return image, "Move the red block next to the blue block."



def build_inputs(interface, instruction, image, oft_model=None):
    if oft_model is not None:
        tokens = oft_model.action_token * oft_model.chunk_len
        instruction = (
            instruction
            + f" Please predict the next {oft_model.chunk_len} robot actions: "
            + f"<action>{tokens}<action>."
        )
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ],
    }]]
    return interface.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(interface.model.device)



@torch.inference_mode()
def prefill(pi_model, inputs):
    outputs = pi_model.qwen_vl_interface(
        **inputs,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
        logits_to_keep=1,
    )
    outputs.attention_mask = inputs["attention_mask"]
    return PrefixState(
        outputs=outputs,
        action_kv=_snapshot_cache(outputs.past_key_values),
        prefix_length=int(inputs["attention_mask"].shape[-1]),
    )



@torch.inference_mode()
def run_language(adapter, processor, prefix, count):
    state = init_language_from_prefix(processor, prefix, stop_on_eos=False)
    for _ in range(count):
        language_step(adapter, state)
    return state



@torch.inference_mode()
def pi_action(model, prefix, steps):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model.action_model.num_inference_timesteps = steps
    hidden = list(prefix.outputs.hidden_states[-model.num_action_dit_layers:])
    hidden = model._project_vl_hidden_for_action(hidden)
    return model.action_model.predict_action(
        hidden, None, encoder_attention_mask=prefix.outputs.attention_mask.bool()
    )



@torch.inference_mode()
def groot_action(head, prefix, steps):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    head.num_inference_timesteps = steps
    state = torch.zeros((1, 1, 7), device=prefix.outputs.logits.device, dtype=torch.bfloat16)
    return head.predict_action(
        prefix.outputs.hidden_states[-1],
        state,
        encoder_attention_mask=prefix.outputs.attention_mask.bool(),
    )



@torch.inference_mode()
def oft_action(oft_model, prefix, input_ids):
    queries = oft_model._gather_action_token_embeddings(
        prefix.outputs.hidden_states[-1], input_ids, oft_model.action_token_id
    )
    return oft_model.action_model.predict_action(queries)
