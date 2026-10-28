"""Load LLaDA-V and turn (image, question) into prompt embeddings, as upstream generate_demo.py does."""
import copy
import time

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

CAPTION_PROMPT = "Please describe the image in detail."


def load(checkpoint="GSAI-ML/LLaDA-V"):
    """Returns (tokenizer, model, image_processor, load_seconds)."""
    t0 = time.perf_counter()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        checkpoint, None, "llava_llada", attn_implementation="sdpa", device_map="cuda:0")
    model.eval()
    return tokenizer, model, image_processor, time.perf_counter() - t0


def prompt_ids(tokenizer, question):
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt").unsqueeze(0).cuda()


@torch.no_grad()
def embed(model, image_processor, input_ids, image_path):
    image = Image.open(image_path)
    image_tensor = [t.to(dtype=torch.float16, device="cuda") for t in
                    process_images([image], image_processor, model.config)]
    return model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, image_tensor, ["image"], image_sizes=[image.size])[4]
