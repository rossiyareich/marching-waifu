import json

import torch
from compel import Compel
from diffusers import AutoencoderKL, ControlNetModel
from diffusers.image_processor import VaeImageProcessor

def load_conf(filepath):
    with open(filepath, "r") as f:
        conf = json.loads(f.read())
    return conf

def load_directions(filepath, image_count):
    directions = [None] * image_count
    directions_json = json.load(open(filepath))["prompt_addition"]
    for direction_json in directions_json[:image_count]:
        directions[direction_json["n"] - 1] = direction_json["direction"]
    return directions


def load_controlnet_models(folder, image_count):
    controlnet_params = [
        ("lllyasviel/control_v11p_sd15_openpose", "openpose_full"),
        ("lllyasviel/control_v11f1p_sd15_depth", "depth"),
        ("lllyasviel/control_v11p_sd15_normalbae", "normals"),
        ("lllyasviel/control_v11p_sd15_lineart", "lineart"),
    ]

    controlnet_models = []
    controlnet_conds = [[] for _ in range(image_count)]

    for repo_id, folder_prefix in controlnet_params:
        controlnet_model = ControlNetModel.from_pretrained(
            repo_id, torch_dtype=torch.float16
        )
        controlnet_model = controlnet_model.to("cuda")
        controlnet_models.append(controlnet_model)

        for i in range(image_count):
            controlnet_cond_path = f"{folder}/{folder_prefix}{(i+1):04}.png"
            controlnet_conds[i].append(controlnet_cond_path)

    return (controlnet_models, controlnet_conds)


@torch.no_grad()
def load_vae(repo_id):
    vae = AutoencoderKL.from_pretrained(repo_id, torch_dtype=torch.float16)
    vae = vae.to("cuda")
    return vae


@torch.no_grad()
def load_image_processor(vae):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    return image_processor


@torch.no_grad()
def load_generator(seed):
    generator = torch.Generator(device="cpu")
    if seed == -1:
        seed = generator.seed()
    else:
        generator = generator.manual_seed(seed)
    return (generator, seed)


@torch.no_grad()
def load_weighted_embeds(pipe, prompt, uncond_prompt):
    compel = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        truncate_long_prompts=False,
    )
    conditioning = compel.build_conditioning_tensor(prompt)
    negative_conditioning = compel.build_conditioning_tensor(uncond_prompt)
    results = compel.pad_conditioning_tensors_to_same_length(
        [conditioning, negative_conditioning]
    )
    return results