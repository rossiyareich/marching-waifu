import os
import PIL.Image

from ..src.utils.file_loader import *
from ..src.utils.image import *
from ..src.workflows.controlnet_unet9_workflow import *
from ..src.workflows.real_esrgan_workflow import *


config_filepath = "inference.json"
prompt_addition_filepath = "../data/prompts/prompt_additions.json"
deepdanbooru_prompt_filepath = "../data/prompts/deepdanbooru_prompt.txt"
textual_inversion_folderpath = "../data/embeddings/"
controlnet_conditions_folderpath = "../data/multi_controlnet/multi_controlnet_data/"
ngp_overview_folderpath = "../data/ngp/overview/"
ngp_train_folderpath = "../data/ngp/train/"


config = load_config(config_filepath)
unet9 = controlnet_unet9_workflow(
    config["models"]["vae_repo_id"],
    config["models"]["ldm_inpaint_repo_id"],
    textual_inversion_folderpath,
    config["ops"],
)

prompt_additions = load_prompt_addition(prompt_addition_filepath)
cc_set = load_controlnet_conditions(controlnet_conditions_folderpath)
controlnet_scales = [
    config["controlnet"]["unit_scales"]["openpose"],
    config["controlnet"]["unit_scales"]["depth"],
    config["controlnet"]["unit_scales"]["normals"],
    config["controlnet"]["unit_scales"]["lineart"],
]
controlnet_scales = [
    config["controlnet"]["unit_scales"]["openpose"],
    config["controlnet"]["unit_scales"]["depth"],
    config["controlnet"]["unit_scales"]["normals"],
    config["controlnet"]["unit_scales"]["lineart"],
]

with open(deepdanbooru_prompt_filepath, "r") as f:
    prompt = f.read()

real_esrgan = real_esrgan_workflow()

def generate_inpaint(indices, gen_rel_index):
    image_set = [
        image(
            PIL.Image.open(os.path.join(ngp_train_folderpath, f"{(i+1):04}.png"))
        ).scale(1.0 / 4.0)
        for i in indices
    ]
    stitched_image = image(image_set[0].to_pil())
    for image_ in image_set[1:]:
        stitched_image = stitched_image.concatenate(image_)
    stitched_image = stitched_image.to_pil()

    controlnet_conditions_set = [
        [image(controlnet_condition) for controlnet_condition in controlnet_conditions]
        for controlnet_conditions in cc_set
    ]
    controlnet_stitched_conditions = [
        image(controlnet_condition.to_pil())
        for controlnet_condition in controlnet_conditions_set[0]
    ]
    for controlnet_conditions in controlnet_conditions_set[1:]:
        for i, controlnet_condition in enumerate(controlnet_conditions):
            controlnet_stitched_conditions[i] = controlnet_stitched_conditions[
                i
            ].concatenate(controlnet_condition)
    controlnet_stitched_conditions = [
        controlnet_stitched_condition.to_pil()
        for controlnet_stitched_condition in controlnet_stitched_conditions
    ]

    stitched_mask = PIL.Image.new("1", (stitched_image.width, stitched_image.height))
    ind_width = stitched_image.width // len(indices)
    inpaint_region = (
        ind_width * gen_rel_index,
        0,
        ind_width * (gen_rel_index + 1),
        stitched_image.height,
    )
    stitched_mask = stitched_mask.paste(True, inpaint_region)

    image_, seed, overview = unet9(
        prompt.format(prompt_additions[indices[gen_rel_index]]),
        config["pipeline"]["restgen"]["negative_prompt"],
        stitched_image,
        stitched_mask,
        config["pipeline"]["restgen"]["steps"],
        config["pipeline"]["restgen"]["cfg_scale"],
        config["pipeline"]["restgen"]["denoising_strength"],
        config["pipeline"]["restgen"]["seed"],
        config["pipeline"]["restgen"]["callback_steps"],
        controlnet_stitched_conditions,
        controlnet_scales,
        config["controlnet"]["soft_exp"],
        config["pipeline"]["restgen"]["inpaint_method"],
    )

    return image_.crop(inpaint_region), seed, overview


for i in range(1, config["pipeline"]["restgen"]["data_size"] + 1):
    image_, seed, overview = generate_inpaint(([i - 1, i, 0] if i > 1 else [0, i]), 1)
    print(f"{(i+1):04}.png : {seed}")
    if overview is not None:
        overview.save(os.path.join(ngp_overview_folderpath, f"{(i+1):04}.png"))
    image_ = real_esrgan(image_)
    image_.save(os.path.join(ngp_train_folderpath, f"{(i+1):04}.png"))