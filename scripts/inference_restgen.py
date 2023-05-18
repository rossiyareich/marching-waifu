import os
import sys

sys.path.append("..")

import PIL.Image

from src.utils.file_loader import *
from src.utils.image_wrapper import *
from src.workflows.controlnet_unet9_workflow import *
from src.workflows.real_esrgan_workflow import *

config_filepath = "inference.json"
prompt_addition_filepath = "../data/prompts/prompt_additions.json"
deepdanbooru_prompt_filepath = "../data/prompts/deepdanbooru_prompt.txt"
textual_inversion_folderpath = "../data/embeddings/"
controlnet_conditions_folderpath = "../data/multi_controlnet/multi_controlnet_data/"
ngp_overview_folderpath = "../data/ngp/overview/"
ngp_train_folderpath = "../data/ngp/train/"


def work_load_config():
    global config
    config = load_config(config_filepath)


def work_load_prompt_additions():
    global prompt_additions
    prompt_additions = load_prompt_addition(prompt_addition_filepath)


def work_load_controlnet_conditions():
    global cc_set, cc_scales
    cc_set = load_controlnet_conditions(controlnet_conditions_folderpath)
    cc_scales = [
        config["controlnet"]["unit_scales"]["openpose"],
        config["controlnet"]["unit_scales"]["depth"],
        config["controlnet"]["unit_scales"]["normals"],
        config["controlnet"]["unit_scales"]["lineart"],
    ]


def work_load_deepdanbooru_prompts():
    global prompt
    with open(deepdanbooru_prompt_filepath, "r") as f:
        prompt = f.read()


def work_load_unet9():
    global unet9
    unet9 = controlnet_unet9_workflow(
        config["models"]["vae_repo_id"],
        config["models"]["ldm_inpaint_repo_id"],
        textual_inversion_folderpath,
        config["ops"],
    )


def work_save_overviews(overviews, filepath):
    if overviews is not None:
        overview_imgs = [image_wrapper(overview, "pil") for overview in overviews]
        overview_img = overview_imgs[0]
        for img in overview_imgs[1:]:
            overview_img.concatenate(img)
        overview_img.to_pil().save(filepath)


def work_generate_inpaint(indices, gen_rel_index):
    image_set = [
        image_wrapper(
            PIL.Image.open(
                os.path.join(ngp_train_folderpath, f"{(i+1):04}.png")
            ).convert("RGB"),
            "pil",
        ).scale(1.0 / 4.0)
        for i in indices
    ]
    stitched_image = image_set[0]
    for image in image_set[1:]:
        stitched_image.concatenate(image)
    stitched_image = stitched_image.to_pil()

    controlnet_conditions_set = [
        [image_wrapper(cc, "pil") for cc in cc_set[i]] for i in indices
    ]
    controlnet_stitched_conditions = controlnet_conditions_set[0]
    for controlnet_conditions in controlnet_conditions_set[1:]:
        for i, controlnet_condition in enumerate(controlnet_conditions):
            controlnet_stitched_conditions[i].concatenate(controlnet_condition)
    controlnet_stitched_conditions = [
        image_wrapper.to_pil(sc) for sc in controlnet_stitched_conditions
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

    image, seed, overview = unet9(
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
        cc_scales,
        config["controlnet"]["soft_exp"],
        config["pipeline"]["restgen"]["inpaint_method"],
    )

    return (
        image_wrapper(image, "pil").crop(inpaint_region).to_pil(),
        seed,
        [image_wrapper(img, "pil").crop(inpaint_region).to_pil() for img in overview],
    )


def work_load_real_esrgan():
    global real_esrgan
    real_esrgan = real_esrgan_workflow()


if __name__ == "__main__":
    work_load_config()
    work_load_prompt_additions()
    work_load_controlnet_conditions()
    work_load_deepdanbooru_prompts()
    work_load_unet9()
    empty_cache()
    work_load_real_esrgan()
    empty_cache()

    for i in range(1, config["pipeline"]["restgen"]["data_size"]):
        image, seed, overview = work_generate_inpaint(
            ([i - 1, i, 0] if i > 1 else [0, i]), 1
        )
        empty_cache()
        print(f"{(i+1):04}.png : {seed}")
        work_save_overviews(
            overview, os.path.join(ngp_overview_folderpath, f"{(i+1):04}.png")
        )
        image = real_esrgan(image)
        empty_cache()
        image.save(os.path.join(ngp_train_folderpath, f"{(i+1):04}.png"))
