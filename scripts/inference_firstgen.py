import os
import sys
import subprocess

sys.path.append("..")

from src.utils.file_loader import *
from src.utils.image_wrapper import *
from src.utils.torch_utils_extended import *
from src.workflows.controlnet_unet4_workflow import *
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


def work_load_unet4():
    global unet4
    unet4 = controlnet_unet4_workflow(
        config["models"]["vae_repo_id"],
        config["models"]["ldm_repo_id"],
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


def work_run_firstgen_prereq():
    global fg_prereq_image
    fg_prereq_image, seed, overviews = unet4(
        config["pipeline"]["firstgen"]["prompt"].format(prompt_additions[0]),
        config["pipeline"]["firstgen"]["negative_prompt"],
        config["pipeline"]["firstgen"]["steps"],
        config["pipeline"]["firstgen"]["cfg_scale"],
        config["pipeline"]["firstgen"]["denoising_strength"],
        config["pipeline"]["firstgen"]["seed"],
        config["pipeline"]["firstgen"]["callback_steps"],
        cc_set[0],
        cc_scales,
        config["controlnet"]["soft_exp"],
    )
    print(seed)
    work_save_overviews(
        overviews, os.path.join(ngp_overview_folderpath, "0001_prereq.png")
    )
    fg_prereq_image.save(os.path.join(ngp_train_folderpath, "0001_prereq.png"))


def work_load_real_esrgan():
    global real_esrgan
    real_esrgan = real_esrgan_workflow()


def work_run_real_esrgan_prereq():
    global fg_prereq_image
    fg_prereq_image = real_esrgan(fg_prereq_image)
    fg_prereq_image.save(os.path.join(ngp_train_folderpath, "0001_prereq.png"))


def work_load_deepdanbooru_prompts():
    global prompt
    with open(deepdanbooru_prompt_filepath, "r") as f:
        prompt = f.read()


def work_run_firstgen():
    global fg_image
    fg_image, seed, overview = unet4(
        prompt.format(prompt_additions[0]),
        config["pipeline"]["firstgen"]["negative_prompt"],
        config["pipeline"]["firstgen"]["steps"],
        config["pipeline"]["firstgen"]["cfg_scale"],
        config["pipeline"]["firstgen"]["img2img_denoising_strength"],
        config["pipeline"]["firstgen"]["img2img_denoising_seed"],
        config["pipeline"]["firstgen"]["callback_steps"],
        cc_set[0],
        cc_scales,
        config["controlnet"]["soft_exp"],
        image_wrapper(fg_prereq_image, "pil").scale(1.0 / 4.0).to_pil(),
    )
    print(seed)
    work_save_overviews(overview, os.path.join(ngp_overview_folderpath, "0001.png"))
    fg_image = real_esrgan(fg_image)
    fg_image.save(os.path.join(ngp_train_folderpath, "0001.png"))


if __name__ == "__main__":
    work_load_config()
    work_load_prompt_additions()
    work_load_controlnet_conditions()
    work_load_unet4()
    work_run_firstgen_prereq()
    subprocess.call(["python", "inference_deepdanbooru.py"])
    work_load_real_esrgan()
    work_run_real_esrgan_prereq()
    work_load_deepdanbooru_prompts()
    work_run_firstgen()
