import os
import sys

sys.path.append("../../")

from src.utils.file_loader import *
from src.utils.image_wrapper import *
from src.workflows.controlnet_unet4_workflow import *
from src.workflows.deepdanbooru_workflow import *
from src.workflows.real_esrgan_workflow import *

config_filepath = "inference.json"
prompt_addition_filepath = "../data/prompts/prompt_additions.json"
deepdanbooru_prompt_filepath = "../data/prompts/deepdanbooru_prompt.txt"
textual_inversion_folderpath = "../data/embeddings/"
controlnet_conditions_folderpath = "../data/multi_controlnet/multi_controlnet_data/"
deepdanbooru_project_folderpath = "../ext/AnimeFaceNotebooks/deepdanbooru_model/"
ngp_overview_folderpath = "../data/ngp/overview/"
ngp_train_folderpath = "../data/ngp/train/"


def save_overview(overviews, filepath):
    if overviews is not None:
        overview_imgs = list(map(image_wrapper, overviews))
        overview_img = overview_imgs[0]
        for img in overview_imgs[1:]:
            overview_img.concatenate(img)
        overview_img.to_pil().save(filepath)


config = load_config(config_filepath)
unet4 = controlnet_unet4_workflow(
    config["models"]["vae_repo_id"],
    config["models"]["ldm_repo_id"],
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

real_esrgan = real_esrgan_workflow()

fg_prereq_image, fg_prereq_seed, fg_prereq_overview = unet4(
    config["pipeline"]["firstgen"]["prompt"],
    config["pipeline"]["firstgen"]["negative_prompt"],
    config["pipeline"]["firstgen"]["steps"],
    config["pipeline"]["firstgen"]["cfg_scale"],
    config["pipeline"]["firstgen"]["denoising_strength"],
    config["pipeline"]["firstgen"]["seed"],
    config["pipeline"]["firstgen"]["callback_steps"],
    cc_set[0],
    controlnet_scales,
    config["controlnet"]["soft_exp"],
)
print(fg_prereq_seed)
save_overview(
    fg_prereq_overview, os.path.join(ngp_overview_folderpath, "0001_prereq.png")
)
fg_prereq_image.save(os.path.join(ngp_train_folderpath, "0001_prereq.png"))

deepdanbooru = deepdanbooru_workflow(deepdanbooru_project_folderpath)
deepdanbooru(
    os.path.join(ngp_train_folderpath, "0001_prereq.png"),
    config["deepdanbooru"]["threshold"],
)
deepdanbooru.load_prompts(
    config["deepdanbooru"]["multiplier"], config["deepdanbooru"]["prefix"]
)
with open(deepdanbooru_prompt_filepath, "w") as f:
    f.write(deepdanbooru.prompt)

fg_prereq_image = real_esrgan(fg_prereq_image)
fg_prereq_image.save(os.path.join(ngp_train_folderpath, "0001_prereq.png"))

fg_image, fg_seed, fg_overview = unet4(
    deepdanbooru.prompt.format(prompt_additions[0]),
    config["pipeline"]["firstgen"]["negative_prompt"],
    config["pipeline"]["firstgen"]["steps"],
    config["pipeline"]["firstgen"]["cfg_scale"],
    config["pipeline"]["firstgen"]["img2img_denoising_strength"],
    config["pipeline"]["firstgen"]["img2img_seed"],
    config["pipeline"]["firstgen"]["callback_steps"],
    cc_set[0],
    controlnet_scales,
    config["controlnet"]["soft_exp"],
    image_wrapper(fg_prereq_image).scale(1.0 / 4.0).to_pil(),
)
print(fg_seed)
save_overview(fg_prereq_overview, os.path.join(ngp_overview_folderpath, "0001.png"))
fg_image = real_esrgan(fg_image)
fg_image.save(os.path.join(ngp_train_folderpath, "0001.png"))
