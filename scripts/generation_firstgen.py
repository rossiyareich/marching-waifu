import os
import sys
import subprocess

sys.path.append("..")

import PIL.Image

from src.utils.file_loader import *
from src.utils.image_wrapper import *
from src.workflows.controlnet_unet4_workflow import *

path = {
    "config_file": "inference.json",
    "prompt_additions_file": "../data/prompts/prompt_additions.json",
    "dd_prompt_file": "../data/prompts/deepdanbooru_prompt.txt",
    "textual_inversion_folder": "../data/embeddings/",
    "controlnet_conditions_folder": "../data/multi_controlnet/multi_controlnet_data/",
    "ngp_overview_folder": "../data/ngp/overview/",
    "ngp_train_folder": "../data/ngp/train/",
}


def process_results(image, seed, interim, filename):
    print(seed)

    image.save(os.path.join(path["ngp_train_folder"], filename))

    if interim is not None and len(interim) > 0:
        interim = [image_wrapper(x, "pil") for x in interim]

        interim_img = interim[0]
        for img in interim[1:]:
            interim_img.concatenate(img)

        interim_img.to_pil().save(os.path.join(path["ngp_overview_folder"], filename))


if __name__ == "__main__":
    # 0. Prepare all pipeline stages
    fl = file_loader()
    config = fl.load_json(path["config_file"])
    prompt_additions = [
        j["prompt_addition"]["direction"]
        for j in fl.load_json(path["prompt_additions_file"])
    ]

    controlnet_conditions = fl.load_controlnet_conditions(
        path["controlnet_conditions_folder"]
    )
    controlnet_scales = [
        config["controlnet"]["unit_scales"]["openpose"],
        config["controlnet"]["unit_scales"]["depth"],
        config["controlnet"]["unit_scales"]["normals"],
        config["controlnet"]["unit_scales"]["lineart"],
    ]

    unet4 = controlnet_unet4_workflow(
        config["models"]["vae_repo_id"],
        config["models"]["ldm_repo_id"],
        path["textual_inversion_folder"],
    )

    # 1. Generate prereq image
    image, seed, interim = unet4(
        config["pipeline"]["firstgen"]["prompt"].format(prompt_additions[0]),
        config["pipeline"]["firstgen"]["negative_prompt"],
        config["pipeline"]["firstgen"]["steps"],
        config["pipeline"]["firstgen"]["cfg_scale"],
        config["pipeline"]["firstgen"]["denoising_strength"],
        config["pipeline"]["firstgen"]["seed"],
        config["pipeline"]["firstgen"]["callback_steps"],
        controlnet_conditions[0],
        controlnet_scales,
        config["controlnet"]["guidance"]["start"],
        config["controlnet"]["guidance"]["end"],
        config["controlnet"]["soft_exp"],
    )
    filename = "prereq.png"
    filepath = os.path.join(path["ngp_train_folder"], filename)
    process_results(image, seed, interim, filename)

    # 2. Run inference on DeepDanbooru & load inferred prompt
    subprocess.call(
        [
            "python",
            "inference_deepdanbooru.py",
            filepath,
            path["dd_prompt_file"],
        ]
    )
    prompt = fl.load_text(path["dd_prompt_file"])

    # 3. Upscale prereq image
    subprocess.call(["python", "inference_realesrgan.py", filepath, filepath])

    # 4. Downscale prereq image & generate first image
    image, seed, interim = unet4(
        prompt.format(prompt_additions[0]),
        config["pipeline"]["firstgen"]["negative_prompt"],
        config["pipeline"]["firstgen"]["steps"],
        config["pipeline"]["firstgen"]["cfg_scale"],
        config["pipeline"]["firstgen"]["img2img_denoising_strength"],
        config["pipeline"]["firstgen"]["img2img_seed"],
        config["pipeline"]["firstgen"]["callback_steps"],
        controlnet_conditions[0],
        controlnet_scales,
        config["controlnet"]["guidance"]["start"],
        config["controlnet"]["guidance"]["end"],
        config["controlnet"]["soft_exp"],
        image_wrapper(PIL.Image.open(filepath), "pil").scale(0.25).to_pil(),
    )
    filename = "0001.png"
    filepath = os.path.join(path["ngp_train_folder"], filename)
    process_results(image, seed, interim, filename)

    # 5. Upscale first image
    subprocess.call(["python", "inference_realesrgan.py", filepath, filepath])
