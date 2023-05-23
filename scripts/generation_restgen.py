import os
import sys
import subprocess

sys.path.append("..")

import PIL.Image

from src.utils.file_loader import *
from src.utils.image_wrapper import *
from src.workflows.controlnet_unet9_workflow import *

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


def generate_mask(width, height, length, index):
    mask = PIL.Image.new("1", (width, height))
    mask.paste(
        True,
        (int(index / length * width), 0, int((index + 1) / length * width), height),
    )

    return mask


if __name__ == "__main__":
    # 0. Prepare all pipeline stages
    fl = file_loader()
    config = fl.load_json(path["config_file"])
    prompt_additions = [
        j["direction"]
        for j in fl.load_json(path["prompt_additions_file"])["prompt_addition"]
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

    prompt = fl.load_text(path["dd_prompt_file"])

    unet9 = controlnet_unet9_workflow(
        config["models"]["vae_repo_id"],
        config["models"]["ldm_repo_id"],
        path["textual_inversion_folder"],
    )

    # 1. Downscale first image
    filename = "0001.png"
    filepath = os.path.join(path["ngp_train_folder"], filename)
    downscaled_firstgen = image_wrapper(PIL.Image.open(filepath), "pil").scale(0.25)

    # 2. Generate second image
    downscaled = image_wrapper(downscaled_firstgen.to_pil().copy(), "pil")
    downscaled = downscaled.concatenate(downscaled_firstgen).to_pil()

    mask = generate_mask(downscaled.width, downscaled.height, 2, 1)

    image, seed, interim = unet9(
        prompt.format(prompt_additions[1]),
        config["pipeline"]["restgen"]["negative_prompt"],
        config["pipeline"]["restgen"]["steps"],
        config["pipeline"]["restgen"]["cfg_scale"],
        config["pipeline"]["restgen"]["denoising_strength"],
        config["pipeline"]["restgen"]["seed"],
        config["pipeline"]["restgen"]["callback_steps"],
        controlnet_conditions[1],
        controlnet_scales,
        config["controlnet"]["guidance"]["start"],
        config["controlnet"]["guidance"]["end"],
        config["controlnet"]["soft_exp"],
        downscaled,
        mask,
        config["pipeline"]["restgen"]["inpaint_method"],
    )
    filename = "0002.png"
    filepath = os.path.join(path["ngp_train_folder"], filename)
    process_results(image, seed, interim, filename)

    # 3. Upscale second image
    subprocess.call(["python", "inference_realesrgan.py", filepath, filepath])

    # 4. Run restgen loop:
    #       Downscale previous image
    #       Generate current image
    #       Upscale current image
    for i in range(2, config["pipeline"]["restgen"]["dataset_size"]):
        filename = f"{i:04}.png"
        filepath = os.path.join(path["ngp_train_folder"], filename)
        downscaled = image_wrapper(PIL.Image.open(filepath), "pil").scale(0.25)
        downscaled = (
            downscaled.concatenate(downscaled).concatenate(downscaled_firstgen).to_pil()
        )

        mask = generate_mask(downscaled.width, downscaled.height, 3, 1)

        controlnet_conditions_ = [
            image_wrapper(controlnet_condition_, "pil")
            for controlnet_condition_ in controlnet_conditions[i - 1]
        ]
        for j in [i, 0]:
            for k, controlnet_condition_ in enumerate(controlnet_conditions_[j]):
                controlnet_conditions_[k].concatenate(controlnet_condition_)

        image, seed, interim = unet9(
            prompt.format(prompt_additions[i]),
            config["pipeline"]["restgen"]["negative_prompt"],
            config["pipeline"]["restgen"]["steps"],
            config["pipeline"]["restgen"]["cfg_scale"],
            config["pipeline"]["restgen"]["denoising_strength"],
            config["pipeline"]["restgen"]["seed"],
            config["pipeline"]["restgen"]["callback_steps"],
            controlnet_conditions_,
            controlnet_scales,
            config["controlnet"]["guidance"]["start"],
            config["controlnet"]["guidance"]["end"],
            config["controlnet"]["soft_exp"],
            downscaled,
            mask,
            config["pipeline"]["restgen"]["inpaint_method"],
        )
        filename = f"{(i+1):04}.png"
        filepath = os.path.join(path["ngp_train_folder"], filename)
        process_results(image, seed, interim, filename)

        subprocess.call(["python", "inference_realesrgan.py", filepath, filepath])
