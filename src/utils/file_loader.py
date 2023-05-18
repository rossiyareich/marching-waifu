import glob
import json
import os
import pathlib

import PIL.Image


def load_config(filepath):
    with open(filepath, "r") as f:
        config = json.load(f)
    return config


def load_prompt_addition(filepath):
    with open(filepath, "r") as f:
        raw = json.load(f)["prompt_addition"]
    prompt_additions = []
    for prompt_addition in raw:
        prompt_additions.append(prompt_addition["direction"])
    return prompt_additions


def load_controlnet_conditions(folderpath):
    prefixes = [
        "openpose_full",
        "depth",
        "normals",
        "lineart",
    ]

    num_controlnet_conditions = len(list(glob.iglob(os.path.join(folderpath, "*"))))
    controlnet_conditions = [[None] * 4 for _ in range(num_controlnet_conditions)]

    for j, prefix in enumerate(prefixes):
        controlnet_files = list(glob.iglob(os.path.join(folderpath, f"{prefix}*")))
        for i, filepath in enumerate(controlnet_files):
            pl = pathlib.Path(filepath)
            if pl.stem in prefixes:
                controlnet_conditions[i][j] = PIL.Image.open(filepath).convert("RGB")

    return controlnet_conditions
