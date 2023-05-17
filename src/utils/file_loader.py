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

    controlnet_conditions = [
        [None] * 4 for _ in range(len(list(glob.iglob(os.path.join(folderpath, "**")))))
    ]

    for j, prefix in enumerate(prefixes):
        for i, filepath in enumerate(
            glob.iglob(os.path.join(folderpath, f"{prefix}*"))
        ):
            pl = pathlib.Path(filepath)
            if not os.path.isfile(filepath):
                continue
            if not pl.suffix in [".png"]:
                continue
            if not pl.stem[:-4] in prefixes:
                continue
            controlnet_conditions[i][j] = PIL.Image.open(filepath, formats=("RGB"))

    return controlnet_conditions
