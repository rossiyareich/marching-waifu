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
        raw = json.load(f)
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
    controlnet_conditions = []
    for filepath in glob.iglob(os.path.join(folderpath, "**")):
        pl = pathlib.Path(filepath)
        if not os.path.isfile(filepath):
            continue
        if not pl.suffix in [".png"]:
            continue
        if not pl.stem[:-4] in prefixes:
            continue
        controlnet_conditions.append(PIL.Image.open(filepath))
    return controlnet_conditions
