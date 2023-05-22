import argparse
import gc
import os
import sys

sys.path.append("..")

import torch

from pipelines.deepdanbooru_pipeline import *
from src.utils.file_loader import *

path = {
    "config_file": "inference.json",
    "dd_project_folder": "../ext/AnimeFaceNotebooks/deepdanbooru_model/",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    args = parser.parse_args()

    fl = file_loader()
    config = fl.load_json(path["config_file"])
    deepdanbooru = deepdanbooru_pipeline(path["dd_project_folder"])

    with open(parser.out_path, "w") as f:
        f.write(
            deepdanbooru(
                args.in_path,
                config["deepdanbooru"]["threshold"],
                config["deepdanbooru"]["multiplier"],
                config["deepdanbooru"]["prefix"],
            )
        )

    del deepdanbooru
    gc.collect()
    torch.cuda.empty_cache()
