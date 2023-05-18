import os
import sys

sys.path.append("..")

from src.workflows.deepdanbooru_workflow import *
from src.utils.file_loader import *

config_filepath = "inference.json"
deepdanbooru_prompt_filepath = "../data/prompts/deepdanbooru_prompt.txt"
deepdanbooru_project_folderpath = "../ext/AnimeFaceNotebooks/deepdanbooru_model/"
ngp_train_folderpath = "../data/ngp/train/"


def work_load_config():
    global config
    config = load_config(config_filepath)


def work_load_deepdanbooru():
    global deepdanbooru
    deepdanbooru = deepdanbooru_workflow(deepdanbooru_project_folderpath)


def work_run_deepdanbooru():
    global prompt
    deepdanbooru(
        os.path.join(ngp_train_folderpath, "0001_prereq.png"),
        config["deepdanbooru"]["threshold"],
    )
    prompt = deepdanbooru.load_prompts(
        config["deepdanbooru"]["multiplier"], config["deepdanbooru"]["prefix"]
    )
    with open(deepdanbooru_prompt_filepath, "w") as f:
        f.write(prompt)


if __name__ == "__main__":
    work_load_config()
    work_load_deepdanbooru()
    work_run_deepdanbooru()
