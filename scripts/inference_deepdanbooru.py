from ..src.inference.deepdanbooru import deepdanbooru
from ..src.utils.loaders import *


project_path = "../ext/AnimeFaceNotebooks/deepdanbooru_model"

inference_config_file = "inference.json"

image_path = "../data/ngp/updowned/0001_preliminary.png"
save_file = "../data/prompts/deepdanbooru_results.json"


conf = load_conf(inference_config_file)["deepdanbooru"]
deepdanbooru(project_path, image_path, save_file)(conf["threshold"])
