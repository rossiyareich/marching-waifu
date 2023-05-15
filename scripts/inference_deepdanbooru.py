from ..src.inference.deepdanbooru import deepdanbooru
from ..src.utils.loaders import *

conf = load_conf("inference.json")["deepdanbooru"]

deepdanbooru(
    "../ext/AnimeFaceNotebooks/deepdanbooru_model",
    "../data/ngp/train/0001_preliminary.png",
    "../data/prompts/deepdanbooru_results.json"
)(conf["threshold"])