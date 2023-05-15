from ..src.inference.txt2img import *
from ..src.utils.loaders import *

conf = load_conf("inference.json")["fg_preliminary"]

direction = load_directions("../data/prompts/prompt_additions.json", 1)[0]
controlnet_models, controlnet_conds = load_controlnet_models(
    "../data/multi_controlnet/multi_controlnet_data", 1
)

image = txt2img(
    conf["params"]["prompt"].format(direction),
    conf["params"]["negative_prompt"],
    conf["models"]["ldm_repo_id"],
    conf["models"]["vae_repo_id"],
    "../data/embeddings",
    "../data/ngp/train/0001_preliminary-overview.png",
    conf["generation"]["ops"],
    conf["params"]["seed"],
    conf["params"]["steps"],
    conf["params"]["cfg_scale"],
    controlnet_models,
    controlnet_conds[0],
    list(conf["controlnet"]["unit_scales"].keys()),
    conf["controlnet"]["soft_exp"],
)()
image.save("../data/ngp/train/0001_preliminary.png")
