import PIL.Image

from ..src.inference.real_esrgan import *
from ..src.inference.txt2img import *
from ..src.utils.loaders import *

embeds_folder = "../data/embeddings"
cond_folder = "../data/multi_controlnet/multi_controlnet_data"

inference_config_file = "inference.json"
directions_file = "../data/prompts/prompt_additions.json"

raw_save_file = "../data/ngp/raw"
overview_save_file = "../data/ngp/overview"
train_save_file = "../data/ngp/train"
updowned_save_file = "../data/ngp/updowned"


conf = load_conf(inference_config_file)["fg_preliminary"]
direction = load_directions(directions_file, 1)[0]
controlnet_models, controlnet_conds = load_controlnet_models(cond_folder, 1)
image = txt2img(
    conf["params"]["prompt"].format(direction),
    conf["params"]["negative_prompt"],
    conf["models"]["ldm_repo_id"],
    conf["models"]["vae_repo_id"],
    embeds_folder,
    overview_save_file if conf["generation"]["overview"] else None,
    conf["generation"]["ops"],
    conf["params"]["seed"],
    conf["params"]["steps"],
    conf["params"]["cfg_scale"],
    controlnet_models,
    controlnet_conds[0],
    list(conf["controlnet"]["unit_scales"].keys()),
    conf["controlnet"]["soft_exp"],
)()
image.save(raw_save_file)

upscaled_image = real_esrgan()(image)
upscaled_image.resize(
    (image.width, image.height), resample=PIL.Image.Resampling.LANCZOS
).save(updowned_save_file)
