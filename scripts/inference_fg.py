import PIL.Image

from ..src.inference.inpaint import *
from ..src.utils.loaders import *
from ..src.inference.real_esrgan import *


embeds_folder = "../data/embeddings"
cond_folder = "../data/multi_controlnet/multi_controlnet_data"

inference_config_file = "inference.json"
directions_file = "../data/prompts/prompt_additions.json"
prompts_file = "../data/prompts/deepdanbooru_results.json"

updowned_fg_file = "../data/ngp/updowned/0001_preliminary.png"
raw_save_file = "../data/ngp/raw/0001.png"
overview_save_file = "../data/ngp/overview/0001.png"
train_save_file = "../data/ngp/train/0001.png"
updowned_save_file = "../data/ngp/updowned/0001.png"


conf = load_conf(inference_config_file)["fg"]
direction = load_directions(directions_file, 1)[0]
controlnet_models, controlnet_conds = load_controlnet_models(cond_folder, 1)

pl_image = PIL.Image.open(updowned_fg_file)
w, h, m = get_image_attrib(pl_image)

mask_image = PIL.Image.new("1", (w * 2, h))
mask_image.paste(True, (w, 0, w * 2, h))

prompt = prepare_prompt(prompts_file, conf["params"]["deepdanbooru_multiplier"]).format(
    direction
)

image = inpaint(
    concat_images([pl_image, pl_image], w * 2, h, m),
    mask_image,
    prompt,
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
    [concat_images([c, c], w * 2, h, m) for c in controlnet_conds[0]],
    list(conf["controlnet"]["unit_scales"].keys()),
    conf["controlnet"]["soft_exp"],
)()
image = image.crop((w, 0, w * 2, h))
image.save(raw_save_file)

upscaled_image = real_esrgan()(image)
upscaled_image.save(train_save_file)
upscaled_image.resize(
    (image.width, image.height), resample=PIL.Image.Resampling.LANCZOS
).save(updowned_save_file)
