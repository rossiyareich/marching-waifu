import PIL

from ..src.inference.inpaint import *
from ..src.utils.loaders import *


conf = load_conf("inference.json")["fg"]

direction = load_directions("../data/prompts/prompt_additions.json", 1)[0]
controlnet_models, controlnet_conds = load_controlnet_models(
    "../data/multi_controlnet/multi_controlnet_data", 1
)

pl_image = PIL.Image.open("../data/ngp/train/0001_preliminary.png")
w, h, m = get_image_attrib(pl_image)

mask_image = PIL.Image.new("1", (w * 2, h))
mask_image.paste(True, (w, 0, w * 2, h))

image = inpaint(
    concat_images([pl_image, pl_image], w * 2, h, m),
    mask_image,
    prepare_prompt(
        "../data/prompts/deepdanbooru_results.json",
        conf["params"]["deepdanbooru_multiplier"],
    ).format(direction),
    conf["params"]["negative_prompt"],
    conf["models"]["ldm_repo_id"],
    conf["models"]["vae_repo_id"],
    "../data/embeddings",
    "../data/ngp/train/0001-overview.png",
    conf["generation"]["ops"],
    conf["params"]["seed"],
    conf["params"]["steps"],
    conf["params"]["cfg_scale"],
    controlnet_models,
    [concat_images([c, c], w * 2, h, m) for c in controlnet_conds[0]],
    list(conf["controlnet"]["unit_scales"].keys()),
    conf["controlnet"]["soft_exp"],
)()
image.crop((w, 0, w * 2, h)).save("../data/ngp/train/0001.png")
