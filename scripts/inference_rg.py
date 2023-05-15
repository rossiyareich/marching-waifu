import PIL

from ..src.inference.inpaint import *
from ..src.utils.loaders import *


conf = load_conf("inference.json")["rg"]
data_size = conf["generation"]["data_size"]

direction = load_directions("../data/prompts/prompt_additions.json", data_size)
controlnet_models, controlnet_conds = load_controlnet_models(
    "../data/multi_controlnet/multi_controlnet_data", data_size
)

fg_image = PIL.Image.open("../data/ngp/train/0001.png")
w, h, m = get_image_attrib(fg_image)

mask_image = PIL.Image.new("1", (w * 3, h))
mask_image.paste(True, (w, 0, w * 2, h))

for i in range(2, data_size + 1):
    lg_image = PIL.Image.open(f"../data/ngp/train/{(i-1):04}.png")
    image = inpaint(
        concat_images([lg_image, lg_image, fg_image], w * 3, h, m),
        mask_image,
        prepare_prompt(
            "../data/prompts/deepdanbooru_results.json",
            conf["params"]["deepdanbooru_multiplier"],
        ).format(direction),
        conf["params"]["negative_prompt"],
        conf["models"]["ldm_repo_id"],
        conf["models"]["vae_repo_id"],
        "../data/embeddings",
        None,
        conf["generation"]["ops"],
        conf["params"]["seed"],
        conf["params"]["steps"],
        conf["params"]["cfg_scale"],
        controlnet_models,
        [
            concat_images([c1, c2, c3], w * 3, h, m)
            for c1, c2, c3 in zip(
                controlnet_conds[i - 1], controlnet_conds[i], controlnet_conds[0]
            )
        ],
        list(conf["controlnet"]["unit_scales"].keys()),
        conf["controlnet"]["soft_exp"],
    )()
    image.crop((w, 0, w * 2, h)).save(f"../data/ngp/train/{i:04}.png")
