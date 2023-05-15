import PIL.Image

from ..src.inference.inpaint import *
from ..src.utils.loaders import *
from ..src.inference.real_esrgan import *


embeds_folder = "../data/embeddings"
cond_folder = "../data/multi_controlnet/multi_controlnet_data"

inference_config_file = "inference.json"
directions_file = "../data/prompts/prompt_additions.json"
prompts_file = "../data/prompts/deepdanbooru_results.json"

updowned_fg_file = "../data/ngp/updowned/0001.png"
raw_save_folder = "../data/ngp/raw"
overview_save_folder = "../data/ngp/overview"
train_save_folder = "../data/ngp/train"
updowned_save_folder = "../data/ngp/updowned"


conf = load_conf(inference_config_file)["rg"]
data_size = conf["generation"]["data_size"]
direction = load_directions(directions_file, data_size)
controlnet_models, controlnet_conds = load_controlnet_models(cond_folder, data_size)

fg_image = PIL.Image.open(updowned_fg_file)
w, h, m = get_image_attrib(fg_image)

mask_image = PIL.Image.new("1", (w * 3, h))
mask_image.paste(True, (w, 0, w * 2, h))

prompt = prepare_prompt(
    prompts_file,
    conf["params"]["deepdanbooru_multiplier"],
).format(direction)

upscale = real_esrgan()

for i in range(2, data_size + 1):
    lg_image = PIL.Image.open(f"{updowned_save_folder}/{(i-1):04}.png")
    image = inpaint(
        concat_images([lg_image, lg_image, fg_image], w * 3, h, m),
        mask_image,
        prompt,
        conf["params"]["negative_prompt"],
        conf["models"]["ldm_repo_id"],
        conf["models"]["vae_repo_id"],
        embeds_folder,
        f"{overview_save_folder}/{i:04}.png"
        if conf["generation"]["overview"]
        else None,
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
    image = image.crop((w, 0, w * 2, h))
    image.save(f"{raw_save_folder}/{i:04}.png")

    upscaled_image = upscale(image)
    upscaled_image.save(f"{train_save_folder}/{i:04}.png")
    upscaled_image.resize(
        (image.width, image.height), resample=PIL.Image.Resampling.LANCZOS
    ).save(f"{updowned_save_folder}/{i:04}.png")

