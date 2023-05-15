import gc
import glob
import os
import pathlib

import PIL.Image
import torch
from diffusers import DPMSolverMultistepScheduler

from ..pipelines.controlnet_unet4_pipeline import StableDiffusionControlNetPipeline
from ..utils.latents import *
from ..utils.loaders import *


class txt2img:
    def __init__(
        self,
        prompt,
        negative_prompt,
        ldm_repo_id,
        vae_repo_id,
        embeddings_folder,
        overview_save_file,
        ops,
        seed,
        steps,
        cfg_scale,
        controlnet_models,
        controlnet_conds,
        controlnet_conds_scale,
        controlnet_soft_exp,
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt

        self.ldm_repo_id = ldm_repo_id
        self.vae_repo_id = vae_repo_id
        self.embeddings_folder = embeddings_folder

        self.overview_save_file = overview_save_file

        self.ops = ops
        self.seed = seed
        self.steps = steps
        self.cfg_scale = cfg_scale

        self.controlnet_models = controlnet_models
        self.controlnet_conds = controlnet_conds
        self.controlnet_conds_scale = controlnet_conds_scale
        self.controlnet_soft_exp = controlnet_soft_exp

    def __call__(self):
        vae = load_vae(self.vae_repo_id)

        if self.overview_save_file is not None:
            image_processor = load_image_processor(vae)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.ldm_repo_id,
            vae=vae,
            controlnet=self.controlnet_models,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

        for file in glob.iglob(f"{self.embeddings_folder}/**"):
            if not os.path.isfile(file):
                continue
            pl = pathlib.Path(file)

            if pl.suffix not in [".safetensors", ".ckpt", ".pt"]:
                continue
            use_safetensors = pl.suffix == ".safetensors"
            pipe.load_textual_inversion(
                file, token=pl.stem, use_safetensors=use_safetensors
            )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )

        if "xformers" in self.ops:
            pipe.enable_xformers_memory_efficient_attention()

        if "cpu_offload" in self.ops:
            pipe.enable_model_cpu_offload()
        elif "cuda" in self.ops:
            pipe.to("cuda")

        seed, generator = load_generator(seed)
        print(seed)

        prompt_embeds, negative_prompt_embeds = load_weighted_embeds(
            pipe, self.prompt, self.negative_prompt
        )

        interim = []

        def cache_interim(step, timestep, latents):
            if self.overview_save_file is not None:
                interim.append(decode_latents(pipe, image_processor, latents))

        image = pipe(
            image=self.controlnet_conds,
            num_inference_steps=self.steps,
            guidance_scale=self.cfg_scale,
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback=cache_interim,
            callback_steps=int(self.steps / 4),
            controlnet_conditioning_scale=self.controlnet_cond_scale,
            controlnet_soft_exp=self.controlnet_soft_exp,
        ).images[0]

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        if self.overview_save_file is not None:
            w, h, m = image.width, image.height, image.mode
            ind_size = (int(w / 4), int(h / 4))

            overview = PIL.Image.new(m, (w + 2 * ind_size[0], h))
            overview.paste(image, (0, 0))
            for i, ind in enumerate(self.controlnet_conds):
                overview.paste(ind.resize(ind_size), (w, i * ind_size[1]))
            for i, ind in enumerate(interim):
                overview.paste(ind.resize(ind_size), (w + ind_size[0], i * ind_size[1]))

            overview.save(self.overview_save_file)

        return image
