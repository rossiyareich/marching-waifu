import torch

from workflows.base_sd_workflow import *

from ..pipelines.controlnet_unet9_pipeline import *
from ..utils.torch_utils_extended import *


class controlnet_unet9_workflow(base_sd_workflow):
    def __init__(self, vae_repo_id, ldm_repo_id, textual_inversion_folderpath, ops):
        super.__init__()

        self.load_vae(vae_repo_id)
        self.load_image_processor()
        self.load_controlnet()

        self.pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
            ldm_repo_id,
            vae=self.vae,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.load_scheduler()
        self.load_textual_inversions(textual_inversion_folderpath)
        self.load_pipeline_optimizations(ops)

    def __call__(
        self,
        prompt,
        negative_prompt,
        image,
        mask_image,
        steps,
        cfg_scale,
        denoising_strength,
        seed,
        callback_steps,
        controlnet_conditions,
        controlnet_scales,
        controlnet_soft_exp,
        fill_mode,
    ):
        seed = self.load_generator(seed)

        prompt_embeds, negative_prompt_embeds = self.load_weighted_embeds(
            prompt, negative_prompt
        )

        interim = []

        def cache_interim(step, timestep, latents):
            interim.append(self.decode_latents(latents))

        image = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            mask_image=mask_image,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            strength=denoising_strength,
            generator=self.generator,
            callback=(cache_interim if callback_steps != 0 else None),
            callback_steps=callback_steps,
            controlnet_conditioning_image=controlnet_conditions,
            controlnet_conditioning_scale=controlnet_scales,
            controlnet_soft_exp=controlnet_soft_exp,
            fill_mode=fill_mode,
        ).images[0]

        empty_cache()

        return (image, seed, interim)
