import glob
import os
import pathlib

import torch
from compel import Compel
from diffusers import AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor


class base_sd_workflow:
    def __init__(self):
        self.vae = None
        self.scheduler = None
        self.image_processor = None
        self.controlnet = None
        self.pipe = None
        self.generator = None

    def load_vae(self, repo_id):
        self.vae = AutoencoderKL.from_pretrained(repo_id, torch_dtype=torch.float16)

    def load_image_processor(self):
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def load_controlnet(self):
        repo_ids = [
            "lllyasviel/control_v11p_sd15_openpose",
            "lllyasviel/control_v11f1p_sd15_depth",
            "lllyasviel/control_v11p_sd15_normalbae",
            "lllyasviel/control_v11p_sd15_lineart",
        ]

        self.controlnet = [
            ControlNetModel.from_pretrained(repo_id, torch_dtype=torch.float16)
            for repo_id in repo_ids
        ]

    def load_scheduler(self):
        self.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True
        )

    def load_textual_inversions(self, folderpath):
        for filepath in sorted(glob.glob(os.path.join(folderpath, "*"))):
            pl = pathlib.Path(filepath)
            if pl.suffix in [".safetensors", ".ckpt", ".pt"]:
                use_safetensors = pl.suffix == ".safetensors"
                self.pipe.load_textual_inversion(
                    filepath, token=pl.stem, use_safetensors=use_safetensors
                )

    def load_pipeline_optimizations(self, ops):
        if "xformers" in ops:
            self.pipe.enable_xformers_memory_efficient_attention()

        if "vae_tiling" in ops:
            self.pipe.enable_vae_tiling()

        if "model_offload" in ops:
            self.pipe.enable_model_cpu_offload()
        elif "cuda" in ops:
            self.pipe.to("cuda")

    def load_generator(self, seed):
        self.generator = torch.Generator(device="cpu")
        if seed == -1:
            seed = self.generator.seed()
        else:
            self.generator = self.generator.manual_seed(seed)

        return seed

    def load_weighted_embeds(self, prompt, negative_prompt):
        compel = Compel(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            truncate_long_prompts=False,
        )

        conditioning = compel.build_conditioning_tensor(prompt)
        negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
        return compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )
