import torch
from diffusers.utils import randn_tensor


@torch.no_grad()
def decode_latents(pipe, image_processor, latents):
    image = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor, return_dict=False
    )[0]
    do_denormalize = [True] * image.shape[0]
    return image_processor.postprocess(
        image, output_type="pil", do_denormalize=do_denormalize
    )[0]


@torch.no_grad()
def encode_latents(pipe, image_processor, generator, image):
    image = image_processor.preprocess(image)
    image = image.to(device=pipe.device, dtype=torch.float16)

    image_latents = pipe.vae.encode(image).latent_dist.sample(generator)
    image_latents = pipe.vae.config.scaling_factor * image_latents
    image_latents = torch.cat([image_latents])
    return image_latents


@torch.no_grad()
def noise_latents(pipe, generator, steps, latents):
    latents = latents.to(pipe.device)

    pipe.scheduler.set_timesteps(steps, device=pipe.device)
    noise = randn_tensor(
        latents.shape, generator=generator, device=pipe.device, dtype=torch.float16
    )
    noise_latents = pipe.scheduler.add_noise(
        latents, noise, pipe.scheduler.timesteps[0]
    )
    return noise_latents
