import gc
from typing import List, Optional, Tuple, Union

import torch


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def zeros_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List[torch.Generator], torch.Generator]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
):
    rand_device = device
    batch_size = shape[0]
    layout = layout or torch.strided
    device = device or torch.device("cpu")
    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.zeros(
                shape,
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for _ in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.zeros(
            shape,
            device=rand_device,
            dtype=dtype,
            layout=layout,
        ).to(device)
    return latents
