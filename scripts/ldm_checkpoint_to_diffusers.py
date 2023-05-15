import gc
import json

import torch

from ..src.pipelines.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)

with open("ldm.json", "r") as f:
    confs = json.loads(f)

for conf in confs:
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path=conf["checkpoint_path"],
        original_config_file=conf["original_config_file"],
        prediction_type="epsilon",
        extract_ema=conf["extract_ema"],
        scheduler_type="dpm-karras",
        device=conf["device"],
        from_safetensors=conf["from_safetensors"],
    )

    if conf["half"]:
        pipe.to(torch_dtype=torch.float16)
    pipe.save_pretrained(conf["dump_path"], safe_serialization=conf["to_safetensors"])

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
