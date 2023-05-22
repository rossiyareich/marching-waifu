import os

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

from src.utils.image_wrapper import *


class realesrgan_pipeline:
    @torch.no_grad()
    def __init__(
        self,
        outscale=4.0,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        face_enhance=True,
        fp32=False,
        gpu_id=0,
    ):
        self.outscale = outscale

        # x4 RRDBNet model with 6 blocks
        model_name = "RealESRGAN_x4plus_anime_6B"
        model = RRDBNet(3, 3, 4, 64, 6, 32)
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]

        # Determine model paths
        model_path = os.path.join("weights", model_name + ".pth")
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(
                    url,
                    os.path.join(ROOT_DIR, "weights"),
                    True,
                    None,
                )

        # Restorer
        self.upsampler = RealESRGANer(
            netscale,
            model_path,
            None,
            model,
            tile,
            tile_pad,
            pre_pad,
            not fp32,
            None,
            gpu_id,
        )

        # Use GFPGAN for face enhancement
        self.face_enhancer = None
        if face_enhance:
            from gfpgan import GFPGANer

            self.face_enhancer = GFPGANer(
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                outscale,
                "clean",
                2,
                self.upsampler,
                None,
            )

    @torch.no_grad()
    def __call__(self, source_img):
        source_img = image_wrapper(source_img, "pil").to_cv2()

        if self.face_enhancer is not None:
            _, _, output = self.face_enhancer.enhance(
                source_img,
                False,
                False,
                True,
            )
        else:
            output, _ = self.upsampler.enhance(source_img, self.outscale)

        return image_wrapper(output, "cv2").to_pil()
