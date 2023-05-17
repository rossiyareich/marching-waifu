import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

from ..utils.image_wrapper import *


class real_esrgan_workflow:
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

        # determine models according to model names
        # x4 RRDBNet model with 6 blocks
        model_name = "RealESRGAN_x4plus_anime_6B"
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]

        # determine model paths
        model_path = os.path.join("weights", model_name + ".pth")
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url,
                    model_dir=os.path.join(ROOT_DIR, "weights"),
                    progress=True,
                    file_name=None,
                )

        # restorer
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=(not fp32),
            gpu_id=gpu_id,
        )

        self.face_enhancer = None
        # Use GFPGAN for face enhancement
        if face_enhance:
            from gfpgan import GFPGANer

            self.face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
            )

    def __call__(self, source_img):
        source_img = image_wrapper(source_img).to_np()

        try:
            if self.face_enhancer is not None:
                _, _, output = self.face_enhancer.enhance(
                    source_img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
            else:
                output, _ = self.upsampler.enhance(source_img, outscale=self.outscale)
        except RuntimeError as error:
            print("Error", error)
            print(
                "If you encounter CUDA out of memory, try to set --tile with a smaller number."
            )
        else:
            extension = extension[1:]
            return image_wrapper(output).to_pil()
