import cv2
import numpy as np
import PIL.Image


class image_wrapper:
    def __init__(self, img):
        if isinstance(img, PIL.Image):
            self.img = img.copy()
        elif isinstance(img, np.array):
            self.img = PIL.Image.fromarray(img)
        elif isinstance(img, cv2.Mat):
            self.img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def to_pil(self):
        return self.img

    def to_np(self):
        return np.array(self.img)

    def to_cv2(self):
        return cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)

    def resize(self, width, height):
        self.img = self.img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        return self

    def scale(self, scale):
        return self.resize(int(self.img.width * scale), int(self.img.height * scale))

    def concatenate(self, other, axis=0):
        width, height = self.img.width, self.img.height

        width, height = (
            width if axis == 0 else width + other.img.width,
            height if axis == 1 else height + other.img.height,
        )

        new_image = PIL.Image.new(other.img.mode, (width, height))
        new_image.paste(self.img, (0, 0))

        if axis == 0:
            new_image.paste(other.img, (width, 0))
        elif axis == 1:
            new_image.paste(other.img, (0, height))

        self.img = new_image
        return self
