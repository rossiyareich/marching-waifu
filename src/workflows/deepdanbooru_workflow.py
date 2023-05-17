import collections

import deepdanbooru as dd
import numpy as np
import tensorflow as tf


class deepdanbooru_workflow:
    def __init__(self, projectpath):
        self.prompt = None
        self.sorted_results = None
        self.project_path = projectpath

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.classif_model = dd.project.load_model_from_project(self.project_path)
        self.all_tags = dd.project.load_tags_from_project(self.project_path)
        self.model_width = self.classif_model.input_shape[2]
        self.model_height = self.classif_model.input_shape[1]
        self.all_tags = np.array(self.all_tags)

    def load_prompts(self, multiplier, prefix):
        self.prompt = prefix
        for tag, prob in self.sorted_results.items():
            tag_strength = prob * multiplier
            self.prompt += f"({tag}){tag_strength}, "
        self.prompt = self.prompt[:-1]

    def __call__(self, imagepath, threshold):
        image = dd.data.load_image_for_evaluate(
            imagepath, width=self.model_width, height=self.model_height
        )
        image = np.array([image])

        # Decode
        result = self.classif_model.predict(image).reshape(-1, self.all_tags.shape[0])[
            0
        ]

        result_tags = {}
        for i in range(len(self.all_tags)):
            if result[i] > threshold:
                result_tags[self.all_tags[i]] = result[i]
        sorted_tags = reversed(sorted(result_tags.keys(), key=lambda x: result_tags[x]))
        self.sorted_results = collections.OrderedDict()
        for tag in sorted_tags:
            self.sorted_results[tag] = result_tags[tag]
