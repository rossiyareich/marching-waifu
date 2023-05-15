import sys

sys.path.append("../ext/AnimeFaceNotebooks/DeepDanbooru")

import collections
import json

import deepdanbooru as dd
import numpy as np
import tensorflow as tf


class deepdanbooru:
    def __init__(self, project_path, image_path, save_file):
        self.project_path = project_path
        self.image_path = image_path
        self.save_file = save_file

    def __call__(self, threshold):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        classif_model = dd.project.load_model_from_project(self.project_path)
        all_tags = dd.project.load_tags_from_project(self.project_path)
        model_width = classif_model.input_shape[2]
        model_height = classif_model.input_shape[1]
        all_tags = np.array(all_tags)

        image = dd.data.load_image_for_evaluate(
            self.image_path, width=model_width, height=model_height
        )
        image_arrays = np.array([image])

        # Decode
        results = classif_model.predict(image_arrays).reshape(-1, all_tags.shape[0])

        result_list = []
        for result_set in results:
            result_tags = {}
            for i in range(len(all_tags)):
                if result_set[i] > threshold:
                    result_tags[all_tags[i]] = result_set[i]
            sorted_tags = reversed(
                sorted(result_tags.keys(), key=lambda x: result_tags[x])
            )
            sorted_results = collections.OrderedDict()
            for tag in sorted_tags:
                sorted_results[tag] = result_tags[tag]
            result_list.append(sorted_results)

        with open(self.save_file, "w") as f:
            f.write(json.dumps(result_list[0], indent=4))
