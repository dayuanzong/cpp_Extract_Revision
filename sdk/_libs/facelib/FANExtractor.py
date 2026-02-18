from .WrapperManager import get_wrapper
import numpy as np

class FANExtractor(object):
    def __init__ (self, landmarks_3D=False, place_model_on_cpu=False, model_path=None, device_id=0):
        if place_model_on_cpu:
            device_id = -1
        self.wrapper = get_wrapper(device_id)

    def extract (self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        if len(rects) == 0:
            return []

        if not is_bgr:
            input_image = input_image[:, :, ::-1]

        landmarks = []
        for rect in rects:
            # rect is [l, t, r, b]
            # wrapper.extract expects [x1, y1, x2, y2]
            lms = self.wrapper.extract(input_image, rect)
            landmarks.append(np.array(lms))
            
        return landmarks
