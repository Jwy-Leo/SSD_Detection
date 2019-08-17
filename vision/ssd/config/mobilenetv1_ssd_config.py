import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
# image_mean = np.array([104,117,123])  # RGB layout
image_std = 128.0
iou_threshold = 0.5 #0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]

'''
specs = [
    SSDSpec(19, 16, SSDBoxSizes(30, 52.5), [1.5, 1]),
    SSDSpec(10, 32, SSDBoxSizes(52.5, 75), [1.5, 1]),
    SSDSpec(5, 64, SSDBoxSizes(75, 97.5), [1.5, 1]),
    SSDSpec(3, 100, SSDBoxSizes(97.5, 120), [1.5, 1]),
    SSDSpec(2, 150, SSDBoxSizes(120, 142.5), [1.5, 1]),
    SSDSpec(1, 300, SSDBoxSizes(142.55, 165), [1.5, 1])
]
'''
'''
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [1, 1.5]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [1, 1.5]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [1, 1.5]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [1, 1.5]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [1, 1.5]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [1, 1.5]),
]
'''
# VIRAT
# specs = [
#     SSDSpec(19, 16, SSDBoxSizes(3, 6), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(6, 9), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(9, 10), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(10, 15), [2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(15, 17), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(17, 20), [2, 3])
# ]
'''
specs = [
    SSDSpec(19, 16, SSDBoxSizes(16, 24), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(24, 36), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(36, 40), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(40, 60), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(60, 68), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(68, 80), [2, 3])
]
'''

priors = generate_ssd_priors(specs, image_size)
