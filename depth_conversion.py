"""
    This file contains the function to convert the depth from Airsim
    to the depth from the image plane. This is necessary because Airsim
    measures the depth from the camera center and we need it from the
    image plane. Source: https://github.com/unrealcv/unrealcv/issues/14#issuecomment-307028752
"""
import numpy as np

def depth_conversion(point_depth, f):
    """Transformation to Airim perspective depth to correct.
        Airsim depth is measured from camera center but we need
        it from the image plane.
    """
    H = point_depth.shape[0]
    W = point_depth.shape[1]
    i_c = np.float64(H) / 2 - 1
    j_c = np.float64(W) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(
        0, W-1, num=W), np.linspace(0, H-1, num=H))
    distance_from_center = np.sqrt((rows - i_c)**2 + (columns - j_c)**2)
    plane_depth = point_depth / np.sqrt(1 + (distance_from_center / (f))**2)
    return plane_depth
