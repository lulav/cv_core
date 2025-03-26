import numpy as np
import scipy as sp
from .rigid3dtform import Rigid3dTform


def _random_3d_transform(n):
    """
    utility function for randomizing 3D transform
    :param n: number of instances
    :return:
    """
    rotations = _random_rotation(n)
    translations = []
    for i in range(n):
        tvec =  (np.random.rand(1, 3) - 0.5) * 10
        translations.append(Rigid3dTform(rotations[i], tvec))
    return translations


def _random_rotation(n):
    """
    utility function for randomizing rotation
    :param n: number of instances
    :return:
    """
    rotations = []
    for i in range(n):
        rot_n = (np.random.rand(1, 3) - 0.5)
        rot_a = np.random.rand(1, 3) * 2 * np.pi
        rvec = rot_a * rot_n / np.linalg.norm(rot_n)
        rotations.append(sp.spatial.transform.Rotation.from_rotvec(rvec.flatten()))
    return rotations
