import os
import random
import traceback
import cv2
import numpy as np
import scipy as sp
from geometry_3D import Rigid3dTform
import pinhole_camera

def _random_3d_transform(n):
    """
    make a random rotation
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
    make a random rotation
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

def _random_pinhole_camera_params(n):
    """
    make a random pinhole camra
    :param n: number of instances
    :return:
    """
    T = _random_3d_transform(n)
    pinhole_camera_params = []
    for i in range(n):
        id = 'camera_{}'.format(i)
        model = np.random.choice(pinhole_camera.CameraModel)
        f = 100 + np.random.rand(2)*2000
        image_size = np.round(100 + np.random.rand(2) * 1000)
        c =  image_size/2 + (np.random.rand(2)-0.5) * image_size/10
        K = np.array([[f[0], 0, c[0]],
                      [0, f[1], c[1]],
                      [0, 0, 1]])
        D = (np.random.rand(4)-0.5) * 0.1
        skew = 0
        pcp = {'id': id,
        'model': model,
        'K': K,
        'D': D,
        'image_size': image_size,
        'skew': skew,
        'TC2B': T[i]}
        pinhole_camera_params.append(pcp)

    return pinhole_camera_params


def test_pinhole_create():
    """
    test id pinhole camera creates correctly
    :return:
    """

    n = 100
    camera_params = _random_pinhole_camera_params(n)

    # sel all params
    res1 = np.zeros((n,1), dtype=bool)
    for i in range(n):
        ph = pinhole_camera.PinholeCamera()
        ph.set(camera_params[i]['id'],
               camera_params[i]['model'],
               camera_params[i]['K'],
               camera_params[i]['D'],
               camera_params[i]['image_size'],
               T_cam_to_body = camera_params[i]['TC2B'].T,
               skew =  camera_params[i]['skew'])

        res1[i] = camera_params[i]['id'] == ph.id and \
                 camera_params[i]['model'] == ph.model and \
                 camera_params[i]['id'] == ph.id and \
                 np.all(np.abs(camera_params[i]['K'] - ph.K) < 1e-9) and \
                 np.all(np.abs(camera_params[i]['D'] - ph.distortion_coefficients) < 1e-9) and \
                 np.all(np.abs(camera_params[i]['image_size'] - ph.image_size) == 1e-9) and \
                 np.all(np.abs(camera_params[i]['TC2B'].T - ph.T_cam_to_body) == 1e-9) and \
                 camera_params[i]['skew'] == ph.skew

    # default T_cam_to_body and skew
    res2 = np.zeros((n, 1), dtype=bool)
    for i in range(n):
        ph = pinhole_camera.PinholeCamera()
        ph.set(camera_params[i]['id'],
               camera_params[i]['model'],
               camera_params[i]['K'],
               camera_params[i]['D'],
               camera_params[i]['image_size'])

        res2[i] = camera_params[i]['id'] == ph.id and \
                  camera_params[i]['model'] == ph.model and \
                  camera_params[i]['id'] == ph.id and \
                  np.all(np.abs(camera_params[i]['K'] - ph.K) < 1e-9) and \
                  np.all(np.abs(camera_params[i]['D'] - ph.distortion_coefficients) < 1e-9) and \
                  np.all(np.abs(camera_params[i]['image_size'] - ph.image_size) == 1e-9) and \
                  camera_params[i]['TC2B'] is None and \
                  camera_params[i]['skew'] is None

    return np.all(res1) and np.all(res2)

def test_pinhole_load_save():
    """
    test id pinhole camera creates correctly
    :return:
    """

    n = 100
    camera_params = _random_pinhole_camera_params(n)

    # sel all params
    res = np.zeros((n,1), dtype=bool)
    for i in range(n):
        ph = pinhole_camera.PinholeCamera()
        ph.set(camera_params[i]['id'],
               camera_params[i]['model'],
               camera_params[i]['K'],
               camera_params[i]['D'],
               camera_params[i]['image_size'],
               T_cam_to_body = camera_params[i]['TC2B'].T,
               skew =  camera_params[i]['skew'])

        params_file = './pinhole_camera_params_{}.yaml'.format(i)
        ph.save(params_file)
        ph2 = pinhole_camera.PinholeCamera(params_file)
        res[i] = ph.equal_to(ph2)

    return res


if __name__ == "__main__":

    try:
        test_pinhole_create()
        print('test_rigid3d_create PASSED!')
    except:
        print('test_rigid3d_create FAILED!')
        traceback.print_exc()

    try:
        test_pinhole_load_save()
        print('test_pinhole_load_save PASSED!')
    except:
        print('test_pinhole_load_save FAILED!')
        traceback.print_exc()