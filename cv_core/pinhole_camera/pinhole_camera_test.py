import os
import traceback
import cv2
import numpy as np
import scipy as sp

import pinhole_camera
import geometry_3D as g3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
        translations.append(g3d.Rigid3dTform(rotations[i], tvec))
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

def _random_pinhole_camera_params(n):
    """
    make a random pinhole camra
    :param n: number of instances
    :return:
    """

    fov_range = [100, 10]  # in degrees
    image_size_range = [200, 2000]  # pixels
    image_aspect_ratio_range = [0.5, 2]  # fx/fy

    T = _random_3d_transform(n)
    pinhole_camera_params = []

    img_size_x = np.round(image_size_range[0] + np.random.rand(n) * (image_size_range[1] - image_size_range[0]))
    aspect = image_aspect_ratio_range[0] + np.random.rand(n) * (image_aspect_ratio_range[1] - image_aspect_ratio_range[0])
    img_size_y = np.round(img_size_x * aspect)

    img_size_x = np.round(img_size_x / 2) * 2
    img_size_y = np.round(img_size_y / 2) * 2


    fovx = fov_range[0] + np.random.rand(n) * (fov_range[1] - fov_range[0])
    fovx = fovx * np.pi / 180
    fovy = fovx * aspect + (np.random.rand(n) - 0.5) * 2 * 0.01 * fovx  # add small pertubation so that fx != fy

    fx = np.divide(img_size_x / 2, np.arctan(fovx / 2))
    fy = np.divide(img_size_y / 2, np.arctan(fovy / 2))

    cx = img_size_x / 2 + (np.random.rand(n) - 0.5) * (2 * img_size_x / 20)
    cy = img_size_y / 2 + (np.random.rand(n) - 0.5) * (2 * img_size_y / 20)

    img_size_x = np.array(img_size_x, dtype=np.uint32)
    img_size_y = np.array(img_size_y, dtype=np.uint32)

    for i in range(n):
        id = 'camera_{}'.format(i)
        model = np.random.choice(pinhole_camera.CameraModel)
        # model = pinhole_camera.CameraModel.PINHOLE

        K = np.array([[fx[i], 0, cx[i]],
                      [0, fy[i], cy[i]],
                      [0, 0, 1]])
        # print('image size: {}, {}   C: {}, {}'.format(img_size_x[i], img_size_y[i], cx[i], cy[i]))
        D = (np.random.rand(4)-0.5) * 0.01
        # D[:] = 0
        skew = 0
        pcp = {'id': id,
        'model': model,
        'K': K,
        'D': D,
        'image_size': [img_size_x[i], img_size_y[i]],
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
                 np.all(np.abs(camera_params[i]['K'] - ph.K) < 1e-9) and \
                 np.all(np.abs(camera_params[i]['D'] - ph.distortion_coefficients) < 1e-9) and \
                 np.all(np.abs(camera_params[i]['image_size'] - ph.image_size) <= 1e-9) and \
                 np.all(np.abs(camera_params[i]['TC2B'].T - ph.T_cam_to_body) <= 1e-9) and \
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
                  np.all(np.abs(camera_params[i]['image_size'] - ph.image_size) <= 1e-9) and \
                  ph.T_cam_to_body is None and \
                  ph.skew == 0

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
        os.remove(params_file)

    return np.all(res)



def test_pinhole_plot():
    """
    test pinhole plot
    :return:
    """

    # camera looks up
    # -----------------
    camera_params = _random_pinhole_camera_params(1)

    camera_pose = g3d.Rigid3dTform(np.eye(3), np.zeros((3,1)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('pinhole camera')
    ax.axis('equal')

    # sel all params
    ph = pinhole_camera.PinholeCamera()
    ph.set(camera_params[0]['id'],
           camera_params[0]['model'],
           camera_params[0]['K'],
           camera_params[0]['D'],
           camera_params[0]['image_size'],
           T_cam_to_body = camera_params[0]['TC2B'].T,
           skew =  camera_params[0]['skew'])
    plt1 = ph.plot(camera_pose, fig, ax, color=(0.5, 0.5, 1), scale=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('pinhole camera looks up')
    ax.set_aspect('equal')
    ax.grid(True)
    plt.show(block=True)
    # plt.draw()
    plt.pause(0.2)


    # camera looks down
    # -----------------
    R, _ = cv2.Rodrigues((-np.pi, 0, 0))
    camera_pose = g3d.Rigid3dTform(R, (0, 0, 10))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('pinhole camera looks down')

    # sel all params
    ph = pinhole_camera.PinholeCamera()
    ph.set(camera_params[0]['id'],
           camera_params[0]['model'],
           camera_params[0]['K'],
           camera_params[0]['D'],
           camera_params[0]['image_size'],
           T_cam_to_body=camera_params[0]['TC2B'].T,
           skew=camera_params[0]['skew'])
    ph.plot(camera_pose, fig2, ax2, color=(0.5, 0.5, 1), scale=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('pinhole camera')
    ax2.set_aspect('equal')
    ax2.grid(True)
    plt.show(block=True)


    # camera looks 45 of nadir
    # -------------------------
    R, _ = cv2.Rodrigues((-3*np.pi/4, 0, 0))
    camera_pose = g3d.Rigid3dTform(R, (0, 0, 10))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('pinhole camera')

    # sel all params
    ph = pinhole_camera.PinholeCamera()
    ph.set(camera_params[0]['id'],
           camera_params[0]['model'],
           camera_params[0]['K'],
           camera_params[0]['D'],
           camera_params[0]['image_size'],
           T_cam_to_body=camera_params[0]['TC2B'].T,
           skew=camera_params[0]['skew'])
    ph.plot(camera_pose, fig, ax, color=(0.5, 0.5, 1), scale=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('pinhole camera')
    ax.set_aspect('equal')
    ax.grid(True)
    plt.show(block=True)

    aa=5


    return True


def test_project_points_manual():
    """
    test id pinhole camera creates correctly
    :return:
    """

    # set camera intrinsics
    fx = 500
    fy = 500
    image_size_x = 1200
    image_size_y = 800
    fovx = np.arctan2(image_size_x / 2, fx) * 180 / np.pi
    fovy = np.arctan2(image_size_y / 2, fy) * 180 / np.pi
    print('camera fov: {:.2f}X{:.2f}[deg]'.format(fovx, fovy))
    cx = float(image_size_x) / 2
    cy = float(image_size_y) / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    D = np.array([0.01, 0.002, 0.003, 0.0004], dtype=np.float32)
    # D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ph = pinhole_camera.PinholeCamera()
    ph.set('cam0', pinhole_camera.CameraModel.PINHOLE, K, D, (image_size_x, image_size_y))

    # set camera pose
    R = np.eye(3, dtype=np.float32)
    t = np.array([0, 0, 0], dtype=np.float32)
    # rvec, _ = cv2.Rodrigues(R)
    # tvec = t.flatten()
    cam_pose = g3d.Rigid3dTform(R, t)

    # set ground plane
    plane = g3d.Plane3D([0, 0, -1], [0, 0, 10])


    #--------------- manual test1 -------------------
    # project world points to pixels and then reproject

    x = np.linspace(-5,5,5, dtype=np.float32)
    y = np.linspace(-5, 5, 5, dtype=np.float32)
    x, y = np.meshgrid(x,y)
    world_points = np.vstack((x.flatten(), y.flatten(), np.zeros(25)+10)).transpose()

    # image_points, _ = cv2.projectPoints(world_points, rvec, tvec, K, D)
    # image_points = image_points.squeeze()
    image_points, _ = ph.project_points(world_points, cam_pose)

    # los_cam_frame = cv2.undistortPoints(image_points, K, D)
    # los_cam_frame = np.hstack((los_cam_frame.squeeze(), np.ones((25, 1)) ))
    los_cam_frame, _ = ph.pixel_to_los(image_points, R)

    # projected_world_points = los_cam_frame
    # for i in range(projected_world_points.shape[0]):
    #     projected_world_points[i,:] = projected_world_points[i, :] * 10 / projected_world_points[i, 2]
    projected_world_points, _ = plane.ray_intersection(np.repeat(cam_pose.t.transpose(), 25, axis=0), los_cam_frame)

    err = projected_world_points - world_points
    err2d = np.sqrt(np.sum(np.power(err, 2), axis=1))
    res1 = np.max(err2d) < 1e-3


    #--------------- manual test2 -------------------
    # project pixels to world points and then reproject back to pixels

    x = np.linspace(0, image_size_x,5, dtype=np.float32)
    y = np.linspace(0, image_size_y,5, dtype=np.float32)
    x, y = np.meshgrid(x,y)
    image_points = np.vstack((x.flatten(), y.flatten())).transpose()

    # los_cam_frame = cv2.undistortPoints(image_points, K, D)
    # los_cam_frame = np.hstack((los_cam_frame.squeeze(), np.ones((25, 1)) ))
    los_cam_frame, _ = ph.pixel_to_los(image_points, R)


    # world_points = los_cam_frame
    # for i in range(world_points.shape[0]):
    #     world_points[i,:] = world_points[i, :] * 10 / world_points[i, 2]
    world_points, _ = plane.ray_intersection(np.repeat(cam_pose.t.transpose(), 25, axis=0), los_cam_frame)

    # image_points_reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, K, D)
    # image_points_reprojected = image_points_reprojected.squeeze()
    image_points_reprojected, _ = ph.project_points(world_points, cam_pose)

    err = image_points_reprojected - image_points
    err2d = np.sqrt(np.sum(np.power(err, 2), axis=1))
    res2 = np.max(err2d) < 1e-3


    #--------------- manual test3 -------------------
    # project world points to pixels and then reproject
    # non-trivial camera pose - camera looks 45 of nadir

    R, _ = cv2.Rodrigues((-3*np.pi/4, 0, 0))
    t = np.array((0, 0, 10), dtype=np.float32)
    cam_pose = g3d.Rigid3dTform(R, (0, 0, 10))
    plane = g3d.Plane3D([0, 0, 1], [0, 0, 0])

    # rvec, _ = cv2.Rodrigues(R.transpose())
    # tvec = t.flatten()

    x = np.linspace(-20,20,7, dtype=np.float32)
    y = np.linspace(0, 100, 20, dtype=np.float32)
    x, y = np.meshgrid(x,y)
    n  = x.size
    world_points = np.vstack((x.flatten(), y.flatten(), np.zeros_like(x.flatten()) )).transpose()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt1 = ph.plot(cam_pose, fig, ax, color=(0.5, 0.5, 1), scale=15)
    # ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], color=(0.5, 0.5, 1))
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('pinhole camera looks up')
    # ax.set_aspect('equal')
    # ax.grid(True)
    # # plt.show(block=True)

    # image_points2, _ = cv2.projectPoints(world_points, rvec, tvec, K, D)
    # image_points2 = image_points.squeeze()
    image_points, is_in_image = ph.project_points(world_points, cam_pose)
    image_points = image_points[is_in_image, :]

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.scatter(image_points[:, 0], image_points[:, 1], color=(0.5, 0.5, 1))
    # ax2.plot([0, 0, ph.image_size[0], ph.image_size[0], 0], [0, ph.image_size[1], ph.image_size[1], 0, 0], color=(0.5, 0.5, 1))
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.invert_yaxis()
    # ax2.set_title('image points')
    # ax2.set_aspect('equal')
    # ax2.grid(True)
    # # plt.show(block=True)

    # los_cam_frame = cv2.undistortPoints(image_points, K, D)
    # los_cam_frame = np.hstack((los_cam_frame.squeeze(), np.ones((n, 1)) ))
    # los_cam_frame = np.matmul(R, los_cam_frame.transpose()).transpose()
    los_cam_frame, _ = ph.pixel_to_los(image_points, R)

    # for l in los_cam_frame:
    #     ray_scale = 30
    #     ray_end_point = [cam_pose.t[0] + l[0] * ray_scale,
    #                      cam_pose.t[1] + l[1] * ray_scale,
    #                      cam_pose.t[2] + l[2] * ray_scale]
    #     ax.plot([cam_pose.t[0], ray_end_point[0]],
    #             [cam_pose.t[1], ray_end_point[1]],
    #             [cam_pose.t[2], ray_end_point[2]], color=(1, 0.5, 0.5))
    # plt.show(block=True)

    # projected_world_points = los_cam_frame
    # for i in range(projected_world_points.shape[0]):
    #     projected_world_points[i,:] = projected_world_points[i, :] * 10 / projected_world_points[i, 2]
    n_in_image = np.sum(is_in_image)
    projected_world_points, _ = plane.ray_intersection(np.repeat(cam_pose.t.transpose(), n_in_image, axis=0), los_cam_frame)

    err = projected_world_points - world_points[is_in_image, :]
    err2d = np.sqrt(np.sum(np.power(err, 2), axis=1))
    res3 = np.max(err2d) < 1e-3


    #--------------- manual test4 -------------------
    # project pixels to world points and then reproject back to pixels
    # non-trivial camera pose - camera looks 45 of nadir

    x = np.linspace(0, image_size_x,7, dtype=np.float32)
    y = np.linspace(0, image_size_y,7, dtype=np.float32)
    x, y = np.meshgrid(x,y)
    n = x.size
    image_points = np.vstack((x.flatten(), y.flatten())).transpose()

    # los_cam_frame = cv2.undistortPoints(image_points, K, D)
    # los_cam_frame = np.hstack((los_cam_frame.squeeze(), np.ones((25, 1)) ))
    los_cam_frame, _ = ph.pixel_to_los(image_points, R)

    # world_points = los_cam_frame
    # for i in range(world_points.shape[0]):
    #     world_points[i,:] = world_points[i, :] * 10 / world_points[i, 2]
    world_points, _ = plane.ray_intersection(np.repeat(cam_pose.t.transpose(), n, axis=0), los_cam_frame)

    # image_points_reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, K, D)
    # image_points_reprojected = image_points_reprojected.squeeze()
    image_points_reprojected, _ = ph.project_points(world_points, cam_pose)

    err = image_points_reprojected - image_points
    err2d = np.sqrt(np.sum(np.power(err, 2), axis=1))
    res4 = np.max(err2d) < 1e-3


    return np.all(np.array([res1, res2, res3, res4]))


def test_project_points_random():
    """
    test id pinhole camera creates correctly
    :return:
    """
    draw = True
    n = 10

    # randomize camera intrinsic params
    camera_params = _random_pinhole_camera_params(n)

    # randomize camera pose
    camera_pose = []
    rot_axis = np.array([1, 0, 0])
    rot_ang = np.pi
    rot_vec = rot_axis * rot_ang
    R, _ = cv2.Rodrigues(rot_vec)
    t = [0, 0, 10]
    camera_pose.append(g3d.Rigid3dTform(R, t))

    m = 20

    # rotate down
    rot_vec1 = np.array([np.pi, 0, 0], dtype=np.float32)
    R1 = sp.spatial.transform.Rotation.from_rotvec(rot_vec1)

    # add general rotation of upto 30 degrees
    rot_axis2 = np.random.rand(m,3)
    rot_axis2 = np.divide(rot_axis2, np.reshape(np.linalg.norm(rot_axis2, axis=1), (m,1)))
    rot_ang2 = (np.random.rand(m,1) - 0.5) * 2 * 45*np.pi/180
    rot_vec2 = np.multiply(rot_axis2, rot_ang2)

    t = [0, 0, 10]
    for i in range(m):
        R2 = sp.spatial.transform.Rotation.from_rotvec(rot_vec2[i, :])
        R = R2 * R1
        camera_pose.append(g3d.Rigid3dTform(R.as_matrix(), t))

    # xy plane
    plane_origin = [0, 0, 0]
    plane_normal = [0, 0, 1]
    plane = g3d.Plane3D(plane_normal, plane_origin)

    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

    res = np.zeros((n, len(camera_pose)), dtype=bool)
    for k, cam_pose in enumerate(camera_pose):

        for i in range(n):
            # set pinhole camera
            ph = pinhole_camera.PinholeCamera()
            ph.set(camera_params[i]['id'],
                   camera_params[i]['model'],
                   camera_params[i]['K'],
                   camera_params[i]['D'],
                   camera_params[i]['image_size'],
                   T_cam_to_body=camera_params[i]['TC2B'].T,
                   skew=camera_params[i]['skew'])
            if draw:
                ax.cla()
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('image points')
                ph.plot(cam_pose, fig, ax, color=(0.5, 0.5, 1), scale=2)

            # select pixels
            px, py = np.meshgrid(np.linspace(0, ph.image_size[0], 7), np.linspace(0, ph.image_size[1], 7))
            image_points = np.vstack((px.flatten(), py.flatten())).transpose()

            # project pixels to line of sight
            los, is_in_image = ph.pixel_to_los(image_points, cam_pose.R)
            los = los * 10
            if draw:
                for l in los:
                    ax.plot([cam_pose.t[0, 0], cam_pose.t[0, 0] + l[0]], [cam_pose.t[1, 0], cam_pose.t[1, 0] + l[1]],
                            [cam_pose.t[2, 0], cam_pose.t[2, 0] + l[2]], color=(1, 0.5, 0.5))
                    ax.set_aspect('equal')
                # plt.show(block=True)

            # intersect los with xy plane
            cam_position = cam_pose.t

            ray_origin = np.repeat(cam_position.transpose(), los.shape[0], 0)
            intersection_points, valid_index = plane.ray_intersection(ray_origin, los, epsilon=1e-9)

            if draw:
                ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2])
                # plt.show(block=True)

            # project world points back to pixels
            projected_image_points, is_in_image = ph.project_points(intersection_points, cam_pose)
            if draw:
                ax2.cla()
                ax2.scatter(image_points[:, 0], image_points[:, 1], c='b', s=50, marker='o')
                ax2.scatter(projected_image_points[:, 0], projected_image_points[:, 1], c='r', s=80, marker='+')
                ax2.plot([0, 0, ph.image_size[0], ph.image_size[0], 0], [0, ph.image_size[1], ph.image_size[1], 0, 0], color=(0.5, 0.5, 1))
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.invert_yaxis()
                ax2.set_title('image points')
                ax2.set_aspect('equal')
                ax2.grid(True)
                # plt.show(block=True)
                plt.draw()
                plt.pause(0.2)


            # calc reproject- project error
            d = projected_image_points - image_points
            error = np.sqrt(np.sum(np.power(d, 2), axis=1))

            res[i, k] = np.max(error) < 1e-3
            if not res[i, k]:
                print('failed: {}: image_size=({},{} max error={})'.format(ph.model, ph.image_size[0], ph.image_size[1],
                                                                           np.max(error)))
                aa=5

            # plt_image_points.set_offsets(np.c_[image_points[:,0], image_points[:,1]])
            # plt_reprojected_points.set_offsets(np.c_[projected_image_points[:,0], projected_image_points[:,1]])
            # ymn = min( min(image_points[:,1]), min(projected_image_points[:,1]))
            # ymx = max( max(image_points[:,1]), max(projected_image_points[:,1]))
            # xmn = min( min(image_points[:,0]), min(projected_image_points[:,0]))
            # xmx = max( max(image_points[:,0]), max(projected_image_points[:,0]))
            # ax2.set(ylim=(ymn, ymx), xlim=(xmn, xmx))

            # ax.scatter(image_points[:,0], image_points[:,1], c='b', s=20)
            # ax.scatter(projected_image_points[:, 0], projected_image_points[:, 1], c='r', s=10)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_title('image points max error {}'.format(np.max(error)))
            # plt.show(block=False)
            # plt.draw()
            # plt.pause(0.2)
            aa = 5

    return np.all(res)


if __name__ == "__main__":

    try:
        res = test_pinhole_create()
        if res:
            print('test_rigid3d_create PASSED!')
        else:
            print('test_rigid3d_create FAILED!')
    except:
        print('test_rigid3d_create FAILED!')
        traceback.print_exc()


    try:
        res = test_pinhole_load_save()
        if res:
            print('test_pinhole_load_save PASSED!')
        else:
            print('test_pinhole_load_save FAILED!')
    except:
        print('test_pinhole_load_save FAILED!')
        traceback.print_exc()

    # try:
    #     test_pinhole_plot()
    #     print('test_pinhole_plot PASSED!')
    # except:
    #     print('test_pinhole_plot FAILED!')
    #     traceback.print_exc()

    try:
        res = test_project_points_manual()
        if res:
            print('test_project_points_manual PASSED!')
        else:
            print('test_project_points_manual FAILED!')
    except:
        print('test_project_points_manual FAILED!')
        traceback.print_exc()


    # TODO: not all testds pass.
    #       opencv project and undistort seems to not be consistent with high distortions
    #       a small portion of fisheye tests fail - see if that is the reason
    #       it seems fails happen when the angle and FOV is so that some pixel rays intersect plane very far away
    try:
        res = test_project_points_random()
        if res:
            print('test_project_points_random PASSED!')
        else:
            print('test_project_points_random FAILED!')
    except:
        print('test_project_points_random FAILED!')
        traceback.print_exc()
