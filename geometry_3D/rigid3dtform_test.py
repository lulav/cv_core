import numpy as np
import scipy as sp
# from scipy.special import dtype

# from numpy.ma.core import reshape
import rigid3dtform as rt3d


def test_rigid3d_create():
    n = 100
    res1 = np.zeros((n,1), dtype=bool)
    res2 = np.zeros((n,1), dtype=bool)
    res3 = np.zeros((n,1), dtype=bool)
    for i in range(n):
        R, t = _random_transform(translation_mean=(0,0,0), translation_scale=100)
        T = np.vstack((np.hstack((R.as_matrix(),t)), [0, 0, 0, 1]))

        T1 = rt3d.Rigid3dTform(R, t)

        # test R, t, T
        res1[i] = np.all(np.abs(T1.R.as_matrix()[0] - R.as_matrix()[0]) < 1e-9)
        res2[i] = np.all((np.abs(T1.t - t)) < 1e-9)
        res3[i] = np.all((np.abs(T1.T - T)) < 1e-9)

    res = np.hstack([res1, res2, res3])
    return np.all(res)


def test_rigid3d_inv():
    n = 100
    res = np.zeros((n,1), dtype=bool)
    for i in range(n):
        R, t = _random_transform(translation_mean=(0,0,0), translation_scale=100)
        T1 = rt3d.Rigid3dTform(R, t)
        T2 = rt3d.Rigid3dTform(R, t)

        T = T1.invert() * T2
        res[i] = np.all(np.abs(T.R.as_matrix() - np.eye(3)) < 1e-9)

    assert np.all(res)


def test_rigid3d_mult():
    n = 100
    res1 = np.zeros((n,1), dtype=bool)
    res2 = np.zeros((n,1), dtype=bool)
    for i in range(n):
        R, t = _random_transform(translation_mean=(0,0,0), translation_scale=100)
        T1 = rt3d.Rigid3dTform(R, t)
        T2 = rt3d.Rigid3dTform(R, t)

        T = T1.invert() * T2
        res1[i] = np.all(np.abs(T.R.as_matrix() - np.eye(3)) < 1e-9)

        T = T2.invert() * T1
        res2[i] = np.all(np.abs(T.R.as_matrix() - np.eye(3)) < 1e-9)

    res = np.hstack([res1, res2])
    assert np.all(res)



def test_rigid3d_transform_points():
    n = 100
    m = 100
    res1 = np.zeros((n,1), dtype=bool)
    res2 = np.zeros((n,1), dtype=bool)
    for i in range(n):
        # random transform
        R, t = _random_transform(translation_mean=(0,0,0), translation_scale=100)
        T1 = rt3d.Rigid3dTform(R, t)

        # random points
        points = (np.random.rand(m, 3) - 0.5) * 100
        p1 = T1.transform_points(points)
        p1_ref = np.matmul(R.as_matrix(), points.transpose()) + t
        p1_ref = p1_ref.transpose()
        res1[i] = np.all(np.abs(p1-p1_ref) < 1e-9)

        p2 = T1.invert().transform_points(p1_ref)
        res2[i] = np.all(np.abs(p2-points) < 1e-9)

    res = np.hstack([res1, res2])
    assert np.all(res)


def _random_transform(translation_scale=1, translation_mean=(0,0,0)):
    """
    generate n random transforms
    Rotation is uniformally randomized
    translation is uniformally randomized in th range of:

    translation_mean - translation_scale.2 < x,y,z < translation_mean + translation_scale/2

    :param translation_scale: scalart
    :param translation_mean: [1x3] array or list
    :return:
    """
    # random rotation
    rotation_axis = np.random.rand(1, 3)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.random.rand(1, 1) * 2 * np.pi
    rotvec = rotation_axis * rotation_angle
    R = sp.spatial.transform.Rotation.from_rotvec(rotvec.flatten())

    # random translation
    t = (np.random.rand(3, 1) - 0.5) * translation_scale + np.reshape(np.array(translation_mean), (3, 1))

    return R, t

# def test_rigid3d_visual():
#     %------------ visual
#     test - ----------------------
#
#     %draw
#     world
#     frame
#     figure;
#     hold
#     on;
#     s = 2;
#     o = [0, 0, 0];
#     x = [1, 0, 0];
#     y = [0, 1, 0];
#     z = [0, 0, 1];
#     draw_frame(o, x, y, z, 'oW', '-');
#
#     %define
#     frame1
#     % a = pi / 7;
#     % n = [1, 0.5, 0];
#     a = pi / 4;
#     n = [0, 0, 1];
#     q21 = quaternion(cos(a / 2), sin(a / 2) * n(1), sin(a / 2) * n(2), sin(a / 2) * n(3));
#     rotm1 = quat2rotm(q21);
#     t1 = [2, 1, 1];
#
#     T1w = rigid3dtform(rotm1, t1); %transform
#     from frame
#
#     1
#     to
#     world
#     o1 = T1w.transformPointsForward([0, 0, 0]);
#     x1 = T1w.transformPointsForward([1, 0, 0]);
#     y1 = T1w.transformPointsForward([0, 1, 0]);
#     z1 = T1w.transformPointsForward([0, 0, 1]);
#     draw_frame(o1, x1, y1, z1, 'o1', '--');
#
#     %define
#     frame2
#     % a = pi / 6;
#     % n = [1, 1, 1];
#     a = pi / 2 + pi / 12;
#     n = [0, 0, 1];
#     q21 = quaternion(cos(a / 2), sin(a / 2) * n(1), sin(a / 2) * n(2), sin(a / 2) * n(3));
#     rotm2 = quat2rotm(q21);
#     t2 = [1, 2, 3];
#
#     T2w = rigid3dtform(rotm2, t2); %transform
#     from frame
#
#     2
#     to
#     world
#     o2 = T2w.transformPointsForward([0, 0, 0]);
#     x2 = T2w.transformPointsForward([1, 0, 0]);
#     y2 = T2w.transformPointsForward([0, 1, 0]);
#     z2 = T2w.transformPointsForward([0, 0, 1]);
#     draw_frame(o2, x2, y2, z2, 'o2', ':');
#
#     %define
#     frame3
#     a = pi / 6;
#     n = [1, 1, 1];
#     q31 = quaternion(cos(a / 2), sin(a / 2) * n(1), sin(a / 2) * n(2), sin(a / 2) * n(3));
#     rotm3 = quat2rotm(q31);
#     t3 = [2, 3, 3];
#
#     T3w = rigid3dtform(rotm3, t3); %transform
#     from frame
#
#     2
#     to
#     world
#     o3 = T3w.transformPointsForward([0, 0, 0]);
#     x3 = T3w.transformPointsForward([1, 0, 0]);
#     y3 = T3w.transformPointsForward([0, 1, 0]);
#     z3 = T3w.transformPointsForward([0, 0, 1]);
#     draw_frame(o3, x3, y3, z3, 'o3', ':');
#
#     axis
#     equal
#     grid
#     on;
#
#
#
#     %---------- test
#     points
#     convertions - -----------------
#
#     %frame
#     1
#     mid
#     axis
#     points in world and in frame
#     1
#     p1w = [(o1 + x1) / 2;
#     ... \
#         (o1 + y1) / 2;
#     ...
#     (o1 + z1) / 2];
#
#     p1 = [[0.5, 0, 0];
#     ...
#     [0, 0.5, 0];
#     ...
#     [0, 0, 0.5]];
#
#     plot3(p1w(:, 1), p1w(:, 2), p1w(:, 3), 'xr');
#
#     %convert
#     points
#     from world to
#
#     frame1
#     q1 = T1w.transformPointsInverse(p1w);
#     err1 = sum(q1(:)-p1(:));
#     fprintf('convertion from world to frame 1: error=%f\n', err1);
#
#     %convert
#     points
#     from frame
#
#     1
#     to
#     world
#     q1w = T1w.transformPointsForward(p1);
#     err1w = sum(q1w(:)-p1w(:));
#     fprintf('convertion from frame 1 to world: error=%f\n', err1w);
#
#     %make
#     a
#     curve in frame
#     1
#     t = linspace(0, 2 * pi, 50)
#     ';
#     p1 = [sin(t), cos(t) * 0.5, ones(size(t)) * 0.5];
#     %convert
#     to
#     world
#     p1w = T1w.transformPointsForward(p1);
#     plot3(p1w(:, 1), p1w(:, 2), p1w(:, 3), '.r');
#
#     %make
#     a
#     curve in frame
#     2
#     p2 = [sin(t), cos(t) * 0.6, ones(size(t)) * 0.5];
#     %convert
#     to
#     world
#     p2w = T2w.transformPointsForward(p2);
#     plot3(p2w(:, 1), p2w(:, 2), p2w(:, 3), '.g');
#
#     %make
#     a
#     curve in frame
#     3
#     p3 = [sin(t), cos(t) * 0.6, ones(size(t)) * 0.5];
#     %convert
#     to
#     world
#     p3w = T3w.transformPointsForward(p3);
#     plot3(p3w(:, 1), p3w(:, 2), p3w(:, 3), '.b');
#
#     %convert
#     from frame
#
#     2
#     to
#     frame
#     1
#     a = pi / 4 + pi / 12;
#     n = [0, 0, 1];
#     q21 = quaternion(cos(a / 2), sin(a / 2) * n(1), sin(a / 2) * n(2), sin(a / 2) * n(3));
#     R21_GT = quat2rotm(q21);
#     t21_GT = [sqrt(2), 0, 2];
#     T21_GT = rigid3dtform(R21_GT, t21_GT); %transform
#     from frame
#
#     2
#     to
#     frame
#     1
#
#     figure('name', 'frame 2 to 1');
#     draw_frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 'o1', '--');
#     hold
#     on;
#
#     % o21 = T21_GT.transformPointsForward([0, 0, 0]);
#     % x21 = T21_GT.transformPointsForward([1, 0, 0]);
#     % y21 = T21_GT.transformPointsForward([0, 1, 0]);
#     % z21 = T21_GT.transformPointsForward([0, 0, 1]);
#     %draw_frame(o21, x21, y21, z21, 'o2', ':');
#     %
#     % p2_to_1 = T21_GT.transformPointsForward(p2);
#     % plot3(p2(:, 1), p2(:, 2), p2(:, 3), '.r');
#     % plot3(p2_to_1(:, 1), p2_to_1(:, 2), p2_to_1(:, 3), '.g');
#     %axis
#     equal
#     %grid
#     on
#
#     % T21 = T1w.Tinv * T2w.T;
#     % R21 = T21(1:3, 1: 3);
#     % t21 = T21(1:3, 4);
#     % T21 = rigid3dtform(R21, t21); %transform
#     from frame
#
#     2
#     to
#     frame
#     1
#     T21 = T1w.invert * T2w;
#
#     o21 = T21.transformPointsForward([0, 0, 0]);
#     x21 = T21.transformPointsForward([1, 0, 0]);
#     y21 = T21.transformPointsForward([0, 1, 0]);
#     z21 = T21.transformPointsForward([0, 0, 1]);
#     draw_frame(o21, x21, y21, z21, 'o2', ':');
#
#     p2_to_1 = T21.transformPointsForward(p2);
#     plot3(p2(:, 1), p2(:, 2), p2(:, 3), '.r');
#     plot3(p2_to_1(:, 1), p2_to_1(:, 2), p2_to_1(:, 3), '.g');
#     axis
#     equal
#     grid
#     on
#
#     err21 = sum(T21.T(:)-T21_GT.T(:));
#     fprintf('convertion from frame 2 to frame 1: error=%f\n', err21);
#
#     %convert
#     fit
#     curves
#     on
#     frames
#     1, 2
#
#     disp('done');
#
#     return


# def draw_frame(o, x, y, z, frame_name, lineformat)
#     hold
#     on;
#     plot3([o(1), x(1)], [o(2), x(2)], [o(3), x(3)], ['r', lineformat]);
#     text(x(1), x(2), x(3), 'X');
#     plot3([o(1), y(1)], [o(2), y(2)], [o(3), y(3)], ['g', lineformat]);
#     text(y(1), y(2), y(3), 'Y');
#     plot3([o(1), z(1)], [o(2), z(2)], [o(3), z(3)], ['b', lineformat]);
#     text(z(1), z(2), z(3), 'Z');
#     plot3(o(1), o(2), o(3), 'ob');
#     text(o(1), o(2), o(3), frame_name, 'HorizontalAlignment', 'right');
#     end


if __name__ == "__main__":

    try:
        test_rigid3d_create()
        print('test_rigid3d_create PASSED!')
    except:
        print('test_rigid3d_create FAILED!')

    try:
        test_rigid3d_inv()
        print('test_rigid3d_inv PASSED!')
    except:
        print('test_rigid3d_inv FAILED!')

    try:
        test_rigid3d_mult()
        print('test_rigid3d_mult PASSED!')
    except:
        print('test_rigid3d_mult FAILED!')

    try:
        test_rigid3d_transform_points()
        print('test_rigid3d_transform_points PASSED!')
    except:
        print('test_rigid3d_transform_points FAILED!')


