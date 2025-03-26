import numpy as np
import pinhole_camera
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    select stereo baseline to achive specific range seperation from a specific range
    """

    camera_fov = (60, 40)  # (width, height) in degrees
    image_width = 1200
    stereo_baseline = 1
    pixel_min_disparity = 0.5

    camera_fov = np.array(camera_fov) * np.pi / 180
    image_height = np.round(float(image_width) * float(camera_fov[1]) / float(camera_fov[1]))
    image_size = (image_width, image_height)
    f = pinhole_camera.fov_to_focal_length(camera_fov, image_size)

    # ------------------------------ theoretic range separation -----------------------------------
    stereo_baseline = 2
    pixel_min_disparity = [0.5, 1, 2, 5]

    r = range(300)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('range')
    ax.set_ylabel('range error')
    ax.set_title('theoretic stereo range error with: b={}, f={:.1f}, dsiparity={}'.format(stereo_baseline, f[0], pixel_min_disparity))

    cmap = plt.cm.plasma

    colors = [cmap(int(np.round(255 * i / len(pixel_min_disparity)))) for i in range(len(pixel_min_disparity))]
    for i, dp in  enumerate(pixel_min_disparity):
        dr = np.power(r, 2) * dp / (stereo_baseline * f[0])
        ax.plot(r, dr, color=colors[i], label='disp = {} pix'.format(dp))

    ax.legend()
    ax.grid(True)
    plt.show(block=True)


    # ------------------------------ set cameras -----------------------------------

    ph_left = pinhole_camera()
    cam_id = 'left'
    cam_model = pinhole_camera.CameraModel.PINHOLE
    f = pinhole_camera.fov_to_focal_length(camera_fov, image_size)
    c = image_size / 2
    K = np.array([[f[0],   0, c[0]],
                  [0,   f[1], c[1]],
                  [0,      0,  1 ]])
    D = [0, 0, 0, 0]
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    ph_left.set(cam_id, cam_model, K, D, image_size, T_cam_to_body=T, skew=0)

    ph_right = pinhole_camera()
    cam_id = 'right'
    T = np.array([[1, 0, 0, stereo_baseline],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    ph_right.set(cam_id, cam_model, K, D, image_size, T_cam_to_body=T, skew=0)


    # ------------------------------ calc range separation  -----------------------------------

    # calc camera range seperation
    r = np.array(range(300))

    n = r.size
    world_points = np.zeros((n,3))
    world_points[2, :] = r[:]

    for i in range(n):

        image_point_left = ph_left
        image_point_right = ph_left







