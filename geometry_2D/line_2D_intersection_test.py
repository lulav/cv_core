import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import line_2D_intersection as l2di


if __name__ == "__main__":

    draw = True

    image_width = 640
    image_height = 480

    if draw:
        fig, ax = plt.subplots(1)
        fig.suptitle('image center ray intersection', fontsize=15)
        ax.plot((0, 0, image_width, image_width, 0), (0, image_height, image_height, 0, 0), color=(0, 0, 0))
        ax.set_ylim(bottom=-100, top=800)
        ax.set_ylim(bottom=-100, top=600)
        ax.set_xlabel(r'X', fontsize=12)
        ax.set_ylabel(r'Y', fontsize=12)
        # ax.set_title('image center ray intersection', fontsize=14)
        ax.grid(True)

    # query points
    points = np.array([[-50, -50],
                        [10, -50],
                        [200, -50],
                        [700, -50],
                        [-40, 550],
                        [70, 550],
                        [250, 550],
                        [750, 550],
                        [-40, 600],
                        [-40, 300],
                        [-40, 100],
                        [-40, -50],
                        [700, 600],
                        [700, 300],
                        [700, 100],
                        [700, -50]])

    intersection_points_ref = np.array([
           [13.79310345, 0],
           [63.44827586, 0],
           [220.68965517, 0],
           [634.48275862, 0],
           [41.29032258, 480],
           [126.4516129, 480],
           [265.80645161, 480],
           [640, 470.69767442],
           [80, 480.],
           [0, 293.33333333],
           [0, 115.55555556],
           [22.06896552, 0],
           [573.33333333, 480],
           [640, 290.52631579],
           [640, 122.10526316],
           [634.48275862, 0]])

    # center point
    center_point = np.array([image_width / 2, image_height / 2])

    if draw:  # draw points
        ax.plot(points[:,0], points[:,1], 'o', color=(0, 0, 1))

    # plot lines between center point to query points
    p1 = (0, 0)
    p2 = (0, image_height)
    p3 = (image_width, image_height)
    p4 = (image_width, 0)
    poly_points = np.array([p1, p2, p3, p4])
    intersection_points = []
    for i in range(points.shape[0]):
        ip, n = l2di.intersect_polygon(poly_points, center_point, points[i, :], epsilon=1e-8)
        # print('point {}: found {} intersection points'.format(i, ip.shape[0]))
        intersection_points.append(ip[0])

        if draw:
            # draw lines from center point to outside points
            ax.plot([center_point[0], points[i, 0]], [center_point[1], points[i, 1]], '-', color=(0, 0, 1))
            # draw intersection points
            for i in range(n):
                ax.plot(ip[i, 0], ip[i, 1], 'o', color=(0.5, 0.5, 1))
            # plot center point
            ax.plot(center_point[0], center_point[1], 'o', color=(1, 0, 0))

    res = np.all(np.abs(intersection_points_ref - intersection_points) < 1e-7)

    if res:
        print('line_2D_intersection test PASSED!')
    else:
        print('line_2D_intersection test FAILED!')

    if draw:
        plt.show(block=True)

    print('Done!')



