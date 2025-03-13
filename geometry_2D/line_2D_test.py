import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import line_2D as l2d


def line_segments_intersect_test(draw=False, n=100):

    # ---------------------- manual test -----------------------------
    # ground truth intersection point
    intersection_points = (np.random.rand(n, 2) - 0.5) * 10
    # segment 1
    d1 = (np.random.rand(n, 2) - 0.5) * 3
    s1p1 = intersection_points + d1
    s1p2 = intersection_points - d1
    # segment 2
    d2 = (np.random.rand(n, 2) - 0.5) * 3
    s2p1 = intersection_points + d2
    s2p2 = intersection_points - d2

    # estimate intersection point
    intersection_points2, intersection_status2 = l2d.intersect_segments(s1p1, s1p2, s2p1, s2p2, epsilon=1e-8, use_shapely=False)

    # calc error
    err = intersection_points - intersection_points2

    if draw:
        fig, ax = plt.subplots(1)
        fig.suptitle('line segments intersection test', fontsize=15)
        for i in range(n):
            col = (np.random.random(), np.random.random(), np.random.random())
            ax.plot((s1p1[i, 0], s1p2[i, 0]), (s1p1[i, 1], s1p2[i, 1]), color=col)
            ax.plot((s2p1[i, 0], s2p2[i, 0]), (s2p1[i, 1], s2p2[i, 1]), color=col)
            ax.scatter((s1p1[i, 0], s1p2[i, 0], s2p1[i, 0], s2p2[i, 0], intersection_points[i, 0]),
                       (s1p1[i, 1], s1p2[i, 1], s2p1[i, 1], s2p2[i, 1], intersection_points[i, 1]), color=col)
        ax.set_xlabel(r'X', fontsize=12)
        ax.set_ylabel(r'Y', fontsize=12)
        # ax.set_title('image center ray intersection', fontsize=14)
        ax.grid(True)

        fig2, ax2 = plt.subplots(1)
        fig2.suptitle('line segments intersection errors', fontsize=10)
        ax2.scatter(err[:, 0], err[:, 1])
        ax2.set_xlabel(r'errors X', fontsize=12)
        ax2.set_ylabel(r'errors Y', fontsize=12)
        # ax.set_title('image center ray intersection', fontsize=14)
        ax2.grid(True)


        plt.show(block=True)

    res1 = ((not np.any(np.isnan(intersection_points2)))
           and (np.max(np.abs(err.flatten())) < 1e-9)) and np.all(intersection_status2 == 0)

    # ---------------------- compare to shapely -----------------------------


    # estimate intersection point
    intersection_points3, intersection_status3 = l2d.intersect_segments(s1p1, s1p2, s2p1, s2p2, epsilon=1e-8, use_shapely=True)


    # calc error
    res2 = (np.max(np.abs(intersection_points3 - intersection_points2)) < 1e-9
            and np.all(intersection_status3 == intersection_status2))



    return res1 and res2



if __name__ == "__main__":

    # res = line_segments_intersect_test(draw=True, n=10)
    res = line_segments_intersect_test(draw=False, n=100)
    if res:
        print('2D line segments intersect test PASSED!')
    else:
        print('2D line segments intersect test FAILED!')

    print('Done!')



