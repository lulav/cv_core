import sys
from logging import warning
import traceback
import numpy as np
import geometry_3D.plane_3d as p3d
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import axes3d

def test_3d_plane_create(n, epsilon = 1e-9):

    plane_normal = (np.random.rand(n, 3) - 0.5) * 10
    plane_origin = (np.random.rand(n, 3) - 0.5) * 10
    res = np.zeros((n,1), dtype=bool)
    for i in range(n):
        plane = p3d.Plane3D(plane_normal[i,:], plane_origin[i,:])
        res[i] = (np.all( max(np.abs(plane.normal - plane_normal[i,:]/np.linalg.norm(plane_normal[i,:]))) < epsilon) and
                  np.all(plane.origin == plane_origin[i,:]))

    return np.all(res)


def test_3d_plane_project_points(n=100, m=20, draw=False):
    """
    test plane point projection on Z axis
    :param n: number of random planes tested
    :param m: number of random points per plane
    :return:
    """

    plane_normal = (np.random.rand(n, 3) - 0.5) * 10
    plane_origin = (np.random.rand(n, 3) - 0.5) * 10

    if draw:
        f1 =  plt.figure()
        ax1 = f1.add_subplot(projection='3d')
        ax1.set_title('plane point projection - x axis')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        f2 =  plt.figure()
        ax2 = f2.add_subplot(projection='3d')
        ax2.set_title('plane point projection - y axis')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        f3 =  plt.figure()
        ax3 = f3.add_subplot(projection='3d')
        ax3.set_title('plane point projection - z axis')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        f4 =  plt.figure()
        ax4 = f4.add_subplot(projection='3d')
        ax4.set_title('plane point projection - normal direction')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')

    res1 = np.zeros((n,1), dtype=bool)
    res2 = np.zeros((n,1), dtype=bool)
    res3 = np.zeros((n,1), dtype=bool)
    res4 = np.zeros((n,1), dtype=bool)
    p11 = None; p12 = None; p13 = None
    p21 = None; p22 = None; p23 = None
    p31 = None; p32 = None; p33 = None
    p41 = None; p42 = None; p43 = None
    for i in range(n):
        plane = p3d.Plane3D(plane_normal[i,:], plane_origin[i,:])

        nx = plane_normal[i, 0]
        ny = plane_normal[i, 1]
        nz = plane_normal[i, 2]
        ox = plane_origin[i, 0]
        oy = plane_origin[i, 1]
        oz = plane_origin[i, 2]

        x = (np.random.rand(m, 1) - 0.5) * 10 + ox
        y = (np.random.rand(m, 1) - 0.5) * 10 + oy
        z = oz - ( nx * (x-ox) + ny * (y-oy) )/nz
        ref_plane_points =  np.hstack((x, y, z))

        # test X axis projection
        xd = x + (np.random.rand(m, 1) - 0.5) * 10
        points = np.hstack((xd, y, z))
        projected_points, is_above_plane = plane.project_point(points, axis=p3d.ProjectionAxis.x)
        res1[i] = np.all(np.linalg.norm(projected_points - ref_plane_points, axis=1) <= 1e-8)
        if draw:
            if p11 is not None:
                p11.remove()
                p12.remove()
                p13.remove()
            p11, _, _ = plane.plot(points, lims_scale_factor=1.5, ax=ax1, color=(0.5, 0.5, 1), alpha=0.6)
            p12 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color=(0.1, 0.7, 0.1))
            p13 = ax1.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], color=(0.2, 0.2, 1))
            ax1.set_aspect('equal')
            ax1.legend(['plane', 'points', 'projected points'])

        # test Y axis projection
        yd = y + (np.random.rand(m, 1) - 0.5) * 10
        points = np.hstack((x, yd, z))
        projected_points, is_above_plane = plane.project_point(points, axis=p3d.ProjectionAxis.y)
        res2[i] = np.all(np.linalg.norm(projected_points - ref_plane_points, axis=1) <= 1e-8)
        if draw:
            if p21 is not None:
                p21.remove()
                p22.remove()
                p23.remove()
            p21, _, _ = plane.plot(points, lims_scale_factor=1.5, ax=ax2, color=(0.5, 0.5, 1), alpha=0.6)
            p22 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], color=(0.1, 0.7, 0.1))
            p23 = ax2.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], color=(0.2, 0.2, 1))
            ax2.set_aspect('equal')
            ax2.legend(['plane', 'points', 'projected points'])

        # test Z axis projection
        zd = z + (np.random.rand(m, 1) - 0.5) * 10
        points = np.hstack((x, y, zd))
        projected_points, is_above_plane = plane.project_point(points, axis=p3d.ProjectionAxis.z)
        res3[i] = np.all(np.linalg.norm(projected_points - ref_plane_points, axis=1) <= 1e-8)
        if draw:
            if p31 is not None:
                p31.remove()
                p32.remove()
                p33.remove()
            p31, _, _ = plane.plot(points, lims_scale_factor=1.5, ax=ax3, color=(0.5, 0.5, 1), alpha=0.6)
            p32 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2], color=(0.1, 0.7, 0.1))
            p33 = ax3.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], color=(0.2, 0.2, 1))
            ax3.set_aspect('equal')
            ax3.legend(['plane', 'points', 'projected points'])

        # test plane normal axis projection
        d = (np.random.rand(m, 1) - 0.5) * 10
        points = ref_plane_points + plane.normal * d
        projected_points, is_above_plane = plane.project_point(points, axis=p3d.ProjectionAxis.plane_normal)
        res4[i] = np.all(np.linalg.norm(projected_points - ref_plane_points, axis=1) <= 1e-8)
        if draw:
            if p41 is not None:
                p41.remove()
                p42.remove()
                p43.remove()
            p41, _, _ = plane.plot(points, lims_scale_factor=1.5, ax=ax4, color=(0.5, 0.5, 1), alpha=0.6)
            p42 = ax4.scatter(points[:, 0], points[:, 1], points[:, 2], color=(0.1, 0.7, 0.1))
            p43 = ax4.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], color=(0.2, 0.2, 1))
            ax4.set_aspect('equal')
            ax4.legend(['plane', 'points', 'projected points'])
            plt.ion()
            plt.draw()
            plt.pause(0.2)
            # plt.show()
            a=5

    return np.all(res1) and np.all(res2) and np.all(res3) and np.all(res4)



if __name__ == "__main__":

    try:
        res = test_3d_plane_create(100)
        if res:
            print('test_3d_plane_create PASSED!')
        else:
            print('test_3d_plane_create FAILED!')
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    try:
        res = test_3d_plane_project_points(100, 20, draw=False)
        if res:
            print('test_3d_plane_project_points PASSED!')
        else:
            print('test_3d_plane_project_points FAILED!')
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
