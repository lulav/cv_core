import numpy as np
import scipy as sp
from fontTools.misc.bezierTools import epsilon
from numpy.ma.core import reshape
from enum import Enum
import matplotlib
import matplotlib.pyplot as plt


class ProjectionAxis(Enum):
    plane_normal = 0
    x = 1
    y = 2
    z = 3

class Plane3D:
    """
    Rigid 3D transform
    """
    def __init__(self, normal, origin, epsilon=1e-9):
        """
        3D plane
        :param origin: point on the plane [3x1]
        :param normal: plane normal [3x1]
        :param epsilon: value close to 0

        3D plane equation is:
        nx*(x-x0) + ny*(y-y0) + nz*(z-z0) = 0
        where
            (nx,ny,nz) is plane normal
            (x0,y0,z0) is plane origin
        This equation satisfied:
        1) origin point is on the plane
        2) any other point on the plane P satisfied:  (P-O) DOT N = 0

        This is equivalent to common 3d plane equation:
        ax+by+cz+d=0
        where
            a = nx
            b = ny
            c = nz
            d = -nx*x0 - ny*y0 - nz*z0
        """

        origin = np.array(origin)
        normal = np.array(normal)

        if origin.size == 3:
            self.origin = np.reshape(origin, (1,3))
        else:
            raise Exception('invalid plane origin!')
        if normal.size == 3:
            self.normal = np.reshape(normal, (1,3))
        else:
            raise Exception('invalid plane normal!')

        # normalize normal
        d = np.linalg.norm(self.normal)
        if d > epsilon:
            self.normal = self.normal / np.linalg.norm(self.normal)

        return

    def project_point(self, points, axis=ProjectionAxis.plane_normal, epsilon=1e-9):
        """
        project point to plane

        :param points: 3D points [nX3]
        :param axis: axis on which to project point to plane
                     must be ProjectionAxis
                     this might be one of the coordinate system axis (x,y,z)
                     or plane normal
        :param epsilon: if a point is closer to plane than epsilon, it is considered on the plane
        :return projected_points
        """

        # check inputs
        points = np.array(points)
        if points.shape[1] != 3:
            raise Exception('invalid point size!')
        if not(isinstance(axis, ProjectionAxis)):
            raise Exception('invalid axis!')
        n = points.shape[0]

        nx = self.normal[0, 0]
        ny = self.normal[0, 1]
        nz = self.normal[0, 2]

        ox = self.origin[0, 0]
        oy = self.origin[0, 1]
        oz = self.origin[0, 2]

        # check if point is above plane
        OP = points - self.origin
        is_above_plane = np.dot(OP, self.normal.flatten()) > 0

        # project point to plane
        if axis == ProjectionAxis.plane_normal:
            OP = points - self.origin
            plane_normal_projection = self.normal * np.reshape(np.dot(OP, self.normal.flatten()), (n,1))
            projected_points = self.origin + (OP - plane_normal_projection)

        elif axis == ProjectionAxis.x:
            if abs(self.normal[0, 0]) < epsilon:
                # plane is parallel to y axis
                # no x or all x coordinates are relevant!
                projected_points = np.zeros((0, 3))
            else:
                d = -np.dot(self.normal.flatten(), self.origin.flatten())
                projected_points = points
                projected_points[:, 0] = -(self.normal[0, 1] * points[:, 1] + self.normal[0, 2] * points[:, 2] + d) / self.normal[0, 0]

        elif axis == ProjectionAxis.y:
            if abs(self.normal[0, 1]) < epsilon:
                # plane is parallel to y axis
                # no y or all y coordinates are relevant!
                projected_points = np.zeros((0, 3))
            else:
                d = -np.dot(self.normal.flatten(), self.origin.flatten())
                projected_points = points
                projected_points[:, 1] = -(nx * points[:, 0] + self.normal[0, 2] * points[:, 2] + d) / self.normal[0, 1]

        elif axis == ProjectionAxis.z:
            if abs(self.normal[0, 2]) < epsilon:
                # plane is parallel to z axis
                # no Z or all Z coordinates are relevant!
                projected_points = np.zeros((0, 3))
            else:
                d = -np.dot(self.normal.flatten(), self.origin.flatten())
                projected_points = points
                projected_points[:, 2] = oz - (nx * (points[:, 0] - ox) + ny * (points[:, 1] - oy)) / nz
        else:
            raise Exception('axis {} not supported!'.format(axis))

        return projected_points, is_above_plane


    def plot(self, points, lims_scale_factor=1, ax=None, color=(0.5, 0.5, 1), alpha=0.6):
        """
        plot plane. plane limita will be taken from the extreme x,y in points

        :param points: 3D points [nX3]
        :param lims_scale_factor: scale factor for inflating the plot area
        :param ax: matplotlib ax for plotting
        :return ax
        """

        points = np.array(points)
        if points.shape[1] != 3:
            raise Exception('invalid points imput! must be [nx3]')

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Plane')
        else:
            fig = None

        # get x,y extreme points
        xmn = min(points[:, 0])
        xmx = max(points[:, 0])
        ymn = min(points[:, 1])
        ymx = max(points[:, 1])

        # inflate by scale factor
        dx = xmx - xmn
        dy = ymx - ymn
        xmn = xmn - dx * (lims_scale_factor-1) / 2
        xmx = xmx + dx * (lims_scale_factor-1) / 2
        ymn = ymn - dy * (lims_scale_factor-1) / 2
        ymx = ymx + dy * (lims_scale_factor-1) / 2

        # get plot 3D polygon
        plot_bbox = np.array([[xmn, ymn, 0],
                              [xmn, ymx, 0],
                              [xmx, ymx, 0],
                              [xmx, ymn, 0]])
        plot_poly, is_above_plane = self.project_point(plot_bbox, axis=ProjectionAxis.z)

        # get plot 3D polygon
        x = np.linspace(xmn, xmx, 5)
        y = np.linspace(ymn, ymx, 5)
        x, y = np.meshgrid(x, y)
        z = np.zeros(x.shape)
        plot_points_tmp = np.array([x.flatten(), y.flatten(), z.flatten()]).transpose()
        plot_points, is_above_plane = self.project_point(plot_points_tmp, axis=ProjectionAxis.z)
        z = np.reshape(plot_points[:, 2], (x.shape))

        # plot
        p = ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color, alpha=alpha, shade=False)

        return p, ax, fig