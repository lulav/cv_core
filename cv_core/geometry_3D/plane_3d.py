import numpy as np
from enum import Enum
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


    def is_point_above_plane(self, points, epsilon=1e-9):
        """
        check if a point is above or below plane

        :param points: 3D points [nX3]
        :param epsilon: if a point is closer to plane than epsilon, it is considered on the plane
        :return point_sign: sign that says if point is above or below plane
                            1 = point is above plane
                            0 = point is on the plane
                            -1 = point is above plane
        """

        # check inputs
        points = np.array(points)
        if points.shape[1] != 3:
            raise Exception('invalid point size!')

        nx = self.normal[0, 0]
        ny = self.normal[0, 1]
        nz = self.normal[0, 2]

        ox = self.origin[0, 0]
        oy = self.origin[0, 1]
        oz = self.origin[0, 2]

        # check if point is above plane
        d = nx * (points[:, 0] - ox) + ny * (points[:, 1] - oy) + nz * (points[:, 2] - oz)

        point_sign = np.sign(d)
        is_on_plane = np.abs(d.flatten()) < epsilon
        point_sign[is_on_plane] = 0

        return point_sign

    def is_point_on_plane(self, points, epsilon=1e-9):
        """
        check if points are on plane
        :param points: 3D points (nx3)
        :param epsilon: small number
        :return: is_on_plane boolean array (nX1)
        """
        # check inputs
        points = np.array(points)
        if points.shape[1] != 3:
            raise Exception('invalid point size!')
        # check if point is on plane
        OP = points - self.origin
        is_on_plane = np.abs(np.dot(OP, self.normal.flatten())) < epsilon

        return is_on_plane

    def ray_intersection(self, ray_origin, ray_direction, epsilon=1e-9):
        """
        intersect ray with the plane

        :param ray_origin: ray start point [m,3]
        :param ray_direction: ray line of sight [m,3]
        :param epsilon: minimal value for testing if ray is parallel to plane

        :return:  intersection_points - 3D world points - intersection of rays with plane [m,3]
                  nan if for rays with no intersection
        :return:  validIndex - 2  ray intersects plane from above
                               1  ray intersects plane from below
                               0  no intersection

        algorithm:
        ----------
        plane is defined by O_plane = (ox,oy,oz)' - point on plane
                                  N = (nx,ny,nz)'     - plane normal
        we use plane equation is:
                 nx*(x-ox) + ny*(y-oy) + nz*(z-oz) = 0

        LOS is defined by: O_cam = (cx,cy,cz)'   - los origin (camera position)
                            los = (lx,ly,lz)'   - los direction
        we use los parametrization:
            (x,y,z) = (cx+lx*t ,cy+ly*t ,cz+lz*t )

        solving t for intersection point P satisfies two equations:
                - ( nx*(cx-ox) + ny*(cy-oy) + nz*(cz-oz) )
        1) tp = -----------------------------------------------
                            nx*lx + ny*ly + nz*lz
        2) P = (cx+lx*tp ,cy+ly*tp ,cz+lz*tp)

        and in vector form:
         1) tp = -  dot( N, (O_cam-O_plane) ) / dot( N, los )
         2) P = O_cam + tp*los
            tp>0 - in front of ray
            tp<0 - behind ray


        """

        # check inputs
        ray_origin = np.array(ray_origin)
        if ray_origin.shape[1] == 3:
            n = ray_origin.shape[0]
        else:
            raise Exception('invalid ray_origin!')

        ray_direction = np.array(ray_direction)
        if ray_direction.shape != (n, 3):
            raise Exception('invalid ray_direction!')

        # normalize ray direction
        ray_direction = ray_direction / np.reshape(np.linalg.norm(ray_direction, axis=1), (n, 1))

        # find rays not parallel to plane
        idx0 = np.abs(np.dot(ray_direction, self.normal.transpose())) > epsilon
        idx0 = idx0.flatten()
        C = ray_origin[idx0, :]
        O = self.origin.transpose()

        tp = np.zeros((n, 1)) + np.nan
        tp_tmp = - np.divide(np.matmul(self.normal, (C.transpose() - O)),
                             np.matmul(self.normal, ray_direction[idx0, :].transpose()))
        tp[idx0, :] = tp_tmp.transpose()

        intersection_points = np.zeros((n, 3)) + np.nan
        intersection_points[idx0, :] = ray_direction[idx0, :] * tp[idx0, :] + C

        # check if intersection is before ray
        idx_point_in_front_of_ray = tp.flatten() > 0
        idx_point_in_behind_ray = np.bitwise_not(idx_point_in_front_of_ray)
        intersection_points[idx_point_in_behind_ray, :] = np.nan

        # check if intersection is above plane
        d = intersection_points - ray_origin
        cosa = np.matmul(d, self.normal.transpose())
        idx_ray_above_plane = cosa.flatten() < 0

        valid_index = np.zeros(n)

        intersect_above_plane = np.bitwise_and(idx_point_in_front_of_ray, idx_ray_above_plane)
        intersect_below_plane = np.bitwise_and(idx_point_in_front_of_ray, np.bitwise_not(idx_ray_above_plane))
        valid_index[intersect_above_plane] = 2
        valid_index[intersect_below_plane] = 1

        return intersection_points, valid_index

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

def intersect_ray_with_planar_polygon(plane:Plane3D, polygon_points, ray_origin, ray_direction, epsilon):
    """
    intersect 3D ray with a 3D plane segment
    :param plane: Plane3D
    :param polygon_points: polygon points (nx3)
    :param ray_origin: ray start point [m,3]
    :param ray_direction: ray line of sight [m,3]
    :param epsilon: minimal value for testing if ray is parallel to plane
    :return:
    """

    points = np.array(polygon_points)
    if polygon_points.shape[1] != 3:
        raise Exception('invalid points input! must be [nx3]')

    is_point_on = plane.is_point_on_plane(polygon_points, epsilon=1e-9)
    if not all(is_point_on):
        raise Exception('not all points are on plane!')

    intersection_point = plane.ray_intersection(ray_origin, ray_direction, epsilon)

    # # transform all polygon and intersection points to XY plane
    # r =
    #
    # polygon_points2 =



    return