import numpy as np
import cv_core.geometry_2D.line_2D as l2d

"""
common notations:
2D points are given as (nx2) array of (x,y)
2D lines are given as (nx3) array of (a, b, c) where ax+by+c=0
"""

def polygon_segment_intersection(poly_points, p1, p2, epsilon=1e-8, use_shapely=False):
    """
    intersect two point segment with polygon
    Does not support self-intersecting polygon (actually this function still works well for self-intersecting polygons,
    but might miss intersection points close to polygon self intersection)

    :param poly_points: [n,2] numpy array - ordered polygon points
    :param p1: (x,y) segment 1 first point
    :param p2: (x,y) segment 1 second point
    :param epsilon: for determining a parameter is close to 0
    :return: intersection_points (x,y)
             num_intersection_points: number of valid intersection points
    """
    # TODO: handle more cases - what if segments are parallel, and overlap? inf number of intersections?

    if not use_shapely:
        num_intersection_points = 0
        intersection_points = []
        for i in range(poly_points.shape[0]-1):
            ip, st = l2d.intersect_segments(p1, p2, poly_points[i, :], poly_points[i+1, :], epsilon=epsilon)
            if st==0:
                d = [np.linalg.norm(x - ip) for x in intersection_points]
                if len(d) == 0 or min(d) > epsilon:
                    intersection_points.append(ip)
                    num_intersection_points = num_intersection_points + 1

        ip, st = l2d.intersect_segments(p1, p2, poly_points[-1, :], poly_points[0, :], epsilon=epsilon)
        if st==0:
            d = [np.linalg.norm(x - ip) for x in intersection_points]
            if len(d) == 0 or min(d) > epsilon:
                intersection_points.append(ip)
                num_intersection_points = num_intersection_points + 1

    else:
        import shapely
        # raise Exception('shapely implementation not supported yet!')
        shapely_poly = shapely.geometry.Polygon(poly_points)
        shapely_line = shapely.geometry.LineString([p1, p2])
        intersection_points = list(shapely_poly.intersection(shapely_line).coords)
        num_intersection_points = len(intersection_points)

    # in case intersection is close to a polygon vertex, two close intersection points will bw found
    # We filter these duplicate points

    return np.array(intersection_points), num_intersection_points


def point_in_polygon(polygon_points, query_points, epsilon=1e-8, use_shapely=False):
    """
    check if points are inside a polygon

    reference:
    https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html

    algorithm:
    Consider a polygon made up of N vertices (xi,yi) where i ranges from 0 to N-1.
    The last vertex (xN,yN) is assumed to be the same as the first vertex (x0,y0), that is, the polygon is closed.
    To determine the status of a point (xp,yp) consider a horizontal ray emanating from (xp,yp) and to the right.
    Check how many times this ray intersects the line segments making up the polygon
     if th number of intersections is even, then the point is outside the polygon.
     Whereas if the number of intersections is odd then the point (xp,yp) lies inside the polygon.
     Note: intersection counts if the ray passes between a segment start and end point or on the start point, but not on the end point. This prevent double counting.

    :param polygon_points: [n,2] numpy array - ordered polygon points
    :param query_points: [n,2] numpy array - query points
    :param epsilon: for determining a parameter is close to 0
    :return: is_in_polygon (n,1)
    """

    if not use_shapely:

        if polygon_points.shape[1] != 2 or query_points.shape[1] != 2:
            raise Exception('invalid input size!')
        n = polygon_points.shape[0]
        m = query_points.shape[0]
        intersection_count = 0
        for j in range(m):
            x, y = query_points[j, :]
            for i in range(n):
                x1, y1 = polygon_points[i % n, :]
                x2, y2 = polygon_points[(i+1) % n, :]

                if abs(y1-y2) < epsilon:
                    is_intersect = (abs(y-y1) < epsilon) and x <= max(x1, x2)

                else:
                    if min(y1, y2) < y and y <= max(y1, y2) :
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1

                    if p1x == p2x or x <= xinters:
                        intersection_count = intersection_count + 1
                        inside = not inside

    else:
        import shapely
        line = shapely.geometry.LineString(polygon_points)
        polygon = shapely.geometry.Polygon(line)
        inside = []
        for qp in query_points:
            point = shapely.geometry.Point(qp[0], qp[1])
            inside.append(polygon.contains(point))
        inside = np.array(inside)

    return inside


def polygon_intersect(poly_points_1, poly_points_2, use_shapely=True):
    """
    intersect two polygons

    :param poly_points_1: (nx2) polygon 2D points
    :param poly_points_2: (mx2) polygon 2D points
    :return:
    """

    if use_shapely:
        import shapely
        line1 = shapely.geometry.LineString(poly_points_1)
        polygon1 = shapely.geometry.Polygon(line1)

        line2 = shapely.geometry.LineString(poly_points_2)
        polygon2 = shapely.geometry.Polygon(line2)
        intersection_polygon_shapely = polygon1.intersection(polygon2)

        # convert from shapely object to simple np array
        intersection_polygon = np.array(intersection_polygon_shapely.exterior.coords)
        # shapely can have it's "geometry" configured to be 2D or 3D
        # it always does only 2D geometric operations,
        # but the coordinates of shapely object has 3 columns in case it's geometry is configured to 3D.
        # The third coordinate might contain nans which may cause failure at the use end.
        # e.g. if user tries to round the points tp plot
        # therefore we make sure to take only the first two columns
        intersection_polygon = intersection_polygon[:-1,:2]
    else:
        raise Exception('explicit implementation not supported yet!')

    return intersection_polygon