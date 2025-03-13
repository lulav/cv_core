import numpy as np

"""
common notations:
2D points are given as (nx2) array of (x,y)
2D lines are given as (nx3) array of (a, b, c) where ax+by+c=0
"""

def line_from_two_points(point1, point2, epsilon=1e-8):
    """
    get line parameters from two points according to:
    ax+by+c=0

    solving two equations:
    a*x1 + b*y1 + c=0
    a*x2 + b*y2 + c=0
    -----------------
    when x1-x2 is not 0:
    a=-b*(y1-y2)/(x1-x2)
    and if x1-x2=0:
    the equation is x = -c/a so a=1, c=-x1

    :param point1: 2D points [nx2] (x,y)
    :param point2: 2D points [nx2] (x,y)
    :param epsilon: for determining a parameter is close to 0
    :return: a,b,c line parameters
    """

    point1 = np.array(point1).reshape(-1, 2)
    point2 = np.array(point2).reshape(-1, 2)
    n = point1.shape[0]
    if point1.shape[1] != 2 or point2.shape != point1.shape:
        raise Exception('invalid input sizes!')

    d = point1 - point2
    dx = d[:, 0].reshape((n, 1))
    dy = d[:, 1].reshape((n, 1))
    p1x =  point1[:, 0].reshape((n, 1))
    p1y =  point1[:, 1].reshape((n, 1))

    a = np.zeros((n, 1))
    b = np.zeros((n, 1))
    c = np.zeros((n, 1))

    idx1 = np.abs(dx) < epsilon
    a[idx1] = 1
    b[idx1] = 0
    c[idx1] = -p1x[idx1]

    idx2 = np.abs(dx) >= epsilon
    b[idx2] = 1
    a[idx2] = - np.divide(dy[idx2], dx[idx2])
    c[idx2] = -np.multiply(a[idx2], p1x[idx2]) - np.multiply(b[idx2], p1y[idx2])

    line_params = np.hstack((a, b, c))
    return line_params


def line_point_parameterization(l, points, epsilon=1e-8):
    """
    find parameterization of point on line

    use parameterization:
    if b > a:
          x = t
          y = -a*t/b - c
    if b < a:
          y = t
          x = -b*t/a - c
    Note: specifically b and a cannot be both 0, so the larger one is not 0

    :param l: line (a,b,c)
    :param points: query points (x,y)
    :param epsilon: for determining a parameter is close to 0
    :return:
    """
    a = l[0]
    b = l[1]
    c = l[2]
    p = np.array(points)

    # check points are on the line
    v = a*p[:,0] + b*p[:,1] + c
    valid_point = abs(v) < epsilon

    # calc parameterization
    if abs(b) < abs(a):
        t = p[:,1]
    else:
        t = p[:,0]

    return t, valid_point

def intersect_lines(l1, l2, epsilon=1e-8):
    """
    find intersection point of two lines

    solve equations:
    a1*x + b1*y + c1 = 0
    a2*x + b2*y + c2 = 0
    --------------------
    when a1 is not 0:
    y = (c2-c1) / (a1*b2-a2*b1)
    x = -(b1*y + c1) / a1
    and when a1 is 0:
    y = -c1 / b1
    x = (b2*c1 - b1*c2) / (a2 * b1)
    in any case if  (a1*b2-a2*b1)=0 lines do not intersect!

    :param l1: line 1 [nx3] (a1,b1,c1)
    :param l2: line 2 [nx3] (a2,b2,c2)
    :param epsilon: for determining a parameter is close to 0
    :return:
    """

    l1 = np.array(l1)
    n = l1.shape[0]
    if l1.shape[1] != 3 or l2.shape != l1.shape:
        raise Exception('invalid input sizes!')

    intersection_point = np.zeros((n, 2))
    for i in range(n):
        a1, b1, c1 = l1[i, :]
        a2, b2, c2 = l2[i, :]

        if abs(a1)<epsilon and abs(b1)<epsilon:
            raise Exception('invalid line 1: a={}, b={}, c={}'.format(a1, b1, c1))
        if abs(a2)<epsilon and abs(b2)<epsilon:
            raise Exception('invalid line 2: a={}, b={}, c={}'.format(a2, b2, c2))

        if abs((a1 * b2) - (a2 * b1)) < epsilon:        # lines do not intersect
            intersection_point[i, :] = np.nan
        elif abs(a1) > epsilon:
            y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
            x = -(b1 * y + c1) / a1
            intersection_point[i, :] = (x, y)
        else:
            # a1==0
            # we know that b1!=0 otherwise line1 is invalid
            # we know that a2!=0 otherwise we go the first condition
            y = -c1 / b1
            x = (b2 * c1 - b1 * c2) / (a2 * b1)
            intersection_point[i, :] = (x, y)

    return intersection_point


def intersect_segments(p1, p2, p3, p4, epsilon=1e-8, use_shapely=False):
    """
    intersect two line segments
    each line segment is given by two points
    1) find line parameters
    2) find two line intersection point
    3) check if intersection point is inside both segments

    :param p1: [nx2] (x,y) segment 1 first point
    :param p2: [nx2] (x,y) segment 1 second point
    :param p3: [nx2] (x,y) segment 2 first point
    :param p4: [nx2] (x,y) segment 2 second point
    :param epsilon: for determining a parameter is close to 0
    :param use_shapely: True - use shapely
                        False - use explicit implementation
    :return: intersection_point (x,y)
             intersection_status: 0 - valid two segment intersection
                                  1 - two line intersection, but not inside the two segments
                                  2 - lines do not intersect
    """

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)
    n = p1.shape[0]
    if p1.shape[1] != 2 or p2.shape != p1.shape or p3.shape != p1.shape or p4.shape != p1.shape:
        raise Exception('invalid input sizes!')

    if not use_shapely:
        intersection_point = np.zeros((n,2))
        intersection_status = np.zeros((n,1))

        # intersect two lines
        l12 = line_from_two_points(p1, p2, epsilon=epsilon)
        l34 = line_from_two_points(p3, p4, epsilon=epsilon)
        p = intersect_lines(l12, l34, epsilon=epsilon)
        # TODO: handle multiple inputs efficiently

        for i in range(n):
            if p is None:
                # lines are parallel
                intersection_point[i, :] = np.nan
                intersection_status[i, :] = 2

            else:
                # check if intersection is within both segments
                # use parameterization:
                # if b is not 0:
                #      x = t
                #      y = -a*t/b - c
                # if b is 0:
                #      y = t
                #      x = -b*t/a - c

                t12, is_valid12 = line_point_parameterization(l12[i, :], [p[i, :], p1[i, :], p2[i, :]], epsilon=1e-8)
                t34, is_valid34 = line_point_parameterization(l34[i, :], [p[i, :], p3[i, :], p4[i, :]], epsilon=1e-8)
                if not all(is_valid12) or not all(is_valid34):
                    raise Exception('invalid points - not on line')
                is_in_12 = (t12[1] - epsilon <= t12[0] <= t12[2] + epsilon) or (t12[2] - epsilon <= t12[0] <= t12[1] + epsilon)
                is_in_34 = (t34[1] - epsilon <= t34[0] <= t34[2] + epsilon) or (t34[2] - epsilon <= t34[0] <= t34[1] + epsilon)

                # this is the same as:
                # is_in_12 = (((p1[0]-epsilon <= p[0] <= p2[0]+epsilon) or (p2[0]-epsilon <= p[0] <= p1[0]+epsilon)) and
                #             ((p1[1]-epsilon <= p[1] <= p2[1]+epsilon) or (p2[1]-epsilon <= p[1] <= p1[1]+epsilon)))
                # is_in_34 = (((p3[0]-epsilon <= p[0] <= p4[0]+epsilon) or (p4[0]-epsilon <= p[0] <= p3[0]+epsilon)) and
                #              ((p3[1]-epsilon <= p[1] <= p4[1]+epsilon) or (p4[1]-epsilon <= p[1] <= p3[1]+epsilon)))
                # but more stable

                if is_in_12 and is_in_34:
                    intersection_point[i, :] = p[i, :]
                    intersection_status[i, :] = 0
                else:
                    intersection_point[i, :] = np.nan
                    intersection_status[i, :] = 1

    else:
        import shapely
        # TODO: handle multiple inputs efficiently
        intersection_point = np.zeros((n,2))
        intersection_status = np.zeros((n,1))
        for i in range(n):
            line1 = shapely.LineString([p1[i, :], p2[i, :]])
            line2 = shapely.LineString([p3[i, :], p4[i, :]])
            point_i = shapely.intersection(line1, line2)
            intersection_point[i, :] = np.array(point_i.xy).flatten()
            if point_i.is_empty:
                intersection_status[i] = 1
            else:
                intersection_status[i] = 0

    return intersection_point, intersection_status
