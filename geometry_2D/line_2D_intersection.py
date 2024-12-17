import numpy as np


def two_points_to_line(point1, point2, epsilon=1e-8):
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

    :param point1: 2D point (x,y)
    :param point2: 2D point (x,y)
    :param epsilon: for determining a parameter is close to 0
    :return: a,b,c line parameters
    """

    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]

    if np.abs(dx) < epsilon:
        a = 1
        b = 0
        c = -point1[0]
    else:
        b = 1
        a = - dy / dx
        c = -a*point1[0] - b*point1[1]

    line_params = (a, b, c)
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

    :param l1: line 1 (a1,b1,c1)
    :param l2: line 2 (a2,b2,c2)
    :param epsilon: for determining a parameter is close to 0
    :return:
    """
    a1 = l1[0]
    b1 = l1[1]
    c1 = l1[2]
    a2 = l2[0]
    b2 = l2[1]
    c2 = l2[2]

    if abs(a1)<epsilon and abs(b1)<epsilon:
        raise Exception('invalid line 1: a={}, b={}, c={}'.format(a1, b1, c1))
    if abs(a2)<epsilon and abs(b2)<epsilon:
        raise Exception('invalid line 2: a={}, b={}, c={}'.format(a2, b2, c2))

    if abs((a1 * b2) - (a2 * b1)) < epsilon:        # lines do not intersect
        intersection_point = None
    elif abs(a1) > epsilon:
        y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
        x = -(b1 * y + c1) / a1
        intersection_point = (x, y)
    else:
        # a1==0
        # we know that b1!=0 otherwise line1 is invalid
        # we know that a2!=0 otherwise we go the first condition
        y = -c1 / b1
        x = (b2 * c1 - b1 * c2) / (a2 * b1)
        intersection_point = (x, y)

    return intersection_point


def intersect_segments(p1, p2, p3, p4, epsilon=1e-8):
    """
    intersect two line segments
    each line segment is given by two points
    1) find line parameters
    2) find two line intersection point
    3) check if intersection point is inside both segments

    :param p1: (x,y) segment 1 first point
    :param p2: (x,y) segment 1 second point
    :param p3: (x,y) segment 2 first point
    :param p4: (x,y) segment 2 second point
    :param epsilon: for determining a parameter is close to 0
    :return: intersection_point (x,y)
             intersection_status: 0 - valid two segment intersection
                                  1 - two line intersection, but not inside the two segments
                                  2 - lines do not intersect
    """
    # intersect two lines
    l12 = two_points_to_line(p1, p2, epsilon=epsilon)
    l34 = two_points_to_line(p3, p4, epsilon=epsilon)
    p = intersect_lines(l12, l34, epsilon=epsilon)

    if p is None:
        # lines are parallel
        intersection_point = None
        intersection_status = 2

    else:
        # check if intersection is within both segments
        # use parameterization:
        # if b is not 0:
        #      x = t
        #      y = -a*t/b - c
        # if b is 0:
        #      y = t
        #      x = -b*t/a - c

        t12, is_valid12 = line_point_parameterization(l12, [p, p1, p2], epsilon=1e-8)
        t34, is_valid34 = line_point_parameterization(l34, [p, p3, p4], epsilon=1e-8)
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
            intersection_point = p
            intersection_status = 0
        else:
            intersection_point = None
            intersection_status = 1

    return intersection_point, intersection_status


def intersect_polygon(poly_points, p1, p2, epsilon=1e-8):
    """
    intersect two point segment with polygon

    :param poly_points: [n,2] numpy array - ordered polygon points
    :param p1: (x,y) segment 1 first point
    :param p2: (x,y) segment 1 second point
    :param epsilon: for determining a parameter is close to 0
    :return: intersection_points (x,y)
             num_intersection_points: number of valid intersection points
    """

    num_intersection_points = 0
    intersection_points = []
    for i in range(poly_points.shape[0]-1):
        ip, st = intersect_segments(p1, p2, poly_points[i, :], poly_points[i+1, :], epsilon=epsilon)
        if st==0:
            intersection_points.append(ip)
            num_intersection_points = num_intersection_points + 1

    ip, st = intersect_segments(p1, p2, poly_points[-1, :], poly_points[0, :], epsilon=epsilon)
    if st==0:
        intersection_points.append(ip)
        num_intersection_points = num_intersection_points + 1

    return np.array(intersection_points), num_intersection_points
