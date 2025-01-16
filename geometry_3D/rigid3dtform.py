import numpy as np
import scipy as sp
# from scipy.special import dtype


# from numpy.ma.core import reshape


class Rigid3dTform:
    """
    Rigid 3D transform
    """
    def __init__(self, R, t):
        """
        set rotation and translation
        :param R: rotation. may be a tuple/list/np.array following scipy default formats:
                                  size 9 - rotation matrix
                                  size 4 - quaternion ordered as (x,y,z,w)
                                  scipy.spatial.transform._rotation.Rotation
        :param t: translation. may be a tuple/list/np.array of size 3.
        """

        if isinstance(R, sp.spatial.transform._rotation.Rotation):
            self.R = R
        elif np.array(R).shape == (3,3):
                self.R = sp.spatial.transform.Rotation.from_matrix(R)
        elif np.array(R).size == 4:
                self.R = sp.spatial.transform.Rotation.from_quat(R)

        self.t = np.reshape(np.array(t, dtype=np.float32), (3,1))

        # matrix notation:
        # T = [  R,   t]
        #     [0,0,0, 1]
        self.T = np.vstack((np.hstack((self.R.as_matrix(), self.t)), (0,0,0,1)))
        return

    def invert(self):
        """
        inverse 3D transform
        """
        R_inv = self.R.inv()
        t_inv = -1 * np.matmul(R_inv.as_matrix(), self.t)
        return Rigid3dTform(R_inv, t_inv)

    def transform_points(self, points):
        """
        transform points forward:
        transformed_points = R * points + t;

        :param points: 3D points (nX3) np.ndarray or list
        :return: transformaed 3D points (nX3) np.ndarray
        """

        p = np.array(points)
        if p.shape[1] != 3:
            raise Exception('invalid points! must be [nx3]!')
        transformed_points = np.matmul(self.R.as_matrix(), points.transpose()) + self.t
        return transformed_points.transpose()

    def __mul__(self, other):
        """
        multiply two transforms
        In matrix notation:
        T = [  R,   t]
           [0,0,0, 1]
        T = T1 * T2 = [ R1,   t1] * [  R2,  t2] = [R1*R2  R1*t2+t1]
                      [0,0,0, 1 ]   [0,0,0, 1 ]   [0,0,0,    1    ]
        So in standard notation we get:
        R = R1*R2
        t = R1*t2 + t1

        :param other:
        :return:
        """
        T = self.T * other.T
        return Rigid3dTform(T[:3,:3], T[:3,3])

