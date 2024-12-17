import os
import cv2
import numpy as np
import scipy as sp
import yaml
from enum import Enum
from geometry_3D.rigid3dtform import Rigid3dTform


class CameraModel(Enum):
    PINHOLE = 0
    FISHEYE = 1


class PinholeCamera:
    """
    Pinhole camera object
    Support pinhole camera model as well as fisheye camera model.
    major members:
    id - camera id
    model - CameraModel.PINHOLE / CameraModel.FISHEYE
    image_size - [width, height] in pixels
    focal_length - [fx, fy] in pixels
    principal_point - [cx, cy] optical axis point in pixels
    skew - skew (scalar)
    K - intrinsic matrix:
        [fx, s,  cx]
        [0,  fy, cy]
        [0, 0,   1 ]
    distortion_coefficients - distortion coefficients given in opencv notation:
                              for pinhole model: k1, k2, p1, p2, k3, k4, k5, k6
                              for fisheye model: k1, k2, k3, k4
    T_cam_to_body - [3x4] 3D rigid transform (rotation and translation) that transforms a point from camera to body
    """

    def __init__(self, camera_intrinsic_params_file=None):

        # camera name
        self.id = None  # camera name

        # camera model
        self.model = None  # camera name

        # camera intrinsic_params
        self.image_size = None  # (width, height)
        self.focal_length = None  # (fx, fy)
        self.principal_point = None  # (cx, cy)
        self.skew = None
        self.K = None  # [3x3] intrinsic matrix
        self.distortion_coefficients = None  # opencv format: [k1, k2, p1, p2, k3, k4, k5, k6, k7, k8] the rest of the parameters are not supported!

        # camera body pose
        self.T_cam_to_body = None  # transforms points from camera to body frame

        if camera_intrinsic_params_file is not None:
            self.load(camera_intrinsic_params_file)

    def load(self, camera_intrinsic_params_file):
        """
        load camera intrinsic params from file
        """

        if not os.path.isfile(camera_intrinsic_params_file):
            Exception('camera calibration file: {} not found!'.format(camera_intrinsic_params_file))

        with open(camera_intrinsic_params_file, 'r') as file:
            data = yaml.safe_load(file)

            if 'id' in data.keys():
                self.id = data['id']
            else:
                Exception('id not found!')

            if ('model' in data.keys()) and (data['model'].upper() in [item.name for item in CameraModel]):
                self.model = CameraModel[data['model'].upper()]
            else:
                Exception('invalid camera model {}!'.format(data['model']))

            if 'image_size' in data.keys():
                self.image_size = np.int32(data['image_size'])
            else:
                Exception('image_size not found!')

            if 'focal_length' in data.keys():
                self.focal_length = data['focal_length']
            else:
                Exception('focal_length not found!')
            if 'principal_point' in data.keys():
                self.principal_point = data['principal_point']
            else:
                Exception('principal_point not found!')
            if 'skew' in data.keys():
                self.skew = data['skew']
            else:
                self.skew = 0
            self.K = np.array(((self.focal_length[0], self.skew, self.principal_point[0]),
                               (0, self.focal_length[1], self.principal_point[1]),
                               (0, 0, 1)))

            if 'distortion_coefficients' in data.keys():
                distortion_coefficients = data['distortion_coefficients']
            else:
                distortion_coefficients = [0, 0, 0, 0]
            if len(distortion_coefficients) > 8:
                raise Exception('only 8 distortion coefficients are supported: k1,k2,p1,p2,k3,k4,k5,k6 !')
            self.distortion_coefficients = np.array(distortion_coefficients)

            if 'T_cam_to_body' in data.keys():
                T = np.array(data['T_cam_to_body'])
                if T.size != 12:
                    raise Exception('invalid T_cam_to_body')
                else:
                    self.T_cam_to_body = np.resize(T, (3, 4))

        return

    def set(self, id, model, intrinsic_matrix, dist_coeffs, image_size, T_cam_to_body=None, skew=0):
        """
        set camera intrinsics
        """
        self.id = id

        if isinstance(model, CameraModel):
            self.model = model
        else:
            raise Exception('ivalid camera model type!')

        if isinstance(intrinsic_matrix, np.ndarray) and intrinsic_matrix.size == 9:
            intrinsic_matrix = np.reshape(intrinsic_matrix, (3, 3))
        elif (isinstance(intrinsic_matrix, list) or isinstance(intrinsic_matrix, tuple)) and len(intrinsic_matrix) == 9:
            intrinsic_matrix = np.reshape(np.array(intrinsic_matrix), (3, 3))
        else:
            raise Exception('invalid intrinsic matrix')
        self.K = intrinsic_matrix
        self.focal_length = (intrinsic_matrix[0, 0], intrinsic_matrix[1, 1])
        self.principal_point = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
        self.skew = skew

        if isinstance(dist_coeffs, np.ndarray) and dist_coeffs.size <= 8:
            self.distortion_coefficients = dist_coeffs.flatten()
        elif (isinstance(dist_coeffs, list) or isinstance(dist_coeffs, tuple)) and len(dist_coeffs) <= 8:
            self.distortion_coefficients = np.array(dist_coeffs).flatten()
        else:
            raise Exception('invalid distortion coefficients')

        if isinstance(image_size, np.ndarray) and image_size.size == 2:
            self.image_size = image_size.flatten()
        elif (isinstance(image_size, list) or isinstance(image_size, tuple)) and len(image_size) == 2:
            self.image_size = np.array(image_size).flatten()
        else:
            raise Exception('invalid distortion coefficients')

        if T_cam_to_body is not None:
            if T_cam_to_body.size != 16:
                raise Exception('invalid T_cam_to_body')
            else:
                self.T_cam_to_body = np.resize(T_cam_to_body, (3, 4))

        return

    def save(self, camera_intrinsics_file):

        output_dir = os.path.dirname(camera_intrinsics_file)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        data = self._to_dict()

        with open(camera_intrinsics_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=None, sort_keys=False)

        return

    def _to_dict(self):
        distortion_coefficients = self.distortion_coefficients
        data = {
            'id': self.id,
            'model': self.model.name,
            'focal_length': [float(self.K[0, 0]), float(self.K[1, 1])],
            'principal_point': [float(self.K[0, 2]), float(self.K[1, 2])],
            'image_size': [int(self.image_size[0]), int(self.image_size[1])],
            'distortion_coefficients': distortion_coefficients.flatten().tolist(),
            'skew': 0}
        if self.T_cam_to_body is not None:
            data['T_cam_to_body'] = self.T_cam_to_body.tolist()

        return data

    def equal_to(self, other):
        """
        compare all values between two PinholeCamera objects
        This does not replace __eq__ which actually checks if it is the same object!
        """
        if isinstance(other, PinholeCamera):
            d1 = self._to_dict()
            d2 = other._to_dict()

            ie_equal = True
            for k in d1.keys():
                if not(k in d2.keys() and d1[k] == d2[k]):
                    ie_equal = False
            for k in d2.keys():
                if k not in d1.keys():
                    ie_equal = False
        else:
            ie_equal = False

        return ie_equal

    def _pixel_in_image(self, pixels):
        """
        check if pixel is inside the image
        :param pixels: [nx2] pixel coordimates
        :return:
        """
        return np.bitwise_and(0 <= pixels[:,0] <= self.image_size[1], 0 <= pixels[:, 1] <= self.image_size[0])

    def project_points(self, world_points, camera_pose:Rigid3dTform):
        """
        project world points to image pixels
        :param world_points: world points [nx3] numpy array (x,y,z)
        :param camera_pose: Rigid3dTform that transforms points from camera coordinates o world coordinates
        :return: image points [nx2] numpy array
        """
        world_points = np.array(world_points)
        if world_points.shape[1] != 3:
            raise Exception('invalid points size! expecting [nx3]')

        # get LOS in camera coordinates
        if self.model is CameraModel.PINHOLE:
            tvec = camera_pose.t
            rvec = cv2.Rodrigues(camera_pose.R.as_matrix())
            image_points = cv2.projectPoints(world_points, rvec, tvec, self.K, self.distortion_coefficients)

        elif self.model is CameraModel.FISHEYE:
            T = camera_pose.T
            image_points = cv2.fisheye.projectPoints(world_points, T, self.K, self.distortion_coefficients)
        else:
            raise Exception('project_points does not support {} camera model'.format(self.model))

        # check if in image
        is_in_image = self._pixel_in_image(image_points)

        return image_points, is_in_image

    def pixel_to_los(self, image_points, camera_rotation):
        """
        back project pixels to line of sight ray
        :param image_points: image points [nx2] numpy array
        :param camera_rotation: rotation that rotates vectors from camera coordinates o world coordinates
                                this may be:
                                - [3x3] rotation_matrix
                                - scipy rotation
        :return: world points [nx3] numpy array
        """
        image_points = np.array(image_points)
        if image_points.shape[1] != 3:
            raise Exception('invalid points size! expecting [nx3]')

        if isinstance(camera_rotation, sp.spatial.transform._rotation.Rotation):
            R = camera_rotation.as_matrix()
        elif isinstance(camera_rotation, np.ndarray):
            R = np.reshape(camera_rotation, (3, 3))
        else:
            raise Exception('invalid rotation format!')

        # get LOS in camera coordinates
        raw_points = cv2.undistortPoints(image_points, self.K, self.distortion_coefficients, R=np.eye(3), P=np.eye(3))

        # normalize
        normalized_points =  raw_points / np.linalg.norm(raw_points, ord=2, axis=1)

        # rotate to world coordinates
        world_points = np.matmul(R, normalized_points.transpose())

        # check if in image
        is_in_image = self._pixel_in_image(image_points)

        return world_points.transpose(), is_in_image