"""reformat record to Kalibr bag"""

import os
import yaml
import calibration_utils as ccu


class StereoInertialCalibrationParams:
    """
    load / save / reformat camera stereo-inertial calibration
    """
    def __init__(self):
        self.camera_ids=[]
        self.camera_params = {}
        self.stereo = None
        self.imu = None
        return

    def load_camera_calibration(self, camera_calibration_file, replace=False):
        """
        load standard camera calibration
        :param camera_side: camera stereo side - 0 or 1
        :param camera_calibration_file: file path
        :param replace: if True, replace existing values from previous loads
                        if False, keep existing values, and add only new ones
        :return:
        """
        if not os.path.isfile(camera_calibration_file):
            raise Exception('file: {} not found!'.format(camera_calibration_file))

        ph = ccu.PinholeCamera(camera_calibration_file)
        if ph.id not in self.camera_ids:
            self.camera_ids.append(ph.id)
            self.camera_params[ph.id] = ph
        else:
            if (self.camera_params[ph.id].image_size is None) or replace:
                self.camera_params[ph.id].image_size = ph.image_size
            if (self.camera_params[ph.id].focal_length is None) or replace:
                self.camera_params[ph.id].focal_length = ph.focal_length
            if (self.camera_params[ph.id].principal_point is None) or replace:
                self.camera_params[ph.id].principal_point = ph.principal_point
            if (self.camera_params[ph.id].distortion_coefficients is None) or replace:
                self.camera_params[ph.id].distortion_coefficients = ph.distortion_coefficients
            if (self.camera_params[ph.id].skew is None) or replace:
                self.camera_params[ph.id].skew = ph.skew
            if ((self.camera_params[ph.id].T_cam_to_body is None) or replace) and (ph.T_cam_to_body is not None):
                self.camera_params[ph.id].T_cam_to_body = ph.T_cam_to_body

        return

    def load_kalibr_camchain(self, camchain_file, replace=False):
        """
        load Kalibr camchain file
        :param camera_side: camera stereo side - 0 or 1
        :param camera_calibration_file: file path
        :param replace: if True, replace existing values from previous loads
                        if False, keep existing values, and add only new ones
        :return:
        """

        if not os.path.isfile(camchain_file):
            raise Exception('file: {} not found!'.format(camchain_file))

        with open(camchain_file, 'r') as file:

            data = yaml.safe_load(file)
            prev_camid = None
            for cam in data.keys():
                cam_id = os.path.split(data[cam]['rostopic'])[0]
                cam_id = cam_id.strip('/')
                if cam_id not in self.camera_ids:
                    self.camera_ids.append(cam_id)

                # intrinsic parameters
                if data[cam]['camera_model'] != 'pinhole':
                    raise Exception('camera model {} not supported!'.format(data[cam]['camera_model']))
                if (self.camera_params[cam_id].image_size is None) or replace:
                    self.camera_params[cam_id].image_size = data[cam]['resolution']
                if (self.camera_params[cam_id].focal_length is None) or replace:
                    self.camera_params[cam_id].focal_length = data[cam]['intrinsics'][:2]
                if (self.camera_params[cam_id].principal_point is None) or replace:
                    self.camera_params[cam_id].principal_point = data[cam]['intrinsics'][2:]
                if data[cam]['distortion_model'] != 'radtan':
                    raise Exception('distortion model {} not supported!'.format(data[cam]['distortion_model']))
                if (self.camera_params[cam_id].distortion_coefficients is None) or replace:
                    self.camera_params[cam_id].distortion_coefficients = data[cam]['distortion_coefficients']
                if (self.camera_params[cam_id].skew is None) or replace:
                    self.camera_params[cam_id].skew = 0

                # camera to imu pose - we assume here that IMU and body are the same!
                if ((self.camera_params[cam_id].T_cam_to_body is None) or replace) and ('T_cam_imu' in data[cam].keys()):
                    self.camera_params[cam_id].T_cam_to_body = data[cam]['T_cam_imu']

                # stereo params
                if ((self.stereo is None) or replace) and ('T_cn_cnm1' in data[cam].keys()):
                    if prev_camid is None:
                        raise Exception('invalid T_cn_cnm1! T_cn_cnm1 should refer to the previos camera, and cannot be defined for the first camera.')
                    self.stereo = {'cam1_id': cam_id, 'cam2_id': prev_camid, 'T_c2_to_c1': data[cam]['T_cn_cnm1']}

                prev_camid = cam_id
        return

    def load_stereo_calibration(self, stereo_calibration_file):
        """
        load standard stereo calibration
        :param stereo_calibration_file: file path
        :return:
        """
        if not os.path.isfile(stereo_calibration_file):
            raise Exception('file: {} not found!'.format(stereo_calibration_file))
        if self.stereo is not None:
            raise Exception('stereo calibration already loaded!')

        with open(stereo_calibration_file, 'r') as file:
            self.stereo = yaml.safe_load(file)
        return

    def load_imu_calibration(self, imu_calibration_file, replace=False):
        """
        load imu calibratrion
        :param imu_calibration_file: file path
        :param replace: if True, replace existing values from previous loads
                        if False, keep existing values, and add only new ones
        :return:
        """
        if not os.path.isfile(imu_calibration_file):
            raise Exception('file: {} not found!'.format(imu_calibration_file))

        if (self.imu is None) or replace:
            with open(imu_calibration_file, 'r') as file:
                self.imu = yaml.safe_load(file)

            if 'rostopic' in self.imu.keys():
                imu_id =  os.path.split(self.imu['rostopic'])[1]
                imu_id = imu_id.strip('/')
                self.imu.pop('rostopic')
            else:
                imu_id = 'imu0'

            self.imu['id'] =  [imu_id]  # This is a list only to make YAML write fields one above the other.

        return


    def save(self, output_file):
        """
        save calibration params to file
        :param output_file: output file path
        :return:
        """

        output_folder = os.path.dirname(output_file)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        data = {}
        for cid in self.camera_ids:
            data[cid] = self.camera_params[cid].to_dict()

        if self.stereo is not None:
            data['stereo'] = self.stereo

        if self.imu is not None:
            data['imu'] = self.imu

        with open(output_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=None, sort_keys=False)

        return

if __name__ =="__main__":
    calibration_folder =  '../examples'
    sicp = StereoInertialCalibrationParams()

    camera_left_calibration_file = os.path.join(calibration_folder, 'intrinsic_calibration_left/camera_intrinsics_left.yaml')
    print('loading {}...'.format(camera_left_calibration_file))
    sicp.load_camera_calibration(camera_left_calibration_file)

    camera_right_calibration_file =  os.path.join(calibration_folder, 'intrinsic_calibration_right/camera_intrinsics_right.yaml')
    print('loading {}...'.format(camera_right_calibration_file))
    sicp.load_camera_calibration(camera_right_calibration_file)

    camera_left_imu_calibration_file =  os.path.join(calibration_folder, 'camera_imu_calibration/kalibr_results/kalibr_format-camchain-imucam.yaml')
    print('loading {}...'.format(camera_left_imu_calibration_file))
    sicp.load_kalibr_camchain(camera_left_imu_calibration_file, replace=False)

    stereo_calibration_file = os.path.join(calibration_folder, 'stereo_calibration/stereo_calibration.yaml')
    print('loading {}...'.format(stereo_calibration_file))
    sicp.load_stereo_calibration(stereo_calibration_file)

    # imu_calibration_file = os.path.join(calibration_folder, 'camera_imu_calibration/imu_BMI270.yaml')
    imu_calibration_file = os.path.join(calibration_folder, 'camera_imu_calibration/imu_BNO085.yaml')
    print('loading {}...'.format(imu_calibration_file))
    sicp.load_imu_calibration(imu_calibration_file, replace=False)

    output_file = os.path.join(calibration_folder, 'stereo_inertial_calibration.yaml')
    print('saving to {}...'.format(output_file))
    sicp.save(output_file)

    print('Done!')
