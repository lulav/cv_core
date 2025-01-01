""" Calibrate Camera intrinsics
"""
import os
import argparse
import cv2
import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt
# import yaml
# from numpy.array_api import float64
import camera_calibration as ccu
import pinhole_camera


class CameraIntrinsicCalibrator:
    def __init__(self,):
        self.calibration_flags = None
        self.image_files = None
        self.calibration_board = None

        self.image_width = None
        self.image_height = None
        self.camera = None

        self.fig1 = None
        self.fig2 = None
        self.fig3 = None

    def calibrate(self, camera_id, images_dir, calibration_board_file, camera_model=pinhole_camera.CameraModel.PINHOLE, draw=True, fig_save_folder=None,
                  num_radial_coeefs=2, use_tangential_coeffs=True, outlier_filter_mahalabobis_hreshold=3):
        """
        Calibrate Camera intrinsics
        :param camera_id: camera id string
        :param camera_model: camera model - as specified by pinhole_camera.CameraModel
                                            currently: pinhole or fisheye
        :param images_dir: folder with all calibration images
        :param calibration_board_file: calibration board file
        :param draw: flag for drawing results
        :param fig_save_folder: folder for saving results images.
                                if None, results are not saved.
        :param num_radial_coeefs: number of radial coeffs
        :param use_tangential_coeffs: number of tangential coeffs
        :param outlier_filter_mahalabobis_hreshold: outliers are filtered by reprojection error rms per image.
                                                    th is mahalanobis distance of each reprojection error rms.
                                                    None means no outlier filtering
        :return:
        """
        # get image files
        self.image_files = ccu.get_images(images_dir)
        if len(self.image_files) == 0:
            raise Exception('no images found in: {}!'.format(images_dir))
        print('----- found {} images\n'.format(len(self.image_files)))

        # prepare object points
        self.calibration_board = ccu.CalibrationBoard(calibration_board_file)

        # detect corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        criteria_sub_pix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        sub_pix_win_size = (9, 9)
        sub_pix_zero_zone = (-1, -1)
        objpoints_tmp, imgpoints_tmp, img_width, img_height = ccu.detect_corners(self.image_files, self.calibration_board, criteria,
                       sub_pix_criteria=criteria_sub_pix, sub_pix_win_size=sub_pix_win_size, sub_pix_zero_zone=sub_pix_zero_zone,
                                                                         draw=True)
        objpoints = [x for x in objpoints_tmp if x is not None]
        imgpoints = [x for x in imgpoints_tmp if x is not None]
        num_images = len(imgpoints)
        print('----- detected corners in {} images'.format(num_images))

        self.image_width = img_width
        self.image_height = img_height
        num_images = len(objpoints)

        # calibrate camera intrinsics
        if camera_model == pinhole_camera.CameraModel.PINHOLE:
            self.set_pinhole_calibration_params(num_radial_coeefs, use_tangential_coeffs)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
            # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_height, img_height),
            #                                                    cameraMatrix=None, distCoeffs=None, flags=calibration_flags)
            retval, mtx, dist, rvecs, tvecs, std_deviations_intrinsics, std_deviations_extrinsics, per_view_errors= cv2.calibrateCameraExtended(
                                                                        objpoints, imgpoints, (img_width, img_height),
                                                                        cameraMatrix=None, distCoeffs=None,
                                                                        rvecs=None, tvecs=None, stdDeviationsIntrinsics=None,
                                                                        stdDeviationsExtrinsics=None, perViewErrors=None,
                                                                        flags = self.calibration_flags, criteria=criteria)

        elif camera_model == pinhole_camera.CameraModel.FISHEYE:
            N_OK = len(objpoints)
            mtx = np.zeros((3, 3))
            dist = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            calibration_flags_fish = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
            self.set_fisheye_calibration_params(num_radial_coeefs)
            retval, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(np.array(objpoints), np.array(imgpoints), (img_width, img_height),
                                 K=mtx, D=dist, rvecs=rvecs, tvecs=tvecs,
                                 flags=calibration_flags_fish,
                                 criteria=criteria)

        else:
            raise Exception('invalid camera model: {}'.format(camera_model))

        # re-projection error
        reprojection_errors, rms_reprojection_error_per_image = reprojection_error(imgpoints, objpoints, rvecs, tvecs, mtx, dist, camera_model)
        reprojection_error_per_image_2d=np.linalg.norm(rms_reprojection_error_per_image, axis=1)

        for i, r in enumerate(reprojection_error_per_image_2d.flatten()):
            print('frame {}: {:.3f} pixels'.format(i, r))
        mean_error = np.sum(reprojection_error_per_image_2d) / len(reprojection_error_per_image_2d)
        print("total mean error: {}".format(mean_error))
        print('\n')

        # find inliers
        is_inlier = ccu.detect_reprojection_outliers(reprojection_error_per_image_2d, outlier_filter_mahalabobis_hreshold)
        objpoints_inliers = []
        imgpoint_inliers = []
        outliers_reprojection_error = []
        for i in range(num_images):
            if is_inlier[i]:
                objpoints_inliers.append(objpoints[i])
                imgpoint_inliers.append(imgpoints[i])
            else:
                outliers_reprojection_error.append({'id': i, 'error': reprojection_error_per_image_2d[i]})
        print('----- filtered {} inlier images'.format(len(imgpoint_inliers)))

        if draw:
            mean_reprojection_error_per_image_2d = np.linalg.norm(rms_reprojection_error_per_image, axis=1)
            self.fig1 = plt.figure()
            ax1 = self.fig1.add_subplot(1, 1, 1)
            ax1.bar(range(0, len(rms_reprojection_error_per_image)), mean_reprojection_error_per_image_2d, width=0.8, bottom=None, align='center', data=None)
            outlier_ids = [x['id'] for x in outliers_reprojection_error]
            outlier_errors = [x['error'] for x in outliers_reprojection_error]
            ax1.bar(outlier_ids, outlier_errors, width=0.8, bottom=None, align='center', data=None, color='r')
            ax1.plot((0, len(rms_reprojection_error_per_image)), (mean_error, mean_error), linestyle='dashed', color='r')
            ax1.text(len(reprojection_error_per_image_2d), mean_error, 'rmse={:.2f}'.format(mean_error), fontsize=9, fontweight=9)
            ax1.set_xlabel('image id', fontsize=10)
            ax1.set_ylabel('reprojection error', fontsize=10)
            ax1.set_title('reprojection errors - all frames', fontsize=18)
            plt.show(block=False)

        # re-calibrate (without outliers)
        if camera_model == pinhole_camera.CameraModel.PINHOLE:
            self.set_pinhole_calibration_params(num_radial_coeefs, use_tangential_coeffs)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
            retval, mtx, dist, rvecs, tvecs, std_deviations_intrinsics, std_deviations_extrinsics, per_view_errors = cv2.calibrateCameraExtended(
                objpoints_inliers, imgpoint_inliers, (img_width, img_height),
                cameraMatrix=None, distCoeffs=None,
                rvecs=None, tvecs=None, stdDeviationsIntrinsics=None,
                stdDeviationsExtrinsics=None, perViewErrors=None,
                flags=self.calibration_flags, criteria=criteria)

        elif camera_model == pinhole_camera.CameraModel.FISHEYE:
            N_OK = len(objpoints)
            mtx = np.zeros((3, 3))
            dist = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            calibration_flags_fish = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
            self.set_fisheye_calibration_params(num_radial_coeefs)
            retval, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints_inliers, imgpoint_inliers, (img_width, img_height),
                                 K=mtx, D=dist, rvecs=rvecs, tvecs=tvecs,
                                 flags=calibration_flags_fish,
                                 criteria=criteria)

        else:
            raise Exception('invalid camera model: {}'.format(camera_model))

        if retval:
            self.camera = pinhole_camera.PinholeCamera()
            self.camera.set(camera_id, camera_model, mtx, dist, (img_width, img_height))
        reprojection_errors, rms_reprojection_error_per_image = reprojection_error(imgpoint_inliers, objpoints_inliers,
                                                                                   rvecs, tvecs, mtx, dist, camera_model)
        reprojection_error_per_image_2d=np.linalg.norm(rms_reprojection_error_per_image, axis=1)
        # reprojection_error_per_image_2d = per_view_errors
        mean_error = np.sum(reprojection_error_per_image_2d) / len(reprojection_error_per_image_2d)
        print("total mean error: {}".format(mean_error))

        if draw:
            # draw reprojection errors per image
            mean_reprojection_error_per_image_2d = np.linalg.norm(rms_reprojection_error_per_image, axis=1)
            self.fig2 = plt.figure()
            ax1 = self.fig2.add_subplot(1, 1, 1)
            ax1.bar(range(0, len(rms_reprojection_error_per_image)), mean_reprojection_error_per_image_2d, width=0.8, bottom=None, align='center', data=None)
            ax1.plot((0, len(rms_reprojection_error_per_image)), (mean_error, mean_error), linestyle='dashed', color='r')
            ax1.text(len(reprojection_error_per_image_2d), mean_error, 'rmse={:.2f}'.format(mean_error), fontsize=9, fontweight=9)
            ax1.set_xlabel('image id', fontsize=10)
            ax1.set_ylabel('reprojection error', fontsize=10)
            ax1.set_title('reprojection errors - inlier images', fontsize=18)

            # draw all image points
            self.fig3 = plt.figure()
            ax2 = self.fig3.add_subplot(1, 1, 1)
            ax2.set_xlabel('pixel X', fontsize=10)
            ax2.set_ylabel('pixel Y', fontsize=10)
            ax2.set_title('all image points', fontsize=18)
            clr_idx = np.uint8(np.round(np.linspace(0,250,len(imgpoint_inliers))))
            for i, p in enumerate(imgpoint_inliers):
                clr = plt.cm.hsv(clr_idx[i])
                # print('color {}: {} = {}'.format(i,clr_idx[i],clr))
                ax2.scatter(p[:,:,0], p[:,:,1], color=clr)

            if fig_save_folder is not None:
                if not(os.path.isdir(fig_save_folder)):
                    os.makedirs(fig_save_folder)
                fig1_file = os.path.join(fig_save_folder, 'calibration_reprojection_errors_all.png')
                self.fig1.savefig(fig1_file)
                fig2_file = os.path.join(fig_save_folder, 'calibration_reprojection_errors_inliers.png')
                self.fig2.savefig(fig2_file)
                fig3_file = os.path.join(fig_save_folder, 'calibration_image_points.png')
                self.fig3.savefig(fig3_file)

            plt.show(block=True)

        # # undistorted images
        # for file_name in image_files:
        #     img = cv2.imread('left12.jpg')
        #     h, w = img.shape[:2]
        #     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        #
        #     dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        #     # crop the image
        #     x, y, w, h = roi
        #     dst = dst[y:y + h, x:x + w]
        #     cv2.imwrite('calibresult.png', dst)

        return

    def set_pinhole_calibration_params(self, num_radial_coeffs, use_tangential_coeffs):
        """
        reformat calibration params to opencv format
        :param num_radial_coeefs: int number of radial coefficients. The rest will be set to 0.
                                  This must be between 0 and 6
        :param num_tangential_coeffs: bool - set True to use tangential coefficients
        :return:
        """

        if num_radial_coeffs > 6:
            raise Exception('number of tangential coefficients must be smaller or equal to 6!')
        self.calibration_flags = 0

        # handle radial distortion params
        if num_radial_coeffs > 3:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_RATIONAL_MODEL
        if num_radial_coeffs < 6:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K6
        if num_radial_coeffs < 5:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K5
        if num_radial_coeffs < 4:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K4
        if num_radial_coeffs < 3:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K3
        if num_radial_coeffs < 2:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K2
        if num_radial_coeffs < 1:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K1

        # handle tangential distortion params
        if not use_tangential_coeffs:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_ZERO_TANGENT_DIST

        return

    def set_fisheye_calibration_params(self, num_distortion_coeffs):
        """
        reformat calibration params to opencv format
        :param num_distortion_coeffs: number of fisheye distortion coefficients.
        :return:
        """

        self.calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        if num_distortion_coeffs > 4:
            raise Exception('number of fisheye distortion coefficients must be smaller or equal to 4!')
        if num_distortion_coeffs < 4:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K4
        if num_distortion_coeffs < 3:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K3
        if num_distortion_coeffs < 2:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K2
        if num_distortion_coeffs < 1:
            self.calibration_flags = self.calibration_flags + cv2.CALIB_FIX_K1

        return

    def save_results(self, output_dir, camera_name):
        camera_intrinsic_calibration_file = os.path.join(output_dir, 'camera_intrinsics_{}.yaml'.format(camera_name))
        self.camera.save(camera_intrinsic_calibration_file)
        return

    def __del__(self):
        cv2.destroyAllWindows()


def reprojection_error(image_points, world_points, rvecs, tvecs, camera_intrinsic_matrix, camera_distortions_model, camera_model):

    n = len(image_points)

    if len(world_points) != n or len(rvecs) != n or len(tvecs) != n:
        Exception('invalid input size!')

    reprojection_errors = []
    rms_per_frame = []
    for i in range(n):
        if camera_model == pinhole_camera.CameraModel.PINHOLE:
            image_points_projected, _ = cv2.projectPoints(world_points[i], rvecs[i], tvecs[i], camera_intrinsic_matrix, camera_distortions_model)
        elif camera_model == pinhole_camera.CameraModel.FISHEYE:
            image_points_projected, _ = cv2.fisheye.projectPoints(world_points[i], rvecs[i], tvecs[i], camera_intrinsic_matrix, camera_distortions_model)
        else:
            raise Exception('invalid camera model {}'.format(camera_model))

        reprojection_error = ( np.squeeze(image_points[i]) - np.squeeze(image_points_projected))
        reprojection_errors.append(reprojection_error)
        rms = np.sqrt(np.mean(reprojection_error**2, axis=0))
        rms_per_frame.append(rms)

    return np.array(reprojection_errors), np.array(rms_per_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics.")
    parser.add_argument("--images_dir", help="Image directory.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--calibration_board_file", help="calibration board file.")
    parser.add_argument("--camera_name", help="camera name.", default='cam0')
    parser.add_argument("--camera_model", help="camera model.", default='pinhole')
    parser.add_argument("--num_radial_coeefs", help="number of radial coefficients [0-6]", default=2)
    parser.add_argument("--use_tangential_coeffs", help="boolean - use 2 tangential distortion coefficients.", default=True)
    parser.add_argument("--num_fisheye_coeefs", help="number of fisheye coefficients [0-4]", default=2)
    args = parser.parse_args()

    images_dir = args.images_dir
    output_dir = args.output_dir
    calibration_board_file = args.calibration_board_file
    camera_name = args.camera_name
    camera_model = args.camera_model
    num_radial_coeefs = args.num_radial_coeefs
    use_tangential_coeffs = args.use_tangential_coeffs
    num_fisheye_coeefs = args.num_fisheye_coeefs

    if camera_model.upper() in [item.name for item in pinhole_camera.CameraModel]:
        camera_model = pinhole_camera.CameraModel[camera_model.upper()]
    else:
        Exception('invalid camera model {}!'.format(camera_model))

    # images_dir = './examples/intrinsic_calibration_left/left/'
    # output_dir = './results/intrinsic_calibration_left/'
    # calibration_board_file = './examples/intrinsic_calibration_left/calibration_chessboard.yaml'
    # camera_name = 'left'
    # camera_model = pinhole_camera.CameraModel.PINHOLE
    # num_radial_coeefs = 2
    # use_tangential_coeffs = True

    cic = CameraIntrinsicCalibrator()

    if camera_model ==pinhole_camera.CameraModel.PINHOLE:
        cic.calibrate(camera_name, images_dir, calibration_board_file, draw=True, fig_save_folder=output_dir,
                  camera_model=pinhole_camera.CameraModel.PINHOLE,
                  num_radial_coeefs=num_radial_coeefs, use_tangential_coeffs=use_tangential_coeffs, outlier_filter_mahalabobis_hreshold=3)
    elif camera_model == pinhole_camera.CameraModel.FISHEYE:
        cic.calibrate(camera_name, images_dir, calibration_board_file, draw=True, fig_save_folder=output_dir,
                  camera_model=pinhole_camera.CameraModel.PINHOLE,
                  num_radial_coeefs=num_fisheye_coeefs, use_tangential_coeffs=True, outlier_filter_mahalabobis_hreshold=3)
    else:
        raise Exception('camera model {} not supported!'.format(camera_model))

    cic.save_results(output_dir, camera_name)


