""" Calibrate Camera intrinsics
"""
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from cv_core import camera_calibration as ccu
import cv_core.pinhole_camera as pinhole_camera

class CameraStereoCalibrator:
    def __init__(self,):
        self.image_files_left = None
        self.image_files_right = None
        self.calibration_board = None

        self.image_width = None
        self.image_height = None
        self.cam_left = None
        self.cam_right = None

        self.stereo_R = None
        self.stereo_t = None

        self.fig1 = None
        self.fig2 = None
        self.fig3 = None

    def calibrate(self, images_folder_left, images_folder_right,
                  intrinsic_calibration_file_left, intrinsic_calibration_file_right,
                  calibration_board_file, draw=True, fig_save_folder=None, mahalabobis_inlier_threshold=3):
        """
        calibrate stereo cameras

        :param images_folder_left: left camera images folder
        :param images_folder_right: right camera images folder
        :param intrinsic_calibration_file_left: what you think it means
        :param intrinsic_calibration_file_right: what you think it means
        :param calibration_board_file: calibration board file
        :param draw: draw corner detection and results
        :param fig_save_folder: folder for saving calibration analysis results (residual errors etc.)
                                if None - no calibration analysis results will not be saved
        :param mahalabobis_inlier_threshold: outliers are filtered by reprojection error rms per image.
                                             th is mahalanobis distance of each reprojection error rms.
                                             None means no outlier filtering
        :return: R,t - transform points from camera right to camera left
                       IMPORTANT NOTE: this is inverse to opencv notation which returns
                       --------------  R,t that transform points from camera left to camera right!

        """
        if not os.path.isdir(images_folder_left):
            raise Exception('folder {} not found!'.format(images_folder_left))
        if not os.path.isdir(images_folder_right):
            raise Exception('folder {} not found!'.format(images_folder_right))

        # get image files
        self.image_files_left = ccu.get_images(images_folder_left)
        self.image_files_right = ccu.get_images(images_folder_right)
        print('----- found {} left images\n'.format(len(self.image_files_left)))
        print('----- found {} right images\n'.format(len(self.image_files_right)))

        # match stereo pairs
        frame_ids_left = [os.path.splitext(os.path.basename(x))[0] for x in self.image_files_left]
        frame_ids_right = [os.path.splitext(os.path.basename(x))[0] for x in self.image_files_right]
        stereo_frames = []
        for i, fid_left in enumerate(frame_ids_left):
            if fid_left in frame_ids_right:
                j = frame_ids_right.index(fid_left)
                stereo_frames.append({'left': self.image_files_left[i],'right': self.image_files_right[j]})
        num_matches = len(stereo_frames)
        if num_matches == 0:
            raise Exception('stereo frames were not found!')

        print('----- found {} stereo_frames\n'.format(num_matches))
        if num_matches != len(frame_ids_left) or num_matches != len(frame_ids_right):
            raise Exception('frame with no stereo match!')

        # get intrinsic calibration
        if not os.path.isfile(intrinsic_calibration_file_left):
            raise Exception('camera intrinsics file: {} not fond!'.format(intrinsic_calibration_file_left))
        if not os.path.isfile(intrinsic_calibration_file_right):
            raise Exception('camera intrinsics file: {} not fond!'.format(intrinsic_calibration_file_right))
        self.cam_left = pinhole_camera.PinholeCamera(intrinsic_calibration_file_left)
        self.cam_right = pinhole_camera.PinholeCamera(intrinsic_calibration_file_right)

        # prepare object points
        self.calibration_board = ccu.CalibrationBoard(calibration_board_file)

        # detect corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        criteria_sub_pix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        sub_pix_win_size = (9, 9)
        sub_pix_zero_zone = (-1, -1)

        left_stereo_frames = [x['left'] for x in stereo_frames]
        objpoints_left_tmp, imgpoints_left_tmp, img_width, img_height = ccu.detect_corners(left_stereo_frames, self.calibration_board, criteria,
                       sub_pix_criteria=criteria_sub_pix, sub_pix_win_size=sub_pix_win_size, sub_pix_zero_zone=sub_pix_zero_zone,
                                                                         draw=True)
        right_stereo_frames = [x['right'] for x in stereo_frames]
        objpoints_right_tmp, imgpoints_right_tmp, img_width, img_height = ccu.detect_corners(right_stereo_frames, self.calibration_board, criteria,
                       sub_pix_criteria=criteria_sub_pix, sub_pix_win_size=sub_pix_win_size, sub_pix_zero_zone=sub_pix_zero_zone,
                                                                         draw=True)
        self.image_width = img_width
        self.image_height = img_height
        # sanity check
        if self.image_width != self.cam_left.image_size[0] or self.image_height != self.cam_left.image_size[1]:
            raise Exception('image size ({},{}) does not fit pinhole camera {}!'.format(self.image_width, self.image_height, intrinsic_calibration_file_left))
        if self.image_width != self.cam_right.image_size[0] or self.image_height != self.cam_right.image_size[1]:
            raise Exception('image size ({},{}) does not fit pinhole camera {}!'.format(self.image_width, self.image_height, intrinsic_calibration_file_right))
        if num_matches != len(objpoints_left_tmp) or num_matches != len(imgpoints_left_tmp) \
            or num_matches != len(objpoints_right_tmp) or num_matches != len(imgpoints_right_tmp):
            raise Exception('internal error! objpoints or imgpoints do not fit the number of matching stereo frame')

        # keep only frames where chessboard was detected in both left and right images
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        for i in range(len(stereo_frames)):
            if objpoints_left_tmp[i] is not None and objpoints_right_tmp[i] is not None:
                objpoints.append(objpoints_left_tmp[i])
                imgpoints_left.append(imgpoints_left_tmp[i])
                imgpoints_right.append(imgpoints_right_tmp[i])
        print('----- detected {} stereo_frames\n'.format(len(objpoints)))

        #-------------------- calibrate stereo extrinsics ----------------------
        # calibrate
        flags_stereo = 0
        flags_stereo |= cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        retS, mtxL, distL, mtxR, distR, Rot, Trns, Emat, Fmat, per_view_errors = cv2.stereoCalibrate(objpoints,
                                                                                    imgpoints_left, imgpoints_right,
                                                                                    self.cam_left.K, self.cam_left.distortion_coefficients,
                                                                                    self.cam_right.K, self.cam_right.distortion_coefficients,
                                                                                    (img_width, img_height),
                                                                                    R=None, T=None, E=None, F=None, perViewErrors=None,
                                                                                    criteria=criteria_stereo,
                                                                                    flags=flags_stereo)
        num_images = len(objpoints)

        # re-projection error
        reprojection_error_per_image_2d = np.linalg.norm(per_view_errors, axis=1)
        print('reprojection error per frame:')
        for i, r in enumerate(reprojection_error_per_image_2d):
            print('frame {}: {:.3f} pixels'.format(i, r))

        # find inliers
        is_inlier = ccu.detect_reprojection_outliers(reprojection_error_per_image_2d, mahalabobis_inlier_threshold)
        mean_error = np.sum(reprojection_error_per_image_2d) / len(reprojection_error_per_image_2d)
        print("total mean error: {}".format(mean_error))
        print('\n')
        objpoints_inliers = []
        imgpoint_left_inliers = []
        imgpoint_right_inliers = []
        outliers_reprojection_error = []
        for i in range(num_images):
            if is_inlier[i]:
                objpoints_inliers.append(objpoints[i])
                imgpoint_left_inliers.append(imgpoints_left[i])
                imgpoint_right_inliers.append(imgpoints_right[i])
            else:
                outliers_reprojection_error.append({'id':i, 'error':reprojection_error_per_image_2d[i]})
        print('----- {} inlier images out of {}'.format(len(objpoints_inliers), len(objpoints)))

        if draw:
            self.fig1 = plt.figure()
            ax1 = self.fig1.add_subplot(1, 1, 1)
            plt.bar(range(0, len(reprojection_error_per_image_2d)), reprojection_error_per_image_2d, width=0.8, bottom=None, align='center', data=None,  color='b')
            outlier_ids = [x['id'] for x in outliers_reprojection_error]
            outlier_errors = [x['error'] for x in outliers_reprojection_error]
            plt.bar(outlier_ids, outlier_errors, width=0.8, bottom=None, align='center', data=None, color='r')
            ax1.plot((0, len(reprojection_error_per_image_2d)), (mean_error, mean_error), linestyle='dashed', color='r')
            ax1.text(len(reprojection_error_per_image_2d), mean_error, 'rmse={:.2f}'.format(mean_error), fontsize=9, fontweight=9)
            ax1.set_xlabel('image id', fontsize=10)
            ax1.set_ylabel('reprojection error', fontsize=10)
            ax1.set_title('reprojection errors - all frames', fontsize=18)
            plt.show(block=False)


        # ----------------- re-calibrate (without outliers) -------------------
        retS, mtxL, distL, mtxR, distR, Rot, Trns, Emat, Fmat, per_view_errors = cv2.stereoCalibrate(objpoints_inliers,
                                                                                                     imgpoint_left_inliers,
                                                                                                     imgpoint_right_inliers,
                                                                                                     self.cam_left.K,
                                                                                                     self.cam_left.distortion_coefficients,
                                                                                                     self.cam_right.K,
                                                                                                     self.cam_right.distortion_coefficients,
                                                                                                     (img_height,
                                                                                                      img_height),
                                                                                                     R=None, T=None,
                                                                                                     E=None, F=None,
                                                                                                     perViewErrors=None,
                                                                                                     criteria=criteria_stereo,
                                                                                                     flags=flags_stereo)
        # re-projection error
        reprojection_error_per_image_2d = np.linalg.norm(per_view_errors, axis=1)
        print('reprojection error per frame:')
        for i, r in enumerate(reprojection_error_per_image_2d):
            print('frame {}: {:.3f} pixels'.format(i, r))

        if retS:
            reprojection_error_per_image_2d = np.linalg.norm(per_view_errors, axis=1)
            mean_reprojection_error = np.sum(reprojection_error_per_image_2d) / len(reprojection_error_per_image_2d)
            print("total mean error: {}".format(mean_reprojection_error))
            self.stereo_R = Rot.transpose()
            self.stereo_t = -1*np.matmul(Rot.transpose(), Trns)

        if draw:
            self.fig2 = plt.figure()
            ax1 = self.fig2.add_subplot(1, 1, 1)
            plt.bar(range(0, len(reprojection_error_per_image_2d)), reprojection_error_per_image_2d, width=0.8, bottom=None, align='center', data=None)
            ax1.plot((0, len(reprojection_error_per_image_2d)), (mean_error, mean_error), linestyle='dashed', color='r')
            ax1.text(len(reprojection_error_per_image_2d), mean_error, 'rmse={:.2f}'.format(mean_error), fontsize=9, fontweight=20)
            ax1.set_xlabel('image id', fontsize=10)
            ax1.set_ylabel('reprojection error', fontsize=10)
            ax1.set_title('reprojection errors - inlier images', fontsize=18)

            # draw all image points
            self.fig3 = plt.figure()
            ax2 = self.fig3.add_subplot(1, 1, 1)
            ax2.set_xlabel('pixel X', fontsize=10)
            ax2.set_ylabel('pixel Y', fontsize=10)
            ax2.set_title('left image points', fontsize=18)
            clr_idx = np.uint8(np.round(np.linspace(0,250,len(imgpoint_left_inliers))))
            for i, p in enumerate(imgpoint_left_inliers):
                clr = plt.cm.hsv(clr_idx[i])
                # print('color {}: {} = {}'.format(i,clr_idx[i],clr))
                ax2.scatter(p[:,:,0], p[:,:,1], color=clr)

            self.fig4 = plt.figure()
            ax3 = self.fig4.add_subplot(1, 1, 1)
            ax3.set_xlabel('pixel X', fontsize=10)
            ax3.set_ylabel('pixel Y', fontsize=10)
            ax3.set_title('right image points', fontsize=18)
            clr_idx = np.uint8(np.round(np.linspace(0, 250, len(imgpoint_right_inliers))))
            for i, p in enumerate(imgpoint_right_inliers):
                clr = plt.cm.hsv(clr_idx[i])
                # print('color {}: {} = {}'.format(i, clr_idx[i], clr))
                ax3.scatter(p[:, :, 0], p[:, :, 1], color=clr)

            if fig_save_folder is not None:
                if not(os.path.isdir(fig_save_folder)):
                    os.makedirs(fig_save_folder)
                fig1_file = os.path.join(fig_save_folder, 'calibration_reprojection_errors_all.png')
                self.fig1.savefig(fig1_file)
                fig2_file = os.path.join(fig_save_folder, 'calibration_reprojection_errors_inliers.png')
                self.fig2.savefig(fig2_file)
                fig3_file = os.path.join(fig_save_folder, 'calibration_image_points_left.png')
                self.fig3.savefig(fig3_file)
                fig4_file = os.path.join(fig_save_folder, 'calibration_image_points_right.png')
                self.fig4.savefig(fig4_file)

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

        return self.stereo_R, self.stereo_t

    def save_results(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        stereo_calibration_file = os.path.join(output_dir, 'stereo_calibration.yaml')
        data = {
            'cam0': self.cam_left.id,
            'cam1': self.cam_right.id,
            'T_c1_to_c0': [[float(self.stereo_R[0,0]), float(self.stereo_R[1,0]), float(self.stereo_R[1,0]), float(self.stereo_t[0,0])],
                           [float(self.stereo_R[1,0]), float(self.stereo_R[1,1]), float(self.stereo_R[1,1]), float(self.stereo_t[1,0])],
                           [float(self.stereo_R[2,0]), float(self.stereo_R[1,2]), float(self.stereo_R[1,2]), float(self.stereo_t[2,0])],
                           [0, 0, 0, 1]]
        }
        with open(stereo_calibration_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=None, sort_keys=False)

        return

    def __del__(self):
        cv2.destroyAllWindows()

def reprojection_error(image_points, world_points, rvecs, tvecs, camera_intrinsic_matrix, camera_distortions_model):

    n = len(image_points)

    if len(world_points) != n or len(rvecs) != n or len(tvecs) != n:
        Exception('invalid input size!')

    reprojection_errors = []
    mean_reprojection_error = []
    for i in range(n):
        image_points_projected, _ = cv2.projectPoints(world_points[i], rvecs[i], tvecs[i], camera_intrinsic_matrix, camera_distortions_model)
        error_xy = (image_points[i] - image_points_projected).squeeze()
        reprojection_errors.append(error_xy)
        mean_reprojection_error.append(np.mean(abs(error_xy), axis=0))

    return reprojection_errors, mean_reprojection_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate stereo cameras.")
    parser.add_argument("--left_images_folder", help="Left camera image directory.",)
    parser.add_argument("--right_images_folder", help="Right camera image directory.",)
    parser.add_argument("--output_dir", help="Output directory.",)
    parser.add_argument("--calibration_board_file", help="calibration board file.",)
    parser.add_argument("--intrinsic_file_left", help="intrinsic calibration file for camera left.",)
    parser.add_argument("--intrinsic_file_right", help="intrinsic calibration file for camera right.",)
    args = parser.parse_args()

    left_images_folder = args.left_images_folder
    right_images_folder = args.right_images_folder
    output_dir = args.output_dir
    calibration_board_file = args.calibration_board_file
    intrinsic_calibration_file_left = args.intrinsic_file_left
    intrinsic_calibration_file_right = args.intrinsic_file_right

    output_dir = 'results/stereo_calibration'
    input_folder = './examples/stereo_calibration'
    left_images_folder = os.path.join(input_folder, 'stereo', 'left')
    right_images_folder = os.path.join(input_folder, 'stereo', 'right')
    intrinsic_calibration_file_left = os.path.join(input_folder, 'camera_intrinsics_left.yaml')
    intrinsic_calibration_file_right = os.path.join(input_folder, 'camera_intrinsics_right.yaml')
    calibration_board_file =  os.path.join(input_folder,'calibration_chessboard.yaml')

    print('calibrating')
    csc = CameraStereoCalibrator()
    R,t = csc.calibrate(left_images_folder, right_images_folder,
                  intrinsic_calibration_file_left, intrinsic_calibration_file_right,
                  calibration_board_file,
                  draw=True, fig_save_folder=output_dir, mahalabobis_inlier_threshold=3)

    T = np.vstack((np.hstack((R, t)), [0,0,0,1]))
    print('transform point from came right to cam left:\n{}'.format(T))
    csc.save_results(output_dir)
