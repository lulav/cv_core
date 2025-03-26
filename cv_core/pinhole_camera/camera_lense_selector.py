import numpy as np

"""
size in mm for each sensor format
"""
SENSOR_FORMATS = {
    '1/4"': [3.6, 2.7],
    '1/3"': [4.8, 3.6],
    '1/2.6"': [5.7, 3.6],
    '1/2.5': [5.76, 4.29],
    '1/1.8"': [7.18, 5.32],
    '2/3"': [8.80,	6.60],
    '3/4"': [18, 13.5],
    '1.1"': [14.10, 10.30]
    }


def get_camera_fov(sensor_format, focal_lengths=[0.76, 1.17, 1.7, 1.8, 2, 2.1, 2.8, 3, 3.5, 3.6, 4, 4.5, 5, 6, 7, 8, 8.5, 10, 12, 13.5, 16, 25, 35, 30, 40, 50],
                   sensor_resolution=None, roi=None):
    """
    calculate camera FOV for each lense focal length given a specific sensor format

    :param sensor_format: sensor format. must be one of the SENSOR_FORMATS keys
    :param focal_length: list of focal lengths in [mm]
    :param sensor_resolution: full sensor resolution
                              if None, full ROI is assumed
    :param roi: usable image ROI
                if None, full ROI is assumed
    :return: fov in deg
    """
    sensor_resolution = np.array(sensor_resolution).flatten()
    if sensor_resolution.size != 2:
        raise Exception('invalid sensor_resolution')

    roi = np.array(roi).flatten()
    if roi.size != 2:
        raise Exception('invalid roi')

    focal_lengths = np.array(focal_lengths).flatten()

    if sensor_format not in SENSOR_FORMATS.keys():
        raise Exception('sensor format {} not supported!'.format(sensor_format))
    sensor_size_mm = SENSOR_FORMATS[sensor_format]

    # calc ROI in mm
    if (sensor_resolution is not None) and (roi is not None):
        roi_ratio = np.divide(np.array(roi).flatten(), np.array(sensor_resolution).flatten())
    else:
        roi_ratio = 1
    roi_size_mm = np.multiply(sensor_size_mm, roi_ratio)

    fovs = []
    for f in focal_lengths:                
        fov = 2 * np.arctan2(roi_size_mm / 2, f) * 180 / np.pi
        fovs.append(fov)

    return np.array(fovs)


if __name__ == "__main__":

    # ------------------ interceptor drone 03.11.2024 -------------
    # camera - rasberrypi camera v2
    # lenses - ardubam lense set
    sensor_format = '1/4"'
    sensor_resolution = [3280, 2464]
    roi = [3280, 2464]

    focal_lengths=[0.76, 1.17, 1.7, 1.8, 2, 2.1, 2.8, 3, 3.5, 3.6, 4, 4.5, 5, 6, 7, 8, 8.5, 10, 12, 13.5, 16, 25, 35, 30, 40, 50]
    fov = get_camera_fov(sensor_format, focal_lengths=focal_lengths, sensor_resolution=sensor_resolution, roi=roi)

    if sensor_format not in SENSOR_FORMATS.keys():
        raise Exception('sensor format {} not supported!'.format(sensor_format))
    sensor_size_mm = SENSOR_FORMATS[sensor_format]
    print('sensor: {}, ({} X {})mm'.format(sensor_format, sensor_size_mm[0], sensor_size_mm[1]))
    print('resolution: {} X {} pixels'.format(sensor_resolution[0], sensor_resolution[1]))
    if roi is not None:
        print('roi: {} X {} pixels'.format(roi[0], roi[1]))

    print('lense fov:')
    for i in range(fov.shape[0]):
        print('focal length {:.2f}mm - fov = ({:.2f} X {:.2f})deg'.format(focal_lengths[i], fov[i, 0], fov[i, 1]))

