import cozmo
import cv2
import os
from threading import Thread
from environment import *


class VisionLocalizer:
    def __init__(self, environment: Environment):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_16H5)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_marker_width = 5  # cm
        self.board_box_width = 1
        self.board_marker_width = 0.8

        self.camera_matrix = np.zeros((3, 3))
        self.camera_distortion = np.zeros((5, 1))
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_pose = Pose2D(0, 0, 0)
        self.environment = environment

    def get_board(self):
        board = cv2.aruco.CharucoBoard_create(7, 7, self.board_box_width, self.board_marker_width, self.aruco_dict)
        return board

    def write_board(self):
        img = self.get_board().draw((2000, 2000))
        cv2.imwrite('boards/board.png', img)

    def read_boards(self, path):
        all_corners = []
        all_ids = []

        images = [path + f for f in os.listdir(path)]

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        decimator = 0

        for image in images:
            frame = cv2.imread(image)

            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = cv2.aruco.detectMarkers(
                grayscale_frame, self.aruco_dict, parameters=self.aruco_params)

            if len(corners) > 0:
                # SUB PIXEL DETECTION
                for corner in corners:
                    cv2.cornerSubPix(grayscale_frame, corner,
                                     winSize=(3, 3),
                                     zeroZone=(-1, -1),
                                     criteria=criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, grayscale_frame, self.get_board())
                if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                    all_corners.append(res2[1])
                    all_ids.append(res2[2])

            decimator += 1

        return all_corners, all_ids, grayscale_frame.shape

    def calibrate_camera(self, all_corners, all_ids, imsize):
        camera_matrix = np.array([[4500, 0., imsize[0] / 2.],
                                  [0., 4500, imsize[1] / 2.],
                                  [0., 0., 1.]])

        distortion_coefficients = np.zeros((5, 1))

        flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
                 cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)

        (_, camera_matrix, distortion_coefficient, _, _, _, _, _) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=self.get_board(),
            imageSize=imsize,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        return camera_matrix, distortion_coefficients

    def run_calibrate(self) -> None:
        all_corners, all_ids, imsize = self.read_boards('captured/')

        self.camera_matrix, self.camera_distortion = self.calibrate_camera(all_corners, all_ids, imsize)

    def process_image(self, image):
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)

        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        if self.camera_matrix is not None:
            rotation_vec, translation_vec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.aruco_marker_width, self.camera_matrix,
                                                                  self.camera_distortion)
            if rotation_vec is not None:
                for i in range(rotation_vec.shape[0]):
                    cv2.drawFrameAxes(image, self.camera_matrix, self.camera_distortion, rotation_vec[i, :, :], translation_vec[i, :, :], 4)

                    print(translation_vec)
                    self.environment.robot_pose = Pose2D(translation_vec[i][0][0], -translation_vec[i][0][1], 0)

    def run(self):
        while True:
            _, image = self.cap.read()
            self.process_image(image)
            cv2.imshow('CAM', image)
            if cv2.waitKey(1) == ord('q'):
                break


class RobotController:
    def __init__(self):
        self.robot = None

    def run(self, robot: cozmo.robot.Robot):
        self.robot = robot


if __name__ == '__main__':
    # CREATE ENVIRONMENT
    obstacle_list = [
        Obstacle(Pose2D(10, 10, 0), 2.5),
        Obstacle(Pose2D(5, 3, 0), 2.5)
    ]

    environment = Environment(100, obstacle_list)
    print('ENVIRONMENT INITIALIZED')

    # CREATE PLANNER, LOCALIZER, CONTROLLER
    rrt_planner = RRTPlanner(environment)
    vision_localizer = VisionLocalizer(environment)
    robot_controller = RobotController()

    # CALIBRATE
    vision_localizer.write_board()
    vision_localizer.run_calibrate()
    print('CAMERA CALIBRATED')

    # RUN COZMO
    cozmo_thread = Thread(target=cozmo.run_program, args=[robot_controller.run])
    cozmo_thread.start()
    print('COZMO STARTED')

    # START WEBSOCKET SERVER
    websocket_thread = Thread(target=environment.start_server)
    websocket_thread.start()
    print('WEBSOCKET SERVER STARTED')

    # START LOCALIZATION
    vision_localizer.run()
    print('LOCALIZATION SERVER STARTED')
