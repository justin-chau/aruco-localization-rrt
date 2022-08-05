import cozmo
import os
from threading import Thread

from environment import *


class VisionLocalizer:
    def __init__(self, environment: Environment):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_calibration_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_16H5)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_marker_width = 6.2  # cm
        self.board_box_width = 1
        self.board_marker_width = 0.8

        self.camera_matrix = np.zeros((3, 3))
        self.camera_distortion = np.zeros((5, 1))
        self.cap = cv2.VideoCapture(1)
        self.cap.read()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -1)
        self.environment = environment
        self.camera_pose_estimates: List[Pose3D] = []
        self.camera_complete_estimates = set()

    def get_board(self):
        board = cv2.aruco.CharucoBoard_create(7, 7, self.board_box_width, self.board_marker_width, self.aruco_calibration_dict)
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
                grayscale_frame, self.aruco_calibration_dict, parameters=self.aruco_params)

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
        camera_matrix = np.array([[1000, 0., imsize[0] / 2.],
                                  [0., 1000, imsize[1] / 2.],
                                  [0., 0., 1.]])

        distortion_coefficients = np.zeros((5, 1))

        flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
                 cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)

        (_, camera_matrix, distortion_coefficients, _, _, _, _, _) = cv2.aruco.calibrateCameraCharucoExtended(
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

                    cv2.drawFrameAxes(image, self.camera_matrix, self.camera_distortion, rotation_vec[i, :, :],
                                      translation_vec[i, :, :], 4)

                    rotation = Rotation3D.from_rotation_vec(rotation_vec[i, :, :])
                    camera_T_marker = Pose3D(translation_vec[i][0][0], translation_vec[i][0][1],
                                            translation_vec[i][0][2], rotation)

                    if ids[i][0] in [4]:
                        if self.environment.world_T_camera is None:

                            marker_T_camera = camera_T_marker.inverse()

                            world_T_camera = Pose3D(0, 0, 0, Rotation3D(0,0,0))

                            if ids[i][0] == 4:
                                world_T_camera = self.environment.world_T_bottom_left.compose(marker_T_camera)

                            # if ids[i][0] == 3:
                            #     world_T_camera = self.environment.world_T_bottom_right.compose(marker_T_camera)
                            #
                            # if ids[i][0] == 2:
                            #     world_T_camera = self.environment.world_T_top_right.compose(marker_T_camera)

                            if ids[i][0] not in self.camera_complete_estimates:
                                self.camera_complete_estimates.add(ids[i][0])
                                self.camera_pose_estimates.append(world_T_camera)

                            if len(self.camera_pose_estimates) == 1:
                                total = np.zeros((4, 4))

                                # for camera_pose_estimate in self.camera_pose_estimates:
                                #     total += camera_pose_estimate.mat
                                #
                                # total = total / 3
                                #
                                # final_world_T_camera = Pose3D(total[0][3], total[1][3], total[2][3], Rotation3D.from_rotation_mat(total[0:3, 0:3]))
                                self.environment.world_T_camera = self.camera_pose_estimates[0]


                    if ids[i][0] == 1:
                        if self.environment.world_T_camera is not None:
                            world_T_robot = self.environment.world_T_camera.compose(camera_T_marker)
                            self.environment.robot_pose = Pose2D(world_T_robot.x, world_T_robot.y, world_T_robot.rotation.get_yaw())

    def run(self):
        while True:
            _, image = self.cap.read()
            self.process_image(image)
            cv2.imshow('CAM', image)
            if cv2.waitKey(1) == ord('q'):
                break


class RobotController:
    def __init__(self, rrt_planner: RRTPlanner, environment: Environment):
        self.robot = None
        self.OFF_PATH_TOLERANCE = 10  #TODO: CHANGE THIS TOLERANCE TO MAKE SENSE
        self.ANGLE_TOLERANCE = np.deg2rad(10)
        self.POSITION_TOLERANCE = 2
        self.ROTATION_SPEED = 20
        self.STRAIGHT_SPEED = 20
        self.rrt_planner = rrt_planner
        self.environment = environment

    def stop(self):
        self.robot.drive_wheel_motors(0, 0)

    def rotate(self, angle):
        self.robot.drive_wheel_motors(-self.ROTATION_SPEED * angle, self.ROTATION_SPEED * angle)

    def drive_straight(self):
        # TODO: ADD P CONTROLLER
        self.robot.drive_wheel_motors(self.STRAIGHT_SPEED, self.STRAIGHT_SPEED)

    def run(self, robot: Optional[cozmo.robot.Robot] = None):
        self.robot = robot

        while True:
            if self.environment.is_at_goal is False:
                print('GENERATING RRT')

                try:
                    self.rrt_planner.plan(time_limit=True)
                    print('RRT GENERATED')
                    path = iter(self.environment.path)
                    current_target = next(path)

                    while True:
                        # print('TARGET: ', current_target.pos())
                        if self.environment.robot_pose.is_within_tolerance(self.environment.goal_pose,
                                                                           self.POSITION_TOLERANCE):
                            # The robot is close enough to goal, so we stop.
                            self.stop()
                            print('ROBOT ARRIVED AT GOAL')
                            environment.is_at_goal = True
                            break

                        if not current_target.is_within_tolerance(self.environment.robot_pose, self.OFF_PATH_TOLERANCE):
                            # The target is too far, so we re-plan from current pose.
                            self.stop()
                            print('ROBOT IS OFF PATH, RECOMPUTING RRT')
                            break

                        if current_target.is_within_tolerance(self.environment.robot_pose, self.POSITION_TOLERANCE):
                            # Robot pose is close enough to the target, so we set the next target in path.
                            try:
                                current_target = next(path)
                            except:
                                # Something went wrong and the robot exhausted the path, but has not reached goal.
                                # We re-plan a new path.
                                print('ROBOT DID NOT REACH GOAL, BUT PATH IS COMPLETE. RETRYING...')
                                break

                        # Calculate the angle from robot_pose to current_pose
                        delta_pos = current_target.pos() - self.environment.robot_pose.pos()
                        world_rotation_to_next = np.arctan2(delta_pos[1], delta_pos[0])
                        delta_rotation = ((world_rotation_to_next - np.pi / 2) - self.environment.robot_pose.theta)

                        if np.abs(delta_rotation) > self.ANGLE_TOLERANCE:
                            print('ROTATION ERROR: ', delta_rotation, 'TOLERANCE: ', self.ANGLE_TOLERANCE)
                            self.rotate(delta_rotation)
                        elif not self.environment.robot_pose.is_within_tolerance(current_target, self.POSITION_TOLERANCE):
                            print('STRAIGHT ERROR: ', self.environment.robot_pose.distance(current_target))
                            self.drive_straight()
                        else:
                            self.stop()

                except Exception as e:
                    print(e)
                    print('COULD NOT GENERATE RRT IN TIME')


if __name__ == '__main__':
    # CREATE ENVIRONMENT
    obstacle_list = [
        Obstacle(Pose2D(29.3, 15.24, 0), 12),
        Obstacle(Pose2D(29.3, 6.35, 0), 12),
        Obstacle(Pose2D(46.99, 41.91, 0), 12),
        Obstacle(Pose2D(54.61, 34.29, 0), 12)
    ]

    environment = Environment(80, 50.165, obstacle_list)
    # environment.add_random_obstacles(10, 3, 5)
    print('ENVIRONMENT INITIALIZED')

    # CREATE PLANNER, LOCALIZER, CONTROLLER
    rrt_planner = RRTPlanner(environment)
    vision_localizer = VisionLocalizer(environment)
    robot_controller = RobotController(rrt_planner, environment)

    # CALIBRATE
    vision_localizer.write_board()
    vision_localizer.run_calibrate()
    print('CAMERA CALIBRATED')

    # RUN COZMO
    cozmo_thread = Thread(target=cozmo.run_program, args=[robot_controller.run])
    cozmo_thread.start()
    print('COZMO STARTED')

    # RUN WITHOUT COZMO
    # robot_thread = Thread(target=robot_controller.run)
    # robot_thread.start()

    # START WEBSOCKET SERVER
    websocket_thread = Thread(target=environment.start_server)
    websocket_thread.start()
    print('WEBSOCKET SERVER STARTED')

    # START LOCALIZATION SERVER
    vision_localizer.run()
    print('LOCALIZATION SERVER STARTED')
