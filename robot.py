import asyncio
import time

import numpy as np
import pygame
import PIL.Image
import cozmo
import cv2
from threading import Thread
from environment import *


class RobotController:
    def __init__(self, environment):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_marker_width = 6# cm

        self.camera_matrix = None
        self.camera_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self.environment = environment

        self.robot = None

        pygame.init()
        self.display = pygame.display.set_mode((320, 240))

    def init_cozmo(self):
        self.robot.camera.image_stream_enabled = True
        self.robot.camera.color_image_enabled = True
        f_x = self.robot.camera.config.focal_length.x
        f_y = self.robot.camera.config.focal_length.y
        c_x = self.robot.camera.config.center.x
        c_y = self.robot.camera.config.center.y

        self.camera_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    def capture(self) -> None:
        img = None

        while img is None:
            if self.robot.world.latest_image is not None:
                img = np.array(self.robot.world.latest_image.raw_image)

        cv2.imwrite('captured/latest.png', img)

    def process_image(self, image):
        cv2_img = np.array(image)

        (corners, ids, rejected) = cv2.aruco.detectMarkers(cv2_img, self.aruco_dict, parameters=self.aruco_params)

        cv2.aruco.drawDetectedMarkers(cv2_img, corners, ids)

        if self.camera_matrix is not None:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.aruco_marker_width, self.camera_matrix,
                                                                self.camera_distortion)

            if tvec is not None:
                for i in range(rvec.shape[0]):
                    cv2.drawFrameAxes(cv2_img, self.camera_matrix, self.camera_distortion, rvec[i], tvec[i], 10)

                    rot_mat, _ = cv2.Rodrigues(rvec[i])
                    cam_rot_mat = np.transpose(rot_mat)

                    cam_trans_vec = np.negative(cam_rot_mat).dot(tvec[i][0])

                    self.environment.robot_pose = Pose2D(-cam_trans_vec[0], cam_trans_vec[2], 0)

        processed_img = PIL.Image.fromarray(cv2_img)

        self.display.blit(
            pygame.image.frombuffer(processed_img.tobytes('raw', 'RGB'), (320, 240), 'RGB'),
            (0, 0))
        pygame.display.update()

    def run(self, robot: cozmo.robot.Robot):
        self.robot = robot
        self.init_cozmo()

        while True:
            if robot.world.latest_image is not None:
                self.capture()
                self.process_image(robot.world.latest_image.raw_image)


if __name__ == '__main__':
    # CREATE ENVIRONMENT
    obstacle_list = [
        Obstacle(Pose2D(10, 10, 0), 2.5),
        Obstacle(Pose2D(5, 3, 0), 2.5)
    ]

    aruco_list = [
        Aruco(Pose2D(0, 10, 0), 0),
    ]

    environment = Environment(100, obstacle_list, aruco_list)

    # RUN COZMO
    robot_controller = RobotController(environment)
    cozmo_thread = Thread(target=cozmo.run_program, args=[robot_controller.run])
    cozmo_thread.start()

    # START WEBSOCKET SERVER
    environment.start_server()

