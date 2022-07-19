import numpy
import pygame
import PIL.Image
import cozmo
import cv2

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

pygame.init()
display = pygame.display.set_mode((320, 240))


def init_robot(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True


def program(robot: cozmo.robot.Robot):
    init_robot(robot)

    while True:
        if robot.world.latest_image is not None:
            cv2_img = numpy.array(robot.world.latest_image.raw_image)

            (corners, ids, rejected) = cv2.aruco.detectMarkers(cv2_img, arucoDict, parameters=arucoParams)

            cv2.aruco.drawDetectedMarkers(cv2_img, corners, ids)

            processed_img = PIL.Image.fromarray(cv2_img)

            display.blit(
                pygame.image.frombuffer(processed_img.tobytes('raw', 'RGB'), (320, 240), 'RGB'),
                (0, 0))
            pygame.display.update()


cozmo.run_program(program)
