from __future__ import annotations

import asyncio
import websockets
import json
from typing import List
from robot import RobotController, VisionLocalizer


class Pose2D:
    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    def dict_rep(self):
        dict_rep = {
            'x': self.x,
            'y': self.y,
            'theta': self.theta
        }

        return dict_rep


class Obstacle:
    def __init__(self, pose: Pose2D, radius: float):
        self.pose = pose
        self.radius = radius

    def dict_rep(self):
        dict_rep = {
            'radius': self.radius,
            'pose': self.pose.dict_rep()
        }

        return dict_rep


class Aruco:
    def __init__(self, pose: Pose2D, aruco_id: int):
        self.pose = pose
        self.aruco_id = aruco_id

    def dict_rep(self):
        dict_rep = {
            'aruco_id': self.aruco_id,
            'pose': self.pose.dict_rep()
        }

        return dict_rep


class Environment:
    def __init__(self, length: int, obstacle_list: List[Obstacle], aruco_list: List[Aruco],
                 robot_controller: RobotController, vision_localizer: VisionLocalizer):
        self.x_dim = length
        self.y_dim = length
        self.obstacle_list = obstacle_list
        self.aruco_list = aruco_list
        self.robot_controller = robot_controller
        self.vision_localizer = vision_localizer

    def json(self):
        obj = {
            'x': self.x_dim,
            'y': self.y_dim,
            'obstacle_list': [obstacle.dict_rep() for obstacle in self.obstacle_list],
            'aruco_list': [aruco.dict_rep() for aruco in self.aruco_list],
            'robot_pose': self.vision_localizer.robot_pose.dict_rep()
        }

        return json.dumps(obj)

    async def serve(self, websocket, path):
        while True:
            await websocket.send(self.json())
            await asyncio.sleep(0.1)

    def start_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_server = websockets.serve(self.serve, 'localhost', 5555)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
