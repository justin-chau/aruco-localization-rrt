from __future__ import annotations

import asyncio
import websockets
import numpy as np
import json
import random
from typing import List, Optional
import time
import cv2
from scipy.spatial.transform import Rotation


class Rotation3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self._rotation = Rotation.from_euler('xyz', [x, y, z])
        self.mat = self._rotation.as_matrix()

    @staticmethod
    def from_rotation_vec(rotation_vec) -> Rotation3D:
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        angles = Rotation.from_matrix(rotation_mat).as_euler('xyz')

        return Rotation3D(angles[0], angles[1], angles[2])

    @staticmethod
    def from_rotation_mat(rotation_mat) -> Rotation3D:
        angles = Rotation.from_matrix(rotation_mat).as_euler('xyz')

        return Rotation3D(angles[0], angles[1], angles[2])

    def get_yaw(self):
        return self._rotation.as_euler('xyz')[2]


class Pose3D:
    def __init__(self, x: float, y: float, z: float, rotation: Rotation3D):
        self.x = x
        self.y = y
        self.z = z
        self.rotation = rotation
        self.mat = np.vstack((np.hstack((self.rotation.mat, [[self.x], [self.y], [self.z]])), [0, 0, 0, 1]))

    def compose(self, other: Pose3D) -> Pose3D:
        result_matrix = np.matmul(self.mat, other.mat)

        x = result_matrix[0][3]
        y = result_matrix[1][3]
        z = result_matrix[2][3]

        rotation_mat = result_matrix[0:3, 0:3]

        return Pose3D(x, y, z, Rotation3D.from_rotation_mat(rotation_mat))

    def inverse(self) -> Pose3D:
        inverse = np.linalg.inv(self.mat)

        return Pose3D(inverse[0][3], inverse[1][3], inverse[2][3], Rotation3D.from_rotation_mat(inverse[0:3, 0:3]))


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

    def pos(self):
        return np.array([self.x, self.y])

    def distance(self, other: Pose2D):
        return np.linalg.norm(self.pos() - other.pos())

    def is_within_tolerance(self, other: Pose2D, tolerance: float):
        distance = self.distance(other)

        if distance <= tolerance:
            return True

        return False

    def step(self, towards: Pose2D, step: float):
        vector = towards.pos() - self.pos()
        length = np.linalg.norm(vector)

        if length < step:
            return towards

        direction_vector = vector / length
        step_vector = direction_vector * step

        stepped_pose = self.pos() + step_vector

        return Pose2D(stepped_pose[0], stepped_pose[1], 0)


class Segment2D:
    def __init__(self, start: Pose2D, end: Pose2D):
        self.start = start
        self.end = end


class Obstacle:
    def __init__(self, pose: Pose2D, radius: float):
        self.pose = pose
        self.radius = radius

    def is_pose_inside(self, pose: Pose2D):
        return self.pose.is_within_tolerance(pose, self.radius)

    # TODO: FIX THIS ALGORITHM, CURRENTLY IT CHECKS THE WHOLE LINE, NOT A SEGMENT
    def does_segment_intersect(self, segment: Segment2D):
        start = segment.start.pos()
        end = segment.end.pos()
        pos = self.pose.pos()

        distance = np.abs(np.linalg.norm(np.cross(end - start, start - pos))) / np.linalg.norm(end - start)
        if distance <= self.radius:
            return True

        return False

    def dict_rep(self):
        dict_rep = {
            'radius': self.radius,
            'pose': self.pose.dict_rep()
        }

        return dict_rep


class Environment:
    def __init__(self, x: float, y: float, obstacle_list: List[Obstacle]):
        self.x_dim = x
        self.y_dim = y
        self.obstacle_list = obstacle_list
        self.robot_pose = Pose2D(0, 0, 0)  # ID 1
        self.goal_pose = Pose2D(72, 40, 0)
        self.world_T_camera = None

        self.BOTTOM_LEFT_ID = 4
        self.BOTTOM_RIGHT_ID = 3
        self.TOP_RIGHT_ID = 2

        self.world_T_bottom_left = Pose3D(40, 25, 0, Rotation3D(0, 0, 0))  # ID 4
        self.world_T_bottom_right = Pose3D(77.25, 3.1, 0, Rotation3D(0, 0, 0))  # ID 3
        self.world_T_top_right = Pose3D(77.25, 47.25, 0, Rotation3D(0, 0, 0))  # ID 2

        self.is_at_goal: bool = False
        self.path: List[Pose2D] = []

    def set_goal(self, goal: Pose2D):
        self.goal_pose = goal
        self.is_at_goal = False

    def add_random_obstacles(self, number: int, min_radius: float, max_radius: float):
        for i in range(number):
            self.obstacle_list.append(Obstacle(self.random_pose(), random.uniform(min_radius, max_radius)))

    def random_pose(self):
        rand_x = random.uniform(0, self.x_dim)
        rand_y = random.uniform(0, self.y_dim)

        return Pose2D(rand_x, rand_y, 0)

    def biased_random_pose(self, biased_pose: Pose2D, biased_chance: float):
        chance = random.random()

        if chance < biased_chance:
            return biased_pose

        return self.random_pose()

    def is_pose_free(self, pose: Pose2D):
        for obstacle in self.obstacle_list:
            if obstacle.is_pose_inside(pose):
                return False

        return True

    def is_segment_free(self, segment: Segment2D):
        for obstacle in self.obstacle_list:
            if obstacle.does_segment_intersect(segment):
                return False

        return True

    def json(self):
        obj = {
            'x': self.x_dim,
            'y': self.y_dim,
            'obstacle_list': [obstacle.dict_rep() for obstacle in self.obstacle_list],
            'robot_pose': self.robot_pose.dict_rep(),
            'goal_pose': self.goal_pose.dict_rep(),
            'path': [pose.dict_rep() for pose in self.path]
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


class RRTPlanner:
    def __init__(self, environment: Environment):
        self.tree: List[RRTNode] = []
        self.STEP_SIZE = 2
        self.GOAL_BIAS_CHANCE = 0.3
        self.GOAL_TOLERANCE = 2
        self.environment = environment

    def nearest_node(self, pose: Pose2D):
        closest = None
        min_distance = np.inf

        for node in self.tree:
            distance = node.pose.distance(pose)
            if distance < min_distance:
                min_distance = distance
                closest = node

        return closest

    def generate_tree(self, goal: Pose2D, time_limit: bool = False):
        root_node = RRTNode(self.environment.robot_pose, None)
        current_node = root_node
        self.tree.append(current_node)
        start_time = time.time()

        while not current_node.pose.is_within_tolerance(goal, self.GOAL_TOLERANCE):
            if time_limit and time.time() > start_time + 5:
                raise Exception('TREE EXCEEDED TIME LIMIT')

            random_pose = self.environment.biased_random_pose(goal, self.GOAL_BIAS_CHANCE)

            nearest_node = self.nearest_node(random_pose)
            stepped_pose = nearest_node.pose.step(random_pose, self.STEP_SIZE)

            nearest_to_step_segment = Segment2D(nearest_node.pose, stepped_pose)

            if self.environment.is_pose_free(stepped_pose) and self.environment.is_segment_free(
                    nearest_to_step_segment):
                current_node = RRTNode(stepped_pose, nearest_node)
                self.tree.append(current_node)

        # Returns the root node and the goal node
        return root_node, current_node

    def recover_path(self, root: RRTNode, goal: RRTNode):
        path: List[Pose2D] = []
        current = goal

        while current is not root:
            path.insert(0, current.pose)
            current = current.parent

        return path

    # def smooth_path(self, path):
    #     if len(path) == 1:
    #         return path
    #
    #     current_pose = self.path[0]
    #     smoothed_path = [current_pose]
    #
    #     for future in range(len(path) - 1, 1, -1):
    #         future_pose = path[future]
    #
    #         current_future_segment = Segment2D(current_pose, future_pose)
    #
    #         if self.environment.is_segment_free(current_future_segment):
    #             smoothed_path.extend(self.smooth_path(path[future: len(path)]))
    #             return smoothed_path
    #
    #     smoothed_path.extend(self.smooth_path(path[1: len(path)]))
    #     return smoothed_path

    def plan(self, time_limit: bool = False):
        try:
            self.tree = []
            root, goal = self.generate_tree(self.environment.goal_pose, time_limit=time_limit)
            path = self.recover_path(root, goal)
            self.environment.path = path
        except Exception as e:
            print(e)

    def start(self):
        while True:
            if self.environment.is_pose_free(self.environment.robot_pose):
                self.plan(time_limit=True)


class RRTNode:
    def __init__(self, pose: Pose2D, parent: Optional[RRTNode]):
        self.pose = pose
        self.parent = parent
