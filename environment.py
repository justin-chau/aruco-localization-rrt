from __future__ import annotations

import asyncio
import websockets
import numpy as np
import json
import random
from typing import List, Optional


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

        return Pose2D(step_vector[0], step_vector[1], 0)


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

    def does_segment_intersect(self, segment: Segment2D):
        distance = np.abs(np.linalg.norm(np.cross(segment.start.pos() - segment.end.pos(), segment.end.pos() - self.pose.pos()))) / np.linalg.norm(segment.start.pos() - segment.end.pos())
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
    def __init__(self, length: int, obstacle_list: List[Obstacle]):
        self.x_dim = length
        self.y_dim = length
        self.obstacle_list = obstacle_list
        self.robot_pose = Pose2D(0, 0, 0)
        self.goal_pose = Pose2D(0, 0, 0)
        self.rrt: List[Segment2D] = []

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
            'robot_pose': self.robot_pose.dict_rep()
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
        self.nodes: List[RRTNode] = []
        self.STEP_SIZE = 2
        self.GOAL_BIAS_CHANCE = 0.1
        self.GOAL_TOLERANCE = 2
        self.environment = environment

    def nearest_node(self, pose: Pose2D):
        closest = None
        min_distance = np.inf

        for node in self.nodes:
            distance = node.pose.distance(pose)
            if distance < min_distance:
                min_distance = distance
                closest = node

        return closest

    def plan(self, goal: Pose2D):
        self.environment.goal_pose = goal

        root = RRTNode(self.environment.robot_pose, None)
        self.nodes.append(root)

        random_pose = self.environment.biased_random_pose(goal, self.GOAL_BIAS_CHANCE)
        nearest_node = self.nearest_node(random_pose)
        stepped_pose = nearest_node.pose.step(random_pose, self.STEP_SIZE)



class RRTNode:
    def __init__(self, pose: Pose2D, parent: Optional[RRTNode]):
        self.pose = pose
        self.parent = parent
