from __future__ import annotations
from typing import List

import numpy as np


class Obstacle:
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius

    def collides_with_line(self, line: Line):
        obstacle_node = Node(self.x, self.y)
        distance_to_center = line.distance_to_node(obstacle_node)

        if distance_to_center <= self.radius:
            return True

        return False


class Environment:
    def __init__(self):
        self.x_dim = 100
        self.y_dim = 100
        self.obstacles = []


class Node:
    def __init__(self, x: float, y: float):
        self.point = np.array([x, y])
        self.parent = None

    def distance(self, other: Node) -> float:
        return np.linalg.norm(self.point - other.point)


class Line:
    def __init__(self, start_node: Node, end_node: Node):
        self.start_node = start_node
        self.end_node = end_node

    def distance_to_node(self, node: Node):
        start = self.start_node.point
        end = self.end_node.point

        return np.abs(np.linalg.norm(np.cross(end - start, start - node.point))) / np.linalg.norm(end - start)


class RRT:
    def __init__(self):
        self.nodes: List[Node] = []


if __name__ == "__main__":
    obstacle = Obstacle(0, 0, 2)
    line = Line(Node(-5, 0), Node(5, 1))

    print(obstacle.collides_with_line(line))
