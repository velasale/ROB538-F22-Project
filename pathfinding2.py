import time
from collections import deque
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

small_orchard = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [-20, -10, -10, -20, -20, -10, -10, -20],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

small_row8 = [-20, -10, -10, -20, -20, -10, -10, -20]


class WeightedGraph():
    def __init__(self, orchard_map, row_description) -> None:
        self.orchard_map = orchard_map
        self.row_description = row_description
        self.obstacles = []
        self.create_checklist()
        pass

    def in_bounds(self, id) -> bool:
        (x, y) = id
        return 0 <= x < len(self.row_description) and 0 <= y < len(self.orchard_map)

    def passable(self, id):
        return id not in self.obstacles

    def create_checklist(self):
        tree_checklist = []
        for i in range(np.shape(self.orchard_map)[0]):
            for j in range(len(self.row_description)):
                if self.orchard_map[i][j] != -20 and self.orchard_map[i][j] != 0:
                    tree_checklist.append((i, j))
        self.obstacles = tree_checklist

    def neighbors(self, id):
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)]  # E W N S
        # see "Ugly paths" section for an explanation:
        if (x + y) % 2 == 0:
            neighbors.reverse()  # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results


class PathFindingASTAR():
    def __init__(self, graph, start) -> None:
        frontier = deque()

        pass


class PathFindingBF():
    def __init__(self) -> None:
        pass

    def breadth_first_search(self, graph, start, goal):
        frontier = deque()
        frontier.append(start)
        came_from = {}
        came_from[start] = None

        while (len(frontier) != 0):
            current = frontier.popleft()
            if current == goal:  # early exit
                break
            for next in graph.neighbors(current):
                if next not in came_from:
                    frontier.append(next)
                    came_from[next] = current
        return came_from


# g = WeightedGraph(small_orchard, small_row8)
# bf = PathFindingBF()

# start = (0, 1)
# goal = (7, 10)

# for i in g.neighbors((goal)):
#     print(i)
# d = bf.breadth_first_search(g, start, goal)

# for i in d:
#     if
# print(d)

def create_checklist():
    tree_checklist = []
    for i in range(np.shape(small_orchard)[0]):
        for j in range(len(small_row8)):
            if small_orchard[i][j] != -20 and small_orchard[i][j] != 0:
                tree_checklist.append((i, j))
    return np.array(tree_checklist)


def create_checklist2():
    tree_checklist = np.copy(small_orchard)
    for i in range(np.shape(small_orchard)[0]):
        for j in range(len(small_row8)):
            if small_orchard[i][j] == -20 or small_orchard[i][j] == 0:
                tree_checklist[i][j] = 1
    return tree_checklist


orch = create_checklist2()
grid = Grid(matrix=orch)


start1 = time.time()

finder = AStarFinder()
grid = Grid(matrix=orch)
for i in range(1):
    start = grid.node(0, 3)
    end = grid.node(7, 10)
    path, _ = finder.find_path(start, end, grid)

print(path)
end1 = time.time()
print(end1 - start1)
