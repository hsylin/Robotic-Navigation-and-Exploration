import cv2
import sys
import heapq
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.open_set = []
        self.closed_set = set()
        self.parent = {}
        self.h = {} 
        self.g = {} 
        self.goal_node = None
        self.moves = [ (0, 1), (0, -1), (1, 0), (-1, 0),  
                  (1, 1), (1, -1), (-1, 1), (-1, -1)
                ]

    def planning(self, start=(100,200), goal=(375,520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize 
        self.initialize()
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)
        heapq.heappush(self.open_set,  (self.g[start]+self.h[start],start))
        
        # A star algorithm
        while self.open_set:
            current_cost, current_node = heapq.heappop(self.open_set)
            if current_node == goal:
                self.goal_node = current_node
                break
            if current_node in self.closed_set:
                continue
            else:
                self.closed_set.add(current_node)
          
            for dx, dy in self.moves:
                neighbor_node = (current_node[0] + dx, current_node[1] + dy)
                
                if neighbor_node in self.closed_set:
                    continue
                if neighbor_node[1]<0 or neighbor_node[1]>=self.map.shape[0] or neighbor_node[0]<0 or neighbor_node[0]>=self.map.shape[1] :
                    continue
                if self.map[int(neighbor_node[1]),int(neighbor_node[0])]<0.5 :
                    continue
                
                neighbor_g_score = self.g[current_node] + utils.distance(current_node, neighbor_node)
                neighbor_h_score = utils.distance(neighbor_node,goal)
                neighbor_f_score =neighbor_g_score + neighbor_h_score

                if neighbor_node in self.g and neighbor_node in self.h :
                    if neighbor_g_score >= self.g[neighbor_node]:
                        continue
                    
                self.g[neighbor_node] = neighbor_g_score
                self.h[neighbor_node] = neighbor_h_score
                heapq.heappush(self.open_set, (neighbor_f_score, neighbor_node))
                self.parent[neighbor_node] = current_node


        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while(True):
            path.insert(0,p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
