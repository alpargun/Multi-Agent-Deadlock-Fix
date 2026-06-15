import numpy as np
import math
import random

random.seed(42)

class Node:
    """
    Represents a single point (node) in the RRT* tree.
    Tracks its position, the cost to reach it from the start, and its parent node.
    """
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.cost = 0.0
        self.parent = None

class RRTStarBridge:
    """
    RRT* Planner with Bridge Sampling.
    Standard RRT struggles with narrow passages because uniform random sampling
    rarely picks points inside small gaps. Bridge sampling actively hunts for narrow
    corridors by finding two obstacle points and checking if the space between them is free.
    """
    def __init__(self, start, goal, obstacles, rand_area, expand_dis=0.5, goal_sample_rate=0.1, bridge_std=1.0):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis             # How far to step toward a sampled point
        self.goal_sample_rate = goal_sample_rate # Bias to sample the goal directly
        self.bridge_std = bridge_std             # Spread of the Gaussian used for the second bridge point
        self.obstacles = obstacles
        self.node_list = []

    def check_collision_point(self, x, y):
        """
        Checks if a specific (x,y) coordinate is inside any rectangular obstacle.
        Includes a small safety buffer so the robot's center doesn't graze the wall.
        """
        buffer = 0.2
        for (ox, oy, w, h) in self.obstacles:
            if (ox - buffer <= x <= ox + w + buffer) and (oy - buffer <= y <= oy + h + buffer):
                return True # Collision (Point is inside obstacle)
        return False # Safe (Point is in free space)

    def check_collision_edge(self, node1, node2):
        """
        Samples points along the line segment between two nodes to ensure 
        the branch does not cut through or jump over an obstacle.
        """
        steps = 10 # Check 10 points along the line
        for i in range(steps + 1):
            t = i / float(steps)
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            if self.check_collision_point(x, y):
                return True # Edge hit a wall!
        return False # Edge is safe

    def get_random_node(self):
        """
        Generates a new target point for the tree to grow towards.
        Mixes Goal Biasing, Bridge Sampling, and Uniform Sampling.
        """
        # Strategy 1: Goal Biasing (Pull the tree toward the goal)
        if random.random() < self.goal_sample_rate:
            return Node(self.goal.x, self.goal.y)
            
        # Strategy 2: Bridge Sampling (Search for narrow passages)
        if random.random() < 0.5: # 50% chance to attempt bridge sampling
            x1 = random.uniform(self.min_rand, self.max_rand)
            y1 = random.uniform(self.min_rand, self.max_rand)
            
            # If the first point is inside an obstacle, check nearby
            if self.check_collision_point(x1, y1):
                # Generate a second point nearby using a Gaussian distribution
                x2 = random.gauss(x1, self.bridge_std)
                y2 = random.gauss(y1, self.bridge_std)
                
                # If the second point is ALSO in an obstacle...
                if self.check_collision_point(x2, y2):
                    # Check the midpoint between the two obstacle points
                    xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    
                    # If the midpoint is in FREE space, we found a "bridge" (a gap)!
                    if not self.check_collision_point(xm, ym):
                        return Node(xm, ym) 

        # Strategy 3: Fallback Uniform Sampling (Explore open areas)
        while True:
            x = random.uniform(self.min_rand, self.max_rand)
            y = random.uniform(self.min_rand, self.max_rand)
            if not self.check_collision_point(x, y):
                return Node(x, y)

    def plan(self, max_iter=1500):
        """Builds the tree and optimizes connections via RRT* rewiring."""
        self.node_list = [self.start]
        for i in range(max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            
            # Ensure the very first branch growth doesn't cut through a wall
            if not self.check_collision_edge(self.node_list[nearest_ind], new_node):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)
                    
        goal_ind = self.search_best_goal_node()
        if goal_ind is not None:
            return self.generate_final_course(goal_ind)
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Takes a bounded step toward the target node."""
        new_node = Node(from_node.x, from_node.y)
        d = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        
        if extend_length > d: extend_length = d
            
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.cost = from_node.cost + extend_length
        new_node.parent = from_node
        return new_node
    
    def get_nearest_node_index(self, node_list, rnd_node):
        return np.argmin([(n.x - rnd_node.x)**2 + (n.y - rnd_node.y)**2 for n in node_list])
        
    def find_near_nodes(self, new_node):
        """Finds all existing tree nodes within a dynamically shrinking search radius."""
        nnode = len(self.node_list) + 1
        r = 5.0 * math.sqrt((math.log(nnode) / nnode)) 
        return [i for i, n in enumerate(self.node_list) if (n.x - new_node.x)**2 + (n.y - new_node.y)**2 <= r**2]
        
    def choose_parent(self, new_node, near_inds):
        """Evaluates all nearby nodes and picks the one providing the cheapest valid path."""
        if not near_inds: return None
        costs = []
        for i in near_inds:
            n = self.node_list[i]
            t_node = self.steer(n, new_node)
            # Check the entire edge, not just the endpoint
            if t_node and not self.check_collision_edge(n, t_node):
                costs.append(n.cost + math.hypot(n.x - new_node.x, n.y - new_node.y))
            else:
                costs.append(float("inf"))
                
        min_cost = min(costs)
        if min_cost == float("inf"): return None
        
        new_node = self.steer(self.node_list[near_inds[costs.index(min_cost)]], new_node)
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        """Checks if routing nearby nodes through the new node reduces their total cost."""
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node: continue
            
            edge_node.cost = new_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
            # Check the entire edge when rewiring
            if near_node.cost > edge_node.cost and not self.check_collision_edge(new_node, near_node):
                near_node.parent = new_node
                near_node.cost = edge_node.cost

    def search_best_goal_node(self):
        """Finds the node with the lowest cost that reached the goal region."""
        goal_inds = [i for i, n in enumerate(self.node_list) if math.hypot(n.x - self.goal.x, n.y - self.goal.y) <= self.expand_dis]
        if not goal_inds: return None
        return goal_inds[np.argmin([self.node_list[i].cost for i in goal_inds])]

    def generate_final_course(self, goal_ind):
        """Traces the parent pointers backwards to build the final path."""
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return np.array(path[::-1])