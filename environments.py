import numpy as np

class DynamicAgent:
    def __init__(self, id, start, goal, color):
        self.id = id
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.color = color
        self.global_path = None
        self.random_base_angle = np.random.uniform(0, 2*np.pi)

        # Minimum rotation speed ensures a full 360 sweep in ~4 seconds
        sign = np.random.choice([-1, 1])
        self.random_drift_rate = sign * np.random.uniform(1.5, 3.0)

def get_scenario(scenario_name):
    """Returns obstacles and agent configurations for a given map."""
    # Corridor
    if scenario_name == "corridor":
        obstacles = [
            (4.8, 5.75, 0.4, 6.25), (4.8, -2.0, 0.4, 6.25),   
            (-2.0, 11.0, 14.0, 0.5), (-2.0, -1.5, 14.0, 0.5),  
            (-1.5, -1.5, 0.5, 13.0), (11.0, -1.5, 0.5, 13.0)   
        ]
        agents = [
            DynamicAgent(1, [0.0, 5.0], [10.0, 5.0], 'blue'),
            DynamicAgent(2, [10.0, 5.0], [0.0, 5.0], 'red')
        ]
    # Maze
    elif scenario_name == "maze":
        obstacles = [
            (-2.0, 7.0, 14.0, 1.0), (-2.0, 2.0, 14.0, 1.0),   
            (3.0, 4.5, 1.0, 2.5), (6.0, 3.0, 1.5, 2.5),     
        ]
        agents = [
            DynamicAgent(1, [0.0, 5.0], [10.0, 5.0], 'blue'),
            DynamicAgent(2, [10.0, 5.0], [0.0, 5.0], 'red')
        ]
    # Intersection
    else:
        obstacles = [
            (-2.0, -2.0, 6.0, 6.0), (-2.0, 6.0, 6.0, 6.0),   
            (6.0, -2.0, 6.0, 6.0), (6.0, 6.0, 6.0, 6.0),    
        ]
        agents = [
            DynamicAgent(1, [0.0, 5.0], [10.0, 5.0], 'blue'),
            DynamicAgent(2, [10.0, 5.0], [0.0, 5.0], 'red'),
            DynamicAgent(3, [5.0, 0.0], [5.0, 10.0], 'green'),
            DynamicAgent(4, [5.0, 10.0], [5.0, 0.0], 'purple')
        ]
    return agents, obstacles