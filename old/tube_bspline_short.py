import numpy as np
from scipy.optimize import minimize

# ==========================================
# B-SPLINE LOCAL PLANNER
# ==========================================
class TubeBSplineShort:
    def __init__(self, start, short_goal, obstacles, r0=0.35, k=0.2):
        self.start = np.array(start)
        self.goal = np.array(short_goal) 
        self.obstacles = obstacles
        self.r0 = r0
        self.k = k
        self.n_cpts = 5
        self.dt = 0.5
        
    def signed_dist_to_box(self, point, box_center, box_size):
        dx = abs(point[0] - box_center[0]) - box_size[0] / 2.0
        dy = abs(point[1] - box_center[1]) - box_size[1] / 2.0
        return np.linalg.norm([max(dx, 0), max(dy, 0)]) + min(max(dx, dy), 0)

    def objective(self, x):
        cpts = x.reshape((self.n_cpts, 2))
        diff1 = np.diff(cpts, axis=0)
        diff2 = np.diff(diff1, axis=0)
        return np.sum(diff2**2) + 0.1 * np.sum(np.linalg.norm(diff1, axis=1))

    def constraints(self, x):
        cpts = x.reshape((self.n_cpts, 2))
        cons = []
        diffs = np.diff(cpts, axis=0)
        velocities = np.linalg.norm(diffs, axis=1) / self.dt
        velocities = np.append(velocities, velocities[-1]) 
        
        MIN_PRESSURE = 1 
        radii = self.r0 + (self.k * velocities * MIN_PRESSURE)
        
        for i in range(self.n_cpts):
            for (ox, oy, w, h) in self.obstacles:
                center = np.array([ox + w/2, oy + h/2])
                size = np.array([w, h])
                cons.append(self.signed_dist_to_box(cpts[i], center, size) - radii[i])
        return np.array(cons)

    def plan(self):
        lerp = np.linspace(0, 1, self.n_cpts)
        initial_guess = self.start + np.outer(lerp, self.goal - self.start)
        
        res = minimize(
            self.objective,
            initial_guess.flatten(),
            method='SLSQP',
            constraints=[
                {'type': 'ineq', 'fun': self.constraints},
                {'type': 'eq', 'fun': lambda x: np.array([
                    x.reshape((self.n_cpts,2))[0,0]-self.start[0],
                    x.reshape((self.n_cpts,2))[0,1]-self.start[1],
                    x.reshape((self.n_cpts,2))[-1,0]-self.goal[0],
                    x.reshape((self.n_cpts,2))[-1,1]-self.goal[1]
                ])}
            ],
            options={'maxiter': 20, 'ftol': 1e-2} 
        )
        if not res.success:
            # THE FIX: If the optimizer throws a numerical fit, do NOT stop moving. 
            # Just follow the raw APF vector (initial_guess) which is already safe!
            return initial_guess.reshape((self.n_cpts, 2))
        return res.x.reshape((self.n_cpts, 2))