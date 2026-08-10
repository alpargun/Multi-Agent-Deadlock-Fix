# Decentralized Multi-Agent Path Planning with Continuous-Time ODEs

This repository implements a completely decentralized, continuous-time navigation architecture for multi-agent systems. It guarantees mathematically collision-free aggregation without relying on inter-agent communication or discrete state-machine switching.

## How It Works

* **Global Planning (RRT*):** Each agent computes an optimal, obstacle-free baseline path using an RRT* planner enhanced with Bridge Sampling to efficiently navigate narrow passages.
* **Local Planning & Deadlock Resolution (B-Splines & APF):** Agents track their global paths using a continuous Ordinary Differential Equation (ODE). The system utilizes Kinematic Artificial Potential Fields (APF) with velocity-scaled uncertainty bubbles to guarantee safety. When a symmetric deadlock is detected (e.g., agents meeting head-on), the system deterministically injects a 2-parameter continuous B-Spline perturbation. This smoothly breaks the deadlock and guarantees an asymptotic return to the optimal path once cleared.

## Showcase and Performance

### 1. Corridor Scenario

<table>
  <tr>
    <th width="48%">Agent Performance</th>
    <th width="52%">Distance to Goal Convergence</th>
  </tr>
  <tr>
    <td valign="middle" align="center">
      <video src="https://github.com/user-attachments/assets/5264ddad-0bca-4c8a-8f90-8400159e7348" controls="controls" width="100%"></video>
    </td>
    <td valign="middle" align="center">
      <img src="output-continuous/distance_error_plot_corridor.png" width="100%">
    </td>
  </tr>
</table>

---

### 2. Maze Scenario

<table>
  <tr>
    <th width="48%">Agent Performance</th>
    <th width="52%">Distance to Goal Convergence</th>
  </tr>
  <tr>
    <td valign="middle" align="center">
      <video src="https://github.com/user-attachments/assets/37dae583-a294-4cd3-967f-e6c6d4f01d88" controls="controls" width="100%"></video>
    </td>
    <td valign="middle" align="center">
      <img src="output-continuous/distance_error_plot_maze.png" width="100%">
    </td>
  </tr>
</table>

---

### 3. Intersection Scenario

<table>
  <tr>
    <th width="48%">Agent Performance</th>
    <th width="52%">Distance to Goal Convergence</th>
  </tr>
  <tr>
    <td valign="middle" align="center">
      <video src="https://github.com/user-attachments/assets/e5a9f370-6d5d-430d-a7db-cd1eb3ed51c7" controls="controls" width="100%"></video>
    </td>
    <td valign="middle" align="center">
      <img src="output-continuous/distance_error_plot_intersection.png" width="100%">
    </td>
  </tr>
</table>
