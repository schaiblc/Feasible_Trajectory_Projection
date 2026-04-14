 Feasible_Trajectory_Projection

ECE 602 Final Project - This project explores the use of nonlinear model predictive control (NMPC) for 
optimal control in autonomous path planning. An existing peer-reviewed formulation 
is examined, which employs the \textit{acados} optimal control framework, 
implementing a real-time iteration (RTI) scheme within a sequential quadratic 
programming (SQP) solver. This approach computes real-time control actions within 
20\,ms planning intervals, enabling autonomous vehicle racing on fixed-width 
closed-course tracks. The bicycle vehicle model, Frenet-frame state representation, 
and time-optimal cost formulation from the original work are reproduced and 
validated against the published results. The NMPC formulation is subsequently extended to the problem of feasible trajectory projection under obstacle avoidance constraints. By incorporating static obstacle positions as state-dependent constraints within the optimal control problem, kinematically feasible trajectories are generated in real time. This extension provides a foundation for integration into end-to-end learned path planning pipelines, where guaranteeing kinematic feasibility is otherwise difficult. Simulation results are presented for both the original and extended formulations, with performance evaluated and critically analyzed. The results demonstrate that RTI-based NMPC is capable of producing real-time solutions to nonlinear optimal control problems, with strong potential for deployment in physical autonomous systems.

Based on acados race_cars example: https://github.com/acados/acados/tree/main/examples/acados_python/race_cars
and accompanying paper: https://www.sciencedirect.com/science/article/pii/S2405896320317845
