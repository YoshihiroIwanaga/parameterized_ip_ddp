# Parameterized Interior-Point iterative Linear Quadratic Regulator (P-IP-iLQR) and Differential Dynamic Programming (P-IP-DDP)

This repository contains an implementations of the P-IP-iLQR and P-IP-DDP. 
The computational cost is reduced by introducing a parametric representation of the control inputs, such as zero-order, linear, and cubic interpolations. 
General nonlinear constraints are handled through the Interior-Point DDP framework [1]. 
This algorithm combines the parametric iLQR (DDP) with the IP-iLQR (IP-DDP) (a Python reimplementation of [this](https://github.com/xapavlov/ipddp)).


## Motivation

The IP-iLQR and IP-DDP algorithm are powerful tool for solving constrained nonlinear optimal control problems. 
To decrease the computational cost, we reduce the dimensionality of the decision variables by introducing a parametric representation of the control inputs. 
In this framework, only the control inputs at specific knot points are optimized, while the remaining control inputs are determined by interpolation functions (zero-order, linear, and cubic).


The general nonlinear constraints are managed using the Interior-Point iLQR (DDP) framework. 
By default, constraints are only considered at each knot point. 
However, this may lead to constraint violations between knot points. 
To address this, we introduce additional constraint points (where constraints are explicitly evaluated, apart from the knots) and margins to ensure constraint satisfaction across the entire prediction horizon.

## Installation
-------------
To use this repository, simply clone it. You'll need to install the required packages, Numpy and Scipy, using the following commands:
```
pip install numpy
```
```
pip install scipy
```

## Examples
To run the car trajectory planning example, use the following command:
```
./scripts/car/run_car_traj_plan.bash
```

The animation below shows the optimization progress. 
The left panel displays the xy-path, while the panels on the right show the time series of states and control inputs. 
The dotted red line represents the original IP-iLQR, and the blue line represents the zero-order-interpoalted P-IP-iLQR. 
As shown, P-IP-iLQR reaches the optimal solution faster than the original IP-iLQR.

![Image](https://github.com/user-attachments/assets/6259bc93-728f-4123-a77f-487735dd17c8)

To run the constraint points and margin finding example, use the following command:
```
./scripts/car/find_margin_and_cp.bash
```
 

## Citations
[1] A. Pavlov, I. Shames and C. Manzie, "Interior Point Differential Dynamic Programming," in IEEE Transactions on Control Systems Technology, vol. 29, no. 6, pp. 2720-2727, Nov. 2021.  


