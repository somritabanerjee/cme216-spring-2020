---
layout: page
title: HW4 Solution
---
## Problem 1


**1** 

Let $$ \lambda = \kappa_i \frac{\Delta t}{\Delta x^2} $$, then we have
$$
A = \begin{bmatrix}
2\lambda_1+1 & -2\lambda_1  &  & & \\
-\lambda_2 & 2\lambda_2+1 & -\lambda_2 & & \\
 & -\lambda_3 & 2\lambda_3 + 1 & -\lambda_3 & \\
& &\ddots & & \\
& & &\ddots & -\lambda_{n-1}\\
&&& -\lambda_n & 2\lambda_n + 1
\end{bmatrix},\quad F^k = \Delta t \begin{bmatrix}
f_1^{k+1} \\
f_2^{k+1} \\
\vdots\\
f_n^{k+1}
\end{bmatrix}
$$

**2** and **3**

See `1DCase/starter1.jl`.

**4** 

$$ a=5 $$, $$ b=2 $$. See `1DCase/starter2.jl`.

## Problem 2

**5**

Let $$\kappa_\theta(x)$$ be the neural network approximation to $$\kappa(x)$$, where $$\theta$$ is the weights and biases.

$$\begin{aligned}
\min_{\theta}\ & \sum_{i=1}^{25} \int_{0}^t ( u(x_i, t)- u_0(x_i,t))^2 dt\\
\mathrm{s.t.}\ & \frac{\partial u(x, t)}{\partial t} = \kappa_\theta(x)\Delta u(x, t) + f(x, t), \quad t\in (0,T), x\in (0,1) \\
& -\kappa(0)\frac{\partial u(0,t)}{\partial x} = 0, t>0\\
& u(1, t) = 0, t>0\\
& u(x, 0) = 0, x\in [0,1]\\
\end{aligned}$$

**6** and **7**

See `1DCase/starter3_pcl.jl`. 


**8**

See `1DCase/starter3_with_noise.jl`. In general, you should observe that the number of total iterations at the time termination for a given stop criterion increases as noise becomes larger. The error in the neural network estimation (e.g., measured by average mean squared error for multiple runs) becomes larger for larger noise. 


## Problem 3

**9** 

Let $$\kappa_\theta(x)$$ be the neural network approximation to $$\kappa(x)$$, where $$\theta$$ is the weights and biases.

$$\begin{aligned}
\min_{\theta}\ & \int_0^1 ( u(0.2, 0.2, t; \theta)- u_1(t))^2 + ( u(0.8, 0.8, t; \theta)- u_1(t))^2 dt\\
\mathrm{s.t.}\ & \frac{\partial u(x,y, t;\theta)}{\partial t} = \kappa_\theta(x)\Delta u(x,y, t;\theta) + f(x,y, t), \quad t\in (0,1), (x,y)\in [0,1]^2 \\
& u(x, y, 0;\theta) = 0, \quad (x,y) \in  \Omega\\
& u(x,y,t;\theta) = 0 ,\quad (x,y)\in \partial \Omega
\end{aligned}$$

Other reasonable formuations are acceptable as well. 

**10**

See `2DCase/starter.jl`.

**11** 

$$a=2, b=3, c=3$$. See `2DCase/starter_pcl.jl`.
