---
layout: page
title: Partial Solutions to Reading Assignments 21-24
---

Here are the solutions to trickier questions from these reading assignments.

RA21 Q1: The form of the function is already given, what we want to determine are the coefficients, hence this is a parameter inverse problem.

RA21 Q3: In general, we cannot express a DNN as a linear combination of basis functions, due to the presence of non-linear activations.

RA21 Q4: In each dimension, we need $$ \frac{1}{h} $$ elements; since there are d dimensions, we need $$ \frac{1}{h^d} $$ elements total.

RA22 Q1: from the recurrence relation we see that $u_{k+1}$ is a function of $\theta$ and $u_k$ only. In reverse-mode, we start from $u_n$. Then we consider its parents nodes, $\theta$ and $u_{n-1}$. This gives us the two independent variables at step $n-1$. Then at every step, the parent nodes of $u_{k+1}$ are $\theta$ and $u_k$. So at step $k$, the two independent variables are $\theta$ and $u_k$.

RA22 Q2: from the hint:

$$ \frac{\partial u_{n,k}}{\partial u_k} = 
\frac{\partial u_{n,k+1}}{\partial \theta} \frac{\partial \theta}{\partial u_k} + 
\frac{\partial u_{n,k+1}}{\partial u_{k+1}} \frac{\partial u_{k+1}}{\partial u_k} $$

At this step $\theta$ and $u_k$ are the independent variables. So

$$ \frac{\partial \theta}{\partial u_k} = 0 $$

So we just get:

$$ \frac{\partial u_{n,k}}{\partial u_k} = \frac{\partial u_{n,k+1}}{\partial u_{k+1}} \frac{\partial u_{k+1}}{\partial u_k} $$

Similarly:

$$ \frac{\partial u_{n,k}}{\partial \theta} = 
\frac{\partial u_{n,k+1}}{\partial \theta} \frac{\partial \theta}{\partial \theta} + 
\frac{\partial u_{n,k+1}}{\partial u_{k+1}} \frac{\partial u_{k+1}}{\partial \theta} $$

Since $\partial \theta / \partial \theta = 1$, we get:

$$ \frac{\partial u_{n,k}}{\partial \theta} = 
\frac{\partial u_{n,k+1}}{\partial \theta} + 
\frac{\partial u_{n,k+1}}{\partial u_{k+1}} \frac{\partial u_{k+1}}{\partial \theta} $$

RA22 Q3: we use Eq. (1) to calculate the derivatives:

$$ \frac{\partial u_{k+1}}{\partial u_k} = \frac{1}{a} $$

and

$$ \frac{\partial u_{k+1}}{\partial \theta} = - \frac{a'}{a^2}(u_k + f_{k+1})
= - \frac{a' u_{k+1}}{a}
$$

RA22 Q4: No. The PDE is not actually used in that method.

RA22 Q5: in the penalty method $u_0$ and $f$ are known. The variables are $\theta$ ($\kappa_\theta$) and $u$. Since we are enforcing the PDE only through a penalty term, unless the loss is exactly 0, we cannot expect the PDE to be satisfied exactly. As $\lambda_1$ increases the error in the PDE decays. But typically it is very difficult to achieve a small error on the PDE.

RA22 Q6: (a) $\lambda_2$, (b) $\lambda_3$, (c) $\lambda_4$.


RA23 Q3: let $$ x = -2(u_\theta - u_0)^T B(\theta)^{-1} $$, we need to rearrange this into the $$ Ax=b $$ form to use the solver: $$ xB(\theta) = -2(u_\theta - u_0)^T $$, $$ B(\theta)^T x^T = -2(u_\theta - u_0) $$. So $$ A = B(\theta)^T $$ and $$ b = -2(u_\theta - u_0) $$.
