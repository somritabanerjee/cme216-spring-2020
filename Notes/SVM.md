---
layout: page
title: Support vector machines
---

### Supervised Learning

We start this lecture with a short introduction about machine learning and supervised learning. For more information, please read Chapter 5 in [Deep Learning](https://www.deeplearningbook.org/). There are broadly speaking two main tasks in machine learning that we will be concerned about.

1. The first one is classification. In this problem, we assume that for each point $$x = (x_1, \dots, x_d)$$ we need to assign a label, which is typically an integer. It may represent for example the type of object represented in an image. In engineering, $$x$$ may represent the result of some experimental measurement for a bridge for example, and the label may represent whether we believe there is a fracture in the structure. Or similarly, $$x$$ may be a time-resolved signal, for example the vibrations of a mechanical part, and the label may represent a mechanical failure: $$+1$$ is a failure is detected, $$-1$$ otherwise.

2. The second one is regression. In that case, we may be interested in some function $$u(x) \in \mathbb R$$ which is real-valued. It may be a scalar quantity like the pressure, a vector loike the velocity, or a tensor like the stress or strain in a solid. This case is more common in engineering.

We are going to start with the problem of regression for simplicity.

In the figure below, the points on the top left in red have a label $$+1$$ and the ones in the bottom right have a label $$-1$$. In this case, $$x = (x_1,x_2)$$ is a vector in $$\mathbb R^2$$.

![labels](2020-03-22-14-54-15.png)

### Support Vector Machines

Consider the problem of separating a data set using a line or hyperplane. 



On one side the label is $$1$$ and the other $$-1$$.



What is the best hyperplane we can find?

The hyperplane is defined by a normal vector $w$ and a bias $b$:

$$ w^\top x + b $$

The best hyperplane is the one that has the largest margin, that is the points are the farthest from the plane. This makes the classifier more robust and accurate:

![](2019-01-09-09-18-20.png){width=50%}

Let's now see how we can calculate the distance of a point to this hyperplane. Take a point $x$ not on the hyperplane. How far is it from the hyperplane?

Exercise: prove that

$$ \delta = \frac{|w^\top x + b|}{\|w\|} $$

We need to search for $(w,b)$ that make $\delta$ as large as possible.

The division by $\|w\|$ indicates that there is a scaling invariance in this problem. We can multiply $w$ and $b$ by any constant $C$ and the hyperplane/classifier is the same. 

To normalize we choose $w$ and $b$ such that

- $y_i (w^\top x_i + b) \ge 1$ for all $i$
- and there must be some $i$ for which $y_i (w^\top x_i + b) = 1$. This happens for the point(s) that is closest to the plane. This point is the one used for normalization.

$$ \rho(w,b) = \min \frac{|w^\top x_i + b|}{\|w\|} $$

becomes

$$ \rho(w,b) = \frac{1}{\|w\|} $$

with our normalization.

To solve this problem, we can re-write it as a quadratic programming problem (we don't need to know what this is in details but what matters is that there efficient methods to solve this type of problems):

$$ (w,b) = \text{argmin}_{w,b} \frac{1}{2} \|w\|_2^2 $$

with the constraint

$$ y_i (w^\top x_i + b) \ge 1 $$

With the optimal solution, there must be at least one $x_i$ for which

$$ y_i (w^\top x_i + b) = 1 $$

These $x_i$s are called **support vectors.**

*End of lecture 3*

<!-- For next year, 2020, add discussion on the approximation-generalization tradeoff.

More complex set of approximation functions -> better chance of approximating the ideal classifier/function

Less complex -> better chance of generalizing to new data (out of sample)

GATech slides
14-model-selection-validation -->

### Soft-margin

For many problems, though, because of noise and complex data it is not possible to have a hyperplane that exactly separates the data. In that case, there is no solution to our optimization problem.

So we need to introduce slack variables so that some constraints can be violated but in a minimal way

$$ y_i (w^\top x_i + b) \ge 1 - \xi_i $$

with $\xi_i \ge 0$. It will be zero if the constraint is satisfied. If $\xi_i > 1$ then the data is **misclassified.** Hopefully this only happens for very few points.

New optimization problem:
 
$$ (w,b,\xi) = \text{argmin}_{w,b,\xi} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \xi_i $$

$$ y_i (w^\top x_i + b) \ge 1 - \xi_i $$

$$ \xi_i \ge 0 $$

$C$ is set by the user and determines how much slack we are allowing. 

Large $C$ means little violation. Very few points are allowed to violate. The hyperplane is strongly determined by the points nearest to the hyperplane. The hyperplane is very sensitive to points that violate.

Small $C$ means a lot of violation are possible. Small $C$ is required when data has a lot of noise that needs to be filtered out. In that case, many violations will be accepted as long as this leads to a large separation $1/\|w\|$.

![](2019-01-09-11-05-55.png){width=50%}

### Optimization

We need to take a small detour through optimization methods to solve our problem.

We will just recall the main results without justifications in the interest of time.

$$ \text{min}_x f(X) $$

$$ g_i(x) \le 0 \qquad h_i(x) = 0 $$

Introduce Lagrangian

$$ L(x;\lambda,\nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_i \nu_i h_i(x) $$

Then we look for the solution in the form:

$$ \min_x \max_{\lambda, \nu; \lambda_i \ge 0} L(x;\lambda,\nu) $$

This is called the primal formulation of the optimization problem.

Note that if $x$ does not satisfy the constraints, then 

$$ \max_{\lambda, \nu; \lambda_i \ge 0} L(x;\lambda,\nu) = +\infty $$

So certainly when computing $\min_x$ we are not going to select those points. However, the min max formulation is important to connect this problem to the dual problem.

## Dual formulation

It turns out that we can formulate this problem not in terms of the primal variables $x$ but in terms of the Lagrange multipliers $\lambda$, $\nu$ (the dual variables). This leads to a dual formulation of the problem:

$$ \max_{\lambda, \nu; \lambda_i \ge 0} \min_x L(x;\lambda,\nu) $$

This is the formulation we will be using later on.

### Karush-Kuhn-Tucker (KKT) conditions

For our classification problem with slack the solution satisfies KKT conditions. They are:

$$ \partial_x L = \partial_{\nu_i} L = 0 $$

This means

$$ \partial_x f + \sum_i \lambda_i \partial_x g_i(x) + \sum_i \nu_i \partial_x h_i(x) = 0 $$

$$ h_i(x) = 0 $$

For $\lambda_i$ it's a bit more complicated. Either the optimal solution satisfies

$$ g_i(x) < 0 $$

in which case $\lambda_i = 0$. But if $g_i(x) = 0$ then the Lagrange multiplier must be "activated" and $\lambda_i > 0$ (the sign of $\lambda_i$ is determined by the sign of the constraint $g_i \le 0$).

So the final conditions are:

$$ g_i \le 0 \qquad \lambda_i \ge 0 \qquad \lambda_i g_i = 0 $$

### SVM primal optimization

We want to solve

$$ (w,b,\xi) = \text{argmin}_{w,b,\xi} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \xi_i $$

$$ y_i (w^\top x_i + b) \ge 1 - \xi_i $$

$$ \xi_i \ge 0 $$

With the Lagrange multipliers:

$$ L = \frac{1}{2} w^\top w + C \sum_{i=1}^n \xi_i$$

$$ + \sum_i \lambda_i (
    1 - \xi_i - y_i (w^\top x_i + b)) - \sum_i \nu_i \xi_i $$

### Solution 

We use the dual formulation for this. First step is taking the minimum over $w$, $b$ and $\xi_i$. Then we will take the maximum over the Lagrange multipliers.

Let's find equations satisfied by the solution.

- $\partial_w$: $w - \sum_i \lambda_i y_i x_i = 0$
- $\partial_b$: $\sum_i \lambda_i y_i = 0$
- $\partial_{\xi_i}$: $C - \lambda_i - \nu_i = 0$

After taking the minimum, the Lagrangian is

$$ L = \frac{1}{2} \sum_{ij} \lambda_i \lambda_j y_i y_j x_i^\top x_j + \sum_i \lambda_i (1 - y_i (\sum_j \lambda_j y_j x_j^\top) x_i) $$

$$ L = - \frac{1}{2} \sum_{ij} \lambda_i \lambda_j y_i y_j x_i^\top x_j + \sum_i \lambda_i $$

$$ \max_{\lambda_i \ge 0, \nu_i \ge 0} L $$

$$ \sum_i \lambda_i y_i = 0, \qquad
\lambda_i + \nu_i = C $$

which can be further simplified to just

$$ \max_{0 \le \lambda_i \le C} - \frac{1}{2} \sum_{ij} \lambda_i \lambda_j y_i y_j x_i^\top x_j + \sum_i \lambda_i $$

$$ \sum_i \lambda_i y_i = 0 $$

<!-- $$ y_i (w^\top x_i + b) \ge 1 - \xi_i \qquad \xi_i \ge 0 $$ -->
<!-- $$ \lambda_i (1 - \xi_i - y_i (w^\top x_i + b)) = 0 $$ -->
<!-- $$ \nu_i \xi_i = 0 $$ -->

*End of lecture 4*

### Training, validation, and testing set

![](2019-01-18-11-54-23.png){width=50%}

Hypothesis set: set of all models that can be used to explain the data.

$\hat{R}(h^\star)$: best model best on observing a sample of the data.

$R(h^\sharp)$: assuming a prior distribution and computing posterior probabilities based on given observations. Optimal if the prior is known.

$R(h^\star)$: error from generalization

In our case, we can vary $C$ to change the "richness". Small $C$ = poor hypothesis set. Large $C$ = overfitting, violations are minimal. $R(h^\star)$ is large.

Goal is to vary $C$ until $R(h^\star)$ starts increasing.

The optimization has essentially two phases:

- $(w,b,\xi)$: optimized based on training set
- $C$: optimized based on validation set

Definitions:

- training set: The sample of data used to fit the model. This is used to calculate $(w,b,\xi)$. If we do data fitting using a polynomial of order $n$, we use the training set to calculate the coefficients of the polynomial.

- Validation set: The sample of data used to provide an unbiased evaluation of a model fit on the training dataset, while tuning model hyperparameters. The evaluation becomes more biased as the error feedback from the validation dataset is incorporated into the model hyper-parameters. This corresponds to tuning $C$ or the polynomial order $n$.

- Test set: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset. No more optimization is done at this stage. This is performed outside of the optimization loop.

Pseudo-code:

```Python
# split data
data = ...
train, validation, test = split(data)

# tune model hyperparameters
parameters = ...
for params in parameters:
    model = fit(train, params)
    # optimize the model based on train
    error = evaluate(model, validation)
    # evaluate the model on validation
    # use error to optimize the hyperparameters

# evaluate final model using test
error = evaluate(model, test)
```

### Approximation-generalization tradeoff

Another terminology that describes something very similar.

Say we want to approximate $\sin(x)$. We are given some samples and we fit a polynomial, for example a constant, linear or higher order.

![](2019-01-18-12-25-37.png){width=50%}

Left: approximation by a constant. Right: approximation by a linear function based on 2 samples.

![](2019-01-18-12-37-58.png){width=50%}

$\bar{g}(x)$: average prediction. Gray region: variance of prediction depending on the training set. Bias: difference between $\bar{g}$ and truth. Variance: variance in prediction due to random training set. Left: constant polynomials. Right: linear polynomials.

**Approximation:** how accurately can your model approximate $f$? That is considering polynomials of order $n$, how closely can they approximate $\sin(x)$?

Typically you want a large $n$ to get a better approximation.

**Generalization:** the problem is you only see a few sample points, not the entire function. This leads to two errors: $E_{in}$ errors for the sample points you have seen. $E_{out}$: errors for other sample points that you have not seen. This is basically the training and validation sets. We say that a model generalizes well if $E_{in} \sim E_{out}$.

Typically you want a small $n$ for the model to generalize.

You need to find the right trade-off. The optimal $n$ depends on how many sample points you are allowed to see. The more sample points the larger $n$ should be.

![](2019-01-18-12-30-00.png){width=50%}

### Kernel trick

A genius idea. You can see that the Lagrangian is based on dot products with $x_i$: $x_i^\top x_j$. The prediction requires

$$ w^\top x + b = \sum_i (\lambda_i y_i) \; x_i^\top x + b $$

and similarly in the definition of $L(\lambda,\nu)$.

The vectors $x_i$ only show up through dot products.

This suggests several extensions.

Instead of considering $x$, we can consider some non-linear function of $x$, $\phi(x)$, called features. Then the dot products $x^\top x'$ become

$$ \phi(x)^\top \phi(x') = \sum_k \phi_k(x) \phi_k(x')$$

We may decide to use a weighted dot product

$$ \sum_k \sigma_k \phi_k(x) \phi_k(x') $$

The downside of this is that we need to calculate all these $\phi_k(x)$. But there is a simple way around this. Define the kernel function

$$ K(x,x') =  \sum_{k=1}^r \sigma_k \phi_k(x) \phi_k(x') $$

This is a simple evaluation; no dot product or function $\phi_k$ to evaluate any more.

But in fact, many kernels we are familiar with can be written in this form. It's just an SVD of the kernel. Any kernel that satisfies the assumption for Mercer's theorem ($K$ is a continuous symmetric positive semi-definite kernel; we skip the technical details here) can be written as

$$ K(x,x') =  \sum_{k=1}^\infty \sigma_k \phi_k(x) \phi_k(x') $$

The optimization Lagrangian becomes:

$$ L = - \frac{1}{2} \sum_{ij} \lambda_i \lambda_j y_i y_j K(x_i,x_j) + \sum_i \lambda_i $$

To evaluate our classifier, we use

$$ \sum_i (\lambda_i y_i) \; K(x,x_i) + b $$

Observe how this is very close to the Gaussian process regression estimate:

$$ f(x) = \sum_i \alpha_i K(x,x_i) + \mu(x) $$

Although different, kernel machines for SVM and GPR are closely related. In SVM, the parameters are obtained through minimization. In GPR, we are parameter-free and the estimate is obtained through averaging (expectation).

*End of lecture 5*