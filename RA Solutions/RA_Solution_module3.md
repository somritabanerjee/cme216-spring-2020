---
layout: page
title: Partial Solutions to Reading Assignments 8-15
---

Here are the solutions to trickier questions from these reading assignments.

RA8 Q1: Note the question is asking for activation function, not loss function. At the output layer, use linear activation for unbounded output, use softplus if output must be positive, and use tanh or sigmoid if output is bounded in some range.

RA9 Q1: To use the CategoricalCrossentropy loss function, we need to output a probability vector for each prediction, with each element non-negative and all elements sum to 1 (note the constraint that each element less than or equal to 1 is implied by these two conditions). This can be achieved with the softmax function. Some students answered one-hot representation, which is the format we want for the labels, but not for our predictions.

RA11 Q3: When $$ \lambda_1 / \lambda_n $$ is close to 1, the optimal $$ \alpha $$ values for each mode are similar, therefore even if we choose the smallest of them all, all modes still converge quickly. When $$ \lambda_1 / \lambda_n $$ is large, the optimal $$ \alpha $$ values for each mode are quite different. For stability, we must choose the smallest learning rate. Note that it will not necessarily diverge if the learning rate is small enough. But the consequence of small learning rate is that some modes will take longer to converge, hence the overall convergence will be slow.


RA13 Q5: To show this is a critical point, we  take the gradient of L: $$ \nabla L = [x_2, x_1, x_4, x_3, ...] $$. Since $$ x=0 $$, $$ \nabla L(x) = 0 $$, x is a critical point. When we take the hessian of L, we notice it is a block diagonal matrix with $$\begin{pmatrix}  0 & 1\\  1 & 0 \end{pmatrix} $$ on the diagonal. So the eigenvalues for the hessian is the same as this 2 by 2 matrix: $$ \lambda^2 - 1 = 0, \lambda = \pm1 $$. Correspondingly, the normalized eigenvectors are $$ [0, ... \frac{\pm\sqrt{2}}{2}, \frac{\sqrt{2}}{2}, ... 0]^T $$, where the pair of non-zero indices are $$(2i-1, 2i), i \in [1, n] $$. Since the hessian has both positive and negative eigenvalues, this point is a saddle point.

RA14 Q5 By definition, $$ [s^{(k+1)}]_i - [s^k]_i = (\frac{\partial L_{k+1}}{\partial x_i})^2 \ge 0 $$, hence $$ [s^{(k+1)}]_i \ge [s^k]_i $$

RA14 Q6 Since A is symmetric, it can be eigendecomposed into $$ A = U\Lambda U^T $$. Then $$ x_n = A^nx_0 = U\Lambda^n U^Tx_0 = \sum_i \lambda_i^n(u_i^Tx_0u_i) $$. When n is large, the sum is dominated by the term with the largest $$ \lambda_i $$, hence $$ x_n \approx \lambda^n(u^Tx_0)u $$.

RA14 Q7 For $$ \lambda = 0.999 $$, $$ n = \frac{\log(0.001)}{\log(0.999)} \approx 6904 $$. For $$ \lambda = 0.9 $$, $$ n = \frac{\log(0.001)}{\log(0.9)} \approx 66 $$. This shows a small change in $$ \lambda $$ can have a large impact on the convergence speed.

RA15 Q1 With the given expression, we can write $$ s_i^{(k)} = \beta^ks_i^{(0)} + (1-\beta) G_i^2 \sum_{l=0}^{k-1} \beta^l $$. The sum is is a geometric series and we can rewrite $$ s_i^{(k)} = \beta^ks_i^{(0)} + (1-\beta^k) G_i^2 $$. Since $$ 0 <\beta < 1 $$, for k large, $$ \beta^k \approx 0 $$, and $$ s_i^{(k)} = G_i^2 $$.

RA15 Q4 Towards the end, RMSProp moves down the hill faster than Adagrad. The difference is due to RMSProp's update rule on $$ s_i $$ uses a moving average, better controlling its growth.

RA15 Q5 After finding an escape direction from the saddle point, the gradients start to align in direction and grows rapidly in magnitude. Since momentum accounts for recent gradient values, this quickly accelerates the learning.