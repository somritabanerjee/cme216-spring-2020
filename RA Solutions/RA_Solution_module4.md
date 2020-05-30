---
layout: page
title: Partial Solutions to Reading Assignments 16-20
---

Here are the solutions to trickier questions from these reading assignments.

**RA16**

Q1: the error is proportional to $O(ux) = O(u)$.

Q2: When computing $f(x+h)$ we have already made an error in $x+h$. So we are computing in fact

$$f(x+h + O(u)) = f(x+h) + f'(x+h) O(u) = f(x+h) + O(u)$$

The roundoff error when evaluating $f$ is $O(uf)$. So in the end the total error is $O(u)$.

Q3: The numerator is $f(x+h) - f(x-h)$. We first observe that we made an error in $O(u)$ when evaluating the function $f$ numerically. Also, as $h$ gets small, $f(x+h) - f(x-h)$ goes to zero (in absence of roundoff errors). 

When $h$ is large compared to $u$, the magnitude of $f(x+h) - f(x-h) + O(u)$ is $O(f) = O(1)$. When $h$ is comparable to $u$, the numerator is of order $O(u)$. When $h$ is very small compared to $u$, we get something different. $x+h = x-h$ in that case because we only carry a limited number of digits. So in fact, the numerator is numerically exactly equal to 0.

Take $h>u$. There are now two errors when computing the ratio. There is an error in $O(u/h)$ because of the error in the numerator, and an error in $O(u/h)$ from the roundoff error when doing the division. So overall, the error is of order $O(u/h)$.

When $h \ll u$, the numerator is 0 so the error is just $O(1)$.

Q4: If $h= 10^{-6}$, and $u=10^{-8}$, we get $u/h = 10^{-2}$. So we can expect approximately two digits of accuracy. When $h = 10^{-16}$, the numerator is numerically exactly equal to 0. So the error is of order 1. We have no significant digits at all in the numerical prediction for the derivative.

Q5: $d/dx (\text{tanh}(1+x^2)) = 2x / \text{cosh}^2(1+x^2)$

Q6: $x^2+1 = 2$, $(x^2+1)' = 2xx' = 16$

Q7: 

$$\text{tanh}(1+x^2) = \text{tanh}(2) = 0.964$$

$$d/dx (\text{tanh}(1+x^2)) = 2xx' / \text{cosh}^2(1+x^2) = 16/\text{cosh}^2(2) = 1.13$$

**RA17**

Q2&3: Note the questions are referring to the generic computational graph in slide with n nodes.

For question 2, we want to traverse the graph from left to right in forward mode. Using the given chain rule relation, we can initialize the derivative to be $$ \frac{\partial y}{\partial x} = 1 $$. Then, in for a loop with i going from 1 to n, we update  $$ \frac{\partial y}{\partial x} \leftarrow \frac{\partial y}{\partial x} \times \frac{\partial z_i}{\partial z_{i-1}} $$, with $$ z_n = y, z_1 = x $$. 

For question 3, we want to traverse the graph from right to left in forward mode. Using the given chain rule relation, we can again initialize the derivative to be $$ \frac{\partial y}{\partial x} = 1 $$. Then, in for a loop with i going from n down to 1, we update  $$ \frac{\partial y}{\partial x} \leftarrow \frac{\partial y}{\partial x} \times \frac{\partial z_i}{\partial z_{i-1}} $$, with $$ z_n = y, z_1 = x $$.

Note that despite the two algorithms look very similar, the orders of computation are reversed, making each more efficient in specific cases as we see in following questions.

Q4-7: Depending on your interpretation, you might get different numeric answers. But using forward mode in one-to-many and using backward mode in many-to-one should result in shorter computation time than the other two cases, due to the ability to reuse shared intermediate values, since in these two cases only one traversal is needed for all the hidden nodes. 