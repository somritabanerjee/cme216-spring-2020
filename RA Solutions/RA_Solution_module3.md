---
layout: page
title: Partial Solutions to Reading Assignments 8-11
---

Here are the solutions to trickier questions from these reading assignments.

RA8 Q1: Note the question is asking for activation function, not loss function. At the output layer, use linear activation for unbounded output, use softplus if output must be positive, and use tanh or sigmoid if output is bounded in some range.

RA9 Q1: To use the CategoricalCrossentropy loss function, we need to output a probability vector for each prediction, with each element non-negative and all elements sum to 1 (note the constraint that each element less than or equal to 1 is implied by these two conditions). This can be achieved with the softmax function. Some students answered one-hot representation, wihch is the format we want for the labels, but not for our predictions.

RA11 Q3: When $$ \lambda_1 / \lambda_n $$ is close to 1, the optimal $$ \alpha $$ values for each mode are similar, therefore even if we choose the smallest of them all, all modes still converge quickly. When $$ \lambda_1 / \lambda_n $$ is large, the optimal $$ \alpha $$ values for each mode are quite different. For stability, we must choose the smallest learning rate. Note that it will not necessarily diverge if the learning rate is small enough. But the consequence of small learning rate is that some modes will take longer to converge, hence the overall conergence will be slow.