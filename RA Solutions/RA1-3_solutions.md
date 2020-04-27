---
layout: page
title: Partial Solutions to Reading Assignments 1-3
---

Here are the solutions to trickier questions from reading assignments 1-3.

RA1 Q6: To uniquely define w and b, an additional constraint of $$y_i(w^T x_i + b) = 1$$ for at least one i (support vectors) is added to the optimization question.

RA2 Q7: For each value (or combination of values) of the hyperparameter(s), a model is fitted using the training set, then evaluated on the validation set based on some metric(score). The best performing combination is selected.

RA3 Q4: Without soft margin, for support vectors, $$\sum_i \alpha_i K(x^{(i)},x) + b$$ can only take on values of $$\pm 1$$