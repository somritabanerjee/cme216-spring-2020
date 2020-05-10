---
layout: page
title: Computational Graph
--- 

Write your answers in a PDF and upload the document on [gradescope](https://www.gradescope.com/courses/102338) for submission. Each question is worth 10 points. Post on [Slack](https://stanford.enterprise.slack.com/) for questions.

Late day policy: 1 late day with a 20% grade penalty.

The next questions refer to video "4.3 Forward Mode."

Consider the following function:

$$ y_1(x_1) = x_1^2 $$

$$ y_2(y_1,x_2) = y_1 + x_2 $$

$$ y_3(y_2) = \sin(y_2) $$

1. Using a sequence of calculations corresponding to forward-mode AD, calculate

$$ \frac{\partial y_3}{\partial x_1} \quad \text{and} \quad \frac{\partial y_3}{\partial x_2} $$

Consider the figure in slide [17](https://ericdarve.github.io/cme216-spring-2020/Slides/AD/AD.pdf#page=17)/47. Denote the input $$x$$ on the left and the output $$y$$ on the right. We will denote $$z_i$$ the output of node $$i$$. The nodes are numbered from 1 to $$n$$, going from left to right. Each node corresponds to some arithmetic operation. 

This graph is very special. This is a linear graph. Most graphs are more complicated but we will use this graph to illustrate a few concepts.

{:start="2"}
1. Show that

$$ \frac{\partial z_{i+1}}{\partial x} = \frac{\partial z_{i+1}}{\partial z_i} \frac{\partial z_i}{\partial x} $$

