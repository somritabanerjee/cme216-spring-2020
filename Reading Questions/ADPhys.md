---
layout: page
title: AD for Physical Simulations
--- 

Write your answers in a PDF and upload the document on [gradescope](https://www.gradescope.com/courses/102338) for submission. Each question is worth 10 points. Post on [Slack](https://stanford.enterprise.slack.com/) for questions.

Late day policy: 1 late day with a 20% grade penalty.

The next question refers to video "4.5 AD for Physical Simulations."

The discretization of the PDE is based on the idea of approximating a derivative using

$$ \frac{\partial u}{\partial x} \approx \frac{u(x+h) - u(x-h)}{2h} $$

If we use a uniform grid with spacing $$\Delta x$$, we get

$$ \frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_{i-1}}{2 \Delta x} $$

To evaluate a second-order derivative like 

$$\triangle u = \frac{\partial^2 u}{\partial x^2}$$

we can use

$$ \triangle u(x_i) \approx \frac{u'(x + \Delta x / 2) - u'(x - \Delta x / 2)}{\Delta x} $$

where $$u'$$ is the derivative.

1. Use the approximation of the derivative and the previous equation to show that:

$$ \triangle u(x_i) \approx \frac{u_{i+1} - 2 u_i + u_{i-1}}{\Delta x^2} $$

{:start="2"}
1. Using the equation on Slide [36](https://ericdarve.github.io/cme216-spring-2020/Slides/AD/AD.pdf#page=40), show that rows of $$A(a,b)$$ (slide [37](https://ericdarve.github.io/cme216-spring-2020/Slides/AD/AD.pdf#page=41)) are of the form (except row 1 and $$n$$):

$$ [ \hspace{2em} -\lambda_i, \quad 2 \lambda_i + 1, \quad - \lambda_i \hspace{2em}] $$

with $$\lambda_i = \kappa_i \frac{\Delta t}{\Delta x^2}$$.