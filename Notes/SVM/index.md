---
layout: page
title: Support vector machines
---

<!-- <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~chris/1638.embed" height="525" width="100%"></iframe> -->

Support vector machine (SVM) is one of the simplest methods for classification. In some sense, it forms a stepping block to neural networks and deep learning.

SVM is a method for binary classification. That is, we are given a point $$x \in \mathbb R^n$$, and we want to predict a label with possible values $$+1$$ or $$-1$$.

In SVM, the space $$\mathbb R^n$$ of possible $$x$$ is subdivided into 2 half-space by a hyperplane: a line in 2D or a plane in 3D. On one side of the hyperplane, the label is $$1$$ and is $$-1$$ on the other.

In the example below, the line that separates the $$-1$$ labels from $$+1$$ is simply $$y = x$$. The $$-1$$ labels are in the top left and the $$+1$$ in the bottom right.

{% include svm2.html %}

However, if the training data that we are given are only the colored dots, we cannot exactly determine the separating line $$y = x$$. So instead, based on the observed data (the dots), we ask what the best hyperplane we can find is.

The hyperplane is defined by a normal vector $$w$$ and a bias $$b$$:

$$ w^T x + b = 0 $$

Once the hyperplane is found, the classification is given as follows. If $$ w^T x + b > 0 $$ then we predict that the label is $$1$$, otherwise the label is $$-1$$.

In SVM, the best hyperplane is defined as the one that has the largest **margin,** that is the one which the points are the farthest from. This makes the classifier more robust and accurate.

In the figure below, the "exact" solution is $$y = x$$ (the solid blue line), but this is unknown to us. Instead, we observe only the colored dots. Based on this, the best line of separation is the black solid line in the middle. We will see later how this can be formulated mathematically and how the black line can be computed.

The dashed lines are two lines that are parallel to the black solid line. The red dot on the top dashed line is at a distance $$h$$ from the black solid line. The blue dots on the bottom dashed line are also at a distance $$h$$ from the black solid line.

![](2020-03-22-18-07-39.png){:width="400px"}

The goal in SVM is to determine the equation of the black solid line such that the distance $$h$$ is maximum. By definition, no training points (the colored dots) can reside between the dashed lines. The black solid line must be equidistant from the two dashed lines.

Let's now see how this can be formulated mathematically.

The first step is calculating the distance of a point to the separating hyperplane (black solid line above). This hyperplane will be defined by the equation $$ w^T x + b = 0 $$ where $$w$$ is a vector in $$\mathbb R^n$$ and $$b$$ is a scalar.

Take a point $$x$$ not on the hyperplane. How far is it from the hyperplane? You can prove that the distance $$\delta$$ is

$$ \delta = \frac{|w^T x + b|}{\|w\|} $$

We then need to search for $$(w,b)$$ that makes $$\delta$$ as large as possible.

The division by $$\|w\|$$ indicates that there is a scaling invariance in this problem. We can multiply $$w$$ and $$b$$ by any constant $$C$$ and the hyperplane/classifier is the same. 

To normalize, we choose $$w$$ and $$b$$ such that

- $$y_i (w^T x_i + b) \ge 1$$ for all $$i$$, and
- there must be some $$i$$ for which $$y_i (w^T x_i + b) = 1$$. This happens for the point(s) that is closest to the plane. This point is the one used for normalization.

The condition $$y_i (w^T x_i + b) = 1$$ may look strange. But recall that when the label $$y_i = -1$$ we expect $$w^T x_i + b < 0$$, and when $$y_i = 1$$, we expect $$w^T x_i + b > 0$$. So at least in terms of the signs, this equation makes sense.

Let's denote by $$\rho$$ the distance of the point **closest** to the hyperplane. Then:

$$ \rho(w,b) = \min \frac{|w^T x_i + b|}{\|w\|} $$

With our normalization this becomes simple:

$$ \rho(w,b) = \frac{1}{\| w \|} $$

since $$ \| w^T x_i + b \| = 1 $$ for the nearest point.

We now want to find $$w$$ such that $$\rho$$ is maximum. To solve this problem, we can re-write it as a quadratic programming problem. We don't need to know what this is in detail but what matters is that there are efficient methods to solve this type of problem. Since we want to maximize the distance of the nearest point $$\rho$$, we can as well minimize $$\| w \|_2$$. So we search for

$$ (w,b) = \text{argmin}_{w,b} \frac{1}{2} \|w\|_2^2 $$

subject to the constraint

$$ y_i (w^T x_i + b) \ge 1 $$

For the optimal solution, there must be at least one $$x_i$$ for which

$$ y_i (w^T x_i + b) = 1 $$

These $$x_i$$s are called **support vectors.** 

In our figure above, the red and blue dots lying on the dashed lines are the support vectors. The black solid line is our best guess $$w^T x + b = 0$$ where $$x$$ is a point in the plane $$x = (x_1,x_2)$$. {% include marginnote.html note='See Section 5.7.2 in [Deep Learning](https://www.deeplearningbook.org/) '%}

### scikit-learn

To demonstrate how SVM works we are going to use [scikit-learn](https://scikit-learn.org/). The results in this section can be reproduced using this shared [notebook](https://colab.research.google.com/drive/1dSo81DdqIkzVyssauB7wYi12kZ0XzuVy). The scikit-learn library can perform many important computations in machine learning including supervised and unsupervised learning. {% include marginnote.html note='See [scikit supervised learning](https://scikit-learn.org/stable/supervised_learning.html) for more details about the functionalities that are supported.'%}

We are going to demonstrate our concept through a simple example. Let's generate 8 random points in the 2D plane. Points in the top left are assigned the label $$-1$$ ($$y > x$$) and points in the bottom right are assigned a label $$-1$$ ($$y < x$$). {% include marginnote.html note='All the examples in this section can be run using Google colab using this [notebook](https://colab.research.google.com/drive/1dSo81DdqIkzVyssauB7wYi12kZ0XzuVy). '%}

In Python, we set up two arrays X (coordinates) and y (labels) with the data:

```python
print('Shape of X: ', X.shape)
print(X)
print('Shape of y: ', y.shape)
print(y)

Shape of X:  (8, 2)
[[-0.1280102  -0.94814754]
 [ 0.09932496 -0.12935521]
 [-0.1592644  -0.33933036]
 [-0.59070273  0.23854193]
 [-0.40069065 -0.46634545]
 [ 0.24226767  0.05828419]
 [-0.73084011  0.02715624]
 [-0.63112027  0.5706703 ]]
Shape of y:  (8,)
[ 1.  1.  1. -1.  1.  1. -1. -1.]
```

Computing the separating hyperplane is done using
```python
from sklearn import svm
clf = svm.SVC(kernel="linear",C=1e6)
clf.fit(X, y)
```

`clf` now contains all the information of the SVM. We will learn later on what the constant `C` is.

To visualize the result, we can plot the black solid line that separates the points. Recall that the equation of the line is

$$ w^T x + b = 0 $$

The vector $$w$$ is given by `clf.coef_`:

```python
print(clf.coef_)
[[ 2.1387469  -2.62113502]]
```

and $$b$$ is given by `clf.intercept_`:

```python
print(clf.intercept_)
[0.63450173]
```

We can plot the points and line using [Plotly](https://plotly.com/python/) syntax

```python
x = np.linspace(-1, 1, 2)
a = -clf.coef_[0,0] / clf.coef_[0,1]
b = -clf.intercept_ / clf.coef_[0,1]
fig.add_trace(go.Scatter(x=x, y=a*x + b))
```

We can also visualize the support vectors.

```python
print(clf.support_vectors_)
[[-0.73084011  0.02715624]
 [-0.40069065 -0.46634545]
 [ 0.24226767  0.05828419]]
```

There are 3 points in this case. The lines going through these points satisfy the equations

Top line

$$ w^T x + b = -1 $$

Bottom line

$$ w^T x + b = 1 $$

These lines can be plotted using

```python
# green line
b1 = -(1 + clf.intercept_) / clf.coef_[0,1]
fig.add_trace(go.Scatter(x=x, y=a*x + b))
# purple line
b2 = -(-1 + clf.intercept_) / clf.coef_[0,1]
fig.add_trace(go.Scatter(x=x, y=a*x + b2))
```

Here is the final plot:

{% include svm1.html %} 

The orange line is the "farthest" away from the red and blue dots. All t  he support vectors are at the same distance from the orange line.

The decision function, equal to $$w^T x + b$$ in our notations, can be computed using

```python
clf.decision_function(X)
```

where `X` contains the coordinates of the points where the function is to be evaluated. This can be used to draw filled contours as shown below.

![](2020-03-22-18-07-39.png){:width="400px"}

Please read [A tutorial on support vector regression](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=5973545F3482D02CFAF0C9DBA1CD7714?doi=10.1.1.114.4288&rep=rep1&type=pdf) by Smola and Sch&ouml;lkopf for more advanced information about these methods.

### Soft-margin

For many problems, though, because of noise and complex data, it is not possible to have a hyperplane that exactly separates the data. In that case, there is no solution to the optimization problem above.

The figure below shows an example where no line divides the red dots from the blue dots. The optimization problem from the previous section has no solution in that case.

{% include svm3.html %}

One solution is to introduce slack variables so that some constraints can be violated but in a minimal way

$$ y_i (w^T x_i + b) \ge 1 - \xi_i $$

with $$\xi_i \ge 0$$. 

If the constraint $$y_i (w^T x_i + b) \ge 1$$ can be satisfied then $$\xi_i = 0$$.

If $$1 > \xi_i > 0$$, then the constraint is weakly violated but the classification is still correct. The sign of $$w^T x_i + b$$ (which is used to predict the label) is still the same as $$y_i$$. 

But, if $$\xi_i > 1$$ then the data is **misclassified.** The sign of $$w^T x_i + b$$ is now different from the sign of $$y_i$$. Hopefully this only happens for very few points.

The new optimization problem becomes:
 
$$ (w,b,\xi) = \text{argmin}_{w,b,\xi} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \xi_i $$

$$ y_i (w^T x_i + b) \ge 1 - \xi_i $$

$$ \xi_i \ge 0 $$

$$C$$ is set by the user and determines how much slack we are allowing. 

A large $$C$$ means little violation is tolerated. Very few points are allowed to violate the condition. The hyperplane is strongly determined by the points nearest to the hyperplane. The hyperplane is very sensitive to points that violate the condition.

A small $$C$$ means a lot of violations are possible. Small $$C$$ is required when data has a lot of noise that needs to be filtered out. In that case, many violations will be accepted as long as this leads to a large separation $$1/\|w\|_2$$.

{% include svm4.html %}

Now we see that points are allowed to lie between the orange and green lines. There are even a few red points below the orange line and a few blue points above. But this cannot be avoided since no line perfectly separates the red points from the blue points.

### Overfitting and underfitting

The value of `C` can be optimized in different ways. This is a broad topic and we will only cover the main ideas.

`C` must be tuned based on how we trust the data. Generally speaking, if the data is very accurate (and a separating hyperplane exists) then `C` must be chosen very large. But if the data is noisy (we do not trust it) then `C` must be small.

Let's start by illustrating the effect of varying `C` in our method. We consider the problem below.

{% include svm5.html %}

We created two well-separated clusters with labels $$-1$$ and $$+1$$. Then we added a blue point on the left and a red point on the right.

The real line of separation is $$y = x$$ as before. So the outlier points can be considered as incorrect data here. Either this data was entered incorrectly in our database, or there was some large error in the measurements.

Let's pick a large value of `C`

```python
# fit the model
clf = svm.SVC(kernel="linear", C = 10^4)
clf.fit(X, y)
```

The SVM decision line has a negative slope as shown below.

{% include svm6.html %}

The red point on the right is classified with a label $$-1$$ (red-orange region). And similarly for the blue point. However, we know that these points are erroneous, and therefore the classification is wrong here.

This is a problem of _overfitting_. We trust too much the data which leads to a large error.

We can try again using a small `C`. However, now the model believes that there is a large error in all the data. As a result, the prediction is quite bad.

```python
clf = svm.SVC(kernel="linear", C = 0.2)
clf.fit(X, y)
```

{% include svm7.html %}

This case corresponds to a situation of _underfitting_. That is we apply too much regularization by reducing `C` and do not trust enough the data.

If we pick `C=0.3`, we get a better fit in this case.

```python
clf = svm.SVC(kernel="linear", C = 0.3)
clf.fit(X, y)
```

This gives us the following plot:

{% include svm8.html %}

which is intermediate between the previous plots. We trust the outlier points but only to a moderate extent. The solid red line is the line $$y = x$$ but because of the outlier points, it is not possible in this case to recover that answer. The SVC model is always biased by the outliers to some extent.

### Training and validation sets

This leads us to the general question of how we should pick $$C$$. This is a problem that we will explore again in the future for other methods.

In machine learning, there are usually errors that arise either from:

- Error in the data; noise in the data may prevent us from finding an exact answer.
- Error in the model; that is, the model cannot possibly reproduce the data because it is too simple. For example, there may not exist a line that separates the blue dots from the red dots.

Because of this, it is therefore common to use a regularization strategy. In our case, this was done by varying $$C$$. A small $$C$$ corresponds to a lot of regularization. This leads to a small vector $$w$$ and if $$C$$ is too small, the vector $$w$$ and scalar $$b$$ may become significantly wrong. A large $$C$$ corresponds to minimal regularization. In that case, we assume the data is accurate and look for a hyperplane that optimally separates the data (e.g., the hyperplane maximizes the distance to all points).

A typical strategy consists of the following. We define two sets of points, called the _training_ set and the _validation_ set.

- Training set: this set is used to fit the model. In our case, this is used to calculate $$(w,b,\xi)$$.
- Validation set: the set is used to tune some of the model parameters, in our case $$C$$. It is usually used to control over- or under-fitting.

The optimization may be based on an iterative loop where we perform in turn

- $$(w,b,\xi)$$: optimized based on training set
- parameter $$C$$: optimized based on validation set

Let's give an example to illustrate how this may work in practice. We consider our previous test case where the input data is randomly perturbed.

```python
# randomly perturb the points X
for i in range(0,X.shape[0]):
  X[i,:] = X[i,:] + ( 2*np.random.rand(1, 2) - 1 ) / 2
```
 
![](2020-03-27-15-09-41.png){:width="400px"}

scikitlearn provides a few functionalities that can be used to simplify the process. Let's start by splitting the input data into a training and validation set.

```python
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)
```

`random_state` is used to initialize the random number generator that is used to randomly generate the sets; `test_size` is the fraction of the input data that will be used for the validation set. The word `test` is used because this function is usually used to split the data into a training set and a test set but for our application we will use the same function to split the data into a training set and a validation set to control over-fitting.

We can verify that the sets have the correct sizes (60% and 40%):

```python
print('Size of X_train: ',X_train.shape[0])
print('Size of X_test: ',X_valid.shape[0])
Size of X_train:  19
Size of X_valid:  13
```

The `clf.score()` function can be used to evaluate the accuracy of our SVM prediction using the test set. The goal is then to find the value of $$C$$ that gives us the highest score.

We obtain the following results in this case.

{% include svm9.html %}

$$a$$ is the slope of the predicted decision line. The exact solution is $$y=x$$ and so $$a=1$$. $$b$$ is the predicted intercept. The exact solution is $$b=0$$.

We see that values of $$C$$ below 0.1 leads to under-fitting. Larger values give consistently good results. Between $$1 \le C \le 10$$ the accuracy is relatively insensitive to the value of $$C$$.

### Cross-validation

We have seen a simple way to decompose the set into a training set and validation set. However, this may lead to some issues. In particular, if we make a unique choice of training and validation set it is quite possible that our prediction becomes biased by our choice of validation set. Indeed, only the validation set is used to estimate how good $$C$$ is and whether it should be adjusted. So to overcome this, it is preferable to optimize our model for many training sets and evaluate its performance on many validation sets. But how can this be done? This would require a tremendous amount of data. In our plot above for example, we had the luxury of generating entirely new sets for each evaluating of $$C$$.

In practice, the method of cross-validation is used. The dataset at our disposal is split into $$K$$ groups. Then we run a series of experiments in which we use $$K-1$$ groups of data for optimizing the model, and the remaining group for validation (i.e., to score our model). By "reusing" our dataset in this fashion, we are able to generate many scores with limited data. These scores can then be averaged to estimate how good $$C$$ is. Based on this result $$C$$ can be optimized. The advantage of this approach is that we minimize a possible bias due to our selection of training and validation sets.

For more information on this, please see [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). There are many variants on this idea.

The [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) section in scikit-learn and the [wikipedia page](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets) on training and validation sets introduce as well the concept of test set. Here is a summary of what each one of these sets is used for

1. Training set: this set is used to opimize the model, in our case $$w$$ and $$b$$.
2. Validation set (also called development set) is used to optimize hyper-parameters in the method, for example the parameter $$C$$ in our example. This can be used to control overfitting and for other optimization of parameters that is not part of step 1 with the training set.
3. Once the model has been computed and that all parameters have been optimized based on the available data, we use a yet-unseen dataset, the test set, to evaluate the accuracy of our model. By definition, the test set is applied to the final model. No further changes are made to the model during that stage. {% include marginnote.html note='See [scikit-learn cross_validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [training, validation, and test sets](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets).'%}

### Kernel trick

An important extension of this method allows treating decision surfaces (the hyperplane that divides points with different colors) into more general surfaces.

To understand this connection, we start from the form of our prediction model

$$ w^T x + b $$

Given some training points $$x^{(i)}$$ we can find coefficients $$\alpha_i$$ such that

$$ w^T x + b = \sum_i \alpha_i [x^{(i)}]^T x + b $$

So far this is just a change in notation. However, it reveals that we can view $$ w^T x $$ as dot products between $$ x^{(i)} $$ and $$x$$. An important extension is to realize that it is possible to use other coordinates than $$x$$. These are sometimes called features. Imagine now that we have at our disposal a function $$ \phi(x) $$ that is vector-valued. This represents a set of "features" representing the data $$x$$. For example, instead of considering $$x = (x_1,x_2)$$ we could define

$$ \phi(x) = (x_1, x_2, x_1^2, x_2^2, x_1 x_2) $$

Then we may use as our model to predict a label

$$ \sum_i \alpha_i \phi(x^{(i)})^T \phi(x) + b $$

When this function is positive, we predict the label $$+1$$ and $$-1$$ otherwise.

However, in many cases, it may be difficult to find an appropriate $$\phi(x)$$, and if $$\phi$$ is very high-dimensional (a lot of features) it may be expensive to compute the dot product 

$$\phi(x^{(i)})^T \phi(x)$$

The kernel trick is an ingenious idea. It replaces the expression above by

$$ \sum_i \alpha_i K(x^{(i)},x) + b $$

There are several theoretical justifications and explanations for this approach but here we will simply demonstrate this approach through examples. {% include marginnote.html note='Please read [Kernel methods in machine learning](https://projecteuclid.org/download/pdfview_1/euclid.aos/1211819561) by Hofman, Sch&ouml;lkopf and Smola for mathematical details on this method.'%}

Many different types of kernels can be used. For example in scikit-learn, we have

Kernel  | Definition
---     | ---
Linear  | $$\langle x, x' \rangle$$
Polynomial | $$(\gamma \langle x, x' \rangle + r)^d$$
Radial basis function (RBF) | $$\exp(-\gamma \lVert x - x' \rVert^2)$$
Sigmoid | $$\tanh(\gamma \langle x, x' \rangle + r)$$

We now demonstrate this method on a simple example.

We consider a situation where the blue points on top are separated from the red points on the bottom by a sine function. Since a linear SVM uses a line to separate these points it cannot make a good prediction.

{% include svm10.html %}

When we use an RBF, the results are much more accurate.{% include marginnote.html note='See Section 5.7.2 in [Deep Learning](https://www.deeplearningbook.org/) for more details on the kernel trick. '%}

{% include svm11.html %}

This is done using

```python
# fit the model
clf = svm.SVC(kernel="rbf", gamma = 10)
clf.fit(X, y)
```

As before, we can tune the regularization parameter `C` to improve the fit; the parameter `gamma` controls the width of the Gaussian function $$\exp(-\gamma \lVert x - x' \rVert^2)$$.