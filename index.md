---
layout: page
title: Stanford CME 216 class material
---

These is the web site for CME 216 Machine Learning in Computational Engineering. This material was created by [Eric Darve](https://me.stanford.edu/people/eric-darve), with the [help](https://github.com/EricDarve/cme216-spring-2020/commits/master) of course staff and students.
{% include marginnote.html id='mn-construction' note='The site is **under construction**! If you find errors or feel that some parts need clarification, please let us know, or submit a pull request with your fixes to our [GitHub repository](https://github.com/EricDarve/cme216-spring-2020).'%} The Jekyll theme is inspired by the [CS228 course notes](https://github.com/ermongroup/cs228-notes) site on GitHub.

## Syllabus

[Syllabus](syllabus)

## Reading assignments

Module 1

1. [Introduction to ML and SVM](Reading Questions/intro_svm)
1. [Soft-margins in SVM](Reading Questions/svm_softmargin)
1. [Kernel trick](Reading Questions/kernel_trick)

- [Module 1 Solutions](RA Solutions/RA1-3_solutions)

Module 2

{:start="4"}
1. [Perceptron](Reading Questions/perceptron)
1. [MLP](Reading Questions/MLP) (multi-layer perceptron)
1. [TensorFlow](Reading Questions/TF)
1. [Subclassing](Reading Questions/subclassing)

Module 3

{:start="8"}
1. [Loss function and cross entropy](Reading Questions/loss)
1. [Loss functions in TF/Keras](Reading Questions/TF_loss)
1. [The backpropagation algorithm](Reading Questions/Backpropagation)
1. [Learning rate and overfitting](Reading Questions/LR_overfitting)
1. [Initializers and regularizers](Reading Questions/initializers_regularizers)
1. [SGD and saddle points](Reading Questions/SGD)
1. [Momentum and ADAGRAD](Reading Questions/Momentum)
1. [RMSProp and Adam](Reading Questions/Adam)

- [Module 3 Solutions](RA Solutions/RA_Solution_module3)

Module 4

{:start="16"}
1. [Automatic differentiation overview](Reading Questions/AD)
1. [Computational graph](Reading Questions/ComplGraph)
1. [Forward and reverse modes](Reading Questions/FwdRevMode)
1. [AD for physical simulation](Reading Questions/ADPhys)
1. [AD through implicit operators](Reading Questions/ImplicitOps)

- [Module 4 Solutions](RA Solutions/RA_Solution_module4)

Module 5

{:start="21"}
1. [Inverse problems](Reading Questions/Inverse)
1. [Training for inverse problems](Reading Questions/InverseTraining)
1. [Physics constrained learning](Reading Questions/PCL)
1. [Physics-informed learning conclusion](Reading Questions/InverseConclusion)

- [Module 5 Solutions](RA Solutions/RA_Solution_module5)

Module 6

{:start="25"}
1. [Generative Adversarial Networks](Reading Questions/GAN)

## Programming Homework

- [Python setup guide](Python Setup Guide)
- [Homework 1](Homework/HW1 Questions) and [starter code](Homework/hw1_starter_code.zip) and [solution](HW Solutions/svm.ipynb)
- [Homework 2](Homework/HW2 Questions) and [starter code](Homework/hw2_starter_code.zip) and [solution](HW Solutions/hw2_solution.zip)
- [Homework 3](Homework/HW3 Questions) and [starter code](Homework/hw3_starter_code.zip) and [solution](HW Solutions/hw3_solution.zip)
- [Homework 4](Homework/HW4/HW4 Questions) and [starter code](Homework/HW4/hw4_starter_code.zip) and [solution writeup](HW Solutions/hw4_solution) and [solution files](HW Solutions/hw4_solution.zip)

## Final Project

[Instructions](Homework/Final Project)

## Lecture slides and code

The videos accompanying these lectures can be found on canvas under ["Course Videos."](https://canvas.stanford.edu/courses/118944/external_tools/3367)

Python tutorial

- [Tutorial code](https://github.com/EricDarve/cme216-spring-2020/tree/master/Code/Python)
- [Python introduction notebook](https://github.com/EricDarve/cme216-spring-2020/blob/master/Code/Python/Python%20basics.ipynb)
- [Numpy notebook](https://github.com/EricDarve/cme216-spring-2020/blob/master/Code/Python/Numpy%20tutorial.ipynb)

Module 1

_Introduction to Machine Learning and Support Vector Machines_

- [SVM code](https://github.com/EricDarve/cme216-spring-2020/blob/master/Code/svm.ipynb)
- [1.1 Brief introduction to machine learning](Slides/ML_introduction/brief_intro)
- [1.2 A few examples of machine learning](Slides/ML_introduction/examples_ML)
- [1.3 Supervised learning](Slides/ML_introduction/supervised_learning)
- [1.4 Machine learning in engineering](Slides/ML_introduction/ml_in_engineering)
- [1.5 Introduction to SVM](Slides/SVM_introduction/)
- [1.6 Scikit-learn](Slides/scikitlearn/scikit)
- [1.7 Soft-margin](Slides/scikitlearn/softmargin)
- [1.8 Overfitting](Slides/scikitlearn/overfitting)
- [1.9 Training and validation sets](Slides/scikitlearn/training_validation)
- [1.10 Kernel trick](Slides/scikitlearn/kernel_trick)

Module 2

_Deep Neural Networks and TensorFlow_

- [2.1 Perceptron](Slides/ANN/perceptron)
- [2.2 Artificial Neural Networks](Slides/ANN/MLP) (multilayer perceptron)
- [2.3 TensorFlow/Keras](Slides/TF_Keras/TF_Pytorch)
- [2.4 Sequential API](Slides/TF_Keras/TF_sequential)
- [2.5 Functional API](Slides/TF_Keras/TF_functional)
- [2.6 Subclassing](Slides/TF_Keras/subclassing)
- [DNN TensorFlow code](https://github.com/EricDarve/cme216-spring-2020/blob/master/Code/DNN_regression.ipynb)
- [Python inheritance example code](https://github.com/EricDarve/cme216-spring-2020/blob/master/Code/Inheritance%20demo.ipynb)

Module 3

_Deep Learning_

- [3.1 Loss function for regression and classification](Slides/Deep_Learning/Loss)
- [3.2 Cross-entropy](Slides/Deep_Learning/Cross_entropy)
- [3.3 TensorFlow loss functions](Slides/Deep_Learning/TF_loss)
- [3.4 Backpropagation](Slides/Deep_Learning/Backprop)
- [3.5 Backpropagation formula](Slides/Deep_Learning/Backprop_formula)
- [3.6 Learning rate for training](Slides/Deep_Learning/Learning_rate)
- [3.7 Empirical method for learning rate](Slides/Deep_Learning/Learning_rate_empirical)
- [3.8 Overfitting](Slides/Deep_Learning/Training_overfitting)
- [3.9 DNN initializers](Slides/Deep_Learning/Training_initializers)
- [3.10 Regularization](Slides/Deep_Learning/Training_regularization)
- [3.11 Stochastic Gradient Descent](Slides/Deep_Learning/SGD)
- [3.12 Saddle points](Slides/Deep_Learning/Saddle_points)
- [3.13 Momentum](Slides/Deep_Learning/Momentum)
- [3.14 Adagrad](Slides/Deep_Learning/Adagrad)
- [3.15 RMSProp and Adam](Slides/Deep_Learning/Adam)
- [Regularization for DNNs example code](https://github.com/EricDarve/cme216-spring-2020/blob/master/Code/DNN_regularization.ipynb)
- [Saddle points illustration code](https://github.com/EricDarve/cme216-spring-2020/blob/master/Code/Saddle%20points.ipynb)
- [ADAGRAD benchmark code](https://github.com/EricDarve/cme216-spring-2020/blob/master/Code/Adagrad.ipynb)

Module 4 and 5

_Physics informed learning, automatic differentiation, inverse modeling_

The slides are assembled into two PDF files. Each lecture will cover one section in one of these PDF files. The lecture videos are on [Canvas](https://canvas.stanford.edu/courses/118944/external_tools/3367).

- [Automatic Differentiation for Computational Engineering](Slides/AD/AD.pdf)
- [Inverse Modeling using ADCME](Slides/AD/Inverse.pdf)

## Contents of class

Highlights of topics to cover

**Supervised learning and SVM**

Module 1 (week 1 and 2, 4/6, 4/13)

- Supervised learning
- SVM; [scikit-learn](https://scikit-learn.org/stable/); kernel trick; radial basis functions
- Overfitting; underfitting; regularization

- Homework 1 (SVM homework)

**Deep learning**

Module 2 (week 3, 4/20)

- NN and DNN; layers; weights and biases; activation function; loss function
<!-- - Universal approximation theorems; [Montufar et al. (2014)](http://papers.nips.cc/paper/5422-on-the-number-of-linear-regions-of-deep-neural-networks.pdf) -->
- [TensorFlow](https://www.tensorflow.org/learn) and [Keras](https://www.tensorflow.org/guide/keras)

Module 3 Part 1 (week 4, 4/27)

- Forward and back-propagation
- Weight initialization
- Regularization; test and validation sets; hyperparameter optimization
- Regularization strategies
<!-- - Batch normalization -->

- Homework 2 (covid-19 modeling)

Module 3 Part 2 (week 5-6, 5/4, 5/11)

- Stochastic gradient methods; SGD, momentum; adaptive algorithms
<!-- - If time allows: convolution nets; pooling; fully-connected nets; DNN and convnet architectures -->

**Physics-informed learning**

Module 4 and 5 (week 6-8, 5/11--5/25)

- Physics-based ML; PhysML
- DNN and numerical PDE solvers
- Automatic differentiation; forward and reverse mode AD; chain rule; computational graph
- Examples of numerical PDE solutions with ADCME
- Physics constrained learning

- Homework 3 (week 6; bathymetry)

**Generative deep networks**

Module 6 (week 9-10, 6/1)

<!-- - Autoencoders and variational autoencoders -->
- PhysGAN and ADCME
- GANs to generate samples from a given probability distribution
- Generator and discriminator networks; WGANs
- TensorFlow example

- Homework 4 (5/31; physics informed learning)

**Reinforcement learning**

Module 7

We won't have enough time to cover this topic unfortunately.

- Reinforcement learning; [Sutton and Barto](http://incompleteideas.net/book/the-book.html); [Mnih 2013](https://arxiv.org/abs/1312.5602)
- Temporal difference learning; deep Q-learning networks
- Policy gradients and actor-critic algorithms

<!-- - Short homework 5 on RL (June 10) -->

## Reading material

###  Books

- [Deep learning](http://www.deeplearningbook.org/) by Ian Goodfellow and Yoshua Bengio and Aaron Courville
- [Deep learning with Python](https://searchworks.stanford.edu/view/13216992) by Fran&ccedil;ois Chollet
- [Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow : concepts, tools, and techniques to build intelligent systems](https://searchworks.stanford.edu/view/13489354) by Aur&eacute;lien G&eacute;ron
- [Fundamentals of deep learning : designing next-generation machine intelligence algorithms](https://searchworks.stanford.edu/view/12112250) by Nikhil Buduma
- [Elements of statistical learning](https://searchworks.stanford.edu/view/12458005) by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
- [Deep learning: an introduction for applied mathematicians](https://epubs.siam.org/doi/pdf/10.1137/18M1165748) by Catherine Higham and Desmond Higham
- [Machine learning: a probabilistic perspective](https://www.cs.ubc.ca/~murphyk/MLbook/) by Kevin Murphy (in [searchworks](https://searchworks.stanford.edu/view/13163347))
- [Deep learning with Python](https://searchworks.stanford.edu/view/13216992) by Fran&#231;ois Cholet
- [Deep learning illustrated: a visual, interactive guide to artificial intelligence](https://searchworks.stanford.edu/view/13463749) by Jon Krohn
- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Foundations of machine learning](https://cs.nyu.edu/~mohri/mlbook/) by Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar
- [Neural networks and learning machines](https://searchworks.stanford.edu/view/8631715) by Simon Haykin
- [The matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) by Kaare Petersen and Michael Pedersen

### Video tutorials

- [Introduction to deep learning: concepts and fundamentals](https://searchworks.stanford.edu/view/13216564) by Laura Graesser
- [Introduction to deep learning models with TensorFlow: learn how to work with TensorFlow to create and run a TensorFlow graph, and build a deep learning model](https://searchworks.stanford.edu/view/13214579) by Lucas Adams
- [Deep learning with TensorFlow: applications of deep neural networks to machine learning tasks](https://searchworks.stanford.edu/view/13215423) by Jon Krohn

### Review papers

- LeCun, Bengio and Hinton, Deep learning, _Nature,_ 521:436-444, 2015
- Schmidhuber, Deep learning in neural networks: an overview, _Neural Networks,_ 61:85-117, 2015
- [Automatic differentiation in machine learning: a survey](https://arxiv.org/pdf/1502.05767.pdf) by At&#305;l&#305;m G&uuml;nes Baydin, Barak Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind
- [A review of the adjoint-state method for computing the gradient of a functional with geophysical applications](https://academic.oup.com/gji/article/167/2/495/559970) by R.-E. Plessix

### Online classes and tutorials

- [Introduction to Deep Learning](http://introtodeeplearning.com/), MIT
- [fast.ai](https://course.fast.ai/)
- [Machine Learning 2014-2015](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/), Oxford, by Nando de Freitas

### Links

- [List of books and tutorials on ML](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md)
- [Online courses](https://github.com/josephmisiti/awesome-machine-learning/blob/master/courses.md)
- [TensorFlow notebooks](https://github.com/the-deep-learners/TensorFlow-LiveLessons) by Jon Krohn
- [TF2 notebooks](https://github.com/jonkrohn/tf2) by Jon Krohn
- [Deep learning illustrated notebooks](https://github.com/the-deep-learners/deep-learning-illustrated) by Jon Krohn
- [TensorFlow playground](http://playground.tensorflow.org/)
- [MNIST visualization](https://www.cs.ryerson.ca/~aharley/vis/conv/) by Adam Harley
- [Distill](https://distill.pub/), a journal for machine learning visualizations