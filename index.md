---
layout: page
title: Stanford cme 216 Lecture notes
---

These are lecture notes for CME 216 Machine Learning in Computational Engineering. They were written by [Eric Darve](https://me.stanford.edu/people/eric-darve), with the [help](https://github.com/EricDarve/cme216-spring-2020/commits/master) of course staff and students.
{% include marginnote.html id='mn-construction' note='The notes are **under construction**! Although we have written up most of the material, you will probably find several typos. If you do, please let us know, or submit a pull request with your fixes to our [GitHub repository](https://github.com/EricDarve/cme216-spring-2020).'%} The Jekyll theme is inspired by the [CS228 course notes](https://github.com/ermongroup/cs228-notes) site on GitHub.

## Content of class

Topics to cover and tentative schedule

**Supervised learning and SVM**

Week 1
- Supervised learning
- SVM; [scikit-learn](https://scikit-learn.org/stable/); kernel trick; radial basis functions
- Overfitting; underfitting; regularization

**Deep learning**

Week 2
- NN and DNN; layers; weights and biases; activation function; loss function
- Universal approximation theorems; [Montufar et al. (2014)](http://papers.nips.cc/paper/5422-on-the-number-of-linear-regions-of-deep-neural-networks.pdf)

Week 3
- [TensorFlow](https://www.tensorflow.org/learn)
- [Keras](https://www.tensorflow.org/guide/keras); [TFLearn](http://tflearn.org/); [TFSlim](https://github.com/google-research/tf-slim); [OpenAI Gym](https://gym.openai.com/)
- [PyTorch](https://pytorch.org/)

Week 4
- Forward and back-propagation
- Regularization; test and validation sets; hyperparameter optimization; capacity

Week 5
- Regularization strategies; bagging, dropout
- Stochastic gradient methods; SGD, momentum; adaptive algorithms; AdaGrad, RMSProp, Adam; 2nd order methods, BFGS (Broyden–Fletcher–Goldfarb–Shanno)

Week 6
- Batch normalization
- Weight initialization
- Convolution nets; pooling; fully-connected nets
- DNN and convnet architectures; LeNet-5, AlexNet, GoogLeNet/Inception, VGGNet, ResNet

**Physics-informed learning**

Week 7
- Physics-based ML; PhysML
- DNN and numerical PDE solvers
- Automatic differentiation; forward and reverse mode AD; computational graph; chain rule
- Examples of numerical PDE solutions with ADCME

**Generative deep networks**

Week 8
- Autoencoders and variational autoencoders
- GAN to model stochastic variables
- Discriminator network

Week 9
- WGANs
- PhysGAN

**Reinforcement learning**

Week 10
- Reinforcement learning; [Sutton and Barto](http://incompleteideas.net/book/the-book.html); [Mnih 2013](https://arxiv.org/abs/1312.5602)
- Temporal difference learning; deep Q-learning networks
- Policy gradients and actor-critic algorithms

## Reading material

###  Books

- [Deep learning](http://www.deeplearningbook.org/) by Ian Goodfellow and Yoshua Bengio and Aaron Courville
- [Deep learning: an introduction for applied mathematicians](https://epubs.siam.org/doi/pdf/10.1137/18M1165748) by Catherine Higham and Desmond Higham
- [Machine learning: a probabilistic perspective]() by Kevin Murphy
- [Deep learning with Python](https://searchworks.stanford.edu/view/13216992) by Fran&#231;ois Cholet
- [Deep learning illustrated: a visual, interactive guide to artificial intelligence](https://searchworks.stanford.edu/view/13463749) by Jon Krohn
- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Foundations of machine learning](https://cs.nyu.edu/~mohri/mlbook/) by Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar
- [Elements of statistical learning](https://searchworks.stanford.edu/view/12458005) by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
- [Neural networks and learning machines](https://searchworks.stanford.edu/view/8631715) by Simon Haykin

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
- [Distill](https://distill.pub/), a journal for machine learning visualizations