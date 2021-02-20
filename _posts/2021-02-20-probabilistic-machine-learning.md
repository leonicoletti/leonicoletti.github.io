---
layout: post
robots: index,follow
published: true

tags: [machine-learning, stochastic-variational-inference, bayesian, statistical-learning, pyro]

title: The statistical foundations of machine learning
description: How probabilistic inference underpins empirical machine learning, and how many rules of thumb relate to structured probabilistic models using the Bayesian framework.
---

## Introduction

This article concerns my take on how the tools of probability can help us give a quantified meaning to the philosophical notion of *causality*. Modern probabilistic frameworks provide useful tools to uncover and quantify the different causes lying behind observations. We propose a short introduction to the field of *probabilistic inference* and how it underpins empirical results in different fields of pratical machine learning.

## Formalism

We consider that our data are realizations of a random variable $x$, which is explained, at least partially, by a set of latent variables $z$. These latent variables act as *causes* for our observations but since they cannot be observed, they must be *infered* from the data. This problem of probabilitic inference requires to encode our knowledge in a model $p$, which reflects the structure of causality.

For example, suppose we want to measure weight $w$ of an object but are only provided with a unreliable scale, which gives slightly different results for each measure. We are facing two different kinds of uncertainty: the one associated with the unknown weight and the one associated with the scale itself. Provided a first *guess* $g$, we can define the following probabilistic model.

$$
\left\{
\begin{align}
w & \mid g \sim \mathcal{N}(g, \sigma) \\
m & \mid g, w \sim \mathcal{N}(w, \tau)
\end{align}
\right.
$$

The measurements $m$ are the only observed random variables, which are determined by the underlying weight, considered a latent cause, with its own uncertainty. Both uncertainties are modeled with Gaussian laws with standard deviations $\sigma$ and $\tau$. The *parameters* $\theta$ of the model correspond to constant scalars or vectors, here $\theta = \{g, \sigma, \tau\}$. Fitting the model means finding the optimal set $\theta^*$ with respect to a given criterion, which is analogous to learning a model with respect to a given loss function in machine learning.

## Maximum likelihood criterion

We call *evidence* the probability of the data $x$ with respect to the model, denoted $p_\theta(x)$. The usual optimization criterion is to find the parameters $\theta$ such as the evidence of the data is maximal, therefore

$$\theta^* := \underset{\theta}{\operatorname{arg max}} p_\theta(x)$$

### Linear regression

Linear regression is a simple model acting as a building block of deep learning models. We suppose that the $d$-dimensioned data are noisy observations of a linear combination of the latent variable $z$. With $w$ the weight vector to be fitted and a normal prior on the observation noise $\epsilon$, the model writes:

$$
\left\{
\begin{align}
x & = z w + \epsilon \\
\epsilon & \sim \mathcal{N}(0, \sigma^2 I_d)
\end{align}
\right.
$$

Using the conditionnal probability notation, we have $x \mid z, w \sim \mathcal{N}(z w, \sigma^2 I_d)$. Therefore, the *maximum likelihood estimator* (MLE) on the parameters $\theta = \{w\}$ boils down to the usual *mean squared error* minimization rule (MSE):

$$
\theta^* := \underset{\theta}{\operatorname{arg max}} p_\theta(x \mid z, w) = \underset{w}{\operatorname{arg max}} \exp(- \lVert x - zw \rVert^2 / \sigma^2) = \underset{w}{\operatorname{arg min}} \lVert x - zw \rVert^2
$$

### Ridge regression

To avoid overfitting the data, we usually add a regularizer on the parameter $w$, *e.g.* a Tikhonov regularizer $\lambda \lVert w \rVert^2$ to elastically control the weight dispersion. This is explained as a maximum likelihood rule by further adding a normal prior on $w$.

$$
\left\{
\begin{align}
x & = z w + \epsilon \\
\epsilon & \sim \mathcal{N}(0, \sigma^2 I_d) \\
w &\sim \mathcal{N}(0, \tau^2 I_d)
\end{align}
\right.
$$

By applying Bayes' rule on the *posterior* $w \mid x,z$, we get $p(w \mid x,z) = p(x \mid z,w) \, p(w) \, / \, p(x \mid z)$ which leads to the usual *ridge regression* formulation:

$$w^* := \underset{w}{\operatorname{arg min}} \lVert x - zw \rVert^2 + \lambda \lVert w \rVert^2$$

Where $\lambda := \sigma^2 \, / \, \tau^2$. This MLE criterion enriched with a prior on the weight parameter is often refered as the *maximum a posteriori* rule (MAP) and underscores how regularization relates to structured uncertainty.

### Unsupervised learning of the latent space

We now have a practical criterion to solve for the best $\theta$ given the uncertainty structure we associate with the problem. This rule concerns a *supervised* context, where we are provided with both $x$ and $z$ for training. A more general - and realistic - setup concerns the case where we only have $x$ observations to infer the causes. It corresponds to *inductive reasoning*, where we try to infer the probables causes of given premises.

The maching learning toolkit for this *unsupervised* learning includes *autoencoders*, a kind of network consisting of two symmetrical parts:
* the *encoder* (parameterized by $\phi$): learns the mapping from observations $x$ to latent variables $z$;
* the *decoder* (parameterized by $\psi$): learns the inverse mapping, from the latent variables $z$ back to the observations $x$.

We train both simultaneously to reconstruct the observed variables through the latent space

$$\phi^*, \psi^* := \underset{\phi,\psi}{\operatorname{arg min}} \lVert x - (\psi \circ \phi) \, x \rVert^2 $$

## Stochastic variational inference

We consider the generic case of a *graphical model* $p_\theta$ with a hierarchy of latent variables $z$ and conditioning relations between them. The two problems of interest are:
* *model fitting*: selecting the best set of parameters $\theta^*$ with respect to the MLE criterion;
* *posterior estimation*: approximating the posterior distributions $p_{\theta^*}(z \mid x)$ for each $z$ given observed data $x$.

We apply the MLE criterion on the *log-evidence*

$$\theta^* := \underset{\theta}{\operatorname{arg max}} \log p_\theta(x) = \underset{\theta}{\operatorname{arg max}} \log \int_z p_\theta(x,z) \, dz$$

The integral over the causes $z$ is often intractable and the associated optimization non-convex, making the whole problem especially difficult to tackle. Furthermore, once $\theta^*$ is estimated, computing the prior also requires to approximate an intractable integral:

$$p_{\theta^*}(z \mid x) = \frac{p_{\theta^*}(x,y)}{\int p_{\theta^*}(x,z) \, dz}$$

### Variational distribution

The idea behind *variational inference* is to solve these two problems by surrogating the posteriors $p_{\theta}(z \mid x)$ by parameterized distributions $q_\psi(z)$ easier to compute. These distributions are called *variational distributions* or *guides* by the authors of the [Pyro](https://pyro.ai/) framework. The approximation criterion is a measure of similarity between distributions $p_{\theta}(\cdot \mid x)$ and $q_\psi$, both living in infinite-dimensional function spaces, hence the name "variational".

We use the *Kullback–Leibler divergence* $D_{KL}(q_\psi \mid\mid p_{\theta}(\cdot \mid x))$ which quantifies the similarity between the true posteriors and their surrogating guides. The KL divergence can be approximated by Monte-Carlo methods since it is defined as the expectation:

$$D_{KL} := \mathbb{E}_{q_\psi} \left[\log \frac{q_\psi(z)}{p_{\theta}(z \mid x)}\right]$$

However, optimizing it with respect to $\psi$ is much harder, since the expectation distribution depends on $\psi$. The trick is that it can be expressed as $\log p_\theta(x) - ELBO$, where ELBO is an *evidence lower bound*. Therefore, maximizing the ELBO amounts to minimizing the divergence between the posterior and its guide.

$$ELBO := \mathbb{E}_{q_\psi} \left[\log \frac{p_{\theta}(x, z)}{q_\psi(z)}\right]$$

ELBO is much easier to compute as it doesn't require to approximate the posterior $p_\theta(z \mid x)$ but only the probability $p_\theta(x,z)$.

## References

* Ulrike von Luxburg, *Statistical Machine Learning (Part 12 - Risk minimization vs. probabilistic approaches)*, Summer Term 2020, University of Tübingen.
* Pyro.ai, *Getting Started With Pyro: Tutorials, How-to Guides and Examples*.

