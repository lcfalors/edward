#!/usr/bin/env python
"""Bayesian logistic regression using Hamiltonian Monte Carlo.

We visualize the fit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal


def build_toy_dataset(N, noise_std=0.1):
  D = 1
  X = np.linspace(-6, 6, num=N)
  y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
  y[y < 0.5] = 0
  y[y >= 0.5] = 1
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D)).astype(np.float32)
  y = y.astype(np.float32)
  return X, y


def model(X):
  w = Normal(loc=tf.zeros(D), scale=3.0 * tf.ones(D), name="w")
  b = Normal(loc=tf.zeros([]), scale=3.0 * tf.ones([]), name="b")
  y = Bernoulli(logits=tf.tensordot(X, w, [[1], [0]]) + b, name="y")


tf.set_random_seed(42)

N = 40  # number of data points
D = 1  # number of features

X_train, y_train = build_toy_dataset(N)

qw = tf.get_variable("qw", [D])
qb = tf.get_variable("qb", [])

new_states, _, _ = ed.hmc(
    model,
    step_size=0.6,
    states=[qw, qb],
    align_latent=lambda name: {"w": "qw", "b": "qb"}.get(name),
    align_data=lambda name: {"y": "y"}.get(name),
    X=X_train,
    y=y_train)

qw_update = qw.assign(new_states[0])
qb_update = qb.assign(new_states[1])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Set up figure.
fig = plt.figure(figsize=(8, 8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

# Build samples from inferred posterior.
n_samples = 50
inputs = np.linspace(-5, 3, num=400, dtype=np.float32).reshape((400, 1))
<<<<<<< HEAD
<<<<<<< HEAD
probs = tf.stack([tf.sigmoid(ed.dot(inputs, qw.sample()) + qb.sample())
=======
# TODO n_samples; will need to store and use last X posterior samples
probs = tf.stack([tf.sigmoid(tf.tensordot(inputs, qw, [[1], [0]]) + qb)
>>>>>>> 25f58be2... replace ed.dot with tensor.dot(..., [[1], [0]])
=======
# TODO n_samples; will need to store and use last X posterior samples
probs = tf.stack([tf.sigmoid(ed.dot(inputs, qw) + qb)
>>>>>>> 4008af0a... update examples/
                  for _ in range(n_samples)])

for t in range(5000):
  sess.run([qw_update, qb_update])
  if t % 10 == 0:
    outputs = sess.run(probs)

    # Plot data and functions
    plt.cla()
    ax.plot(X_train[:], y_train, 'bx')
    for s in range(n_samples):
      ax.plot(inputs[:], outputs[s], alpha=0.2)

    ax.set_xlim([-5, 3])
    ax.set_ylim([-0.5, 1.5])
    plt.draw()
    plt.pause(1.0 / 60.0)
