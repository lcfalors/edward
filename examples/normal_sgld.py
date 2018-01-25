#!/usr/bin/env python
"""Correlated normal posterior. Inference with stochastic gradient
Langevin dynamics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import MultivariateNormalTriL


def model():
  z = MultivariateNormalTriL(
      loc=tf.ones(2),
      scale_tril=tf.cholesky(tf.constant([[1.0, 0.8], [0.8, 1.0]])),
      name="z")


tf.set_random_seed(42)

qz = tf.get_variable("qz", [2])
counter = tf.get_variable("counter", initializer=0.)
qz_mom = tf.get_variable("qz_mom", [2], initializer=tf.zeros_initializer())
# TODO what's up with the samples?
new_states, new_counter, new_momentums = ed.sgld(
    model,
    states=[qz],
    counter=counter,
    momentums=[qz_mom],
    learning_rate=1e-3,
    align_latent=lambda name: "qz" if name == "z" else None,
    align_data=lambda name: None)
qz_update = qz.assign(new_states[0])
counter_update = counter.assign(new_counter)
qz_mom_update = qz_mom.assign(new_momentums[0])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
samples = []
for t in range(2500):
  sample, _, _ = sess.run([qz_update, counter_update, qz_mom_update])
  samples.append(sample)
  if t % 100 == 0:
    print("Step {}".format(t))

samples = samples[500:]

mean = np.mean(samples)
stddev = np.std(samples)
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior stddev:")
print(stddev)
