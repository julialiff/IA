from random import choice
from numpy import array, dot, random
from matplotlib import pylab

unit_step = lambda x: 0 if x < 0 else 1

training_data = [
  (array([0,0,1]), 0),
  (array([0,1,1]), 1),
  (array([1,0,1]), 1),
  (array([1,1,1]), 1),
]

w = random.rand(3)

errors = []
eta = 0.2
n = 100

for i in range(n):
  x, expected = choice(training_data)
  result = dot(w, x)
  error = expected - unit_step(result)
  errors.append(error)
  w += eta * error * x

for x, _ in training_data:
  result = dot(x, w)
  print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

# [0 0]: -0.0714566687173 -> 0
# [0 1]: 0.829739696273 -> 1
# [1 0]: 0.345454042997 -> 1
# [1 1]: 1.24665040799 -> 1

import matplotlib.pyplot as plt
plt.plot(errors)
plt.show()

from pylab import plot, ylim
ylim([-1,1])
p = plot(errors)

