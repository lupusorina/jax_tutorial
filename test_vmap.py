# By eliminating explicit loops in favor of operations that can be parallelized,
# vmap can significantly speed up computations.

from jax import vmap
import jax.numpy as jnp
import time

# Define a function to add two numbers
def add(x, y):
    return x + y

time_1 = time.time()
x_vector = jnp.arange(10000000)
y_vector = jnp.arange(10000000)
add_vectorized = vmap(add)
print("Time for vmap:", time.time() - time_1)

# Compute element-wise addition
time_2 = time.time()
result_vectorized = add_vectorized(x_vector, y_vector)
time_3 = time.time()
print("JAX Summation Time taken:{} seconds".format(time_3 - time_2))
# print("Vectorized Addition:", result_vectorized)

# do the same with numpy
import numpy as np

x_np = np.arange(10000000)
y_np = np.arange(10000000)
time_4 = time.time()
sum = 0
for i in range(len(x_np)):
    sum += x_np[i] + y_np[i]
print("Numpy Summation Time taken:{} seconds".format(time.time() - time_4))
 