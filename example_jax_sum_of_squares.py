from jax import jit
import jax.numpy as jnp
import time

# Define a simple function
def sum_squares(x):
    return jnp.sum(x ** 2)

time_1 = time.time()
# JIT compile the function
sum_squares_jit = jit(sum_squares)

# Call the JIT compiled function
result = sum_squares_jit(jnp.array([1, 2, 3, 4]))

print("JAX Time taken:{} seconds".format(time.time() - time_1))

# Numpy equivalent
import numpy as np

def sum_squares_np(x):
    return np.sum(x ** 2)

time_1 = time.time()
result = sum_squares_np(np.array([1, 2, 3, 4]))

print("NP Time taken:{} seconds".format(time.time() - time_1))
