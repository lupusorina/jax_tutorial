# JAX.
import time
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0, 4.0])
y = jnp.array([5.0, 6.0, 7.0, 8.0])

time_1 = time.time()
z = x + y
print("Sum:", z)

w = x * y
print("Product:", w)
time_2 = time.time()
print("JAX Time taken:{} seconds".format(time_2 - time_1))

# numpy
import numpy as np

x_np = np.array([1.0, 2.0, 3.0, 4.0])
y_np = np.array([5.0, 6.0, 7.0, 8.0])

time_1 = time.time()
z_np = x_np + y_np
print("Sum:", z_np)

w_np = x_np * y_np
print("Product:", w_np)
time_2 = time.time()
print("Numpy Time taken:{} seconds".format(time_2 - time_1))