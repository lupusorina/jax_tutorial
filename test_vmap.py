# By eliminating explicit loops in favor of operations that can be parallelized,
# vmap can significantly speed up computations.

from jax import vmap
import jax.numpy as jnp
import time

def add(x, y):
    return x + y

times = []
N = 18

for i in range(0, N):
    print("Power of 10:", i)
    x_vector = jnp.arange(10**i)
    y_vector = jnp.arange(10**i)
    add_vectorized = vmap(add)

    # Compute element-wise addition
    time_2 = time.time()
    result_vectorized = add_vectorized(x_vector, y_vector)
    times.append(time.time() - time_2)

# Plot
import matplotlib.pyplot as plt
plt.plot(range(0, N), times, 'o-')
plt.xlabel("Power of 10")
plt.ylabel("Time taken in seconds")
plt.title("Time taken for vmap")
plt.xticks(range(0, N))
plt.grid()
plt.savefig("vmap_time.png")
plt.show()

# Check the time taken for numpy
import numpy as np
x_np = np.arange(10000000)
y_np = np.arange(10000000)
time_4 = time.time()
sum = 0
for i in range(len(x_np)):
    sum += x_np[i] + y_np[i]
print("Numpy Summation Time taken:{} seconds".format(time.time() - time_4))
 