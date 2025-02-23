import jax.numpy as jnp

# Original array
original_array = jnp.array([1, 2, 3, 4])

# Update the first element
updated_array = original_array.at[0].set(10)

print("Original Array:", original_array)
print("Updated Array:", updated_array)