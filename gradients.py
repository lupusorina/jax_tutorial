from jax import grad

# Define a function for which to compute the gradient
def f(x):
    return x**3 + 2*x**2 - 3*x + 1

# Compute gradient
df = grad(f)

# Evaluate the gradient at x = 1
gradient_at_1 = df(1.0)
print("df/dx at x=1:", gradient_at_1)