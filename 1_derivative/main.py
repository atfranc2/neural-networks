import matplotlib.pyplot as plt
import numpy as np

def f_single_var(x):
    """A function f(x) = 3*x^2 + 4*x + 6"""
    return 3*(x**2) + 4*x + 6

def der_f_single_var(x):
    """Derivative of function f(x) = 6*x + 4"""
    return 6*x + 4

data = np.arange(-5,5, 0.25)
f_x = f_single_var(data)
fig, axes = plt.subplots(1, 3, figsize=(24, 5))  # 1 row, 3 columns

axes[0].set_title("Simple Function")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")

# Plot the original function
axes[0].plot(data, f_single_var(data))

def __half(x, nums, min):
    """Recursive Helper Function With Stop Condition"""
    if x < min:
        return
    
    __half(x*0.5, nums, min)
    nums.append(x)

def half(x, min=0.000001):
    """Recursive Helper Function to Generate Values Towards 0"""
    nums = []

    __half(x, nums, min)

    return nums

# Evaluate the derivative of f(x) at 10
x = 2
h = np.array(half(1))

h_data = (f_single_var(h + x) - f_single_var(x)) / h

axes[1].set_title(f"Limit of [f(x + h) - f(x)] / h as h -> 0")
axes[1].set_xlabel("h")
axes[1].set_ylabel(f"[f(x + h) - f(x)] / h (x={x})")


print(f"f({x})=", f_single_var(x))
print(f"d/dx ~= {h_data[0]}")

# Plot the original function
axes[1].plot(h, h_data) #plot example

der_at = np.array([-3, 0, 3])
f_xs = f_single_var(der_at)
df_xs = der_f_single_var(der_at)
bs = f_xs - df_xs * der_at

f_x = f_single_var(data)

axes[2].set_title("Simple Function With Tangent Derivative Curves")
axes[2].set_xlabel("x")
axes[2].set_ylabel("f(x)")

# Plot the original function
axes[2].plot(data, f_single_var(data)) #plot example

for x, dx, b in zip(der_at, df_xs, bs):
    line_x = np.arange(x-1, x+2, 0.25)
    fun = dx * line_x + b
    axes[2].plot(line_x, fun)

plt.tight_layout()

plt.savefig('/app/derivative/figures/derivative_plots.png', dpi=300, bbox_inches="tight")

plt.show()
