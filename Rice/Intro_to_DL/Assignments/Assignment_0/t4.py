import numpy as np
import matplotlib.pyplot as plt

# Define x range
x = np.linspace(-10, 10, 400)

# Three equations
y1 = x**2  # quadratic
y2 = np.sin(x)  # sine wave
y3 = np.exp(0.2 * x)  # exponential growth

# Plot all three
plt.figure(figsize=(8, 6))
plt.plot(x, y1, label="y = x²")
plt.plot(x, y2, label="y = sin(x)")
plt.plot(x, y3, label="y = exp(0.2x)")

# Labels, legend, grid
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-5, 10)
plt.title("Custom Plot with Three Equations")
plt.legend()
plt.grid(True)

plt.show()
