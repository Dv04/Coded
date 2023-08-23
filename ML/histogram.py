import numpy as np
import matplotlib.pyplot as plt

# Generate data
data = np.random.uniform(0, 100, 1000)

# Create a histogram
plt.hist(data, bins=10, range=(0, 100), density=True)

# Add labels and title
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Symmetric and Uniform Histogram")

# Save the histogram to a PNG file
plt.savefig("histogram.png")
