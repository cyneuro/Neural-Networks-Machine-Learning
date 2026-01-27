import numpy as np
import matplotlib.pyplot as plt

# Define the cost function J(theta) = theta^2 + 2*theta + 5
def J(theta):
    return theta**2 + 2*theta + 5

# Theta values for plotting
theta = np.linspace(-3, 1, 100)
J_values = J(theta)

# First plot: Just the function
plt.figure(figsize=(8, 6))
plt.plot(theta, J_values, label=r'$J(\theta) = \theta^2 + 2\theta + 5$')
plt.axvline(x=-1, linestyle='--', color='green', label='Minimum at θ=-1')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J(\theta)$')
plt.title('Quadratic Cost Function')
plt.legend()
plt.grid(True)
plt.savefig('function_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Gradient descent points
thetas = [0, -0.8, -0.96, -0.992]
J_thetas = [J(t) for t in thetas]

# Second plot: Function with gradient descent steps
plt.figure(figsize=(8, 6))
plt.plot(theta, J_values, label=r'$J(\theta) = \theta^2 + 2\theta + 5$')
plt.scatter(thetas, J_thetas, color='red', zorder=5, label='Gradient Descent Steps')
plt.axvline(x=-1, linestyle='--', color='green', label='Minimum at θ=-1')

# Add arrows for gradient descent steps
for i in range(len(thetas)-1):
    plt.arrow(thetas[i], J_thetas[i], thetas[i+1] - thetas[i], J_thetas[i+1] - J_thetas[i],
              head_width=0.05, head_length=0.1, fc='red', ec='red', length_includes_head=True)

# Labels and title
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J(\theta)$')
plt.title('Gradient Descent on Quadratic Function')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('gd_plot.png', dpi=300, bbox_inches='tight')
plt.close()