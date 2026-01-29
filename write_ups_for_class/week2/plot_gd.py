import numpy as np
import matplotlib.pyplot as plt

# Week 2: Quadratic function plots
# Define the cost function J(theta) = theta^2 + 2*theta + 5
def J_quad(theta):
    return theta**2 + 2*theta + 5

# Theta values for plotting
theta = np.linspace(-3, 1, 100)
J_values = J_quad(theta)

# First plot: Just the function
plt.figure(figsize=(8, 6))
plt.plot(theta, J_values, label=r'$J(\theta) = \theta^2 + 2\theta + 5$')
plt.axvline(x=-1, linestyle='--', color='green', label='Minimum at θ=-1')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J(\theta)$')
plt.title('Quadratic Cost Function')
plt.legend()
plt.grid(True)
plt.savefig('week2_function_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Gradient descent points
thetas_quad = [0, -0.8, -0.96, -0.992]
J_thetas_quad = [J_quad(t) for t in thetas_quad]

# Second plot: Function with gradient descent steps
plt.figure(figsize=(8, 6))
plt.plot(theta, J_values, label=r'$J(\theta) = \theta^2 + 2\theta + 5$')
plt.scatter(thetas_quad, J_thetas_quad, color='red', zorder=5, label='Gradient Descent Steps')
plt.axvline(x=-1, linestyle='--', color='green', label='Minimum at θ=-1')

# Add arrows for gradient descent steps
for i in range(len(thetas_quad)-1):
    plt.arrow(thetas_quad[i], J_thetas_quad[i], thetas_quad[i+1] - thetas_quad[i], J_thetas_quad[i+1] - J_thetas_quad[i],
              head_width=0.05, head_length=0.1, fc='red', ec='red', length_includes_head=True)

# Labels and title
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J(\theta)$')
plt.title('Gradient Descent on Quadratic Function')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('week2_gd_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Week 3: Linear regression plots
# Data points
x = [1, 2, 3]
y = [2, 4, 6]

# Plot data points and linear fit
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot([0, 4], [0, 8], color='red', label='y = 2x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Points and Linear Fit')
plt.legend()
plt.grid(True)
plt.savefig('week3_data_plot.png')
plt.close()

# Cost function J(theta) = (7/3)theta^2 - (28/3)theta + 28/3
def J_lin(theta):
    return (7/3) * theta**2 - (28/3) * theta + 28/3

theta_vals = np.linspace(0, 4, 100)
J_vals = J_lin(theta_vals)

# Gradient descent for linear regression
theta_lin = 0.0
alpha = 0.3
thetas_lin = [theta_lin]
Js_lin = [J_lin(theta_lin)]

for i in range(3):
    grad = (14/3) * (theta_lin - 2)
    theta_lin = theta_lin - alpha * grad
    thetas_lin.append(theta_lin)
    Js_lin.append(J_lin(theta_lin))

# Plot gradient descent on cost function
plt.figure(figsize=(8, 6))
plt.plot(theta_vals, J_vals, color='blue', label='Cost function')
plt.scatter(thetas_lin, Js_lin, color='red', label='Gradient descent steps')
for i in range(len(thetas_lin) - 1):
    plt.arrow(thetas_lin[i], Js_lin[i], thetas_lin[i+1] - thetas_lin[i], Js_lin[i+1] - Js_lin[i],
              head_width=0.05, head_length=0.1, fc='red', ec='red')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J(\theta)$')
plt.title('Gradient Descent on Linear Regression Cost Function')
plt.legend()
plt.grid(True)
plt.savefig('week3_gd_plot.png')
plt.close()