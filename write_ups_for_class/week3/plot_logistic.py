import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Figure 1: Sigmoid Function
# =============================================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-6, 6, 200)
sig_z = sigmoid(z)

plt.figure(figsize=(8, 5))
plt.plot(z, sig_z, 'b-', linewidth=2, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label=r'$\sigma(0) = 0.5$')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axhline(y=1, color='k', linewidth=0.5, linestyle=':')
plt.xlabel(r'$z$', fontsize=12)
plt.ylabel(r'$\sigma(z)$', fontsize=12)
plt.title('The Sigmoid Function', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.savefig('sigmoid_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: Why Linear Regression Fails for Classification
# =============================================================================
# Simple example showing linear regression vs logistic regression
x_lin = np.array([1, 2, 3, 4, 5, 6])
y_lin = np.array([0, 0, 0, 1, 1, 1])

# Fit linear regression (simple line fit)
slope = 0.2
intercept = -0.1
x_plot = np.linspace(0, 7, 100)
y_linear = slope * x_plot + intercept

plt.figure(figsize=(8, 5))
plt.scatter(x_lin, y_lin, c=['green' if yi == 0 else 'blue' for yi in y_lin], 
            s=100, zorder=5, edgecolors='k')
plt.plot(x_plot, y_linear, 'r--', linewidth=2, label='Linear Regression')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axhline(y=1, color='k', linewidth=0.5)
plt.fill_between(x_plot, -0.3, 0, alpha=0.1, color='red', label='Invalid region (y < 0)')
plt.fill_between(x_plot, 1, 1.3, alpha=0.1, color='red', label='Invalid region (y > 1)')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$y$', fontsize=12)
plt.title('Problem: Linear Regression for Classification', fontsize=14)
plt.legend(fontsize=10)
plt.ylim(-0.3, 1.3)
plt.xlim(0, 7)
plt.grid(True, alpha=0.3)
plt.savefig('linear_vs_logistic.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 3: 2D Classification Data with Decision Boundary
# =============================================================================
# Simple 2D dataset: 4 well-separated points
# Class 0 (green): (1, 1), (2, 2)
# Class 1 (blue): (3, 1), (4, 2)
X = np.array([[1, 1], [2, 2], [3, 1], [4, 2]])
y = np.array([0, 0, 1, 1])

plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='green', marker='^', s=150, 
            label='Class 0', edgecolors='k', zorder=5)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', s=150, 
            label='Class 1', edgecolors='k', zorder=5)
plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title('2D Classification Dataset', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 5)
plt.ylim(0, 3)
plt.savefig('data_2d.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: Gradient Descent Iterations with Decision Boundary
# =============================================================================
# Perform actual gradient descent on the 4-point dataset
def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

def compute_cost(X, y, w, b):
    m = len(y)
    predictions = predict(X, w, b)
    # Add small epsilon to avoid log(0)
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    cost = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Initialize parameters
w = np.array([0.0, 0.0])
b = 0.0
alpha = 0.5
n_iterations = 100

# Store history
w_history = [w.copy()]
b_history = [b]
cost_history = [compute_cost(X, y, w, b)]

# Gradient descent
for i in range(n_iterations):
    m = len(y)
    predictions = predict(X, w, b)
    
    dw = (1/m) * np.dot(X.T, (predictions - y))
    db = (1/m) * np.sum(predictions - y)
    
    w = w - alpha * dw
    b = b - alpha * db
    
    w_history.append(w.copy())
    b_history.append(b)
    cost_history.append(compute_cost(X, y, w, b))

# Plot decision boundary evolution at specific iterations
iterations_to_plot = [0, 10, 50, 99]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, iteration in enumerate(iterations_to_plot):
    ax = axes[idx]
    w_iter = w_history[iteration]
    b_iter = b_history[iteration]
    
    ax.scatter(X[y==0, 0], X[y==0, 1], c='green', marker='^', s=150, 
               label='Class 0', edgecolors='k', zorder=5)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', s=150, 
               label='Class 1', edgecolors='k', zorder=5)
    
    # Decision boundary: w1*x1 + w2*x2 + b = 0
    # => x2 = -(w1*x1 + b) / w2
    if abs(w_iter[1]) > 0.01:  # Avoid division by zero
        x1_vals = np.linspace(0, 5, 100)
        x2_vals = -(w_iter[0] * x1_vals + b_iter) / w_iter[1]
        valid = (x2_vals >= 0) & (x2_vals <= 3)
        ax.plot(x1_vals[valid], x2_vals[valid], 'k-', linewidth=2, label='Decision Boundary')
    
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_title(f'Iteration {iteration}: Cost = {cost_history[iteration]:.4f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)

plt.tight_layout()
plt.savefig('gd_iterations.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 5: Cost Function Convergence
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history)), cost_history, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
plt.title('Cost Function Convergence', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('cost_convergence.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 6: Final Decision Boundary with Probability Regions
# =============================================================================
# Create mesh grid for probability surface
h = 0.02
x1_min, x1_max = 0, 5
x2_min, x2_max = 0, 3
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                        np.arange(x2_min, x2_max, h))
grid_points = np.c_[xx1.ravel(), xx2.ravel()]

# Final weights
w_final = w_history[-1]
b_final = b_history[-1]
probs = predict(grid_points, w_final, b_final)
probs = probs.reshape(xx1.shape)

plt.figure(figsize=(10, 7))
contour = plt.contourf(xx1, xx2, probs, 25, cmap='RdYlGn_r', alpha=0.8)
plt.colorbar(contour, label='P(y=1)')

# Decision boundary
if abs(w_final[1]) > 0.01:
    x1_vals = np.linspace(0, 5, 100)
    x2_vals = -(w_final[0] * x1_vals + b_final) / w_final[1]
    valid = (x2_vals >= 0) & (x2_vals <= 3)
    plt.plot(x1_vals[valid], x2_vals[valid], 'k--', linewidth=2, label='Decision Boundary (P=0.5)')

plt.scatter(X[y==0, 0], X[y==0, 1], c='white', marker='^', s=200, 
            label='Class 0', edgecolors='k', linewidths=2, zorder=5)
plt.scatter(X[y==1, 0], X[y==1, 1], c='black', marker='s', s=200, 
            label='Class 1', edgecolors='white', linewidths=2, zorder=5)
plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title('Logistic Regression Probability Surface', fontsize=14)
plt.legend(fontsize=11)
plt.savefig('probability_surface.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Print values for LaTeX document
# =============================================================================
print("=" * 60)
print("VALUES FOR LATEX DOCUMENT")
print("=" * 60)
print(f"\nDataset:")
print(f"  Point 1: x = (1, 1), y = 0")
print(f"  Point 2: x = (2, 2), y = 0")
print(f"  Point 3: x = (3, 1), y = 1")
print(f"  Point 4: x = (4, 2), y = 1")

print(f"\nInitial parameters: w = [0, 0], b = 0")
print(f"Learning rate: α = {alpha}")

print(f"\nIteration results:")
for i in [0, 1, 2, 3, 10, 50, 99]:
    if i < len(w_history):
        print(f"  Iter {i}: w = [{w_history[i][0]:.4f}, {w_history[i][1]:.4f}], "
              f"b = {b_history[i]:.4f}, Cost = {cost_history[i]:.4f}")

print(f"\nFinal parameters:")
print(f"  w = [{w_final[0]:.4f}, {w_final[1]:.4f}]")
print(f"  b = {b_final:.4f}")
print(f"  Final Cost = {cost_history[-1]:.4f}")

# Decision boundary equation
print(f"\nDecision boundary: {w_final[0]:.4f}*x1 + {w_final[1]:.4f}*x2 + {b_final:.4f} = 0")
if abs(w_final[1]) > 0.01:
    slope = -w_final[0] / w_final[1]
    intercept = -b_final / w_final[1]
    print(f"  => x2 = {slope:.4f}*x1 + {intercept:.4f}")

print("\nPlots saved: sigmoid_plot.png, linear_vs_logistic.png, data_2d.png,")
print("             gd_iterations.png, cost_convergence.png, probability_surface.png")
