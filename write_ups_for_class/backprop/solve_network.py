import numpy as np
import matplotlib.pyplot as plt

"""
Neural Network Forward and Backward Pass Calculator
Solves the simple network with verification of all math
"""

# Define sigmoid function
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z))"""
    return z * (1 - z)

# Network parameters
print("=" * 70)
print("NEURAL NETWORK FORWARD AND BACKWARD PASS")
print("=" * 70)

# Inputs
x1 = 0.35
x2 = 0.7
print(f"\n--- INPUTS ---")
print(f"x1 = {x1}")
print(f"x2 = {x2}")

# Weights
w_x1_h1 = 0.2
w_x2_h1 = 0.2
w_x1_h2 = 0.3
w_x2_h2 = 0.3
w_h1_o = 0.3
w_h2_o = 0.9

print(f"\n--- WEIGHTS ---")
print(f"x1 -> h1: {w_x1_h1}")
print(f"x2 -> h1: {w_x2_h1}")
print(f"x1 -> h2: {w_x1_h2}")
print(f"x2 -> h2: {w_x2_h2}")
print(f"h1 -> o:  {w_h1_o}")
print(f"h2 -> o:  {w_h2_o}")

# Target and learning rate
y = 0.5  # Target output
eta = 1.0  # Learning rate

print(f"\n--- TRAINING PARAMETERS ---")
print(f"Target output (y): {y}")
print(f"Learning rate (eta): {eta}")

# ============================================================================
# FORWARD PASS
# ============================================================================
print("\n" + "=" * 70)
print("FORWARD PASS")
print("=" * 70)

# Compute z_h1 (pre-activation for hidden node 1)
z_h1 = x1 * w_x1_h1 + x2 * w_x2_h1
print(f"\n--- Hidden Node 1 (h1) ---")
print(f"z_h1 = x1 * w_x1_h1 + x2 * w_x2_h1")
print(f"z_h1 = {x1} * {w_x1_h1} + {x2} * {w_x2_h1}")
print(f"z_h1 = {x1 * w_x1_h1} + {x2 * w_x2_h1}")
print(f"z_h1 = {z_h1}")

# Apply sigmoid to get h1
h1 = sigmoid(z_h1)
print(f"\nh1 = sigmoid(z_h1) = sigmoid({z_h1})")
print(f"h1 = 1 / (1 + e^(-{z_h1}))")
print(f"h1 = {h1:.6f}")

# Compute z_h2 (pre-activation for hidden node 2)
z_h2 = x1 * w_x1_h2 + x2 * w_x2_h2
print(f"\n--- Hidden Node 2 (h2) ---")
print(f"z_h2 = x1 * w_x1_h2 + x2 * w_x2_h2")
print(f"z_h2 = {x1} * {w_x1_h2} + {x2} * {w_x2_h2}")
print(f"z_h2 = {z_h2}")
print(f"h2 = sigmoid({z_h2}) = {sigmoid(z_h2):.6f}")
h2 = sigmoid(z_h2)

# Compute z_o (pre-activation for output)
z_o = h1 * w_h1_o + h2 * w_h2_o
print(f"\n--- Output Node (o) ---")
print(f"z_o = h1 * w_h1_o + h2 * w_h2_o")
print(f"z_o = {h1:.6f} * {w_h1_o} + {h2:.6f} * {w_h2_o}")
print(f"z_o = {h1 * w_h1_o:.6f} + {h2 * w_h2_o:.6f}")
print(f"z_o = {z_o:.6f}")

# Apply sigmoid to get output
o = sigmoid(z_o)
print(f"\no = sigmoid(z_o) = sigmoid({z_o:.6f})")
print(f"o = 1 / (1 + e^(-{z_o:.6f}))")
print(f"o = {o:.6f}")

# ============================================================================
# BACKWARD PASS (BACKPROPAGATION)
# ============================================================================
print("\n" + "=" * 70)
print("BACKWARD PASS")
print("=" * 70)

# Compute sigmoid derivative for output
sigma_prime_o = sigmoid_derivative(o)
print(f"\n--- Output Sigmoid Derivative ---")
print(f"sigma'(z_o) = o * (1 - o)")
print(f"sigma'(z_o) = {o:.6f} * (1 - {o:.6f})")
print(f"sigma'(z_o) = {sigma_prime_o:.6f}")

# Output delta (with sigmoid derivative - proper MSE loss)
delta_o = (o - y) * sigma_prime_o
print(f"\n--- Output Delta ---")
print(f"For MSE loss: delta_o = (o - y) * sigma'(z_o)")
print(f"delta_o = ({o:.6f} - {y}) * {sigma_prime_o:.6f}")
print(f"delta_o = {o - y:.6f} * {sigma_prime_o:.6f}")
print(f"delta_o = {delta_o:.6f}")

# Compute sigmoid derivatives for hidden nodes
sigma_prime_h1 = sigmoid_derivative(h1)
sigma_prime_h2 = sigmoid_derivative(h2)

print(f"\n--- Sigmoid Derivatives (Hidden Nodes) ---")
print(f"sigma'(z_h1) = h1 * (1 - h1)")
print(f"sigma'(z_h1) = {h1:.6f} * (1 - {h1:.6f})")
print(f"sigma'(z_h1) = {h1:.6f} * {1 - h1:.6f}")
print(f"sigma'(z_h1) = {sigma_prime_h1:.6f}")

print(f"\nsigma'(z_h2) = h2 * (1 - h2)")
print(f"sigma'(z_h2) = {h2:.6f} * (1 - {h2:.6f})")
print(f"sigma'(z_h2) = {sigma_prime_h2:.6f}")

# Hidden deltas (using chain rule)
delta_h1 = delta_o * w_h1_o * sigma_prime_h1
delta_h2 = delta_o * w_h2_o * sigma_prime_h2

print(f"\n--- Hidden Deltas (Using Chain Rule) ---")
print(f"delta_h1 = delta_o * w_h1_o * sigma'(z_h1)")
print(f"delta_h1 = {delta_o:.6f} * {w_h1_o} * {sigma_prime_h1:.6f}")
print(f"delta_h1 = {delta_h1:.6f}")

print(f"\ndelta_h2 = delta_o * w_h2_o * sigma'(z_h2)")
print(f"delta_h2 = {delta_o:.6f} * {w_h2_o} * {sigma_prime_h2:.6f}")
print(f"delta_h2 = {delta_h2:.6f}")

# ============================================================================
# WEIGHT UPDATES
# ============================================================================
print("\n" + "=" * 70)
print("WEIGHT UPDATES")
print("=" * 70)

# Store old weights for comparison
old_weights = {
    'w_x1_h1': w_x1_h1,
    'w_x2_h1': w_x2_h1,
    'w_x1_h2': w_x1_h2,
    'w_x2_h2': w_x2_h2,
    'w_h1_o': w_h1_o,
    'w_h2_o': w_h2_o
}

# Update input -> hidden weights (Gradient Descent: w_new = w_old - eta * delta * input)
delta_w_x1_h1 = eta * delta_h1 * x1
w_x1_h1_new = w_x1_h1 - delta_w_x1_h1

delta_w_x2_h1 = eta * delta_h1 * x2
w_x2_h1_new = w_x2_h1 - delta_w_x2_h1

delta_w_x1_h2 = eta * delta_h2 * x1
w_x1_h2_new = w_x1_h2 - delta_w_x1_h2

delta_w_x2_h2 = eta * delta_h2 * x2
w_x2_h2_new = w_x2_h2 - delta_w_x2_h2

# Update hidden -> output weights
delta_w_h1_o = eta * delta_o * h1
w_h1_o_new = w_h1_o - delta_w_h1_o

delta_w_h2_o = eta * delta_o * h2
w_h2_o_new = w_h2_o - delta_w_h2_o

print(f"\n--- Input -> Hidden Weights ---")
print(f"For gradient descent: w_new = w_old - eta * delta * input")
print(f"delta_w_x1_h1 = eta * delta_h1 * x1 = {eta} * {delta_h1:.6f} * {x1} = {delta_w_x1_h1:.6f}")
print(f"w_x1_h1: {w_x1_h1} - {delta_w_x1_h1:.6f} = {w_x1_h1_new:.6f}")

print(f"\ndelta_w_x2_h1 = eta * delta_h1 * x2 = {eta} * {delta_h1:.6f} * {x2} = {delta_w_x2_h1:.6f}")
print(f"w_x2_h1: {w_x2_h1} - {delta_w_x2_h1:.6f} = {w_x2_h1_new:.6f}")

print(f"\ndelta_w_x1_h2 = eta * delta_h2 * x1 = {eta} * {delta_h2:.6f} * {x1} = {delta_w_x1_h2:.6f}")
print(f"w_x1_h2: {w_x1_h2} - {delta_w_x1_h2:.6f} = {w_x1_h2_new:.6f}")

print(f"\ndelta_w_x2_h2 = eta * delta_h2 * x2 = {eta} * {delta_h2:.6f} * {x2} = {delta_w_x2_h2:.6f}")
print(f"w_x2_h2: {w_x2_h2} - {delta_w_x2_h2:.6f} = {w_x2_h2_new:.6f}")

print(f"\n--- Hidden -> Output Weights ---")
print(f"delta_w_h1_o = eta * delta_o * h1 = {eta} * {delta_o:.6f} * {h1:.6f} = {delta_w_h1_o:.6f}")
print(f"w_h1_o: {w_h1_o} - {delta_w_h1_o:.6f} = {w_h1_o_new:.6f}")

print(f"\ndelta_w_h2_o = eta * delta_o * h2 = {eta} * {delta_o:.6f} * {h2:.6f} = {delta_w_h2_o:.6f}")
print(f"w_h2_o: {w_h2_o} - {delta_w_h2_o:.6f} = {w_h2_o_new:.6f}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF ALL VALUES")
print("=" * 70)

print("\n--- Forward Pass Activations ---")
print(f"{'Node':<10} {'Pre-activation':<20} {'Activation':<20}")
print("-" * 50)
print(f"{'h1':<10} {z_h1:<20.6f} {h1:<20.6f}")
print(f"{'h2':<10} {z_h2:<20.6f} {h2:<20.6f}")
print(f"{'o':<10} {z_o:<20.6f} {o:<20.6f}")

print("\n--- Backward Pass Deltas ---")
print(f"{'Node':<10} {'Delta':<20}")
print("-" * 30)
print(f"{'delta_o':<10} {delta_o:<20.6f}")
print(f"{'delta_h1':<10} {delta_h1:<20.6f}")
print(f"{'delta_h2':<10} {delta_h2:<20.6f}")

print("\n--- Weight Updates ---")
print(f"{'Weight':<15} {'Old':<15} {'Delta':<15} {'New':<15}")
print("-" * 60)
print(f"{'w_x1_h1':<15} {w_x1_h1:<15.6f} {delta_w_x1_h1:<15.6f} {w_x1_h1_new:<15.6f}")
print(f"{'w_x2_h1':<15} {w_x2_h1:<15.6f} {delta_w_x2_h1:<15.6f} {w_x2_h1_new:<15.6f}")
print(f"{'w_x1_h2':<15} {w_x1_h2:<15.6f} {delta_w_x1_h2:<15.6f} {w_x1_h2_new:<15.6f}")
print(f"{'w_x2_h2':<15} {w_x2_h2:<15.6f} {delta_w_x2_h2:<15.6f} {w_x2_h2_new:<15.6f}")
print(f"{'w_h1_o':<15} {w_h1_o:<15.6f} {delta_w_h1_o:<15.6f} {w_h1_o_new:<15.6f}")
print(f"{'w_h2_o':<15} {w_h2_o:<15.6f} {delta_w_h2_o:<15.6f} {w_h2_o_new:<15.6f}")

# ============================================================================
# SECOND FORWARD PASS (with updated weights)
# ============================================================================
print("\n" + "=" * 70)
print("SECOND FORWARD PASS (WITH UPDATED WEIGHTS)")
print("=" * 70)

# Use updated weights
z_h1_new = x1 * w_x1_h1_new + x2 * w_x2_h1_new
h1_new = sigmoid(z_h1_new)

z_h2_new = x1 * w_x1_h2_new + x2 * w_x2_h2_new
h2_new = sigmoid(z_h2_new)

z_o_new = h1_new * w_h1_o_new + h2_new * w_h2_o_new
o_new = sigmoid(z_o_new)

print(f"\n--- Hidden Nodes (Updated Weights) ---")
print(f"z_h1 = x1 * w_x1_h1_new + x2 * w_x2_h1_new")
print(f"z_h1 = {x1} * {w_x1_h1_new:.6f} + {x2} * {w_x2_h1_new:.6f}")
print(f"z_h1 = {z_h1_new:.6f}")
print(f"h1 = sigmoid({z_h1_new:.6f}) = {h1_new:.6f}")

print(f"\nz_h2 = {x1} * {w_x1_h2_new:.6f} + {x2} * {w_x2_h2_new:.6f}")
print(f"z_h2 = {z_h2_new:.6f}")
print(f"h2 = sigmoid({z_h2_new:.6f}) = {h2_new:.6f}")

print(f"\n--- Output Node (Updated Weights) ---")
print(f"z_o = h1 * w_h1_o_new + h2 * w_h2_o_new")
print(f"z_o = {h1_new:.6f} * {w_h1_o_new:.6f} + {h2_new:.6f} * {w_h2_o_new:.6f}")
print(f"z_o = {z_o_new:.6f}")
print(f"o = sigmoid({z_o_new:.6f}) = {o_new:.6f}")

# Compare errors
error_before = abs(o - y)
error_after = abs(o_new - y)
error_reduction = error_before - error_after
percent_reduction = (error_reduction / error_before) * 100

print(f"\n--- Error Comparison ---")
print(f"Target output: y = {y}")
print(f"Output before weight update: {o:.6f}")
print(f"Error before: |{o:.6f} - {y}| = {error_before:.6f}")
print(f"\nOutput after weight update: {o_new:.6f}")
print(f"Error after: |{o_new:.6f} - {y}| = {error_after:.6f}")
print(f"\nError reduction: {error_reduction:.6f}")
print(f"Percent reduction: {percent_reduction:.2f}%")

print("\n" + "=" * 70)
