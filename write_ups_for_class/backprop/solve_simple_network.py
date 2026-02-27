import numpy as np

"""
Simple 1-to-1 Neural Network: Forward and Backward Pass
Verifies all mathematical calculations shown in the LaTeX document
"""

print("=" * 80)
print("SIMPLE 1-TO-1 NEURAL NETWORK: BACKPROPAGATION EXAMPLE")
print("=" * 80)

# ============================================================================
# PARAMETERS
# ============================================================================
x = 0.5         # Input
w_initial = 0.8 # Initial weight
y = 0.6         # Target output
eta = 1.0       # Learning rate

print(f"\n{'INPUT PARAMETERS':=^80}")
print(f"Input value:           x = {x}")
print(f"Initial weight:        w = {w_initial}")
print(f"Target output:         y = {y}")
print(f"Learning rate:         η = {eta}")

# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))"""
    sig = sigmoid(z)
    return sig * (1 - sig)

# ============================================================================
# FIRST FORWARD PASS
# ============================================================================
print(f"\n{'FIRST FORWARD PASS':=^80}")

# Pre-activation
z = x * w_initial
print(f"\nStep 1a: Compute pre-activation")
print(f"  z = x · w = ({x}) · ({w_initial}) = {z}")

# Activation
o = sigmoid(z)
print(f"\nStep 1b: Apply sigmoid activation")
print(f"  σ(z) = 1 / (1 + e^(-z))")
print(f"  σ({z}) = 1 / (1 + e^({-z}))")
print(f"  e^({-z}) = {np.exp(-z):.6f}")
print(f"  1 + e^({-z}) = {1 + np.exp(-z):.6f}")
print(f"  o = {o:.6f}")

# Loss
loss = 0.5 * (o - y) ** 2
error = o - y
print(f"\nStep 2: Compute MSE Loss")
print(f"  L = (1/2) * (o - y)²")
print(f"  o - y = {o:.6f} - {y} = {error:.6f}")
print(f"  (o - y)² = ({error:.6f})² = {(o - y)**2:.8f}")
print(f"  L = (1/2) * {(o - y)**2:.8f} = {loss:.8f}")

print(f"\n{'FIRST FORWARD PASS SUMMARY':^80}")
print(f"  Input: x = {x}")
print(f"  Weight: w = {w_initial}")
print(f"  Pre-activation: z = {z}")
print(f"  Output: o = {o:.6f}")
print(f"  Target: y = {y}")
print(f"  Error: (o - y) = {error:.6f}")
print(f"  Loss: L = {loss:.8f}")

# ============================================================================
# BACKWARD PASS: CHAIN RULE
# ============================================================================
print(f"\n{'BACKWARD PASS: COMPUTING GRADIENT':=^80}")

# Term 1: dL/do
dL_do = o - y
print(f"\nTerm 1: ∂L/∂o (Loss with respect to output)")
print(f"  L = (1/2) * (o - y)²")
print(f"  ∂L/∂o = (o - y) = {dL_do:.6f}")

# Term 2: do/dz (sigmoid derivative)
do_dz = sigmoid_derivative(z)
print(f"\nTerm 2: ∂o/∂z (Sigmoid derivative)")
print(f"  σ'(z) = σ(z) * (1 - σ(z))")
print(f"  σ'({z}) = {o:.6f} * (1 - {o:.6f})")
print(f"  σ'({z}) = {o:.6f} * {1 - o:.6f}")
print(f"  ∂o/∂z = {do_dz:.6f}")

# Verify sigmoid derivative formula
sig_deriv_check = np.exp(-z) / ((1 + np.exp(-z))**2)
print(f"  Verification: e^(-z) / (1 + e^(-z))² = {sig_deriv_check:.6f} ✓")

# Term 3: dz/dw
dz_dw = x
print(f"\nTerm 3: ∂z/∂w (Pre-activation with respect to weight)")
print(f"  z = x · w")
print(f"  ∂z/∂w = x = {dz_dw}")

# Full gradient via chain rule
dL_dw = dL_do * do_dz * dz_dw
print(f"\nChain Rule Multiplication:")
print(f"  ∂L/∂w = (∂L/∂o) · (∂o/∂z) · (∂z/∂w)")
print(f"  ∂L/∂w = ({dL_do:.6f}) · ({do_dz:.6f}) · ({dz_dw})")
print(f"  ∂L/∂w = ({dL_do:.6f}) · ({do_dz * dz_dw:.6f})")
print(f"  ∂L/∂w = {dL_dw:.6f}")

# ============================================================================
# WEIGHT UPDATE: GRADIENT DESCENT
# ============================================================================
print(f"\n{'WEIGHT UPDATE (GRADIENT DESCENT)':=^80}")

w_new = w_initial - eta * dL_dw
print(f"\nWeight update rule:")
print(f"  w_new = w_old - η · (∂L/∂w)")
print(f"  w_new = {w_initial} - ({eta}) · ({dL_dw:.6f})")
print(f"  w_new = {w_initial} - ({eta * dL_dw:.6f})")
print(f"  w_new = {w_initial} + {-eta * dL_dw:.6f}")
print(f"  w_new = {w_new:.6f}")

print(f"\nWeight change: Δw = {w_new - w_initial:.6f}")

# ============================================================================
# SECOND FORWARD PASS
# ============================================================================
print(f"\n{'SECOND FORWARD PASS (WITH UPDATED WEIGHT)':=^80}")

# Pre-activation with new weight
z_new = x * w_new
print(f"\nStep 1a: Compute pre-activation with new weight")
print(f"  z_new = x · w_new = ({x}) · ({w_new:.6f}) = {z_new:.6f}")

# Activation with new pre-activation
o_new = sigmoid(z_new)
print(f"\nStep 1b: Apply sigmoid activation")
print(f"  o_new = σ({z_new:.6f}) = {o_new:.6f}")

# Loss with new output
loss_new = 0.5 * (o_new - y) ** 2
error_new = o_new - y
print(f"\nStep 2: Compute MSE Loss")
print(f"  o_new - y = {o_new:.6f} - {y} = {error_new:.6f}")
print(f"  L_new = (1/2) * ({error_new:.6f})² = {loss_new:.8f}")

print(f"\n{'SECOND FORWARD PASS SUMMARY':^80}")
print(f"  Weight: w_new = {w_new:.6f}")
print(f"  Pre-activation: z_new = {z_new:.6f}")
print(f"  Output: o_new = {o_new:.6f}")
print(f"  Target: y = {y}")
print(f"  Error: (o_new - y) = {error_new:.6f}")
print(f"  Loss: L_new = {loss_new:.8f}")

# ============================================================================
# COMPARISON: BEFORE VS AFTER
# ============================================================================
print(f"\n{'ERROR REDUCTION ANALYSIS':=^80}")

print(f"\n{'':<30} {'Before Update':<20} {'After Update':<20} {'Change':<15}")
print("-" * 85)
print(f"{'Weight':<30} {w_initial:<20.6f} {w_new:<20.6f} {w_new - w_initial:<15.6f}")
print(f"{'Pre-activation (z)':<30} {z:<20.6f} {z_new:<20.6f} {z_new - z:<15.6f}")
print(f"{'Output (o)':<30} {o:<20.6f} {o_new:<20.6f} {o_new - o:<15.6f}")
print(f"{'Error (o - y)':<30} {error:<20.6f} {error_new:<20.6f} {error_new - error:<15.6f}")
print(f"{'Absolute Error |o - y|':<30} {abs(error):<20.6f} {abs(error_new):<20.6f} {abs(error_new) - abs(error):<15.6f}")
print(f"{'Loss':<30} {loss:<20.8f} {loss_new:<20.8f} {loss_new - loss:<15.8f}")

# ============================================================================
# STATISTICS
# ============================================================================
abs_error_reduction = abs(error) - abs(error_new)
percent_error_reduction = (abs_error_reduction / abs(error)) * 100
loss_reduction = loss - loss_new
percent_loss_reduction = (loss_reduction / loss) * 100

print(f"\n{'IMPROVEMENT METRICS':=^80}")
print(f"Absolute error reduced by:     {abs_error_reduction:.6f} ({percent_error_reduction:.2f}%)")
print(f"Loss reduced by:               {loss_reduction:.8f} ({percent_loss_reduction:.2f}%)")
print(f"Distance to target decreased:  {percent_error_reduction:.2f}%")

print(f"\n{'CONCLUSION':=^80}")
print(f"✓ The output moved closer to the target")
print(f"✓ The loss decreased (moved in correct direction)")
print(f"✓ Backpropagation successfully computed gradients")
print(f"✓ Gradient descent successfully updated the weight")
print(f"\nOne iteration complete. Repeating this process many times will")
print(f"gradually reduce the loss to near-zero and train the network.")
print("=" * 80)
