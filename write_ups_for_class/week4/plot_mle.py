import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma as gamma_dist
from scipy.optimize import minimize
import networkx as nx

np.random.seed(42)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

def neg_ll_normal(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(norm.logpdf(data, mu, sigma))

def fit_mle_normal(data):
    res = minimize(neg_ll_normal, [np.mean(data), np.std(data)],
                   args=(data,), bounds=[(None, None), (1e-6, None)])
    return res.x

def neg_ll_gamma(params, data):
    k, theta = params
    if k <= 0 or theta <= 0:
        return np.inf
    return -np.sum(gamma_dist.logpdf(data, a=k, scale=theta))

# ── 1. MLE SAMPLE SIZE ────────────────────────────────────────────────────────
for n in [10, 100, 10000]:
    data = np.random.normal(0, 1, n)
    mu_fit, sigma_fit = fit_mle_normal(data)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(data, bins=30, density=True, alpha=0.5, color='steelblue', label='Samples')
    x = np.linspace(-4, 4, 200)
    ax.plot(x, norm.pdf(x, 0, 1), 'k--', lw=1.5, label='True N(0,1)')
    ax.plot(x, norm.pdf(x, mu_fit, sigma_fit), 'r-', lw=2,
            label=f'MLE: N({mu_fit:.2f},{sigma_fit:.2f})')
    ax.set_title(f'MLE Fit — n = {n}')
    ax.set_xlabel('x'); ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'mle_fit_{n}.png', dpi=120)
    plt.close()

# ── 2. MOM vs MLE — Gamma ─────────────────────────────────────────────────────
true_k, true_theta = 3.0, 2.0
np.random.seed(7)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for idx, n in enumerate([20, 100, 1000]):
    data = np.random.gamma(true_k, true_theta, n)
    m1 = np.mean(data); m2 = np.var(data, ddof=1)
    k_mom = m1**2 / m2; theta_mom = m2 / m1
    res = minimize(neg_ll_gamma, [k_mom, theta_mom], args=(data,), method='Nelder-Mead')
    k_mle, theta_mle = res.x
    x = np.linspace(1e-3, data.max() * 1.25, 300)
    axes[idx].hist(data, bins=20, density=True, alpha=0.35, color='gray', label='Data')
    axes[idx].plot(x, gamma_dist.pdf(x, a=true_k,  scale=true_theta),
                   'k--', lw=2, label=f'True Γ({true_k},{true_theta})')
    axes[idx].plot(x, gamma_dist.pdf(x, a=k_mom,   scale=theta_mom),
                   'b-',  lw=2, label=f'MoM k={k_mom:.2f}')
    axes[idx].plot(x, gamma_dist.pdf(x, a=k_mle,   scale=theta_mle),
                   'r-',  lw=2, label=f'MLE k={k_mle:.2f}')
    axes[idx].set_title(f'n = {n}'); axes[idx].set_xlabel('x'); axes[idx].set_ylabel('Density')
    axes[idx].legend(fontsize=7)
plt.suptitle('Method of Moments vs MLE — Gamma Distribution', fontsize=12)
plt.tight_layout()
plt.savefig('mom_vs_mle.png', dpi=120, bbox_inches='tight')
plt.close()

# ── 3. COVARIANCE ─────────────────────────────────────────────────────────────
np.random.seed(42)
n = 150
X = np.random.randn(n)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
datasets = [
    (X, X + 0.5 * np.random.randn(n), '#2ecc71', 'Positive'),
    (np.random.randn(n), np.random.randn(n), '#3498db', 'Near Zero'),
    (X, -X + 0.5 * np.random.randn(n), '#e74c3c', 'Negative'),
]
for ax, (xi, yi, color, label) in zip(axes, datasets):
    cov = np.cov(xi, yi)[0, 1]
    ax.scatter(xi, yi, alpha=0.5, color=color, s=18)
    ax.set_title(f'{label} Covariance\nCov = {cov:.2f}')
    ax.set_xlabel('Feature X'); ax.set_ylabel('Feature Y')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('covariance_examples.png', dpi=120)
plt.close()

cov_matrix = np.array([
    [1.0,  0.8,  0.2, -0.3,  0.1],
    [0.8,  1.0,  0.3, -0.2,  0.2],
    [0.2,  0.3,  1.0,  0.1, -0.4],
    [-0.3,-0.2,  0.1,  1.0,  0.3],
    [0.1,  0.2, -0.4,  0.3,  1.0],
])
data_mv = np.random.multivariate_normal(np.zeros(5), cov_matrix, 200)
corr_matrix = np.corrcoef(data_mv.T)
labels_f = [f'F{i}' for i in range(5)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, mat, title in zip(axes, [cov_matrix, corr_matrix], ['Covariance Matrix', 'Correlation Matrix']):
    im = ax.imshow(mat, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(labels_f); ax.set_yticklabels(labels_f)
    ax.set_title(title)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                    color='white' if abs(mat[i,j]) > 0.5 else 'black', fontsize=8)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('covariance_matrix.png', dpi=120)
plt.close()

np.random.seed(42); m = 100; X1 = np.random.randn(m)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X1, X1 + 0.1*np.random.randn(m), alpha=0.5, s=20)
axes[0].plot([-3,3],[-3,3],'r--',alpha=0.6)
axes[0].set_title('Redundant Features\n(High Covariance)')
axes[0].set_xlabel('Feature X1'); axes[0].set_ylabel('Feature X2')
axes[0].grid(True,alpha=0.3); axes[0].set_xlim([-3,3]); axes[0].set_ylim([-3,3])
axes[1].scatter(np.random.randn(m), np.random.randn(m), alpha=0.5, s=20, color='#e67e22')
axes[1].set_title('Independent Features\n(Near-Zero Covariance)')
axes[1].set_xlabel('Feature X1'); axes[1].set_ylabel('Feature X2')
axes[1].grid(True,alpha=0.3); axes[1].set_xlim([-3,3]); axes[1].set_ylim([-3,3])
plt.tight_layout()
plt.savefig('covariance_feature_selection.png', dpi=120)
plt.close()

# ── 4. KL DIVERGENCE ─────────────────────────────────────────────────────────
kl_values, diffs = [], []
for mu in [0.5, 1.0, 2.0]:
    kl = 0.5 * mu**2
    kl_values.append(kl); diffs.append(mu)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.linspace(-5, 6, 300)
    ax.plot(x, norm.pdf(x, 0, 1), lw=2, label='P = N(0,1)')
    ax.plot(x, norm.pdf(x, mu, 1), lw=2, label=f'Q = N({mu},1)')
    ax.fill_between(x, norm.pdf(x, 0, 1), norm.pdf(x, mu, 1), alpha=0.2)
    ax.set_title(f'KL(P||Q) = {kl:.3f}  (mean diff = {mu})')
    ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('Density')
    plt.tight_layout()
    plt.savefig(f'kl_dist_mu{str(mu).replace(".","p")}.png', dpi=120)
    plt.close()

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(diffs, kl_values, 'o-', lw=2, ms=8)
ax.set_xlabel('Mean Difference'); ax.set_ylabel('KL Divergence')
ax.set_title('KL Divergence vs Mean Difference')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kl_vs_diff.png', dpi=120)
plt.close()

# ── 5. SOFTMAX FIGURES ────────────────────────────────────────────────────────
bar_c = ['#ff6b6b','#4ecdc4','#45b7d1']

# 5a. Examples
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (logits, sub, col) in zip(axes, [
        (np.array([1,2,3]), 'Unequal logits', '#1f77b4'),
        (np.array([0,0,0]), 'Equal logits',   '#ff7f0e'),
        (np.array([5,1,1]), 'One dominant',   '#2ca02c')]):
    p = softmax(logits)
    ax.bar(range(3), p, color=col, alpha=0.75)
    for i,v in enumerate(p): ax.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_ylim([0,1]); ax.set_xticks(range(3)); ax.set_xticklabels(['C0','C1','C2'])
    ax.set_title(f'{sub}\nLogits: {logits}'); ax.set_ylabel('Probability')
plt.tight_layout()
plt.savefig('softmax_examples.png', dpi=120)
plt.close()

# 5b. Pipeline
logits = np.array([2.1, 0.5, -1.3]); probs = softmax(logits); pred = np.argmax(logits)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].bar(['C0','C1','C2'], logits, color=bar_c, alpha=0.75)
axes[0].axhline(0, color='k', lw=0.5); axes[0].set_title('1. NN Logits'); axes[0].set_ylabel('Value')
axes[0].grid(True,alpha=0.3,axis='y')
axes[1].bar(['C0','C1','C2'], probs, color=bar_c, alpha=0.75)
for i,p in enumerate(probs): axes[1].text(i, p+0.02, f'{p:.3f}', ha='center', fontsize=9)
axes[1].set_ylim([0,1]); axes[1].set_title('2. After Softmax'); axes[1].set_ylabel('Probability')
axes[1].grid(True,alpha=0.3,axis='y')
pp = np.zeros(3); pp[pred] = 1
axes[2].bar(['C0','C1','C2'], pp, color=bar_c, alpha=0.75)
axes[2].set_ylim([0,1]); axes[2].set_title(f'3. Predict → Class {pred}'); axes[2].set_ylabel('Probability')
axes[2].grid(True,alpha=0.3,axis='y')
plt.tight_layout()
plt.savefig('softmax_pipeline.png', dpi=120)
plt.close()

# 5c. Binary sigmoid curve
z = np.linspace(-5,5,200); sig = 1/(1+np.exp(-z))
fig, ax = plt.subplots(figsize=(6,3.5))
ax.plot(z, sig, lw=2, label='P(Class 1)')
ax.plot(z, 1-sig, lw=2, label='P(Class 0)')
ax.axhline(0.5, color='r', ls='--', alpha=0.5, label='Decision boundary')
ax.set_xlabel('Logit z'); ax.set_ylabel('Probability')
ax.set_title('Sigmoid — Binary Classification')
ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('softmax_curve.png', dpi=120)
plt.close()

# 5d. Temperature
fig, axes = plt.subplots(1, 3, figsize=(12,3.5))
lt = np.array([2.0,1.0,-0.5])
for ax, T, ttl in zip(axes, [0.5,1.0,2.0], ['T=0.5 (sharp)','T=1.0 (normal)','T=2.0 (soft)']):
    p = softmax(lt/T)
    ax.bar(['C0','C1','C2'], p, color=bar_c, alpha=0.75)
    for i,v in enumerate(p): ax.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_ylim([0,1]); ax.set_title(ttl); ax.set_ylabel('Probability')
    ax.grid(True,alpha=0.3,axis='y')
plt.tight_layout()
plt.savefig('softmax_temperature.png', dpi=120)
plt.close()

# 5e. Cross-entropy
zv = np.linspace(-5,5,200); sp, ce = [], []
for zval in zv:
    p = softmax(np.array([zval,0.0]))
    sp.append(p[0]); ce.append(-np.log(p[0]+1e-10))
fig,(a1,a2) = plt.subplots(1,2,figsize=(11,4))
a1.plot(zv,sp,lw=2,color='#2e86de'); a1.fill_between(zv,0,sp,alpha=0.2,color='#2e86de')
a1.axvline(0,color='r',ls='--',alpha=0.5); a1.set_xlabel('Logit (true class)'); a1.set_ylabel('P(true class)')
a1.set_title('Softmax Probability'); a1.grid(True,alpha=0.3)
a2.plot(zv,ce,lw=2,color='#a29bfe'); a2.fill_between(zv,0,np.minimum(ce,5),alpha=0.2,color='#a29bfe')
a2.axvline(0,color='r',ls='--',alpha=0.5); a2.set_xlabel('Logit (true class)'); a2.set_ylabel('Loss')
a2.set_ylim([0,5]); a2.set_title('Cross-Entropy Loss'); a2.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('softmax_cross_entropy.png', dpi=120)
plt.close()

# ── 6. NETWORK DIAGRAM — sigmoid vs softmax ───────────────────────────────────
def draw_network(ax, output_nodes, output_label, out_color, title):
    G = nx.DiGraph()
    layer_sizes = [3, 4, output_nodes]
    layer_names = ['input', 'hidden', 'output']
    pos, node_colors = {}, {}
    for l, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        for i in range(size):
            nid = f'{name}_{i}'
            G.add_node(nid)
            pos[nid] = (l * 2.5, i - (size - 1) / 2.0)
            node_colors[nid] = {'input': '#74b9ff', 'hidden': '#55efc4', 'output': out_color}[name]
    for l in range(len(layer_sizes) - 1):
        for i in range(layer_sizes[l]):
            for j in range(layer_sizes[l + 1]):
                G.add_edge(f'{layer_names[l]}_{i}', f'{layer_names[l+1]}_{j}')
    color_list = [node_colors[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_list, node_size=650)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=10, edge_color='#636e72', alpha=0.5)
    input_labels = {f'input_{i}': f'$x_{i}$' for i in range(3)}
    nx.draw_networkx_labels(G, pos, labels=input_labels, font_size=9, ax=ax)
    for l, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        ax.text(l * 2.5, -(size-1)/2.0 - 0.75, name.capitalize(),
                ha='center', fontsize=9, style='italic', color='#2d3436')
    x_out = (len(layer_sizes)-1) * 2.5
    y_top  = (layer_sizes[-1]-1) / 2.0 + 1.0
    ax.text(x_out, y_top, output_label, ha='center', fontsize=10, fontweight='bold',
            color=out_color, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=out_color))
    ax.set_title(title, fontsize=11, pad=14)
    ax.axis('off')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
draw_network(ax1, output_nodes=1, output_label='Sigmoid\n(Binary: 0 or 1)',
             out_color='#e17055', title='Sigmoid Output Layer\n(Binary Classification)')
draw_network(ax2, output_nodes=3, output_label='Softmax\n(Multi-class: C0/C1/C2)',
             out_color='#6c5ce7', title='Softmax Output Layer\n(Multi-class Classification)')
fig.suptitle('Neural Network Output Layer: Sigmoid vs Softmax', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('network_diagram.png', dpi=130, bbox_inches='tight')
plt.close()

print("All plots generated successfully.")
