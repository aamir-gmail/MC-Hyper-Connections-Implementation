# Manifold-Constrained Hyper-Connections (mHC) Implementation

A PyTorch implementation of the **Manifold-Constrained Hyper-Connections (mHC)** architecture proposed by DeepSeek-AI (2025).

This repository demonstrates how projecting residual connections onto the **Birkhoff polytope** (via the Sinkhorn-Knopp algorithm) allows for wider, multi-stream network topologies without the gradient explosion problems typical of standard Hyper-Connections.

## 📄 Paper Reference
**Title:** Manifold-Constrained Hyper-Connections  
**Authors:** Zhenda Xie, Yixuan Wei, et al. (DeepSeek-AI)  
**Link:** [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)

## 🚀 Key Features

* **Manifold Constraint:** Implements a custom `SinkhornProjection` layer that forces residual mixing matrices to be doubly stochastic.
* **Dynamic Routing:** Connection coefficients ($\mathcal{H}^{pre}, \mathcal{H}^{post}, \mathcal{H}^{res}$) are generated dynamically per-token based on input content.
* **Multi-Stream Topology:** Expands the residual stream by a factor of $n=4$ to increase expressivity while maintaining the memory footprint of smaller layers.
* **Stability:** Unlike standard residuals which can explode in deep/wide networks, mHC ensures signal norms remain bounded ($\le 1$).

## 🧠 Architecture

The model replaces the standard residual connection $x_{l+1} = x_l + F(x_l)$ with a manifold-constrained highway:

$$x_{l+1} = \mathcal{H}^{res} x_l + \mathcal{H}^{post} F(\mathcal{H}^{pre} x_l)$$

Where $\mathcal{H}^{res}$ is projected onto the Birkhoff polytope using the Sinkhorn-Knopp algorithm:

```python
# Core Stability Mechanism
def sinkhorn_projection(log_matrix, iterations=20):
    matrix = torch.exp(log_matrix - log_matrix.max()) # Numerical stability
    for _ in range(iterations):
        matrix /= matrix.sum(dim=-1, keepdim=True) # Row Norm
        matrix /= matrix.sum(dim=-2, keepdim=True) # Col Norm
    return matrix

# If you are looking for better results, please increase the size of the text fed into the model by at least 10x. 
