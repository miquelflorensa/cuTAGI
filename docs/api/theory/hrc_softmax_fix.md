# Hierarchical Softmax Normalization Fix

## Problem

For a hidden/output layer with $\mathtt{H}$ classes, the previous HRC Softmax used a complete tree with

$$
\mathtt{K}=\left\lceil \log_2(\mathtt{H}) \right\rceil
$$

levels, which implies $2^{\mathtt{K}}$ leaves.

When $\log_2(\mathtt{H})$ is not an integer, there are unused leaves:

$$
2^{\mathtt{K}}-\mathtt{H} > 0.
$$

The probability mass assigned to those unused leaves was not assigned to any class. Therefore, the class probabilities could satisfy

$$
\sum_{i=1}^{\mathtt{H}} P(h_i) < 1.
$$

Example:

$$
\mathtt{H}=10,\quad \mathtt{K}=4,\quad 2^{\mathtt{K}}=16,
$$

so with zero gate logits,

$$
\sum_{i=1}^{10} P(h_i)=\frac{10}{16}=0.625.
$$

## Solution

The fix replaces the complete tree with a hierarchical binary tree containing exactly $\mathtt{H}$ leaves and $\mathtt{H}-1$ internal gates:

$$
\bm{g}\in\mathbb{R}^{\mathtt{G}},\quad \mathtt{G}=\mathtt{H}-1.
$$

Each class $h_i$ is represented by one root-to-leaf path. The path length may differ across classes. Shorter paths are padded with dummy entries, and dummy entries are ignored during the backward update.

For each internal gate $g_j$, the left branch probability is

$$
\sigma_\mathtt{r}(g_j),
$$

and the right branch probability is

$$
1-\sigma_\mathtt{r}(g_j).
$$

The probability of class $h_i$ is the product of the branch probabilities along its path:

$$
P(h_i)=\prod_{j\in\operatorname{path}(h_i)} p_j.
$$

Since the tree has exactly $\mathtt{H}$ leaves and every leaf is a valid class,

$$
\sum_{i=1}^{\mathtt{H}} P(h_i)=1.
$$

## Algorithm

Given class interval $[a,b)$:

1. If $b-a=1$, stop. The interval is a leaf.
2. Create one gate for $[a,b)$.
3. Split the interval as

$$
\mathtt{L}=\left\lceil\frac{b-a}{2}\right\rceil,\quad
m=a+\mathtt{L}.
$$

4. Assign classes $[a,m)$ to the left branch:

$$
p_j=\sigma_\mathtt{r}(g_j).
$$

5. Assign classes $[m,b)$ to the right branch:

$$
p_j=1-\sigma_\mathtt{r}(g_j).
$$

6. Recurse on $[a,m)$ and $[m,b)$.

This algorithm fixes the unused-leaf normalization bug.

## Uniform prior via per-gate bias

With the new tree, $\sum_i P(h_i)=1$, but odd splits ($L\neq b-a-L$) make leaves at different depths carry different prior mass when gate logits are zero. To recover a uniform prior $P(h_i)=1/\mathtt{H}$, each gate $g_j$ that splits an interval of size $\mathtt{N}=b-a$ into $\mathtt{L}=\lceil \mathtt{N}/2 \rceil$ and $\mathtt{R}=\mathtt{N}-\mathtt{L}$ carries a fixed prior offset

$$
b_j = \frac{1}{\alpha}\,\Phi^{-1}\!\left(\frac{\mathtt{L}}{\mathtt{N}}\right),
$$

with $\alpha=3$ matching the slope used inside `obs_to_class`. The gate activation becomes

$$
p_j = \Phi\!\left(\frac{m_{z,j}+b_j}{\sqrt{(1/\alpha)^2+S_{z,j}}}\right),
$$

so at initialization ($m_{z,j}=0$, $S_{z,j}\to 0$) we get $p_j=\mathtt{L}/\mathtt{N}$ and every leaf reaches probability $1/\mathtt{H}$. Even splits keep $b_j=0$.

In the backward pass, the bias is absorbed by subtracting $b_j$ from the fictive observation in `labels_to_hrs`, so the updater's residual $(\,\text{obs} - b_j) - m_{z,j}$ stays consistent with the shifted forward activation. The network learns deviations from the uniform prior rather than the prior itself.

## Worked example: $\mathtt{H}=10$

The recursive split with $\mathtt{L}=\lceil \mathtt{N}/2 \rceil$ produces a 9-gate, depth-4 tree:

```
                              g1 (N=10, L=5, b=0.0000, p=0.500)
                              /                              \
                  g2 (N=5, L=3,                          g6 (N=5, L=3,
                  b=0.0844, p=0.600)                     b=0.0844, p=0.600)
                  /              \                      /              \
       g3 (N=3, L=2,         g5 (N=2,            g7 (N=3, L=2,      g9 (N=2,
       b=0.1436, p=0.667)    b=0, p=0.500)       b=0.1436, p=0.667) b=0, p=0.500)
         /          \           /     \             /          \       /     \
    g4 (N=2,      class 2   class 3 class 4    g8 (N=2,     class 7  class 8 class 9
    b=0, p=0.500)                              b=0, p=0.500)
     /     \                                    /     \
 class 0 class 1                            class 5  class 6
```

Only three distinct bias values appear, all driven by the subtree size $\mathtt{N}$:

| Gate role | $\mathtt{N}\to\mathtt{L},\mathtt{R}$ | $\mathtt{L}/\mathtt{N}$ | $b_j$ | $p_{\text{left}}$ at init |
|---|---|---|---|---|
| g1 (root) | $10\to 5,5$ | 0.5000 | 0.0000 | 0.5000 |
| g2, g6 | $5\to 3,2$ | 0.6000 | 0.0844 | 0.6000 |
| g3, g7 | $3\to 2,1$ | 0.6667 | 0.1436 | 0.6667 |
| g4, g5, g8, g9 | $2\to 1,1$ | 0.5000 | 0.0000 | 0.5000 |

### Verifying the prior

The biases compensate exactly for differing path lengths. A shorter (depth-3) path:

$$P(\text{class 2}) = \underbrace{0.500}_{g_1\to L}\,\times\,\underbrace{0.600}_{g_2\to L}\,\times\,\underbrace{0.333}_{g_3\to R} = 0.1$$

A longer (depth-4) path in the same subtree:

$$P(\text{class 0}) = 0.500 \times 0.600 \times 0.667 \times 0.500 = 0.1$$

All ten classes land at exactly $1/10$. The extra half-step that depth-4 leaves take is balanced by the parent gates ($g_3, g_7$) sending more mass into the larger subtree.
