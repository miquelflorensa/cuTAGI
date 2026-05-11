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

This algorithm fixes the unused-leaf normalization bug. It does not yet implement prior gate-bias balancing for uniform prior class probabilities $P(h_i)=1/\mathtt{H}$.
