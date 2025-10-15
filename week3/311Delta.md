# CIS 311: Neural Networks

## **Online vs. Batch Learning for Single-Layer Perceptrons (Delta/Least Mean Squares Rule)**

------

## 1) Overview

A single-layer **linear neuron** (Adaline) computes a linear output
$$
 o = s = \sum_{i=0}^{d} w_i,x_i \qquad (\text{bias } x_0=1).
$$

We train the weights by minimizing the **Mean Squared Error (MSE)**:
$$
 E(w) = \tfrac{1}{2}\sum_{e=1}^{N}\big(y^{(e)} - o^{(e)}\big)^2,
 \qquad
 o^{(e)} = \sum_{i=0}^{d} w_i,x_i^{(e)}.
$$

Taking a gradient step on this objective yields the **Delta rule** (also called **Least Mean Squares** or **Widrow–Hoff**).

------

## 2) Motivation (as requested)

Awesome — here’s a **drop-in replacement** you can paste right where that paragraph appears in your lecture. It keeps the spirit of the original but is **technically correct for Least Mean Squares/Adaline** and **clear for students**.

------

### **Why incremental (stochastic) gradient descent?**

Batch gradient descent can face two practical issues:

- **Slow convergence:** especially on ill-conditioned problems or large datasets.
- **Local minima (in general):** for **non-convex** models (e.g., multilayer nets with nonlinear activations), gradient descent can get stuck in poor local minima or flat regions.

A **stochastic (incremental) version** addresses these concerns by updating the weights **after each training example**:

- **Faster wall-clock progress:** many cheap updates instead of one expensive full-batch step.
- **Better scalability:** works well with large/streaming data.
- **Helpful noise:** stochasticity can move the iterate through flat/shallow regions.

> **Context note (LMS/Adaline in this course):**
>  With **linear output + MSE**, the loss
>  $$E(w) = \tfrac{1}{2}\sum_{e=1}^{N}\big(y^{(e)} - w^\top x^{(e)}\big)^2$$ 
>  is a **convex quadratic** (Hessian $X^\top X \succeq 0$).
>  Therefore, the “multiple local minima” issue does **not** arise here — any local minimum is global.
>  Incremental updates are mainly beneficial for **speed and efficiency**, not for escaping local minima.

------

------

## 3) The Delta (**Least Mean Squares** / Widrow–Hoff) Rule

From the MSE objective, the per-example gradient is
$$
 \frac{\partial E}{\partial w_i} = -\big(y^{(e)} - o^{(e)}\big)x_i^{(e)}.
$$
A gradient step with learning rate $\eta>0$ gives the **Delta rule**:
$$
\boxed{w_i \leftarrow w_i + \eta\big(y^{(e)} - o^{(e)}\big)x_i^{(e)}}
$$
Two standard application modes:

### 3.1 Online (Incremental) **Least Mean Squares**

Update **after each example**:
$$
\text{for } e=1,\dots,N:\quad
 o^{(e)}=\sum_{i=0}^{d} w_ix_i^{(e)},\quad
 w_i \leftarrow w_i + \eta\big(y^{(e)} - o^{(e)}\big)x_i^{(e)}
 \quad\text{for all } i.
$$

- Reacts immediately; steps are noisier but often faster in practice.

### 3.2 Batch **Least Mean Squares**

Accumulate over the dataset, then update **once per epoch**:
$$
 g_i \leftarrow 0,\qquad
 g_i \leftarrow g_i + \big(y^{(e)} - o^{(e)}\big),x_i^{(e)}\ \ (\text{for all } e),\qquad
 w_i \leftarrow w_i + \eta,g_i .
$$

- Smoother, more stable updates aligned with the average gradient.

> **Don’t mix styles:** In **online**, multiply by $\eta$ inside each update.
>  In **batch**, accumulate raw $g_i$ first, then multiply by $\eta$ once.

------

## 4) Algorithms (pseudocode)

### 4.1 Online (Incremental) Gradient Descent — Delta Rule

**Initialization:** training set ${(x^{(e)},y^{(e)})}_{e=1}^N$; set $x_0^{(e)}=1$; initialize $w_i$ (small random); choose $\eta$ (e.g., $0.01$–$0.1$).

**Repeat until termination (max epochs, small change in $E$, or validation stop):**

1. For each $(x^{(e)},y^{(e)})$:
   $$
    o^{(e)} = \sum_{i=0}^{d} w_ix_i^{(e)},\qquad
    w_i \leftarrow w_i + \eta\big(y^{(e)} - o^{(e)}\big)x_i^{(e)}\ \ \text{for all } i.
   $$

### 4.2 Batch Gradient Descent — Delta Rule

**Initialization:** as above.

**Repeat until termination:**

1. Set $g_i = 0$ for all $i$.

2. For each $(x^{(e)},y^{(e)})$:
   $$
    o^{(e)} = \sum_{i=0}^{d} w_ix_i^{(e)},\qquad
    g_i \leftarrow g_i + \big(y^{(e)} - o^{(e)}\big)x_i^{(e)}.
   $$

3. Update once:
   $$
    w_i \leftarrow w_i + \eta g_i.
   $$

------

## 5) Manual Online Example (matches your numbers)

Inputs $x_1=2,\ x_2=1$; bias $x_0=1$.
 Weights $w_1=0.5,\ w_2=0.3,\ w_0=-1$; target $y=0$.
 Use $\eta=1$ for easy arithmetic.

**Output (linear):**
$$
 o = 2(0.5) + 1(0.3) + 1(-1) = 0.3.
$$

**Update (Delta rule):**
$$
 \delta = y - o = -0.3,\qquad
 w_1 \leftarrow 0.5 + (-0.3)\cdot 2 = -0.1,\quad
 w_2 \leftarrow 0.3 + (-0.3)\cdot 1 = 0,\quad
 w_0 \leftarrow -1 + (-0.3)\cdot 1 = -1.3.
$$

> The “output is 1” statement only applies if you apply a **threshold** to $o=0.3$.
>  **LMS updates must use the linear output $o$**, not the thresholded class.

------

## 6) Why use Online (Incremental) updates?

- **Lower per-step cost** (one example at a time).
- **Faster feedback** (weights adapt immediately).
- **Stochasticity** can help traverse plateaus/poor directions that slow pure batch GD.
- Still, tune $\eta$ carefully and consider learning-rate decay or averaging.

------

## 7) Practical tips

- Always include the **bias** by setting $x_0=1$ (so $w_0$ learns).
- **Normalize/standardize features** for stability.
- Start with modest $\eta$ (e.g., $10^{-3}$–$10^{-1}$); adjust by validation.
- Use a clear **stopping criterion** (max epochs, tolerance on $E$, or validation early-stop).

------

## 8) Why LMS/Adaline (linear output + MSE) has **no local minima**

**Model and loss (with bias in $x$):**
$$
 o^{(e)} = w^\top x^{(e)}, \qquad
 E(w) = \tfrac{1}{2}\sum_{e=1}^N \big(y^{(e)} - w^\top x^{(e)}\big)^2
 = \tfrac{1}{2},|y - Xw|_2^2,
$$
 where $X\in\mathbb{R}^{N\times(d+1)}$ stacks inputs and $y\in\mathbb{R}^N$ stacks targets.

**Convexity:** the Hessian is
$$
 \nabla^2 E(w) = X^\top X \succeq 0,
$$
 so $E$ is **convex**. In a convex function, every local minimum is **global**.

**Uniqueness:** if $X^\top X$ is positive definite (full column rank),
 $w^\star = (X^\top X)^{-1}X^\top y$ is **unique**. If not full rank, there is a **flat valley** (many global minimizers).

------

## 9) Why batch GD can be **slow** (even though convex)

Batch GD update:
$$
 w_{t+1} = w_t - \eta,\nabla E(w_t) = w_t - \eta,X^\top,(Xw_t - y).
$$

Convergence speed depends on the **condition number** of $X^\top X$.
 If features are poorly scaled or nearly collinear, $X^\top X$ is **ill-conditioned**:

- you must choose a **small** $\eta$ to stay stable;
- steps **zig–zag** and progress can be slow.

**Fixes:** feature scaling/standardization, step-size schedules, momentum, conjugate gradients, or second-order methods.

------

## 10) Why **incremental (stochastic) updates** help

**Online/SGD (Delta/LMS):**
$$
 w \leftarrow w + \eta,(y^{(e)} - w^\top x^{(e)}),x^{(e)} \quad \text{(after each example)}.
$$

Benefits:

- **Cheaper per step**; many updates per unit time.
- **Faster wall-clock progress** in large datasets.
- **Noise** can help traverse flat/ill-conditioned regions.
- **Anytime** behavior: you can stop early with a reasonable solution.

Nuance: for LMS the objective is convex, so SGD isn’t “escaping local minima” (there aren’t bad ones). It mainly helps **speed and scalability**.

------

## 11) Quick mental picture

- **LMS/MSE surface:** a smooth **bowl** (or flat-bottomed valley). A ball rolls to the **global** bottom.
- **Multilayer nets:** a **rugged landscape** with many basins and saddles.

------

## 12) Takeaways

- **Linear output + MSE (LMS/Adaline) ⇒ convex loss.** No bad local minima; a global solution exists.
- **Batch GD** can be **slow** on ill-conditioned problems (scale features!).
- **Incremental/SGD** improves **speed** and **scalability**; use appropriate $\eta$ and stopping criteria.

------

## 13) Suggested Readings

- Widrow, B., & Hoff, M. E. (1960). *Adaptive switching circuits* (LMS / Delta rule).
- Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*, pp. 95–98.
- Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation* (2nd ed.).

------

### One-line summary

**Delta/LMS** updates are $w \leftarrow w + \eta (y - o) x$.
 Use **online** (per example) or **batch** (sum first, update once).
 For linear output + MSE, the loss is **convex**; incremental updates mainly improve **speed/practicality**.