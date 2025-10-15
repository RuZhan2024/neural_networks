# CIS 311: Neural Networks — Multilayer Perceptrons (Typora-ready)



# 1. The Multilayer Perceptron

The **multilayer perceptron (MLP)** is a hierarchical network of perceptrons that overcomes the limits of single-layer networks. It learns **nonlinear** function mappings and can represent a rich variety of **nonlinear decision surfaces**.

Key idea: Nonlinear functions require **nonlinear activation functions** in hidden units. Stacking linear units alone still yields a **linear** mapping.

### 1.1 Differentiable Activation Functions

Training multilayer networks via gradient methods requires **continuous, differentiable** nonlinear activations.

- **Logistic (sigmoid)**
  $$
  o = \sigma(s) = \frac{1}{1 + e^{-s}}
  $$
  $$
  s = \sum_{i=0}^{d} w_i x_i
  $$

  $\sigma$ is also called a **squashing** function because it maps a large input domain to the range $(0,1)$.

  

  ![image-20251015100607141](/Users/ruzhan/Library/Application Support/typora-user-images/image-20251015100607141.png)

- **Hyperbolic tangent**

  $$
   o = \tanh(s) = \frac{e^{s} - e^{-s}}{e^{s} + e^{-s}}
   $$

  $\tanh$ is zero-centered (range $(-1,1)$), which often makes training slightly easier.

  ![image-20251015100742758](/Users/ruzhan/Library/Application Support/typora-user-images/image-20251015100742758.png)

### 1.2 Multilayer Network Structure

A network with one or more layers **between** the input and output is a **multilayer** network.

- **Architecture / topology:** input layer → one or more **hidden** layers → output layer.
   Inputs pass to the first hidden layer, then on layer-by-layer until outputs are produced.

- Naming convention:

  - Input + **1 hidden** + Output → **two-layer** network (counting only adaptive layers).
  - Input + **2 hidden** + Output → **three-layer** network, etc.
     (The input layer is a pass-through, so we discount it.)

  ![image-20251015100858613](/Users/ruzhan/Library/Application Support/typora-user-images/image-20251015100858613.png)

A common two-layer (one hidden) MLP computes

$$
 f(\mathbf{x}) = \sigma\Bigg( \sum_{k} w_{0k} + \sum_{j} w_{jk}\ \sigma\Big( w_{0j} + \sum_{i} w_{ij} x_i \Big) \Bigg)
 $$

where

- $\mathbf{x}$: input vector
- $w_{0j}$ and $w_{0k}$: biases (thresholds)
- $w_{ij}$: weights input → hidden
- $w_{jk}$: weights hidden → output
- $\sigma$: sigmoid activation

Hidden units let the network **extract progressively more meaningful features**, enabling complex tasks to be learned.

**Connectivity.** MLPs are typically **fully connected** layer-to-layer (every unit connects to all units in the next layer).

**Signal flow.** Inputs propagate **forward** layer-by-layer → **feedforward**. During training, **error signals** propagate **backward**.

Two kinds of signals:

- **Function signals:** inputs transformed by hidden units and activations to produce outputs.
- **Error signals:** output errors back-propagate layer-by-layer so each unit receives its error contribution from the next layer.

### 1.3 Representation Power of MLP

Proven (informal summaries):

- **Arbitrary functions:** a **three-layer** network can approximate very general functions to arbitrary accuracy.
- **Bounded continuous functions:** a **two-layer** network can approximate any bounded continuous function with small error (enough hidden units needed).
- **Boolean functions:** any Boolean function can be represented exactly by a **two-layer** network (number of hidden units can grow exponentially with input dimension).

------

## 2. Backpropagation Learning Algorithm

MLPs became practical after the discovery of **backpropagation**, a supervised training algorithm that generalizes LMS to nonlinear (logistic) units. Training is **iterative**; weights are adjusted after data presentations.

Backprop consists of **two passes**:

1. **Forward pass:** compute activations to produce outputs.
2. **Backward pass:** compute and propagate **errors**; **update** weights by an error-correction rule to bring outputs closer to targets.

Backprop performs **gradient descent**: it moves opposite the gradient of the error, i.e., along the direction of **steepest decrease** while varying all weights **simultaneously** in proportion to their contribution.

Gradient vector:

$$
 \nabla E(\mathbf{w}) = \Big[ \frac{\partial E}{\partial w_0}\ \frac{\partial E}{\partial w_1}\ \dots\ \frac{\partial E}{\partial w_d} \Big]
 $$

**Notes.**

- Continuous, differentiable activations are required.
- Initialize weights to **small random** values.
- Present training data (online or in mini-batches).
- Update until convergence or acceptable cost.

### 2.1 Backpropagation — Training Algorithm (Sigmoid Units)

**Initialization**

- Training set ${(\mathbf{x}^{(e)}, \mathbf{y}^{(e)})}_{e=1}^{N}$
- Small random weights, learning rate $\eta = 0.1$

**Repeat** (until a termination condition)

For each example $(\mathbf{x}, \mathbf{y})$:

**Forward (sigmoid at hidden and output)**

$$
 s_j = \sum_{i=0}^{d} w_{ij}\ o_i \qquad \text{with } o_0 \equiv 1 \ \text{and}\ o_i = x_i
 $$

$$
 o_j = \sigma(s_j) \qquad \text{hidden unit } j
 $$

$$
 s_k = \sum_{j=0}^{m} w_{jk}\ o_j \qquad \text{with } o_0 \equiv 1
 $$

$$
 o_k = \sigma(s_k) \qquad \text{output unit } k
 $$

**Backward (define deltas)**

Output layer:

$$
 \delta_k = o_k \big(1 - o_k\big) \big( y_k - o_k \big)
 $$

Hidden layer:

$$
 \delta_j = o_j \big(1 - o_j\big) \sum_{k} \delta_k w_{jk}
 $$

**Weight updates**

$$
 \Delta w_{jk} = \eta\ \delta_k\ o_j \qquad \Delta w_{0k} = \eta\ \delta_k
 $$

$$
 \Delta w_{ij} = \eta\ \delta_j\ o_i \qquad \Delta w_{0j} = \eta\ \delta_j
 $$

and

$$
 w \leftarrow w + \Delta w
 $$

*(For mini-batch training, sum or average the per-example $\Delta w$ over the batch.)*

### 2.2 Derivation (Generalized Delta Rule)

Per-example squared error:

$$
 E_e = \tfrac12 \sum_{k} \big( y_k - o_k \big)^2
 $$

For a unit with net input $s = \sum_{\ell} w_{\ell} o_{\ell}$ and output $o = \sigma(s)$:
$$
\frac{\partial E_e}{\partial w}
= \frac{\partial E_e}{\partial s}\,\frac{\partial s}{\partial w}
= \frac{\partial E_e}{\partial s}\,o_{\text{pre}}
\qquad \text{since } \frac{\partial s}{\partial w} = o_{\text{pre}}
$$
**Output-layer weights $w_{jk}$**
$$
 \frac{\partial E_e}{\partial w_{jk}}
 = \frac{\partial E_e}{\partial s_k}\ o_j
 \qquad
 \frac{\partial E_e}{\partial s_k}
 = \frac{\partial E_e}{\partial o_k}\ \frac{\partial o_k}{\partial s_k}
$$
Compute the parts:

$$
 \frac{\partial E_e}{\partial o_k}
 = \frac{\partial}{\partial o_k}\Big( \tfrac12 (y_k - o_k)^2 \Big)
 = -\big( y_k - o_k \big)
 \qquad
 \frac{\partial o_k}{\partial s_k}
 = \sigma'(s_k) = o_k \big(1 - o_k\big)
 $$

Hence

$$
 \frac{\partial E_e}{\partial s_k}
 = -\big( y_k - o_k \big)\ o_k \big(1 - o_k\big)
 $$

Define

$$
 \delta_k = o_k \big(1 - o_k\big) \big( y_k - o_k \big)
 $$

then

$$
 \Delta w_{jk} = \eta\ \delta_k\ o_j
 $$

**Hidden-layer weights $w_{ij}$**

$$
 \frac{\partial E_e}{\partial w_{ij}}
 = \frac{\partial E_e}{\partial s_j}\ o_i
 \qquad
 \frac{\partial E_e}{\partial s_j}
 = \sum_{k} \frac{\partial E_e}{\partial s_k}\ \frac{\partial s_k}{\partial o_j}\ \frac{\partial o_j}{\partial s_j}
 = \sum_{k} \big[ -(y_k - o_k) o_k (1 - o_k) \big]\ w_{jk}\ \sigma'(s_j)
 $$

Define
$$
 \delta_j = \sigma'(s_j) \sum_{k} \delta_k w_{jk}
 \qquad
 \Rightarrow \quad
 \Delta w_{ij} = \eta\ \delta_j\ o_i
$$
**Dataset level:** sum or average gradients over all examples:
$$
\frac{\partial E_{\text{total}}}{\partial w}
 = \sum_{e} \frac{\partial E_e}{\partial w}
$$

------

## Example — XOR with One Hidden Layer (Threshold Illustration)

A one-hidden-layer network (two hidden units, one output) can classify XOR:

![image-20251015101121152](/Users/ruzhan/Library/Application Support/typora-user-images/image-20251015101121152.png)

| $(x_1,x_2)$ | $Y$  |
| ----------- | ---- |
| $(0,0)$     | $0$  |
| $(0,1)$     | $1$  |
| $(1,0)$     | $1$  |
| $(1,1)$     | $0$  |

With **threshold** activations, the following weights (illustrative) achieve correct XOR classification:

- **Biases:**
   $w_{10} = -\tfrac{3}{2}$  $w_{20} = -\tfrac{1}{2}$  $w_{30} = -\tfrac{1}{2}$
- **Weights (sample layout shown as given):**
   $w_{11} = 1$  $w_{21} = 1$  $w_{31} = -2$
   $w_{21} = 1$  $w_{22} = 1$  $w_{32} = 1$

Interpretation:

- $(0,0) \to 0$ output unit off because both inputs are off
- $(0,1) \to 1$ output on due to positive excitation from the second input’s pathway
- $(1,0) \to 1$ output on similarly
- $(1,1) \to 0$ output off due to an inhibitory effect from the first pathway

*(Exact indexing depends on your diagram; keep consistent mapping of indices to connections.)*

------

## Suggested Readings

- Bishop, C. (1995). *Neural Networks for Pattern Recognition*, OUP, pp. 116–149 (Ch. 4).
- Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation* (2nd ed.), Prentice-Hall.