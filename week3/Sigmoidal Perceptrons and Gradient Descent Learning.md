## **Sigmoidal Perceptrons and Gradient Descent Learning**

------

## **1. Introduction**

Early perceptrons used **threshold** or **linear** activation functions, which could only classify **linearly separable** data (e.g., AND, OR).
 They fail on **non-linear** problems (like XOR).

To overcome this limitation, we use a **sigmoid activation function**, which is smooth and differentiable.
 This enables **gradient-based learning** and allows extension into **multilayer neural networks**.

------

## **2. The Sigmoid Activation Function**

The **sigmoid** (logistic) function squashes any real number into a value between 0 and 1:

$$
 \sigma(s) = \frac{1}{1 + e^{-s}}
 $$

In a perceptron:

$$
 s = \sum_{i=0}^{d} w_i x_i
 $$

where:

- $x_i$ = input features
- $w_i$ = corresponding weights
- $x_0 = 1$, $w_0$ = bias input and bias weight
- $d$ = number of input features

------

### **2.1 Properties of the Sigmoid**

| Property       | Description                             |
| -------------- | --------------------------------------- |
| **Range**      | $(0, 1)$                                |
| **At $s=0$**   | $\sigma(0) = 0.5$                       |
| **Derivative** | $\sigma'(s) = \sigma(s)(1 - \sigma(s))$ |
| **Continuity** | Smooth and differentiable everywhere    |
| **Usefulness** | Enables gradient-based optimization     |

**Proof of derivative:**

$$
 \frac{d}{ds}\left(\frac{1}{1 + e^{-s}}\right)
 = \frac{e^{-s}}{(1 + e^{-s})^2}
 = \frac{1}{1 + e^{-s}}\left(1 - \frac{1}{1 + e^{-s}}\right)
 = \sigma(s)(1 - \sigma(s))
 $$

------

## **3. Error Function (Objective)**

To train the perceptron, we minimize the **Mean Squared Error (MSE)** between the target output $y_e$ and predicted output $o_e$:

$$
 E = \frac{1}{2} \sum_{e=1}^{N} (y_e - o_e)^2
 $$

The factor $\frac{1}{2}$ simplifies differentiation, as it cancels the 2 from the power.

------

## **4. Gradient Descent Principle**

We adjust each weight $w_i$ to reduce the total error $E$:

$$
 w_i \leftarrow w_i - \eta \frac{\partial E}{\partial w_i}
 $$

where $\eta$ is the **learning rate** (typically $0.01$ to $0.1$).

The gradient $\frac{\partial E}{\partial w_i}$ shows how sensitive the error is to that weight.

------

## **5. Derivation of the Weight Update Rule**

### **Step 1. Start with the error function**

$$
 E = \frac{1}{2} \sum_e (y_e - o_e)^2
 $$

Differentiate with respect to $w_i$:

$$
 \frac{\partial E}{\partial w_i} = \sum_e (y_e - o_e) \frac{\partial (y_e - o_e)}{\partial w_i}
 $$

------

### **Step 2. Simplify**

Since $y_e$ is constant:

$$
 \frac{\partial (y_e - o_e)}{\partial w_i} = -\frac{\partial o_e}{\partial w_i}
 $$

Thus:

$$
 \frac{\partial E}{\partial w_i} = -\sum_e (y_e - o_e) \frac{\partial o_e}{\partial w_i}
 $$

------

### **Step 3. Chain Rule**

Since $o_e = \sigma(s_e)$ and $s_e = \sum_i w_i x_{ie}$:

$$
 \frac{\partial o_e}{\partial w_i} = \frac{d\sigma(s_e)}{ds_e} \cdot \frac{\partial s_e}{\partial w_i} = \sigma'(s_e) x_{ie}
 $$

and $\sigma'(s_e) = o_e (1 - o_e)$.

So:

$$
 \frac{\partial o_e}{\partial w_i} = o_e (1 - o_e) x_{ie}
 $$

------

### **Step 4. Substitute Back**

$$
 \boxed{\frac{\partial E}{\partial w_i} = -\sum_e (y_e - o_e) o_e (1 - o_e) x_{ie}}
 $$

This expression is the foundation for both **Batch** and **Online** gradient descent.

------

## **6. Learning Modes Overview**

There are two common ways to use this gradient in practice:

| Type                                            | Description                                                  |
| ----------------------------------------------- | ------------------------------------------------------------ |
| **Batch Gradient Descent (BGD)**                | Accumulate gradients from all examples, then update weights once per epoch. |
| **Online (Incremental) Gradient Descent (OGD)** | Update weights immediately after each example.               |

------

## **7. Batch Gradient Descent (BGD)**

### **Update Rule**

1. Accumulate raw gradients (without $\eta$ inside the sum):
   $$
   g_i = g_i + (y_e - o_e) o_e (1 - o_e) x_{ie}
   $$

2. After processing all examples:
   $$
    w_i = w_i + \eta g_i
   $$

------

### **Algorithm**

**Repeat until convergence:**

1. Initialize $g_i = 0$ for all $i$.
2. For each example $(x_e, y_e)$:
   - Compute $s_e = \sum_i w_i x_{ie}$
   - Compute $o_e = \sigma(s_e)$
   - Update accumulator: $g_i = g_i + (y_e - o_e) o_e (1 - o_e) x_{ie}$
3. After all examples: update weights
    $w_i = w_i + \eta g_i$

------

### **Worked Example — Batch Gradient Descent**

Gotcha — here’s the **same combined section**, cleaned and formatted with `$...$` and `$$...$$` for Typora/KaTeX. No spacing tricks, just plain LaTeX.

------

## **Batch Gradient Descent — Manual Example (Two Samples, Two Epochs)**

**Dataset (two items):**

| Example | $x_1$ | $x_2$ | Target $y$ |
| ------- | ----- | ----- | ---------- |
| 1       | 2     | 1     | 0          |
| 2       | 1     | 2     | 1          |

**Conventions:** bias input $x_0 = 1$; weights $(w_0, w_1, w_2)$; sigmoid $\sigma(s)=\dfrac{1}{1+e^{-s}}$; learning rate $\eta = 1$ (for easy arithmetic).

**Batch rule:**

1. Accumulate raw gradients
   $$
   g_i \leftarrow g_i + (y_e - o_e) o_e (1 - o_e), x_{ie}
   $$

2. One update after all examples
   $$
   w_i \leftarrow w_i + \eta g_i
   $$

------

### **Epoch 1**

**Initial weights:**
$$
 w_0 = -1,\quad w_1 = 0.5,\quad w_2 = 0.3
$$

#### 1) Forward pass (use the same initial weights for both examples)

- **Example 1:** $x_1=2,\ x_2=1,\ y=0$
  $$
   s_1 = 2(0.5) + 1(0.3) - 1 = 0.3,\quad
   o_1 = \sigma(0.3) = 0.5744,\quad
   o_1(1 - o_1) = 0.2446
  $$

- **Example 2:** $x_1=1,\ x_2=2,\ y=1$
  $$
   s_2 = 1(0.5) + 2(0.3) - 1 = 0.1,\quad
   o_2 = \sigma(0.1) = 0.5250,\quad
   o_2(1 - o_2) = 0.2494
  $$

#### 2) Per-example gradient contributions

(using $\Delta w_i^{(e)} = (y_e - o_e) o_e (1 - o_e), x_{ie}$)

- **Example 1:**
  $$
   \Delta w_1^{(1)} = -0.2808,\quad
   \Delta w_2^{(1)} = -0.1404,\quad
   \Delta w_0^{(1)} = -0.1404
  $$

- **Example 2:**
  $$
   \Delta w_1^{(2)} = +0.1185,\quad
   \Delta w_2^{(2)} = +0.2370,\quad
   \Delta w_0^{(2)} = +0.1185
  $$

#### 3) Sum across the batch

$$
g_1 = -0.1623,\quad g_2 = +0.0966,\quad g_0 = -0.0219
$$

#### 4) Single batch update ($\eta = 1$)

$$ w_1 = 0.5 + (-0.1623) = 0.3377$$ 
 $$w_2 = 0.3 + (+0.0966) = 0.3966$$ 
 $$w_0 = -1 + (-0.0219) = -1.0219$$ 

**After Epoch 1:**
$$
\boxed{w_0 = -1.0219,\quad w_1 = 0.3377,\quad w_2 = 0.3966}
$$

------

### **Epoch 2**

**Start from Epoch-1 weights:**
$$
 w_0 = -1.0219,\quad w_1 = 0.3377,\quad w_2 = 0.3966
$$

#### 1) Forward pass (again, same weights for both examples)

- **Example 1:** $x_1=2,\ x_2=1,\ y=0$
  $$
  s_1 = 2(0.3377) + 1(0.3966) - 1.0219 = 0.0501,\quad
   o_1 = \sigma(0.0501) = 0.5125,\quad
   o_1(1 - o_1) = 0.2498
  $$

- **Example 2:** $x_1=1,\ x_2=2,\ y=1$
  $$
   s_2 = 1(0.3377) + 2(0.3966) - 1.0219 = 0.1090,\quad
   o_2 = \sigma(0.1090) = 0.5272,\quad
   o_2(1 - o_2) = 0.2493
  $$

#### 2) Per-example gradient contributions

- **Example 1:**
  $$
   \Delta w_1^{(1)} = -0.2561,\quad
   \Delta w_2^{(1)} = -0.1281,\quad
   \Delta w_0^{(1)} = -0.1281
  $$

- **Example 2:**
  $$
   \Delta w_1^{(2)} = +0.1178,\quad
   \Delta w_2^{(2)} = +0.2357,\quad
   \Delta w_0^{(2)} = +0.1178
  $$

#### 3) Sum across the batch

$$
 g_1 = -0.1383,\quad g_2 = +0.1076,\quad g_0 = -0.0102
$$

#### 4) Single batch update ($\eta = 1$)

$$
 \begin{aligned}
 w_1 &= 0.3377 + (-0.1383) = 0.1994\
 w_2 &= 0.3966 + (+0.1076) = 0.5042\
 w_0 &= -1.0219 + (-0.0102) = -1.0321
 \end{aligned}
 $$

**After Epoch 2:**
 $$
 \boxed{w_0 \approx -1.0321,\quad w_1 \approx 0.1994,\quad w_2 \approx 0.5042}
 $$

------

### **What to notice**

- **One update per epoch:** both examples influence $g_i$ before any weight change.
- **Opposing pulls:** the $y=0$ sample pushes outputs down; the $y=1$ sample pushes them up.
- **Feature influence:** because Example 2 emphasizes $x_2$, $w_2$ tends to increase more.
- **Convergence:** repeating epochs reduces error; updates generally shrink over time.

------

### **BGD formula summary (with bias)**

$$
 \boxed{
 \begin{aligned}
 g_i &= \sum_{e=1}^{N} (y_e - o_e), o_e, (1 - o_e), x_{ie},\quad x_{0e} = 1 \
 w_i^{\text{new}} &= w_i^{\text{old}} + \eta, g_i
 \end{aligned}
 }
$$

*(In practice, choose $\eta < 1$; here $\eta = 1$ is used to keep the arithmetic simple.)*

------

## **8. Online (Incremental) Gradient Descent (OGD)**

### **Update Rule**

Update immediately after each example:
$$
 w_i = w_i + \eta (y_e - o_e) o_e (1 - o_e) x_{ie}
$$
This version reacts faster to new data but is noisier.

------

### **Algorithm**

**Repeat until convergence:**

For each example $(x_e, y_e)$:

1. Compute $s_e = \sum_i w_i x_{ie}$
2. Compute $o_e = \sigma(s_e)$
3. Update weight:
    $w_i = w_i + \eta (y_e - o_e) o_e (1 - o_e) x_{ie}$

------

### **Worked Example — Online Gradient Descent**

**Initial weights:**
 $w_1 = 0.5$, $w_2 = 0.3$, $w_0 = -1$

#### Step 1 — Example 1

$x_1 = 2$, $x_2 = 1$, $y = 0$
$$
 s = 2(0.5) + 1(0.3) - 1 = 0.3
$$
$$
 o = 0.5744,\quad o(1 - o) = 0.2446
$$

| Weight | Formula                   | Initial $w_i$ | Δ$w_i$  | New $w_i$   |
| ------ | ------------------------- | ------------- | ------- | ----------- |
| $w_1$  | $(0 - 0.5744)(0.2446)(2)$ | 0.5000        | −0.2809 | **0.2191**  |
| $w_2$  | $(0 - 0.5744)(0.2446)(1)$ | 0.3000        | −0.1404 | **0.1596**  |
| $w_0$  | $(0 - 0.5744)(0.2446)(1)$ | −1.0000       | −0.1404 | **−1.1404** |

#### Step 2 — Example 2

Using updated weights:
 $w_1 = 0.2191$, $w_2 = 0.1596$, $w_0 = -1.1404$
 $x_1 = 1$, $x_2 = 2$, $y = 1$
$$
 s = 1(0.2191) + 2(0.1596) - 1.1404 = -0.6021
$$
$$
 o = 0.3539,\quad o(1 - o) = 0.2289
$$

| Weight | Formula                   | Initial $w_i$ | Δ$w_i$  | New $w_i$   |
| ------ | ------------------------- | ------------- | ------- | ----------- |
| $w_1$  | $(1 - 0.3539)(0.2289)(1)$ | 0.2191        | +0.1477 | **0.3669**  |
| $w_2$  | $(1 - 0.3539)(0.2289)(2)$ | 0.1596        | +0.2955 | **0.4551**  |
| $w_0$  | $(1 - 0.3539)(0.2289)(1)$ | −1.1404       | +0.1477 | **−0.9927** |

✅ **Final Weights:**
 $w_1 = 0.3669$, $w_2 = 0.4551$, $w_0 = -0.9927$

✅ **Online Summary:**
 Each example updates weights immediately, leading to faster adaptation.

------

## **9. Comparison of BGD vs OGD**

| Aspect                  | **Batch GD**                                            | **Online GD**                                    |
| ----------------------- | ------------------------------------------------------- | ------------------------------------------------ |
| **When weights update** | After all examples (per epoch)                          | After each example                               |
| **Equation**            | $w_i = w_i + \eta \sum_e (y_e - o_e)o_e(1 - o_e)x_{ie}$ | $w_i = w_i + \eta (y_e - o_e)o_e(1 - o_e)x_{ie}$ |
| **Gradient uses**       | Old weights for all examples                            | Latest weights after each step                   |
| **Behavior**            | Smooth, stable updates                                  | Faster, more adaptive but noisier                |
| **Best for**            | Small static datasets                                   | Large or streaming datasets                      |

------

## **10. Key Points**

- Always include the **bias term** ($x_0 = 1$).
- Keep the learning rate small for smoother convergence.
- **Never multiply by $\eta$ twice** — use it either during accumulation (online) or after accumulation (batch).
- The sign of $(y - o)$ controls whether weights increase or decrease.
- Sigmoid derivative $\sigma'(s) = \sigma(s)(1 - \sigma(s))$ is essential for learning.

------

## **11. References**

- Bishop, C. M. (1995). *Neural Networks for Pattern Recognition.* Oxford University Press.
- Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation (2nd Edition).* Prentice Hall.