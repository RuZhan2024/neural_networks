[TOC]





# Week 1

## Introduction to Connectionist Learning

### 1. Neural Computation

Neural computation is an approach to problem solving inspired by the structure and function of the human brain. It provides a framework for obtaining approximate solutions to real-world problems that are often ill-defined, noisy, or too complex for traditional algorithms.

Artificial Neural Networks (ANNs) are models that attempt to reproduce, in simplified form, the principles of biological neural processing. While biological neurons are individually slow compared to silicon logic gates, the brain achieves remarkable performance through:

- **Massive parallelism**: billions of neurons working simultaneously.
- **Robustness and fault tolerance**: partial damage does not lead to total failure.
- **Adaptability**: the ability to adjust synaptic strengths in response to experience.

**Core abilities of biological neural systems include:**

- **Learning** by adapting synaptic weights to environmental changes.
- **Generalization** from known examples to new, unseen inputs.
- **Robust handling** of fuzzy, noisy, and probabilistic information.

Artificial neural networks attempt to replicate these characteristics. Unlike classical computers that rely on a sequential instruction set, ANNs store knowledge in their connection weights.

Each artificial neuron functions as a simple processing unit. It sums its weighted inputs, applies a transformation (activation function), and produces an output. When combined in networks, assemblies of neurons can perform universal computation. In fact, suitably constructed neural networks are computationally equivalent to digital computers.

**Key elements that characterize an ANN include:**

- **Architecture (topology)**: how neurons are organized and connected.
- **Node properties**: the computational function of each neuron.
- **Connections (weights)**: parameters determining the influence between neurons.
- **Learning rules**: algorithms for updating weights and neuron states.

**Advantages of ANNs compared with traditional von Neumann computers:**

- Distributed representation and computation
- Learning capability
- Generalization ability
- Adaptability
- Fault tolerance
- Massive parallelism

#### Table 1: Von Neumann Computers vs. Biological Neural Systems

| Feature     | Von Neumann Computer                    | Biological Neural System                                  |
| ----------- | --------------------------------------- | --------------------------------------------------------- |
| Processor   | Complex, high-speed, few units          | Simple, slow, many units                                  |
| Memory      | Separate, localized, non-addressable    | Integrated into neurons, distributed, content-addressable |
| Computing   | Centralized, sequential, stored-program | Distributed, parallel, self-learning                      |
| Reliability | Vulnerable to failure                   | Robust, fault tolerant                                    |
| Expertise   | Symbolic, numerical manipulation        | Perceptual, pattern recognition                           |
| Environment | Well-defined, constrained               | Ill-defined, unconstrained                                |

###  2. Biological Neural Networks

Biological neural networks consist of **neurons**, which are specialized cells that process and transmit information.

**Structure of a neuron:**

- **Cell body (soma):** contains the nucleus and metabolic machinery.
- **Dendrites (inputs):** branching structures that receive signals from other neurons.
- **Axon (output):** a single extension that carries signals to other neurons.
- **Synapses:** junctions where axons connect to dendrites of other neurons.

**Signal transmission process:**

- A neuron integrates incoming signals through its dendrites.
- If the combined input exceeds a threshold, the neuron generates an action potential.
- Action potentials are **all-or-none**: the neuron either fires fully or not at all.
- Excitatory synapses increase the likelihood of firing, while inhibitory synapses decrease it.

**Learning in the brain:**

- Evidence suggests that synaptic changes are the basis of learning.
- Synaptic plasticity adjusts the strength of transmission between neurons.
- “Neurons that fire together, wire together” (Hebbian principle).

**Timing and parallelism:**

- Neurons fire at a few to several hundred hertz—far slower than electronic circuits.
- Yet humans recognize faces in a fraction of a second, implying the brain uses massively parallel processing with fewer than ~100 sequential steps.
- Information is not transmitted as large, detailed symbols but is distributed in connection strengths.

This distributed, connectionist style of computation inspires the design of ANNs.

### 3. Computational Model of a Neuron

In ANNs, neurons are modeled mathematically:

1. Each input $x_i$ is multiplied by a connection weight $w_i$.
2. The neuron sums the weighted inputs:
    $$ u = \sum_i w_i x_i $$
3. An **activation function** transforms $u$ into an output $y$.

- A **threshold function** outputs 1 if $u$ exceeds a threshold, else 0.
- A **sigmoid or ReLU function** provides smoother or nonlinear transformations.

**Interpretation of weights:**

- Large positive weight → strong excitatory influence.
- Small negative weight → weak inhibitory influence.

###  4. Neural Network Architectures

ANNs can be represented as **weighted, directed graphs**:

- **Nodes:** artificial neurons.
- **Edges:** weighted connections.

#### Two broad categories:

1. **Feedforward networks (static):**
   - No cycles; signals flow in one direction.
   - Examples: single-layer perceptron, multilayer perceptron (MLP), radial-basis function (RBF) networks, polynomial networks.
2. **Recurrent networks (dynamic):**
   - Contain feedback loops.
   - Examples: competitive networks, self-organizing maps (SOM), Hopfield networks, adaptive resonance theory (ART) models.

### 5. Connectionist Learning Algorithms

**Definition of learning:** improvement in network performance through weight adjustment.

**Learning in ANNs involves:**

- A computational model (e.g., inductive learning from examples).
- Updating weights according to available data and rules.
- Converging to a configuration that produces desired outputs.

### Two learning paradigms:

- **Supervised learning:**
  - A teacher provides inputs and corresponding outputs.
  - The goal is to infer a general mapping.
- **Unsupervised learning:**
  - No explicit outputs; the network organizes patterns or clusters.

### Two learning modes:

- **Incremental (online):** update after each example.
- **Batch:** update after processing many examples.

### Four basic types of learning rules:

- **Error-correction rules** (e.g., perceptron, backpropagation).
- **Boltzmann rule** (stochastic relaxation in recurrent nets).
- **Hebbian rule** (strengthening correlated activity).
- **Competitive rule** (winner-takes-all learning).

------

## 6. Applications of Artificial Neural Networks

ANNs are applied across domains such as:

- **Classification:** speech recognition, medical diagnosis.
- **Regression:** financial forecasting, sensor calibration.
- **System identification and control.**
- **Pattern recognition:** handwriting, image analysis.
- **Data mining and dimensionality reduction.**
- **Time-series prediction:** stock markets, weather.

### Example mappings of paradigms to tasks:

| Paradigm     | Learning Rule    | Architecture             | Algorithm           | Tasks                       |
| ------------ | ---------------- | ------------------------ | ------------------- | --------------------------- |
| Supervised   | Error-correction | Single-layer perceptron  | Perceptron learning | Pattern classification      |
| Supervised   | Error-correction | Multilayer perceptron    | Backpropagation     | Regression, time-series     |
| Supervised   | Boltzmann        | Recurrent                | Boltzmann learning  | Pattern classification      |
| Supervised   | Hebbian          | Multilayer feedforward   | Linear discriminant | Classification, data mining |
| Supervised   | Competitive      | Competitive              | LVQ                 | Data compression            |
| Unsupervised | Error-correction | Multilayer feedforward   | Simmon’s projection | Data analysis               |
| Unsupervised | Hebbian          | Feedforward, competitive | PCA                 | Data compression, analysis  |
| Unsupervised | Competitive      | SOM                      | Kohonen SOM         | Categorization, analysis    |

------

## Suggested Readings

- Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation* (2nd ed.). Prentice-Hall.
- Jain, A. K., & Mao, J. (1996). Artificial Neural Networks: A Tutorial. *IEEE Computer*, 29(3), 31–44.



# The Classification Problem

## 1. Statistical Learning

Statistical learning methods provide a formal framework for addressing **classification problems**. The general task is:

- **Given:** a feature (pattern) vector
   $x = (x_1, x_2, \ldots, x_d)$
- **Determine:** the class $c_k$ (category or label) from which this feature vector was generated.

### 1.1 Interpretations of Class Definitions

Classification can be interpreted in different ways depending on the context:

1. **Population labels:** Classes correspond to distinct populations, and membership is defined by an external authority (supervisor).
2. **Prediction problems:** Classes represent outcomes to be predicted from attributes; in this view, the class is a random variable.
3. **Partition of feature space:** Classes are defined as partitions of the input space itself, essentially representing functions of the attributes.

In all cases, the goal is to **learn a rule that closely mimics the true (unknown) rule** that originally generated the class assignments.

------

### 1.2 Classifiers and Decision Regions

A **classifier** is a rule that maps each feature vector into one of $K$ classes:
 $c_k, \quad k = 1, 2, \ldots, K$

This implies that the feature space is divided into **decision regions**. Any vector that falls within region $R_k$ is assigned to class $c_k$. The surfaces separating these regions are known as **decision boundaries**.

------

### 1.3 Bayes Decision Rule

The optimal classification rule, in terms of minimizing the probability of misclassification, is the **Bayes decision rule**:

$$
 \hat{c}(x) = \arg \max_k P(c_k \mid x)
 $$

That is, assign $x$ to the class with the largest posterior probability.

The posterior probabilities are given by **Bayes’ theorem**:

$$
 P(c_k \mid x) = \frac{P(c_k) , p(x \mid c_k)}{p(x)}
 $$

where:

- $P(c_k \mid x)$ = posterior probability of class $c_k$ given $x$.
- $P(c_k)$ = prior probability of class $c_k$.
- $p(x \mid c_k)$ = class-conditional density of $x$.
- $p(x)$ = overall probability density of $x$.

**Assumptions:**

- Classes are mutually exclusive.
- Every $x$ belongs to exactly one class.

Thus:

$$
 \sum_{k=1}^K P(c_k) = 1, \quad
 \sum_{k=1}^K P(c_k \mid x) = 1, \quad
 \sum_{k=1}^K P(c_k) , p(x \mid c_k) = p(x)
 $$

**Key point:** Decision rules based on Bayes’ theorem are *optimal*. No other rule achieves a lower expected error rate or misclassification cost. While Bayes rules are rarely attainable in practice (since the true distributions are unknown), they provide the theoretical foundation for most statistical classifiers.

------

## 2. Discriminant Functions

Classification can also be expressed in terms of **discriminant functions** $y_k(x)$.

- Rule: assign $x$ to class $c_k$ if
   $$
   y_k(x) > y_j(x), \quad \forall j \neq k
   $$
- Common choices:
   $$
   y_k(x) = P(c_k \mid x) \quad \text{or} \quad y_k(x) = P(c_k) , p(x \mid c_k)
   $$
- **Decision boundaries** are located where two discriminant functions are equal.

Discriminant functions are useful because they can be designed to be computationally efficient and can often be estimated directly from data.

------

## 3. Classifier Construction Approaches

Different classifiers can be grouped according to how they model the problem:

### 3.1 A Posteriori Classifiers

- Directly model the posterior probabilities $P(c_k \mid x)$.
- Based on Bayes’ theorem, these are theoretically ideal but assume complete knowledge of distributions.
- In practice, statistical procedures attempt to approximate missing information.

### 3.2 Probability Density Classifiers

- Model the likelihood functions $p(x \mid c_k)$.
- Assume a distributional form (commonly Gaussian).
- Assign a new observation to the class with the highest likelihood times prior probability.

**Example: Two-class Gaussian case**

$$
 p(x \mid c_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}}
 \exp \left[ -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right], \quad k=1,2
 $$

- $\mu_k$: mean vector of class $c_k$.
- $\Sigma_k$: covariance matrix of class $c_k$.

If parameters are unknown, they can be estimated from training data.

### 3.3 Decision Boundary Classifiers

- Construct decision surfaces directly, without explicit probability modeling.
- Simpler, but less powerful since they lack a full probabilistic foundation.
- Flexible in adapting when new examples are provided.

------

### 3.4 Choosing a Classifier

- The choice depends on the assumptions about the data distribution and available information.
- In some situations, it is easier to estimate **posteriors** $P(c_k \mid x)$ directly; in others, it is easier to estimate **likelihoods** $p(x \mid c_k)$.
- Note that $P(c_k \mid x)$ and $p(x \mid c_k)$ may have very different shapes, making the choice of modeling approach critical.

------

## Suggested Readings

- Bishop, C. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press.
- Michie, D., Spiegelhalter, D. J., & Taylor, C. C. (1994). *Machine Learning, Neural and Statistical Classification*. Ellis Horwood.
- Nilsson, N. (1996). *Introduction to Machine Learning*. Stanford University.



# The Regression Problem

## 1. Inductive Learning and Regression

The problem of **inductive learning** can be formulated as a **multivariate regression** problem.

- **Given:** a dataset of $N$ examples
   $$
   D = {(x_i, y_i)}_{i=1}^N
   $$

  where:

  - $x_i = (x_{i1}, x_{i2}, \ldots, x_{id}) \in \mathbb{R}^d$ is a vector of independent variables (features),
  - $y_i \in \mathbb{R}$ is the corresponding dependent variable (target).

- **Determine:** a function $y = f(x)$ that models the relationship between features and outputs.

### 1.1 Noise model

A common assumption is that examples are drawn independently from an unknown probability distribution, and outputs are generated according to:

$$
 y = f(x) + \epsilon
 $$

where $\epsilon$ is a random noise term with mean zero and constant variance $\sigma^2$.

The ideal regression function is:

$$
 f^*(x) = \mathbb{E}[y \mid x]
 $$

that is, the conditional expectation of $y$ given $x$.

### 1.2 Practical goal

Because the underlying distribution is unknown and real data are noisy, the practical objective is to find a good approximation $\hat{f}(x)$ to $f^*(x)$ by **minimizing empirical error** on the training set.

We choose $\hat{f}$ from a **model family** (e.g., linear models, neural networks), selecting the function that best fits the training data.

### 1.3 Error function

A common criterion is the **average squared residual (ASR)**, also known as the mean squared error (MSE):

$$
 ASR = \frac{1}{N} \sum_{i=1}^N \big( y_i - f(x_i) \big)^2
 $$

where $y_i$ is the observed value, $f(x_i)$ is the model prediction, and $N$ is the sample size.

------

## 2. Model Function Families

Several families of regression models are widely used:

### 2.1 Linear Models

$$
 f(x) = w_0 + \sum_{i=1}^d w_i x_i = w^T x + w_0
 $$

### 2.2 Polynomial Models

$$
 f(x) = w_0 + \sum_{i=1}^p w_i x_i
 $$
 (where higher-order terms allow more complex relationships).

### 2.3 Generalized Linear Models (GLMs)

$$
 f(x) = w_0 + \sum_{i=1}^d w_i h_i(x)
 $$
 where $h_i(x)$ are fixed, prespecified basis functions.

### 2.4 Neural Network Models

$$
 f(x) = \varphi \left( \sum_{i=1}^d w_i , \varphi \Big( \sum_{j=1}^d w_j , \varphi(\ldots) \Big) \right)
 $$

where $\varphi(\cdot)$ is a nonlinear activation function (commonly the sigmoid or ReLU). Neural networks are universal approximators, capable of modeling highly complex nonlinear mappings.

------

## 3. Linear vs. Nonlinear Models

- **Linear regression**:
  - Solution can be obtained analytically by solving a system of linear equations.
  - Efficient, unique solution.
  - Enables rapid experimentation.
- **Nonlinear regression**:
  - Requires iterative optimization methods.
  - Multiple local optima are possible.
  - Training is computationally expensive.

Thus, linear models are fast but limited, while nonlinear models are more expressive but harder to optimize.

------

## 4. Accuracy and Overfitting

Since training data are only a subset of all possible inputs, there is always a risk of **overfitting**, where the model adapts too closely to training data but generalizes poorly to unseen data.

Two important notions for understanding this trade-off are **bias** and **variance**:

- **Model bias**:
  - The error due to restricting the hypothesis space.
  - If the true function lies outside the chosen model family, the model is biased.
  - High bias leads to **underfitting**.
- **Model variance**:
  - The variability of model predictions across different training sets.
  - Small model families → low variance.
  - Large, flexible families → high variance.
  - High variance leads to **overfitting**.

**Goal:** Achieve a balance between bias and variance to ensure good generalization.

------

## Suggested Readings

- Bishop, C. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press.
- Seber, G. A. F. (1977). *Linear Regression Analysis*. John Wiley & Sons.
- Wild, C. J., & Seber, G. A. F. (1989). *Nonlinear Regression*. John Wiley & Sons.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*. Chapman & Hall.



# Week 2

# The Perceptron

## 1. Single-Layer Perceptrons

### 1.1 Definition

The **Perceptron** is one of the earliest models of an artificial neuron. It is a **single-layer neural network** consisting of a single threshold unit (neuron).

The perceptron computes a **linear combination** of its real-valued or Boolean inputs and passes the result through a **threshold activation function**:

$$
 o = \text{Threshold}\left( \sum_{i=0}^d w_i x_i \right)
 $$

- $x = (x_1, x_2, \ldots, x_d)$ is the input vector,
- $w = (w_0, w_1, \ldots, w_d)$ are the weights,
- $o$ is the output of the perceptron.

The threshold activation function is defined as:

$$
 \text{Threshold}(s) =
 \begin{cases}
 1, & s > 0 \
 -1, & \text{otherwise}
 \end{cases}
 $$

Here $w_0$ is treated as a **bias weight**, with a constant input $x_0 = 1$.

------

### 1.2 Perceptron as a Linear Discriminant

The perceptron is equivalent to a **linear discriminant classifier**. It decides whether an input pattern belongs to one of two classes based on whether the weighted sum is above or below the threshold.

Geometrically, the perceptron defines a **linear decision boundary** (a hyperplane):

$$
 \sum_{i=1}^d w_i x_i + w_0 = 0
 $$

- Inputs on one side of the hyperplane are classified as one class.
- Inputs on the other side are classified as the other class.

------

### 1.3 Boolean Function Representation

A single perceptron can represent many **primitive Boolean functions** such as **AND, OR, NAND, NOR**.
 However, it **cannot represent XOR**, since XOR is not linearly separable.

- Single perceptrons are limited to **linearly separable functions**.
- Networks of perceptrons (multi-layer perceptrons) can represent any Boolean function.

------

## 2. Perceptron Learning Algorithms

### 2.1 General Principle

The perceptron learning task is to adjust the weights $w_i$ such that the perceptron classifies training examples correctly.

- No assumptions are made about data distributions (non-parametric).
- Training examples are presented sequentially.
- Errors lead to weight updates; correct classifications leave weights unchanged.

Three related rules are commonly used:

- The **Perceptron Rule**
- The **Gradient Descent Rule**
- The **Delta Rule**

All belong to the family of **error-correction learning procedures**.

------

### 2.2 The Perceptron Rule

**Algorithm (sequential learning procedure):**

1. **Initialization**
   - Given training examples ${(x^{(e)}, y^{(e)})}_{e=1}^N$
   - Initialize weights $w_i$ to small random values
   - Choose a learning rate $\eta > 0$ (e.g., $\eta = 0.1$)
2. **Repeat** for each training example $(x^{(e)}, y^{(e)})$:
   - Compute output:
      $$
      o^{(e)} = \text{Threshold}\left(\sum_{i=0}^d w_i x_i^{(e)}\right)
      $$
   - If $o^{(e)} \neq y^{(e)}$, update the weights:
      $$
      w_i \leftarrow w_i + \eta \big( y^{(e)} - o^{(e)} \big) x_i^{(e)}
      $$
3. **Stop** when all training examples are correctly classified or when a termination condition is met.

The learning rate $\eta$ controls the magnitude of weight adjustments.

------

### 2.3 Perceptron Convergence Theorem

The **Perceptron Convergence Theorem** states:

- If the training data are **linearly separable**, then the perceptron learning algorithm will converge to a correct solution in a finite number of steps.
- That is, the algorithm is guaranteed to find a separating hyperplane (a weight vector) that classifies all training examples correctly.

A function is **linearly separable** if its outputs can be separated by a hyperplane in feature space.



------

### 2.4 Example

Suppose a perceptron with two inputs:

- Inputs: $x_1 = 2$, $x_2 = 1$
- Weights: $w_1 = 0.5$, $w_2 = 0.3$, bias $w_0 = -1$

Compute the output:

$$
 o = 2 \cdot 0.5 + 1 \cdot 0.3 - 1 = 0.3
 $$

Since $o > 0$, the perceptron outputs **1**.

Suppose the correct output is $0$ (or equivalently $-1$ depending on convention).
 The weights are updated:

$$
 w_1 = 0.5 + (0 - 1) \cdot 2 = -1.5
 $$

$$
 w_2 = 0.3 + (0 - 1) \cdot 1 = -0.7
 $$

$$
 w_0 = -1 + (0 - 1) \cdot 1 = -2
 $$

The new weights shift the decision boundary so that the previously misclassified example may now be correctly classified.

------

## Suggested Readings

- Bishop, C. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press, pp. 98–102.
- Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation* (2nd ed.). Prentice Hall.
- Nilsson, N. (1996). *Introduction to Machine Learning*, Chapter 4. [Available online](http://robotics.stanford.edu/people/nilsson/mlbook.html).



Here’s a **rewritten, academic, and detailed version** of your *Gradient Descent Training* notes. All equations are formatted with `$...$` for **Typora**.

------

# Gradient Descent Training for Single-Layer Perceptrons

## 1. Motivation: Unthresholded Perceptron

The classic perceptron learning rule converges only if the training data are **linearly separable**. To extend learning to cases where data may not be perfectly separable, we use **gradient descent**.

The idea is to treat the perceptron as an **unthresholded linear unit**:

$$
 o = \sum_{i=0}^d w_i x_i
 $$

where:

- $x_i$ are input features,
- $w_i$ are the weights,
- $o$ is the raw (unthresholded) output.

The training objective is to adjust the weights so that $o$ approximates the target $y$ as closely as possible, even when exact classification is impossible.

------

## 2. The Gradient Descent Rule

### 2.1 Error Function

Define the **error function** (squared error over all examples):

$$
 E(w) = \frac{1}{2} \sum_e \big( y^{(e)} - o^{(e)} \big)^2
 $$

where:

- $y^{(e)}$ is the target output for example $e$,
- $o^{(e)}$ is the perceptron output for example $e$.

The factor $\tfrac{1}{2}$ simplifies differentiation.

------

### 2.2 Gradient of the Error

The **gradient** of the error with respect to the weights is:

$$
 \nabla E(w) = \left[ \frac{\partial E}{\partial w_0}, \frac{\partial E}{\partial w_1}, \ldots, \frac{\partial E}{\partial w_d} \right]
 $$

Weight updates move in the opposite direction of the gradient:

$$
 w_i \leftarrow w_i - \eta , \frac{\partial E}{\partial w_i}
 $$

where $\eta > 0$ is the **learning rate**.

------

### 2.3 Derivation of Weight Update

$$
 \frac{\partial E}{\partial w_i}
 = \frac{\partial}{\partial w_i} \left( \frac{1}{2} \sum_e ( y^{(e)} - o^{(e)} )^2 \right)
 $$

$$
 = \sum_e ( y^{(e)} - o^{(e)} ) , \frac{\partial (y^{(e)} - o^{(e)})}{\partial w_i}
 $$

$$
 = \sum_e ( y^{(e)} - o^{(e)} ) , (-x_i^{(e)})
 $$

Thus, the **gradient descent update rule** is:

$$
 w_i \leftarrow w_i + \eta \sum_e ( y^{(e)} - o^{(e)} ) , x_i^{(e)}
 $$

This corresponds to **batch gradient descent**, since weight updates accumulate over all training examples before applying.

------

### 2.4 Learning Rate Considerations

- If $\eta$ is too large → overshooting on the error surface.
- If $\eta$ is too small → very slow convergence.
- In practice, $\eta$ may be gradually reduced as training progresses.

------

## 3. Gradient Descent Learning Algorithm

**Initialization**

- Training examples ${ (x^{(e)}, y^{(e)}) }_{e=1}^N$
- Initialize weights $w_i$ to small random values
- Choose learning rate $\eta$ (e.g., $\eta = 0.1$)

**Repeat until convergence:**

1. For each training example $(x^{(e)}, y^{(e)})$:

   - Compute output:
      $$
      o^{(e)} = \sum_{i=0}^d w_i x_i^{(e)}
      $$
   - Accumulate error corrections:
      $$
      \Delta w_i = \Delta w_i + \eta , ( y^{(e)} - o^{(e)} ) , x_i^{(e)}
      $$

2. After processing all examples, update weights:
   $$
    w_i \leftarrow w_i + \Delta w_i
   $$

------

# 4. Perceptron Rule vs. Gradient Descent Rule

| Aspect                    | **Perceptron Rule**                                          | **Gradient Descent Rule**                                    |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Formulation**           | Updates weights only when misclassification occurs           | Updates weights to minimize squared error over all examples  |
| **Weight Update**         | $w_i \leftarrow w_i + \eta ,(y^{(e)} - o^{(e)}), x_i^{(e)}$  | $w_i \leftarrow w_i + \eta \sum_e (y^{(e)} - o^{(e)}), x_i^{(e)}$ |
| **Output Function**       | Thresholded perceptron: $o = \text{sign}!\left(\sum w_i x_i\right)$ | Unthresholded linear unit: $o = \sum w_i x_i$                |
| **Assumptions**           | Works only if training data are **linearly separable**       | Works even if data are **not linearly separable**            |
| **Convergence Guarantee** | Converges in finite steps (Perceptron Convergence Theorem) if separable | Converges (in expectation) to least-squares solution, even for non-separable data |
| **Learning Mode**         | Online (updates one example at a time)                       | Typically batch (updates accumulated over all examples), though online variants exist |
| **Pros**                  | Simple, efficient, guaranteed convergence for separable data | More general, finds best-fit approximation for non-separable data |
| **Cons**                  | Fails if data are not linearly separable                     | Slower convergence, sensitive to learning rate $\eta$        |

## 5. Example

Suppose a perceptron with inputs $x_1, x_2$, weights $w_1 = 0.5$, $w_2 = 0.3$, and bias $w_0 = -1$.

### Example 1

Inputs: $x_1 = 2, x_2 = 1$, target $y = 0$

- Output:
   $$
   o = 2 \cdot 0.5 + 1 \cdot 0.3 - 1 = 0.3
   $$

- Errors and corrections:
  $$
   \Delta w_1 = (0 - 0.3) \cdot 2 = -0.6
  $$
  $$
   \Delta w_2 = (0 - 0.3) \cdot 1 = -0.3
  $$
  $$
   \Delta w_0 = (0 - 0.3) \cdot 1 = -0.3
  $$

### Example 2

Inputs: $x_1 = 1, x_2 = 2$, target $y = 1$

- Output:
   $$
   o = 1 \cdot 0.5 + 2 \cdot 0.3 - 1 = 0.1
   $$
- Errors and corrections:
   $$
   \Delta w_1 = -0.6 + (1 - 0.1)\cdot 1 = 0.3
   $$
   $$
   \Delta w_2 = -0.3 + (1 - 0.1)\cdot 2 = 1.5
   $$
   $$
   \Delta w_0 = -0.3 + (1 - 0.1)\cdot 1 = 0.6
   $$

### Updated Weights

After batch update:
 $$
 w_1 = 0.5 + 0.3 = 0.8
 $$

$$
 w_2 = 0.3 + 1.5 = 1.8
 $$

$$
 w_0 = -1 + 0.6 = -0.4
 $$

------

## Suggested Readings

- Bishop, C. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press, pp. 98–105.
- Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation* (2nd ed.). Prentice Hall.



# Error-Correction Learning

## 1. The Perceptron

A **single-layer perceptron** computes a weighted sum of its inputs and applies a threshold activation function:

$$
 o = \text{Threshold}(w_0x_0 + w_1x_1 + w_2x_2)
 $$

where:

- $x_0 = 1$ is the bias input,
- $w_0, w_1, w_2$ are the weights,
- The threshold function is:

$$
 \text{Threshold}(s) =
 \begin{cases}
 1, & s > 0 \
 0, & s \leq 0
 \end{cases}
 $$

The learning rule is an **error-correction update**:

$$
 w_i ;\leftarrow; w_i + \eta,(y^{(e)} - o^{(e)}),x_i^{(e)}
 $$

where $y^{(e)}$ is the target, $o^{(e)}$ is the prediction, and $\eta$ is the learning rate.

------

## 2. Worked Example: Learning a NOR Function

Target truth table:

| $(x_1, x_2)$ | Desired $Y$ |
| ------------ | ----------- |
| (0,0)        | 1           |
| (0,1)        | 0           |
| (1,0)        | 0           |
| (1,1)        | 0           |

Initial weights: $w_0=0.5, w_1=0.5, w_2=0.5$.

------

### Iteration 1

- Example (0,0) → target 1
   $o = 0.5 \Rightarrow 1 ;;\checkmark$
- Example (0,1) → target 0
   $o = 1 \Rightarrow 1 ;\text{(Error)}$
   Update:
   $w_0=-0.5,;w_1=0.5,;w_2=-0.5$
- Example (1,0) → target 0
   $o = 0 \Rightarrow 0 ;;\checkmark$
- Example (1,1) → target 0
   $o=-0.5 \Rightarrow 0 ;;\checkmark$

------

### Iteration 2

- Example (0,0) → target 1
   $o=-0.5 \Rightarrow 0 ;\text{(Error)}$
   Update: $w_0=0.5$
- Example (0,1) → target 0
   $o=0 \Rightarrow 0 ;;\checkmark$
- Example (1,0) → target 0
   $o=1 \Rightarrow 1 ;\text{(Error)}$
   Update: $w_0=-0.5,;w_1=-0.5$
- Example (1,1) → target 0
   $o=-1.5 \Rightarrow 0 ;;\checkmark$

------

### Learned Function

After convergence, the perceptron implements:

$$
 o = \text{Threshold}(0.5 - 0.5x_1 - 0.5x_2)
 $$

This is the **NOR function**, which is `true` only when $(x_1,x_2)=(0,0)$.

------

## 3. Limitation: XOR Problem

Consider the XOR truth table:

| $(x_1, x_2)$ | Desired $Y$ |
| ------------ | ----------- |
| (0,0)        | 0           |
| (0,1)        | 1           |
| (1,0)        | 1           |
| (1,1)        | 0           |

- Points with $Y=1$: $(0,1), (1,0)$
- Points with $Y=0$: $(0,0), (1,1)$

No single line can separate these groups in 2D.
 Thus, **XOR is not linearly separable**, and a perceptron cannot converge to a correct solution.
 Training causes weights to **oscillate** without convergence.

------

## 4. Limitations of Single-Layer Perceptrons

- Can only solve **linearly separable** problems.
- Equivalent to a linear classifier with a hyperplane decision boundary.
- Cannot represent higher-order relations (e.g., XOR, parity).

**Historical note:**

- Minsky & Papert (1969) showed these limitations, slowing neural network research for decades.
- The solution: **Multilayer Perceptrons (MLPs)** with nonlinear activations.

------

## 5. Multilayer Perceptrons (MLPs)

### 5.1 Key Idea

- Add **hidden layers** between inputs and outputs.
- Hidden units apply nonlinear transformations, enabling networks to learn **nonlinear decision boundaries**.
- An MLP with **one hidden layer** is already a **universal approximator**.

------

## 6. Worked Example: MLP Solving XOR

### Architecture

- Inputs: $x_1, x_2$
- Hidden layer: 2 units ($h_1,h_2$)
- Output: 1 unit $y$

Hidden units:

$$
 h_1 = \sigma(20x_1 + 20x_2 - 10) \quad \text{(acts like OR)}
 $$

$$
 h_2 = \sigma(20x_1 + 20x_2 - 30) \quad \text{(acts like AND)}
 $$

Output unit:

$$
 y = \sigma(20h_1 - 20h_2 - 10)
 $$

with sigmoid activation $\sigma(z)=\frac{1}{1+e^{-z}}$.

------

### Forward Pass

- $(0,0) \to y \approx 0$ ✔
- $(0,1) \to y \approx 1$ ✔
- $(1,0) \to y \approx 1$ ✔
- $(1,1) \to y \approx 0$ ✔

------

### Why It Works

- $h_1$ detects when *either* input is 1 (OR-like).
- $h_2$ detects when *both* inputs are 1 (AND-like).
- Output unit computes **OR − AND**, which is XOR.

Thus, an MLP can solve XOR, demonstrating why **hidden layers are essential**.

------

## 7. Summary

- **Error-correction perceptrons** can learn linearly separable Boolean functions (e.g., AND, OR, NOR).
- They **fail** on non-separable problems like XOR.
- This limitation motivated **Multilayer Perceptrons (MLPs)**, which extend expressive power by introducing hidden layers and nonlinear activations.
- With MLPs, we can model **any continuous function**, including XOR.



# Week 3

Great — let’s rewrite your **Delta Rule section** into a polished, professional, and Typora-ready lecture note, with math in `$...$`, clear structure, and example walkthrough.

## The Delta Rule

## 1. Introduction

The **gradient descent rule** (batch version) suffers from two main drawbacks in practice:

1. **Slow convergence** — updating only after accumulating the error across the entire dataset can be computationally inefficient.
2. **Local minima** — if the error surface contains multiple local minima, there is no guarantee that batch gradient descent will reach the global minimum.

To address these issues, a **stochastic (incremental) version** of gradient descent is often used.

------

## 2. Incremental Gradient Descent (Delta Rule)

Instead of updating the weights after processing **all training examples** (batch mode), the **incremental gradient descent** rule updates weights **after each example**.

This incremental version is called the **Delta Rule**, also known as:

- **LMS rule** (Least-Mean-Square)
- **Widrow–Hoff rule**

------

### 2.1 Mathematical Formulation

For a perceptron with input vector $x^{(e)} = (x_0, x_1, \dots, x_d)$ and weights $w = (w_0, w_1, \dots, w_d)$:

- Output:

$$
 o^{(e)} = \sum_{i=0}^d w_i x_i^{(e)}
 $$

- Weight update (Delta Rule):

$$
 w_i ;\leftarrow; w_i + \eta ,(y^{(e)} - o^{(e)}),x_i^{(e)}
 $$

where:

- $y^{(e)}$ = desired target output,
- $o^{(e)}$ = perceptron output,
- $\eta$ = learning rate ($0 < \eta < 1$).

------

### 2.2 Algorithm

**Incremental Gradient Descent (Delta Rule)**

1. **Initialization:**
   - Training examples ${(x^{(e)}, y^{(e)})}_{e=1}^N$
   - Initial weights $w_i$ set to small random values
   - Learning rate $\eta = 0.1$ (typical choice)
2. **Repeat until termination condition is met:**
    For each training example $(x^{(e)}, y^{(e)})$:
   - Compute output:
      $$
      o^{(e)} = \sum_{i=0}^d w_i x_i^{(e)}
      $$
   - Update weights:
      $$
      w_i ;\leftarrow; w_i + \eta ,(y^{(e)} - o^{(e)}),x_i^{(e)}
      $$

------

## 3. Worked Example

Suppose a perceptron with two inputs $x_1=2$, $x_2=1$, and bias $x_0=1$.

Initial weights:
 $$
 w_1 = 0.5,\quad w_2 = 0.3,\quad w_0 = -1
 $$

------

### Step 1: Compute Output

$$
 o = w_1 x_1 + w_2 x_2 + w_0 \cdot x_0
 = 2 \cdot 0.5 + 1 \cdot 0.3 + (-1)\cdot 1
 = 1 + 0.3 - 1 = 0.3
 $$

------

### Step 2: Compare with Target

If the correct output is $y=0$, then:
 $$
 \text{Error} = (y - o) = (0 - 0.3) = -0.3
 $$

------

### Step 3: Update Weights

Using the Delta Rule with $\eta = 1$ for simplicity:

- Bias weight:
   $$
   w_0 = -1 + (-0.3)\cdot 1 = -1.3
   $$
- First input weight:
   $$
   w_1 = 0.5 + (-0.3)\cdot 2 = -0.1
   $$
- Second input weight:
   $$
   w_2 = 0.3 + (-0.3)\cdot 1 = 0
   $$

------

### Final Updated Weights

$$
 w_0 = -1.3, \quad w_1 = -0.1, \quad w_2 = 0
 $$

## 4. Batch Gradient Descent vs. Delta Rule (SGD)

| Aspect                   | **Batch Gradient Descent**                                   | **Delta Rule / Incremental Gradient Descent (SGD)**          |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Update Frequency**     | Updates weights **after the whole dataset** is processed (per epoch). | Updates weights **after each training example**.             |
| **Equation**             | $w_i \leftarrow w_i + \eta \sum_{e=1}^N (y^{(e)} - o^{(e)}) x_i^{(e)}$ | $w_i \leftarrow w_i + \eta (y^{(e)} - o^{(e)}) x_i^{(e)}$    |
| **Stability**            | More stable updates (less noisy).                            | More stochastic, updates can oscillate.                      |
| **Convergence Speed**    | Slow, especially for large datasets.                         | Faster per update; can converge more quickly.                |
| **Risk of Local Minima** | Higher risk of getting stuck in local minima.                | Can sometimes escape local minima due to noise in updates.   |
| **Computational Cost**   | Expensive per update (requires full pass over data).         | Cheaper per update (uses only one example).                  |
| **When Useful**          | Small datasets, convex problems, or when stable convergence is desired. | Large datasets, online learning, non-convex problems (typical in deep learning). |
| **Modern Usage**         | Rarely used directly for neural networks (too slow).         | Foundation of modern training (SGD and its variants: SGD+Momentum, Adam, RMSProp, etc.). |

------

## 5. Key Takeaway

- **Batch Gradient Descent**: stable but computationally heavy → impractical for large neural networks.
- **Delta Rule (SGD)**: lightweight, adaptive, and the basis for nearly all modern deep learning optimizers.

## 6. Summary

- The **batch gradient descent rule** updates weights after processing all examples → stable but slow.
- The **Delta Rule (incremental gradient descent)** updates weights after each example → faster and often avoids local minima.
- Also called the **LMS** or **Widrow–Hoff** rule.
- Foundation for **stochastic gradient descent (SGD)**, which underlies modern neural network training.

Perfect — here’s a polished, **Typora-ready lecture note** on **Steepest Gradient Descent Training for Sigmoidal Perceptrons**. I’ve structured it to be clear, professional, and easy to follow with equations in `$...$`.

------

# Steepest Gradient Descent Training

*CIS 311: Neural Networks*

------

## 1. Sigmoidal Perceptrons

The classic single-layer perceptron with **threshold** or **linear activation** is not sufficient for more powerful learning mechanisms, such as those used in multilayer neural networks.

To address this limitation, we introduce the **sigmoidal perceptron**, which uses a **sigmoid activation function**:

- Net input:

$$
 s = \sum_{i=0}^d w_i x_i
 $$

- Output:

$$
 o = \sigma(s) = \frac{1}{1+e^{-s}}
 $$

The sigmoid function has several desirable properties:

- Smooth and differentiable everywhere.
- Maps real values to $(0,1)$.
- Derivative is easy to compute:

$$
 \sigma'(s) = \sigma(s),\big(1 - \sigma(s)\big)
 $$

------

## 2. Gradient Descent Training Rule

The training objective is to minimize the **squared error**:

$$
 E(w) = \frac{1}{2} \sum_e \big(y^{(e)} - o^{(e)}\big)^2
 $$

where $y^{(e)}$ is the desired target output and $o^{(e)}$ is the network output.

### Derivative of the Error

$\frac{\partial E}{\partial w_i}
 = \sum_e \big(y^{(e)} - o^{(e)}\big) , \sigma'(s^{(e)}) , (-x_i^{(e)})$

Thus, the **steepest descent weight update** becomes:
$$
 w_i ;\leftarrow; w_i + \eta \sum_e \big(y^{(e)} - o^{(e)}\big),\sigma(s^{(e)})\big(1 - \sigma(s^{(e)})\big),x_i^{(e)}
$$

------

## 3. Batch Gradient Descent Algorithm

**Initialization:**

- Training examples ${(x^{(e)}, y^{(e)})}_{e=1}^N$
- Initial weights $w_i$ small random values
- Learning rate $\eta = 0.1$

**Repeat until convergence:**

1. For each training example $(x^{(e)}, y^{(e)})$:
   - Compute output:
      $$
      o^{(e)} = \sigma!\left(\sum_{i=0}^d w_i x_i^{(e)}\right)
      $$
   - Accumulate corrections:
      $$
      \Delta w_i ;+=; \eta ,(y^{(e)} - o^{(e)}),o^{(e)},(1-o^{(e)}),x_i^{(e)}
      $$
2. Update weights:
    $$
    w_i ;\leftarrow; w_i + \Delta w_i
    $$

------

## 4. Worked Example (Batch Update)

Suppose:

- Inputs: $x_1=2,;x_2=1,;x_0=1$ (bias)
- Initial weights: $w_0=-1,;w_1=0.5,;w_2=0.3$
- Target: $y=0$

### Step 1: Compute output

$
 s = -1 + (2)(0.5) + (1)(0.3) = 0.3
 $

$
 o = \sigma(0.3) = \frac{1}{1+e^{-0.3}} \approx 0.5744
$

### Step 2: Compute weight corrections

$
 \Delta w_0 = (0-0.5744)(0.5744)(1-0.5744)(1) = -0.1404
 $

$
 \Delta w_1 = (0-0.5744)(0.5744)(1-0.5744)(2) = -0.2808
 $

$
 \Delta w_2 = (0-0.5744)(0.5744)(1-0.5744)(1) = -0.1404
 $

------

### Step 3: Add another training example

- Inputs: $x_1=1,;x_2=2,;y=1$

Compute:
 $
 s = -1 + (1)(0.5) + (2)(0.3) = 0.1
 $

$
 o = \sigma(0.1) \approx 0.525
 $

Corrections (accumulated):

$
 \Delta w_0 = -0.1404 + (1-0.525)(0.525)(0.475)(1) = -0.0219
 $

$
 \Delta w_1 = -0.2808 + (1-0.525)(0.525)(0.475)(1) = -0.1623
 $

$
 \Delta w_2 = -0.1404 + (1-0.525)(0.525)(0.475)(2) = +0.0966
 $

### Step 4: Final weight update

$
 w_0 = -1 + (-0.0219) = -1.0219
 $

$
 w_1 = 0.5 + (-0.1623) = 0.3966
 $

$
 w_2 = 0.3 + 0.0966 = 0.3966
 $

------

## 5. Incremental Gradient Descent (Online Version)

In **incremental training**, weights are updated immediately after each example:

$
 w_i ;\leftarrow; w_i + \eta ,(y^{(e)} - o^{(e)}),o^{(e)},(1-o^{(e)}),x_i^{(e)}
 $

This version is faster and often preferred for large datasets.

------

## 6. Summary

- **Sigmoidal perceptrons** extend simple perceptrons by introducing smooth nonlinear activations.
- Their derivative enables gradient-based learning.
- Training can be done using **batch gradient descent** or **incremental (online) gradient descent**.
- These principles form the foundation of **backpropagation** in multilayer neural networks.

