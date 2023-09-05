
# Background 

## Linear Algebra

Linear algebra is the the maths of data, with vectors and matrices of numbers. Classical methods such as linear regression are linear algebra methods, and other methods such as PCA were born from the marriage of linear algebra and stats.

To understand machine learning, you need to be able to read and understand linear algebra.

Linear equation is a serie of terms and operations where some terms are unknown, for example:

\begin{equation}
y = 4 \times x + 1
\end{equation}

This equation is linear as it describes a line on a two-dimensional graph. The line comes from plugging in different values into the unknown $x$ to find out what the model does to the value of $y$.

We can have a system of equations with the same form with three unknowns for example:

\begin{equation}
y = 0.1 \times x_1 + 0.4 \times x_2 \\
y = 0.3 \times x_2 + 0.9 \times x_2 \\
y = 0.2 \times x_1 + 0.3 \times x2
\end{equation}

The column of $y$ values can be taken as a column vector of the outputs from the equation.

The two columns of integer are the data columns, which can be for example $a_1$ and $a_2$, and can be taken as a matrix $A$.

The two unknown values $x_1$ and $x_2$ can be taken as the coefficients of the equation and form a vector of unknowns $b$ to be solve. This can be written compactly using linear algebra:

\begin{equation}
y = A \cdot b
\end{equation}

However, in real life, we have generally more unknowns than equations to solve, and we often need to approximate the solutions (ie. finding a solution approximating $y$).


### Vectors

### Norms

### Matrices

### Types of Matrices

### Matrix Operations

### Sparse Matrices

## Statistics 


## Probability 

### Distributions

### Maximum Likelihood

### Bayesian Probability

### Information Theory

### Classification

## Calculus

### Limits and Differential Calculus

### Multivariate Calculus 

### Mathematical Programming

### Approximation

### Gradient Descent

## Optimization