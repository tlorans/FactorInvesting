
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

Vectors are built from components, which are ordinary numbers. We can think of a vector as a list of numbers and vector algebra as operations performed on the numbers in the list.

Vectors are often represent using a lowercase character, such as $v$ for example:

\begin{equation}
v = \begin{pmatrix}
v1 & v2 & v3
\end{pmatrix}
\end{equation}

Where $v_1$, $v_2$ and $v_3$ are scalar values.

Vectors can also be shown using a column representation:

\begin{equation}
v = 
\begin{pmatrix}
v_1 \\
v_2 \\
v_3
\end{pmatrix}
\end{equation}

It is a common practice to represent the target variable as a vector with the lowercase $y$ when describing a ML algorithm.

### Norms

Calculating the length or magnitude of vectors is ofen required either directly as a regularization method in machine learning or as part of broader vector or matrix operations. Vector lengths or magnitude are also called the vector norm. In summary:

- The $L^1$ norm is calculated as the sum of the absolute values of the vector
- The $L^2$ norm is calculated as the square root of the sum of the squared vector values
- The max norm is calculated as the maximum vector values.

#### Vector Norm

Calculating the size or length of a vector is often required either directly or as part of a broader vector-matrix opertion. The length of the vector is referred to as the vector norm or the vector's magnitude.

The length of the vector is always a positive number, expect for a vector of all zero values. It is calculated using some measure that summarizes the distance of the vector from the origin of the vector space. 

#### Vector $L^1$ Norm

The length of a vector can be calculated using the $L^1$ norm, where the 1 is a superscript of the $L$. The notation for the $L^1$ norm of a vector is $||v||_1$. As such, this length is sometimes called the Manhattan norm.

\begin{equation}
L^1(v) = ||v||_1
\end{equation}

The $L^1$ norm is calculated as the sum of the absolute vector values, where the absolute value of a scalar uses the notation $|a_1|$. In effect, the norm is a calculation of the Manhattan distance from the origin of the vector space:

\begin{equation}
||v||_1 = |a_1| + |a_2| + |a_3|
\end{equation}

In many ML applications, it is important to discriminate between elements that are exactly zero and elements that are small but nonzero. In these cases, we turn to a function that grows at the same rate in all locations, but retains mathematical simplicity: the $L^1$ norm.

*Example: Let's have the vector $ a = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$. We have $L^1(a) = 6$.*

The $L^1$ norm is often used when fitting ML algorithms as a regularization method, ie a method to keep the coefficients of the model small and thus the model less complex.

#### Vector $L^2$ Norm

The length of a vector can be calculated using the $L^2$ norm. The notation for the $L^2$ norm vector is the following:

\begin{equation}
L^2(v) = ||v||_2
\end{equation}

The $L^2$ norm calculates the distance of the vector coordinate from the origin of the vector space. As such, it is also known as the Euclidean norm as it is calculated as the Euclidean distance from the origin. The result is a positive distance value. The $L^2$ norm is calculated as the square root of the sum of the squared vector values:

\begin{equation}
||v||_2 = \sqrt{a^2_1 + a_2^2 + a^2_3}
\end{equation}

*Example: For a vector $a = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$, we have $L^2(a) \approx 3.74$.* 

The $L^2$ norm is also used as a regularization method (keeping the coefficients of the model small). The $L^2$ norm is the most commonly used vector norms method in ML.

#### Vector Max Norm

The length of a vector can be calculated using the maximum or max norm. It is referred to as $L^{inf}$. 

The notation is:

\begin{equation}
L^{inf}(v) = ||v||_{inf}
\end{equation}

It is calculated as returning the maximum value of the vector:

\begin{equation}
||v||_{inf} = \max a_1, a_2, a_3
\end{equation}

*Example: For a vector $a = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$, we have $L^{inf}(a) = 3$.*

It is also used as a regularization method in ML, such as on neural network weights, called max norm regularization.
### Types of Matrices

#### Orthogonal Matrix

Two vectors are orthogonal when their dot product equals zero. The length of each vector is 1 then the vector are called orthonormal vecause they are both orthogonal and normalized.

\begin{equation}
v \cdot w = 0
\end{equation}

An orthogonal matrix is a type of square matrix whose columns and rows are orthonormal unit vectors, ie. perpedicular and have a length of magnitude of 1.

An orthogonal matrix is a square matrix whose rows are mutually orthonormal and whose columns are mutually orthonomal.

An Orthogonal matrix is often denoted as $Q$. 

The Orthogonal matrix is defined as follows:

\begin{equation}
Q^T \cdot Q = Q \cdot Q^T = I
\end{equation}

Orthogonal matrices are used a lot for linear transformations, such as reflections and permtations. 

*Example: We have the following orthogonal matrix:*

\begin{equation}
Q = 
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
\end{equation}

\begin{equation}
Q \cdot Q^T = \begin{pmatrix}
1 & 0 \\
0 & 1 
\end{pmatrix}
\end{equation}
### Matrix Operations

#### Trace 

A trace of a square matrix is the sum of the values on the main diagonal of the matrix.

It is described using the notation $tr(A)$ where $A$ is the square matrix on which the operation is performed.

The trace is calculated as the sum of the diagonal values, for example in the case of $3 \times 3$ matrix:

\begin{equation}
tr(A) = a_{1,1} + a_{2,2} + a_{3,3}
\end{equation}

*Example: we have the following matrix:*

\begin{equation}
A = \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}
\end{equation}

\begin{equation}
tr(A) = 15
\end{equation}

#### Determinant

The determinant of a square matrix is a scalar representation of the volume of the matrix. 

It is denoted by the $det(A)$ or $|A|$ notation, where $A$ is the matrix on which we are calculating the determinant.

The determinant of a square matrix is calculated from the elements of the matrix.
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