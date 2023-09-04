# Linear Algebra Review

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

## NumPy

### NumPy N-dimensional Array

Arrays are the main data strucure used in ML. In Python, arrays from the `NumPy` library are used. 

The main structure in `NumPy` is the `ndarray`, a shorthand for N-dimensional array. 

The data type supported by an array can be accessed via the `dtype` attribute of the array. 

The dimensions of an array can be accessed via the `shape` attribute that returns a tuple describing the length of each dimension. 

A simple way to create an array from data is to use the `array()` function.

```Python
import numpy as np

l = np.array([1.0, 2.0, 3.0, 4.0])
print(l)
print(l.shape)
print(l.dtype)
```

which output:
```
[1. 2. 3. 4.]
(4,)
float64
```

#### Functions to Create Arrays

The `empty()` function will create a new array of the specified shape. The argument to the function is an array of tuples that specifies the length of each dimension of the array to create.

The values or content of the created array will be random and will need to be assigned before use. 

```Python
a = np.empty([3,3])
print(a)
```
which output:
```
[[4.86661498e-310 0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 0.00000000e+000]]
```

We can use the `zeros()` function to create a new array of the specified size with the contents filled with zero values. The argument to the function is an array of tuple that specifies the length of each dimension of the array to create.

```Python
a = np.zeros([3,5])
print(a)
```
which output:
```
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
```

The `ones()` function will create a new array of the specified size with the contents filled with one values. The argument to the function is an array or tuple that specifies the length of each dimension of the array to create. 

```Python
a = np.ones([5])
print(a)
```
which output:
```
[1. 1. 1. 1. 1.]
```

#### Combining Arrays

Given two or more existing arrays, you can stack them vertically using the `vstack()` function. For example, given two one-dimensional arrays, you can create a new two-dimensional array with two rows by vertically stacking them.

```Python
ar1 = np.array([1,2,3])
print(ar1)
ar2 = np.array([4,5,6])
print(ar2)
ar3 = np.vstack((ar1,ar2))
print(ar3)
print(ar3.shape)
```
which output:
```
[1 2 3]
[4 5 6]
[[1 2 3]
 [4 5 6]]
(2, 3)
```

You can also stack them horizontally using the `hstack()` function. For example, you can create new one-dimensional array with the columns of the first and second arrays concatenated.

```Python
ar1 = np.array([1,2,3])
print(ar1)
ar2 = np.array([4,5,6])
print(ar2)
ar3 = np.hstack((ar1,ar2))
print(ar3)
print(ar3.shape)
```
which output:
```
[1 2 3]
[4 5 6]
[1 2 3 4 5 6]
(6,)
```

### Index, Slice and Reshape NumPy Arrays

#### From List to Arrays

You can load you data and have access to it as a list. You can convert this one-dimensional list of data to an array by calling the `array()` function.

```
data = [11, 45, 67]
data = np.array(data)
print(data)
print(type(data))
```
which output:
```
[11 45 67]
<class 'numpy.ndarray'>
```

It is however more likely that you will have two-dimensional data. For example, a table of data where each row represents a new observation and each column a new feature. For example, you can have a list of lists. Each list represents a new observation. You can convert you list of lists to a `NumPy` array by calling the `array()` function:

```Python
data = [[1,2,3],
        [3,4,5]]
data = np.array(data)
print(data)
print(type(data))
```
which output:
```
[[1 2 3]
 [3 4 5]]
<class 'numpy.ndarray'>
```

#### Array Indexing

You can access elements using the bracket operator `[]` specifying the zero-offset index for the value to retrieve.

```Python
data = np.array([1,2,3])
print(data[0])
```
which output:
```
1
```

You can use negative indexes to retrieve values offset from the end of the array. For example, the index -1 refers to the last item in the array. 
The index -2 returns the second last item all the way back to -5 for the first item in the current example.

```Python
data = np.array([2,3,4,4,6,7])
print(data[-1])
print(data[-2])
```
which outputs:
```
7
6
```

Indexing two-dimensional data is similar to indexing one-dimensional data, except that a comma is used to separate the index for each dimension.

```Python
data = np.array([[1,2,3],
                 [4,5,6]])
print(data[0,1])
```
which output:
```
2
```

If we want all items in the first row, we can leave the second dimension index empty.

```Python
data = np.array([[1,2,3],
                 [4,5,6]])
print(data[0,])
```
which output:
```
[1 2 3]
```

#### Array Slicing

Slicing means that a subsequence of the structure can be indexed and retrieved. This is most useful in ML when specifying input variables and output variables, or splitting training rows from testing rows. 

Slicing is specified using the colon operate `:` with a from and to index before and after the column respectively.


## Vectors

## Norms

## Matrices

## Types of Matrices

## Matrix Operations

## Sparse Matrices

