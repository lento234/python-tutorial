
# **Numpy -  multidimensional data arrays**

J.R. Johansson (jrjohansson at gmail.com) modified by **Lento Manickathan**

- The latest version of this [IPython notebook](http://ipython.org/notebook.html) lecture is available at [http://github.com/jrjohansson/scientific-python-lectures](http://github.com/jrjohansson/scientific-python-lectures).
- The other notebooks in this lecture series are indexed at [http://jrjohansson.github.io](http://jrjohansson.github.io).

**Table of content:**

1. [Introduction](#introduction)
2. [Creating numpy array](#creating_numpy_array)
3. [Using array functions](#using_array_functions)
4. [Random data](#random)
5. [File I/O](#file_io)
6. [Numpy array properties](#numpy_array_properties)
7. [Manipulating arrays](#manipulating_arrays)
8. [Functions for extracting data from arrays and creating arrays](#extracting)
9. [Linear algebra](#linear_algebra)
10. [Array transformations](#array_transformations)
11. [Matrix computations](#matrix_computations)
12. [Data processing](#data_processing)
13. [Computations on subsets of arrays](#array_subset)
14. [Calculations with higher-dimensional data](#high_dim_data)
15. [Reshaping, resizing and stacking arrays](#reshaping_resize_stacking)
16. [Adding a new dimension](#newaxis)
17. [Stacking and repeating arrays](#stacking_repeating)
18. [Copy and deep-copy](#copy_deepcopy)
19. [Iterating over array elements](#iterating)
20. [Vectorizing functions](#vectorizing)
21. [Using arrays in conditions](#array_conditions)
22. [Type casting](#type_casting)
23. [Futher reading](#further_reading)

<a id='introduction'></a>
# Introduction


```python
%matplotlib inline
import matplotlib.pyplot as plt
```

The `numpy` package (module) is used in almost all numerical computation using Python. It is a package that provide high-performance vector, matrix and higher-dimensional data structures for Python. It is implemented in C and Fortran so when calculations are vectorized (formulated with vectors and matrices), performance is very good. 

To use `numpy` you need to import the module, using for example:

*WARNING: We must not do this*  **(bad)**

```python
from numpy import *
```

*keep you namespace clean silly*  **(good)**

```python
import numpy as np
```


```python
import numpy as np
```

In the `numpy` package the terminology used for vectors, matrices and higher-dimensional data sets is *array*. 



<a id='creating_numpy_array'></a>
# Creating `numpy` arrays

There are a number of ways to initialize new numpy arrays, for example from

* a Python list or tuples
* using functions that are dedicated to generating numpy arrays, such as `arange`, `linspace`, etc.
* reading data from files

### **From lists**

For example, to create new vector and matrix arrays from Python lists we can use the `numpy.array` function.


```python
# a vector: the argument to the array function is a Python list
v = np.array([1,2,3,4])

v
```




    array([1, 2, 3, 4])




```python
# a matrix: the argument to the array function is a nested Python list
M = np.array([[1, 2], [3, 4]])

M
```




    array([[1, 2],
           [3, 4]])



The `v` and `M` objects are both of the type `ndarray` that the `numpy` module provides.


```python
type(v), type(M)
```




    (numpy.ndarray, numpy.ndarray)



The difference between the `v` and `M` arrays is only their shapes. We can get information about the shape of an array by using the `ndarray.shape` property.


```python
v.shape
```




    (4,)




```python
M.shape
```




    (2, 2)



The number of elements in the array is available through the `ndarray.size` property:


```python
M.size
```




    4



Equivalently, we could use the function `numpy.shape` and `numpy.size`


```python
np.shape(M)
```




    (2, 2)



or


```python
M.shape
```




    (2, 2)




```python
np.size(M) # or M.size
```




    4



So far the `numpy.ndarray` looks awefully much like a Python list (or nested list). Why not simply use Python lists for computations instead of creating a new array type? 

There are several reasons:

* Python lists are very general. They can contain any kind of object. They are dynamically typed. They do not support mathematical functions such as matrix and dot multiplications, etc. Implementing such functions for Python lists would not be very efficient because of the dynamic typing.
* Numpy arrays are **statically typed** and **homogeneous**. The type of the elements is determined when the array is created.
* Numpy arrays are memory efficient.
* Because of the static typing, fast implementation of mathematical functions such as multiplication and addition of `numpy` arrays can be implemented in a compiled language (C and Fortran is used).

Using the `dtype` (data type) property of an `ndarray`, we can see what type the data of an array has:


```python
M.dtype
```




    dtype('int64')



We get an error if we try to assign a value of the wrong type to an element in a numpy array:


```python
M[0,0] = "hello"
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-14-e1f336250f69> in <module>
    ----> 1 M[0,0] = "hello"
    

    ValueError: invalid literal for int() with base 10: 'hello'


If we want, we can explicitly define the type of the array data when we create it, using the `dtype` keyword argument: 


```python
M = np.array([[1, 2], [3, 4]], dtype=complex)

M
```




    array([[1.+0.j, 2.+0.j],
           [3.+0.j, 4.+0.j]])



Common data types that can be used with `dtype` are: `int`, `float`, `complex`, `bool`, `object`, etc.

We can also explicitly define the bit size of the data types, for example: `int64`, `int16`, `float128`, `complex128`.

<a id='using_array_functions'></a>
# Using array functions

For larger arrays it is inpractical to initialize the data manually, using explicit python lists. Instead we can use one of the many functions in `numpy` that generate arrays of different forms. Some of the more common are:

### **arange**


```python
# create a range

x = np.arange(0, 10, 1) # arguments: start, stop, step

x
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
x =np. arange(-1, 1, 0.1)

x
```




    array([-1.00000000e+00, -9.00000000e-01, -8.00000000e-01, -7.00000000e-01,
           -6.00000000e-01, -5.00000000e-01, -4.00000000e-01, -3.00000000e-01,
           -2.00000000e-01, -1.00000000e-01, -2.22044605e-16,  1.00000000e-01,
            2.00000000e-01,  3.00000000e-01,  4.00000000e-01,  5.00000000e-01,
            6.00000000e-01,  7.00000000e-01,  8.00000000e-01,  9.00000000e-01])



### **linspace and logspace**


```python
# using linspace, both end points ARE included
np.linspace(0, 10, 25)
```




    array([ 0.        ,  0.41666667,  0.83333333,  1.25      ,  1.66666667,
            2.08333333,  2.5       ,  2.91666667,  3.33333333,  3.75      ,
            4.16666667,  4.58333333,  5.        ,  5.41666667,  5.83333333,
            6.25      ,  6.66666667,  7.08333333,  7.5       ,  7.91666667,
            8.33333333,  8.75      ,  9.16666667,  9.58333333, 10.        ])




```python
np.logspace(0, 10, 10, base=np.e)
```




    array([1.00000000e+00, 3.03773178e+00, 9.22781435e+00, 2.80316249e+01,
           8.51525577e+01, 2.58670631e+02, 7.85771994e+02, 2.38696456e+03,
           7.25095809e+03, 2.20264658e+04])



### **mgrid**


```python
x, y = np.mgrid[0:5, 0:5] # similar to meshgrid in MATLAB
```


```python
x
```




    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])




```python
y
```




    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])



<a id='random'></a>
# Random data

For reproducable code, we use a seed in random number generator.


```python
np.random.seed(123) 
```


```python
# uniform random numbers in [0,1]
np.random.rand(5,5)
```




    array([[0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897],
           [0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752],
           [0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426],
           [0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759],
           [0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338]])




```python
# standard normal distributed random numbers
np.random.randn(5,5)
```




    array([[-1.10098526, -1.4103012 , -0.74765132, -0.98486761, -0.74856868],
           [ 0.24036728, -1.85563747, -1.7794548 , -2.75022426, -0.23415755],
           [-0.69598118, -1.77413406,  2.36160126,  0.03499308, -0.34464169],
           [-0.72503229,  1.03960617, -0.24172804, -0.11290536, -1.66069578],
           [ 0.01353855,  0.33737412, -0.92662298,  0.27574741,  0.37085233]])



### **diag**


```python
# a diagonal matrix
np.diag([1,2,3])
```




    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])




```python
# diagonal with offset from the main diagonal
np.diag([1,2,3], k=1) 
```




    array([[0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 3],
           [0, 0, 0, 0]])



### **zeros and ones**


```python
np.zeros((3,3))
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
np.ones((3,3))
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])



<a id='file_io'></a>
# File I/O

### **Comma-separated values (CSV)**

A very common file format for data files is comma-separated values (CSV), or related formats such as TSV (tab-separated values). To read data from such files into Numpy arrays we can use the `numpy.genfromtxt` function. For example, 


```python
!head data/stockholm_td_adj.dat
```

    1800  1  1    -6.1    -6.1    -6.1 1
    1800  1  2   -15.4   -15.4   -15.4 1
    1800  1  3   -15.0   -15.0   -15.0 1
    1800  1  4   -19.3   -19.3   -19.3 1
    1800  1  5   -16.8   -16.8   -16.8 1
    1800  1  6   -11.4   -11.4   -11.4 1
    1800  1  7    -7.6    -7.6    -7.6 1
    1800  1  8    -7.1    -7.1    -7.1 1
    1800  1  9   -10.1   -10.1   -10.1 1
    1800  1 10    -9.5    -9.5    -9.5 1



```python
data = np.genfromtxt('data/stockholm_td_adj.dat')
```


```python
data.shape
```




    (77431, 7)



### **Plotting**


```python
fig, ax = plt.subplots(figsize=(14,4))
ax.plot(data[:,0]+data[:,1]/12.0+data[:,2]/365, data[:,5])
ax.axis('tight')
ax.set_title('tempeatures in Stockholm')
ax.set_xlabel('year')
ax.set_ylabel('temperature (C)');
```


![png](data/output_64_0.png)


Using `numpy.savetxt` we can store a Numpy array to a file in CSV format:


```python
M = np.random.rand(3,3)

M
```




    array([[0.54506801, 0.34276383, 0.30412079],
           [0.41702221, 0.68130077, 0.87545684],
           [0.51042234, 0.66931378, 0.58593655]])




```python
np.savetxt("random-matrix.csv", M)
```


```python
!cat random-matrix.csv
```

    5.450680064664649160e-01 3.427638337743084129e-01 3.041207890271840908e-01
    4.170222110247016056e-01 6.813007657927966365e-01 8.754568417951749115e-01
    5.104223374780111344e-01 6.693137829622722856e-01 5.859365525622128867e-01



```python
np.savetxt("random-matrix.csv", M, fmt='%.5f') # fmt specifies the format

!cat random-matrix.csv
```

    0.54507 0.34276 0.30412
    0.41702 0.68130 0.87546
    0.51042 0.66931 0.58594


### **Numpy's native file format**

Useful when storing and reading back numpy array data. Use the functions `numpy.save` and `numpy.load`:


```python
np.savez("random-matrix", M)

!file random-matrix.npz
```

    random-matrix.npz: Zip archive data, at least v2.0 to extract



```python
f = np.load("random-matrix.npz")['arr_0']
print(f)
```

    [[0.54506801 0.34276383 0.30412079]
     [0.41702221 0.68130077 0.87545684]
     [0.51042234 0.66931378 0.58593655]]


<a id='numpy_array_properties'></a>
# Numpy array properties

Data type:


```python
M.dtype 
```




    dtype('float64')



Item byte size:


```python
M.itemsize # bytes per element
```




    8



Number of bytes:


```python
M.nbytes # number of bytes
```




    72



Number of dimensions (or axes)


```python
M.ndim # number of dimensions
```




    2



Array stride size (row, col)


```python
M.strides # array stride
```




    (24, 8)



<a id='manipulating_arrays'></a>
# Manipulating arrays

### **Indexing**

We can index elements in an array using square brackets and indices:

<p><span style="color:red"><em>WARNING:</em> Array starts at 0 not 1.</span> We are no longer in MATLAB or FORTRAN.</p>


```python
# v is a vector, and has only one dimension, taking one index
v[0]
```




    1




```python
# M is a matrix, or a 2 dimensional array, taking two indices 
M[0,0]
```




    0.5450680064664649



If we omit an index of a multidimensional array it returns the whole row (or, in general, a N-1 dimensional array) 


```python
M
```




    array([[0.54506801, 0.34276383, 0.30412079],
           [0.41702221, 0.68130077, 0.87545684],
           [0.51042234, 0.66931378, 0.58593655]])




```python
M[0]
```




    array([0.54506801, 0.34276383, 0.30412079])



The same thing can be achieved with using `:` instead of an index: 


```python
M[0,:] # row 0
```




    array([0.54506801, 0.34276383, 0.30412079])




```python
M[:,0] # column 0
```




    array([0.54506801, 0.41702221, 0.51042234])



We can assign new values to elements in an array using indexing:


```python
M[0,0] = 1
```


```python
M
```




    array([[1.        , 0.34276383, 0.30412079],
           [0.41702221, 0.68130077, 0.87545684],
           [0.51042234, 0.66931378, 0.58593655]])




```python
# also works for rows and columns
M[1,:] = 0
M[:,2] = -1
```


```python
M
```




    array([[ 1.        ,  0.34276383, -1.        ],
           [ 0.        ,  0.        , -1.        ],
           [ 0.51042234,  0.66931378, -1.        ]])



### **Index slicing**

Index slicing is the technical name for the syntax `M[lower:upper:step]` to extract part of an array:


```python
A = np.array([1,2,3,4,5])
A
```




    array([1, 2, 3, 4, 5])




```python
A[1:3]
```




    array([2, 3])



Array slices are *mutable*: if they are assigned a new value the original array from which the slice was extracted is modified:


```python
A[1:3] = [-2,-3]

A
```




    array([ 1, -2, -3,  4,  5])



We can omit any of the three parameters in `M[lower:upper:step]`:


```python
A[::] # lower, upper, step all take the default values
```




    array([ 1, -2, -3,  4,  5])




```python
A[::2] # step is 2, lower and upper defaults to the beginning and end of the array
```




    array([ 1, -3,  5])




```python
A[:3] # first three elements
```




    array([ 1, -2, -3])




```python
A[3:] # elements from index 3
```




    array([4, 5])



Negative indices counts from the end of the array (positive index from the begining):


```python
A = np.array([1,2,3,4,5])
```


```python
A[-1] # the last element in the array
```




    5




```python
A[-3:] # the last three elements
```




    array([3, 4, 5])



Index slicing works exactly the same way for multidimensional arrays:


```python
A = np.array([[n+m*10 for n in range(5)] for m in range(5)])

A
```




    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [20, 21, 22, 23, 24],
           [30, 31, 32, 33, 34],
           [40, 41, 42, 43, 44]])




```python
# a block from the original array
A[1:4, 1:4]
```




    array([[11, 12, 13],
           [21, 22, 23],
           [31, 32, 33]])




```python
# strides
A[::2, ::2]
```




    array([[ 0,  2,  4],
           [20, 22, 24],
           [40, 42, 44]])



### **Fancy indexing**

Fancy indexing is the name for when an array or list is used in-place of an index: 


```python
row_indices = [1, 2, 3]
A[row_indices]
```




    array([[10, 11, 12, 13, 14],
           [20, 21, 22, 23, 24],
           [30, 31, 32, 33, 34]])




```python
col_indices = [1, 2, -1] # remember, index -1 means the last element
A[row_indices, col_indices]
```




    array([11, 22, 34])



We can also use index masks: If the index mask is an Numpy array of data type `bool`, then an element is selected (True) or not (False) depending on the value of the index mask at the position of each element: 


```python
B = np.array([n for n in range(5)])
B
```




    array([0, 1, 2, 3, 4])




```python
row_mask = np.array([True, False, True, False, False])
B[row_mask]
```




    array([0, 2])




```python
# same thing
row_mask = np.array([1,0,1,0,0], dtype=bool)
B[row_mask]
```




    array([0, 2])



This feature is very useful to conditionally select elements from an array, using for example comparison operators:


```python
x = np.arange(0, 10, 0.5)
x
```




    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ,
           6.5, 7. , 7.5, 8. , 8.5, 9. , 9.5])




```python
mask = (5 < x) * (x < 7.5)

mask
```




    array([False, False, False, False, False, False, False, False, False,
           False, False,  True,  True,  True,  True, False, False, False,
           False, False])




```python
x[mask]
```




    array([5.5, 6. , 6.5, 7. ])



<a id='extracting'></a>
# Functions for extracting data from arrays and creating arrays

### **where**

The index mask can be converted to position index using the `where` function


```python
indices = np.where(mask)

indices
```




    (array([11, 12, 13, 14]),)




```python
x[indices] # this indexing is equivalent to the fancy indexing x[mask]
```




    array([5.5, 6. , 6.5, 7. ])



### **diag**

With the diag function we can also extract the diagonal and subdiagonals of an array:


```python
np.diag(A)
```




    array([ 0, 11, 22, 33, 44])




```python
np.diag(A, -1)
```




    array([10, 21, 32, 43])



### **take**

The `take` function is similar to fancy indexing described above:


```python
v2 = np.arange(-3,3)
v2
```




    array([-3, -2, -1,  0,  1,  2])




```python
row_indices = [1, 3, 5]
v2[row_indices] # fancy indexing
```




    array([-2,  0,  2])




```python
v2.take(row_indices)
```




    array([-2,  0,  2])



But `take` also works on lists and other objects:


```python
np.take([-3, -2, -1,  0,  1,  2], row_indices)
```




    array([-2,  0,  2])



### **choose**

Constructs an array by picking elements from several arrays:


```python
which = [1, 0, 1, 0]
choices = [[-2,-2,-2,-2], [5,5,5,5]]

np.choose(which, choices)
```




    array([ 5, -2,  5, -2])



<a id='linear_algebra'></a>
# Linear algebra

Vectorizing code is the key to writing efficient numerical calculation with Python/Numpy. That means that as much as possible of a program should be formulated in terms of matrix and vector operations, like matrix-matrix multiplication.

### **Scalar-array operations**

We can use the usual arithmetic operators to multiply, add, subtract, and divide arrays with scalar numbers.


```python
v1 = np.arange(0, 5)
```


```python
v1 * 2
```




    array([0, 2, 4, 6, 8])




```python
v1 + 2
```




    array([2, 3, 4, 5, 6])




```python
A * 2, A + 2
```




    (array([[ 0,  2,  4,  6,  8],
            [20, 22, 24, 26, 28],
            [40, 42, 44, 46, 48],
            [60, 62, 64, 66, 68],
            [80, 82, 84, 86, 88]]), array([[ 2,  3,  4,  5,  6],
            [12, 13, 14, 15, 16],
            [22, 23, 24, 25, 26],
            [32, 33, 34, 35, 36],
            [42, 43, 44, 45, 46]]))



### **Element-wise array-array operations**

When we add, subtract, multiply and divide arrays with each other, the default behaviour is **element-wise** operations:


```python
A * A # element-wise multiplication
```




    array([[   0,    1,    4,    9,   16],
           [ 100,  121,  144,  169,  196],
           [ 400,  441,  484,  529,  576],
           [ 900,  961, 1024, 1089, 1156],
           [1600, 1681, 1764, 1849, 1936]])




```python
v1 * v1
```




    array([ 0,  1,  4,  9, 16])



If we multiply arrays with compatible shapes, we get an element-wise multiplication of each row:


```python
A.shape, v1.shape
```




    ((5, 5), (5,))




```python
A * v1
```




    array([[  0,   1,   4,   9,  16],
           [  0,  11,  24,  39,  56],
           [  0,  21,  44,  69,  96],
           [  0,  31,  64,  99, 136],
           [  0,  41,  84, 129, 176]])



### **Matrix algebra**

What about matrix mutiplication? There are two ways. We can either use the `dot` function, which applies a matrix-matrix, matrix-vector, or inner vector multiplication to its two arguments: 


```python
np.dot(A, A)
```




    array([[ 300,  310,  320,  330,  340],
           [1300, 1360, 1420, 1480, 1540],
           [2300, 2410, 2520, 2630, 2740],
           [3300, 3460, 3620, 3780, 3940],
           [4300, 4510, 4720, 4930, 5140]])




```python
np.dot(A, v1)
```




    array([ 30, 130, 230, 330, 430])




```python
np.dot(v1, v1)
```




    30



See also the related functions: `inner`, `outer`, `cross`, `kron`, `tensordot`. Try for example `help(kron)`.

<a id='array_transformations'></a>
# Array transformations

Above we have used the `.T` to transpose the matrix object `v`. We could also have used the `transpose` function to accomplish the same thing. 

Other mathematical functions that transform matrix objects are:

### **Complex array**


```python
C = np.array([[1j, 2j], [3j, 4j]])
C
```




    array([[0.+1.j, 0.+2.j],
           [0.+3.j, 0.+4.j]])



### **Complex conjugate**


```python
np.conjugate(C)
```




    array([[0.-1.j, 0.-2.j],
           [0.-3.j, 0.-4.j]])



We can extract the real and imaginary parts of complex-valued arrays using `real` and `imag`:

### **Real and imaginary parts**


```python
np.real(C) # same as: C.real
```




    array([[0., 0.],
           [0., 0.]])




```python
np.imag(C) # same as: C.imag
```




    array([[1., 2.],
           [3., 4.]])



Or the complex argument and absolute value

### **Angle and magnitude**


```python
np.angle(C+1) # heads up MATLAB Users, angle is used instead of arg
```




    array([[0.78539816, 1.10714872],
           [1.24904577, 1.32581766]])




```python
np.abs(C)
```




    array([[1., 2.],
           [3., 4.]])



<a id='matrix_computations'></a>

# Matrix computations

### **Inverse**


```python
np.linalg.inv(C) # equivalent to C.I 
```




    array([[0.+2.j , 0.-1.j ],
           [0.-1.5j, 0.+0.5j]])




```python
np.linalg.inv(C) * C
```




    array([[-2. +0.j,  2. +0.j],
           [ 4.5+0.j, -2. +0.j]])



### **Determinant**


```python
np.linalg.det(C)
```




    (2.0000000000000004+0j)




```python
np.linalg.det(np.linalg.inv(C))
```




    (0.49999999999999967+0j)



<a id='data_processing'></a>
# Data processing

Often it is useful to store datasets in Numpy arrays. Numpy provides a number of functions to calculate statistics of datasets in arrays. 

For example, let's calculate some properties from the Stockholm temperature dataset used above.


```python
# reminder, the tempeature dataset is stored in the data variable:
np.shape(data)
```




    (77431, 7)



### **mean**


```python
# the temperature data is in column 3
np.mean(data[:,3])
```




    6.197109684751585



The daily mean temperature in Stockholm over the last 200 years has been about 6.2 C.

### **standard deviations and variance**


```python
np.std(data[:,3]), np.var(data[:,3])
```




    (8.282271621340573, 68.59602320966341)



### **min and max**


```python
# lowest daily average temperature
data[:,3].min()
```




    -25.8




```python
# highest daily average temperature
data[:,3].max()
```




    28.3



### **sum, prod, and trace**


```python
d = np.arange(0, 10)
d
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# sum up all elements
np.sum(d)
```




    45




```python
# product of all elements
np.prod(d+1)
```




    3628800




```python
# cummulative sum
np.cumsum(d)
```




    array([ 0,  1,  3,  6, 10, 15, 21, 28, 36, 45])




```python
# cummulative product
np.cumprod(d+1)
```




    array([      1,       2,       6,      24,     120,     720,    5040,
             40320,  362880, 3628800])




```python
# same as: diag(A).sum()
np.trace(A)
```




    110



<a id='array_subset'></a>
# Computations on subsets of arrays

We can compute with subsets of the data in an array using indexing, fancy indexing, and the other methods of extracting data from an array (described above).

For example, let's go back to the temperature dataset:


```python
!head -n 3 data/stockholm_td_adj.dat
```

    1800  1  1    -6.1    -6.1    -6.1 1
    1800  1  2   -15.4   -15.4   -15.4 1
    1800  1  3   -15.0   -15.0   -15.0 1


The dataformat is: year, month, day, daily average temperature, low, high, location.

If we are interested in the average temperature only in a particular month, say February, then we can create a index mask and use it to select only the data for that month using:


```python
np.unique(data[:,1]) # the month column takes values from 1 to 12
```




    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.])




```python
mask_feb = data[:,1] == 2
```


```python
# the temperature data is in column 3
np.mean(data[mask_feb,3])
```




    -3.212109570736596



With these tools we have very powerful data processing capabilities at our disposal. For example, to extract the average monthly average temperatures for each month of the year only takes a few lines of code: 

### **Plot montly average**


```python
months = np.arange(1,13)
monthly_mean = [np.mean(data[data[:,1] == month, 3]) for month in months]

fig, ax = plt.subplots()
ax.bar(months, monthly_mean)
ax.set_xlabel("Month")
ax.set_ylabel("Monthly avg. temp.");
```


![png](data/output_220_0.png)


<a id='high_dim_data'></a>
# Calculations with higher-dimensional data

When functions such as `min`, `max`, etc. are applied to a multidimensional arrays, it is sometimes useful to apply the calculation to the entire array, and sometimes only on a row or column basis. Using the `axis` argument we can specify how these functions should behave: 


```python
m = np.random.rand(3,3)
m
```




    array([[0.6249035 , 0.67468905, 0.84234244],
           [0.08319499, 0.76368284, 0.24366637],
           [0.19422296, 0.57245696, 0.09571252]])




```python
# global max
m.max()
```




    0.8423424376202573




```python
# max in each column
m.max(axis=0)
```




    array([0.6249035 , 0.76368284, 0.84234244])




```python
# max in each row
m.max(axis=1)
```




    array([0.84234244, 0.76368284, 0.57245696])



Many other functions and methods in the `array` and `matrix` classes accept the same (optional) `axis` keyword argument.

<a id='reshaping_resize_stacking'></a>
# Reshaping, resizing and stacking arrays

The shape of an Numpy array can be modified without copying the underlaying data, which makes it a fast operation even for large arrays.


```python
A
```




    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [20, 21, 22, 23, 24],
           [30, 31, 32, 33, 34],
           [40, 41, 42, 43, 44]])




```python
n, m = A.shape
```


```python
B = A.reshape((1,n*m))
B
```




    array([[ 0,  1,  2,  3,  4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30,
            31, 32, 33, 34, 40, 41, 42, 43, 44]])




```python
B[0,0:5] = 5 # modify the array

B
```




    array([[ 5,  5,  5,  5,  5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30,
            31, 32, 33, 34, 40, 41, 42, 43, 44]])




```python
A # and the original variable is also changed. B is only a different view of the same data
```




    array([[ 5,  5,  5,  5,  5],
           [10, 11, 12, 13, 14],
           [20, 21, 22, 23, 24],
           [30, 31, 32, 33, 34],
           [40, 41, 42, 43, 44]])



We can also use the function `flatten` to make a higher-dimensional array into a vector. But this function create a copy of the data.


```python
B = A.flatten()

B
```




    array([ 5,  5,  5,  5,  5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31,
           32, 33, 34, 40, 41, 42, 43, 44])




```python
B[0:5] = 10

B
```




    array([10, 10, 10, 10, 10, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31,
           32, 33, 34, 40, 41, 42, 43, 44])




```python
A # now A has not changed, because B's data is a copy of A's, not refering to the same data
```




    array([[ 5,  5,  5,  5,  5],
           [10, 11, 12, 13, 14],
           [20, 21, 22, 23, 24],
           [30, 31, 32, 33, 34],
           [40, 41, 42, 43, 44]])



<a id='newaxis'></a>
# Adding a new dimension: newaxis

With `newaxis`, we can insert new dimensions in an array, for example converting a vector to a column or row matrix:


```python
v = np.array([1,2,3])
```


```python
np.shape(v)
```




    (3,)




```python
# make a column matrix of the vector v
v[:, np.newaxis]
```




    array([[1],
           [2],
           [3]])




```python
# column matrix
v[:,np.newaxis].shape
```




    (3, 1)




```python
# row matrix
v[np.newaxis,:].shape
```




    (1, 3)



<a id='stacking_repeating'></a>
# Stacking and repeating arrays

Using function `repeat`, `tile`, `vstack`, `hstack`, and `concatenate` we can create larger vectors and matrices from smaller ones:

### **tile and repeat**


```python
a = np.array([[1, 2], [3, 4]])
```


```python
# repeat each element 3 times
np.repeat(a, 3)
```




    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])




```python
# tile the matrix 3 times 
np.tile(a, 3)
```




    array([[1, 2, 1, 2, 1, 2],
           [3, 4, 3, 4, 3, 4]])



### **concatenate**


```python
b = np.array([[5, 6]])
```


```python
np.concatenate((a, b), axis=0)
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
np.concatenate((a, b.T), axis=1)
```




    array([[1, 2, 5],
           [3, 4, 6]])



### **hstack and vstack**


```python
np.vstack((a,b))
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
np.hstack((a,b.T))
```




    array([[1, 2, 5],
           [3, 4, 6]])



<a id='copy_deepcopy'></a>
# Copy and **"deep copy"**

To achieve high performance, assignments in Python usually do not copy the underlaying objects. This is important for example when objects are passed between functions, to avoid an excessive amount of memory copying when it is not necessary (technical term: pass by reference). 

### **Shallow-copy or referecing**

<p><span style="color:red"><em>WARNING:</em> This is the default in python. </span>A major source of errors for new python users.</p>


```python
A = np.array([[1, 2], [3, 4]])

A
```




    array([[1, 2],
           [3, 4]])




```python
# now B is referring to the same array data as A 
B = A 
```


```python
# changing B affects A
B[0,0] = 10

B
```




    array([[10,  2],
           [ 3,  4]])




```python
A
```




    array([[10,  2],
           [ 3,  4]])



See, changing `B` resulted in a change in `A`. This is because `B` **points** to `A`

### **Deep-copy**

If we want to avoid this behavior, so that when we get a new completely independent object `B` copied from `A`, then we need to do a so-called "deep copy" using the function `copy`:


```python
B = np.copy(A)
```


```python
# now, if we modify B, A is not affected
B[0,0] = -5

B
```




    array([[-5,  2],
           [ 3,  4]])




```python
A
```




    array([[10,  2],
           [ 3,  4]])



See, the `A` did not change, because `B` has it's own **copy** of the data

<a id='iterating'></a>

# Iterating over array elements

Generally, we want to avoid iterating over the elements of arrays whenever we can (at all costs). The reason is that in a interpreted language like Python (or MATLAB), iterations are really slow compared to vectorized operations. 

However, sometimes iterations are unavoidable. For such cases, the Python `for` loop is the most convenient way to iterate over an array:


```python
v = np.array([1,2,3,4])

for element in v:
    print(element)
```

    1
    2
    3
    4



```python
M = np.array([[1,2], [3,4]])

for row in M:
    print("row", row)
    
    for element in row:
        print(element)
```

    row [1 2]
    1
    2
    row [3 4]
    3
    4


When we need to iterate over each element of an array and modify its elements, it is convenient to use the `enumerate` function to obtain both the element and its index in the `for` loop: 


```python
for row_idx, row in enumerate(M):
    print("row_idx", row_idx, "row", row)
    
    for col_idx, element in enumerate(row):
        print("col_idx", col_idx, "element", element)
       
        # update the matrix M: square each element
        M[row_idx, col_idx] = element ** 2
```

    row_idx 0 row [1 2]
    col_idx 0 element 1
    col_idx 1 element 2
    row_idx 1 row [3 4]
    col_idx 0 element 3
    col_idx 1 element 4



```python
# each element in M is now squared
M
```




    array([[ 1,  4],
           [ 9, 16]])



<a id='vectorizing'></a>
# Vectorizing functions

As mentioned several times by now, to get good performance we should try to avoid looping over elements in our vectors and matrices, and instead use vectorized algorithms. The first step in converting a scalar algorithm to a vectorized algorithm is to make sure that the functions we write work with vector inputs.


```python
def Theta(x):
    """
    Scalar implemenation of the Heaviside step function.
    """
    if x >= 0:
        return 1
    else:
        return 0
```


```python
Theta(np.array([-3,-2,-1,0,1,2,3]))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-159-b49266106206> in <module>
    ----> 1 Theta(np.array([-3,-2,-1,0,1,2,3]))
    

    <ipython-input-158-f72d7f42be84> in Theta(x)
          3     Scalar implemenation of the Heaviside step function.
          4     """
    ----> 5     if x >= 0:
          6         return 1
          7     else:


    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()


OK, that didn't work because we didn't write the `Theta` function so that it can handle a vector input... 

To get a vectorized version of Theta we can use the Numpy function `vectorize`. In many cases it can automatically vectorize a function:


```python
Theta_vec = np.vectorize(Theta)
```


```python
Theta_vec(np.array([-3,-2,-1,0,1,2,3]))
```




    array([0, 0, 0, 1, 1, 1, 1])



We can also implement the function to accept a vector input from the beginning (requires more effort but might give better performance):


```python
def Theta(x):
    """
    Vector-aware implemenation of the Heaviside step function.
    """
    return 1 * (x >= 0)
```


```python
Theta(np.array([-3,-2,-1,0,1,2,3]))
```




    array([0, 0, 0, 1, 1, 1, 1])




```python
# still works for scalars as well
Theta(-1.2), Theta(2.6)
```




    (0, 1)



<a id='array_conditions'></a>
# Using arrays in conditions

When using arrays in conditions,for example `if` statements and other boolean expressions, one needs to use `any` or `all`, which requires that any or all elements in the array evalutes to `True`:


```python
M
```




    array([[ 1,  4],
           [ 9, 16]])




```python
if (M > 5).any():
    print("at least one element in M is larger than 5")
else:
    print("no element in M is larger than 5")
```

    at least one element in M is larger than 5



```python
if (M > 5).all():
    print("all elements in M are larger than 5")
else:
    print("all elements in M are not larger than 5")
```

    all elements in M are not larger than 5


<a id='type_casting'></a>
# Type casting

Since Numpy arrays are *statically typed*, the type of an array does not change once created. But we can explicitly cast an array of some type to another using the `astype` functions (see also the similar `asarray` function). This always create a new array of new type:


```python
M.dtype
```




    dtype('int64')




```python
M2 = M.astype(float)

M2
```




    array([[ 1.,  4.],
           [ 9., 16.]])




```python
M2.dtype
```




    dtype('float64')




```python
M3 = M.astype(bool)

M3
```




    array([[ True,  True],
           [ True,  True]])



# Cleanup


```python
!rm random-matrix.csv random-matrix.npz
```

<a id='further_reading'></a>
# Further reading

* http://numpy.scipy.org
* http://scipy.org/Tentative_NumPy_Tutorial
* http://scipy.org/NumPy_for_Matlab_Users - A Numpy guide for MATLAB users.
