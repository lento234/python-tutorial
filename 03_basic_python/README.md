
# Introduction to Python programming

J.R. Johansson (jrjohansson at gmail.com)

Modified by Lento Manickathan

The latest version of this [IPython notebook](http://ipython.org/notebook.html) lecture is available at [http://github.com/jrjohansson/scientific-python-lectures](http://github.com/jrjohansson/scientific-python-lectures).

The other notebooks in this lecture series are indexed at [http://jrjohansson.github.io](http://jrjohansson.github.io).

## Python program files

* Python code is usually stored in text files with the file ending "`.py`":

        myprogram.py

* Every line in a Python program file is assumed to be a Python statement, or part thereof. 

    * The only exception is comment lines, which start with the character `#` (optionally preceded by an arbitrary number of white-space characters, i.e., tabs or spaces). Comment lines are usually ignored by the Python interpreter.


* To run our Python program from the command line we use:

        $ python myprogram.py

* On UNIX systems it is common to define the path to the interpreter on the first line of the program (note that this is a comment line as far as the Python interpreter is concerned):

        #!/usr/bin/env python

  If we do, and if we additionally set the file script to be executable, we can run the program like this:

        $ myprogram.py

### Example:


```python
ls .
```

    03_basic_python.ipynb



```python
!head ../README.md
```

    
    [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lento234/python-tutorial/master?urlpath=lab)
    [![Build Status](https://travis-ci.com/lento234/python-tutorial.svg?branch=master)](https://travis-ci.com/lento234/python-tutorial)
    
    # Python Workshop
    
    Sources (derived and modified from):
    - [Scipy 2019 jupyterlab tutorial](https://github.com/jupyterlab/scipy2019-jupyterlab-tutorial)
    - [Lecture series on scientific python by jrjohansson](https://github.com/jrjohansson/scientific-python-lectures)
    - [Scipy Lecture notes](https://scipy-lectures.org/)



```python
!python -c 'print("hello world!")'
```

    hello world!


## IPython notebooks

This file - an IPython notebook -  does not follow the standard pattern with Python code in a text file. Instead, an IPython notebook is stored as a file in the [JSON](http://en.wikipedia.org/wiki/JSON) format. The advantage is that we can mix formatted text, Python code and code output. It requires the IPython notebook server to run it though, and therefore isn't a stand-alone Python program as described above. Other than that, there is no difference between the Python code that goes into a program file or an IPython notebook.

## Modules

Most of the functionality in Python is provided by *modules*. The Python Standard Library is a large collection of modules that provides *cross-platform* implementations of common facilities such as access to the operating system, file I/O, string management, network communication, and much more.


```python
import math
```

This includes the whole module and makes it available for use later in the program. For example, we can do:


```python
import math

x = math.cos(2 * math.pi)

print(x)
```

    1.0


Alternatively, we can chose to import all symbols (functions and variables) in a module to the current namespace (so that we don't need to use the prefix "`math.`" every time we use something from the `math` module:


```python
from math import * # bad

x = cos(2 * pi)

print(x)
```

    1.0


This pattern can be very convenient, but in large programs that include many modules it is often a good idea to keep the symbols from each module in their own namespaces, by using the `import math` pattern. This would elminate potentially confusing problems with name space collisions.

As a third alternative, we can chose to import only a few selected symbols from a module by explicitly listing which ones we want to import instead of using the wildcard character `*`:

**WARNING don't import everything and pollute your namespace, keep it clean**


```python
from math import cos, pi # better

x = cos(2 * pi)

print(x)
```

    1.0


### Looking at what a module contains, and its documentation

Once a module is imported, we can list the symbols it provides using the `dir` function:


```python
import math

print(dir(math))
```

    ['__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'e', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 'gcd', 'hypot', 'inf', 'isclose', 'isfinite', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'modf', 'nan', 'pi', 'pow', 'radians', 'remainder', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'tau', 'trunc']


And using the function `help` we can get a description of each function (almost .. not all functions have docstrings, as they are technically called, but the vast majority of functions are documented this way). 


```python
help(math.log)
```

    Help on built-in function log in module math:
    
    log(...)
        log(x, [base=math.e])
        Return the logarithm of x to the given base.
        
        If the base not specified, returns the natural logarithm (base e) of x.
    



```python
log(10)
```




    2.302585092994046




```python
log(10, 2)
```




    3.3219280948873626



We can also use the `help` function directly on modules: Try

    help(math) 

Some very useful modules form the Python standard library are `os`, `sys`, `math`, `shutil`, `re`, `subprocess`, `multiprocessing`, `threading`. 

A complete lists of standard modules for Python 3 is available at http://docs.python.org/3/library/, .

## Variables and types

### Symbol names 

Variable names in Python can contain alphanumerical characters `a-z`, `A-Z`, `0-9` and some special characters such as `_`. Normal variable names must start with a letter. 

By convention, variable names start with a lower-case letter, and Class names start with a capital letter. 

In addition, there are a number of Python keywords that cannot be used as variable names. These keywords are:

    and, as, assert, break, class, continue, def, del, elif, else, except, 
    exec, finally, for, from, global, if, import, in, is, lambda, not, or,
    pass, print, raise, return, try, while, with, yield

Note: Be aware of the keyword `lambda`, which could easily be a natural variable name in a scientific program. But being a keyword, it cannot be used as a variable name.

### Assignment



The assignment operator in Python is `=`. Python is a dynamically typed language, so we do not need to specify the type of a variable when we create one.

Assigning a value to a new variable creates the variable:


```python
# variable assignments
x = 1.0
my_variable = 12.2
```

Although not explicitly specified, a variable does have a type associated with it. The type is derived from the value that was assigned to it.


```python
type(x)
```




    float



If we assign a new value to a variable, its type can change.


```python
x = 1
```


```python
type(x)
```




    int



If we try to use a variable that has not yet been defined we get an `NameError`:


```python
print(y)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-20-d9183e048de3> in <module>
    ----> 1 print(y)
    

    NameError: name 'y' is not defined


### Fundamental types


```python
# integers
x = 1
type(x)
```




    int




```python
# float
x = 1.0
type(x)
```




    float




```python
# boolean
b1 = True
b2 = False

type(b1)
```




    bool




```python
# complex numbers: note the use of `j` to specify the imaginary part
x = 1.0 - 1.0j
type(x)
```




    complex




```python
print(x)
```

    (1-1j)



```python
print(x.real, x.imag)
```

    1.0 -1.0


### Type utility functions


The module `types` contains a number of type name definitions that can be used to test if variables are of certain types:


```python
import types

# print all types defined in the `types` module
print(dir(types))
```

    ['AsyncGeneratorType', 'BuiltinFunctionType', 'BuiltinMethodType', 'ClassMethodDescriptorType', 'CodeType', 'CoroutineType', 'DynamicClassAttribute', 'FrameType', 'FunctionType', 'GeneratorType', 'GetSetDescriptorType', 'LambdaType', 'MappingProxyType', 'MemberDescriptorType', 'MethodDescriptorType', 'MethodType', 'MethodWrapperType', 'ModuleType', 'SimpleNamespace', 'TracebackType', 'WrapperDescriptorType', '_GeneratorWrapper', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_calculate_meta', 'coroutine', 'new_class', 'prepare_class', 'resolve_bases']



```python
x = 1.0

# check if the variable x is a float
type(x) is float
```




    True




```python
# check if the variable x is an int
type(x) is int
```




    False



We can also use the `isinstance` method for testing types of variables:


```python
isinstance(x, float)
```




    True



### Type casting


```python
x = 1.5

print(x, type(x))
```

    1.5 <class 'float'>



```python
x = int(x)

print(x, type(x))
```

    1 <class 'int'>



```python
z = complex(x)

print(z, type(z))
```

    (1+0j) <class 'complex'>



```python
x = float(z)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-34-19c840f40bd8> in <module>
    ----> 1 x = float(z)
    

    TypeError: can't convert complex to float


Complex variables cannot be cast to floats or integers. We need to use `z.real` or `z.imag` to extract the part of the complex number we want:


```python
y = bool(z.real)

print(z.real, " -> ", y, type(y))

y = bool(z.imag)

print(z.imag, " -> ", y, type(y))
```

    1.0  ->  True <class 'bool'>
    0.0  ->  False <class 'bool'>


## Operators and comparisons

Most operators and comparisons in Python work as one would expect:

* Arithmetic operators `+`, `-`, `*`, `/`, `//` (integer division), '**' power



```python
1 + 2, 1 - 2, 1 * 2, 1 / 2
```




    (3, -1, 2, 0.5)



**We see that python 3 dynamically converts int division to float**


```python
1.0 + 2.0, 1.0 - 2.0, 1.0 * 2.0, 1.0 / 2.0
```




    (3.0, -1.0, 2.0, 0.5)



Force an integer division


```python
# Integer division of float numbers
1 // 2
```




    0




```python
# Note! The power operators in python isn't ^, but **
2 ** 2
```




    4



Note: The `/` operator always performs a floating point division in Python 3.x.
This is not true in Python 2.x, where the result of `/` is always an integer if the operands are integers.
to be more specific, `1/2 = 0.5` (`float`) in Python 3.x, and `1/2 = 0` (`int`) in Python 2.x (but `1.0/2 = 0.5` in Python 2.x).

* The boolean operators are spelled out as the words `and`, `not`, `or`. 


```python
True and False
```




    False




```python
not False
```




    True




```python
True or False
```




    True



* Comparison operators `>`, `<`, `>=` (greater or equal), `<=` (less or equal), `==` equality, `is` identical.


```python
2 > 1, 2 < 1
```




    (True, False)




```python
2 > 2, 2 < 2
```




    (False, False)




```python
2 >= 2, 2 <= 2
```




    (True, True)




```python
# equality
[1,2] == [1,2]
```




    True




```python
# objects identical?
l1 = l2 = [1,2]

l1 is l2
```




    True



## Compound types: Strings, List and dictionaries

### Strings

Strings are the variable type that is used for storing text messages. 


```python
s = "Hello world"
type(s)
```




    str




```python
# length of the string: the number of characters
len(s)
```




    11




```python
# replace a substring in a string with something else
s2 = s.replace("world", "test")
print(s2)
```

    Hello test


We can index a character in a string using `[]`:


```python
s[0]
```




    'H'



**Heads up MATLAB users:** Indexing start at 0!

**I repeated, indexing starts at 0 not 1**

We can extract a part of a string using the syntax `[start:stop]`, which extracts characters between index `start` and `stop` -1 (the character at index `stop` is not included):


```python
s[0:5]
```




    'Hello'




```python
s[4:5]
```




    'o'



If we omit either (or both) of `start` or `stop` from `[start:stop]`, the default is the beginning and the end of the string, respectively:


```python
s[:5]
```




    'Hello'




```python
s[6:]
```




    'world'




```python
s[:]
```




    'Hello world'



We can also define the step size using the syntax `[start:end:step]` (the default value for `step` is 1, as we saw above):


```python
s[::1]
```




    'Hello world'




```python
s[::2]
```




    'Hlowrd'



#### String formatting examples


```python
print("str1", "str2", "str3")  # The print statement concatenates strings with a space
```

    str1 str2 str3



```python
print("str1", 1.0, False, -1j)  # The print statements converts all arguments to strings
```

    str1 1.0 False (-0-1j)



```python
print("str1" + "str2" + "str3") # strings added with + are concatenated without space
```

    str1str2str3



```python
print("value = {:f}".format(1.0))       # we can use C-style string formatting
```

    value = 1.000000



```python
print("value = {:g}".format(1.0))       # we can use C-style string formatting
```

    value = 1



```python
# this formatting creates a string
s2 = "value1 = {0:.2f}, value2 = {1:.0f}".format(3.1415, 1.5)

print(s2)
```

    value1 = 3.14, value2 = 2



```python
# alternative, more intuitive way of formatting a string 
s3 = 'value1 = {0}, value2 = {1}'.format(3.1415, 1.5)

print(s3)
```

    value1 = 3.1415, value2 = 1.5


### List

Lists are very similar to strings, except that each element can be of any type.

The syntax for creating lists in Python is `[...]`:


```python
l = [1,2,3,4]

print(type(l))
print(l)
```

    <type 'list'>
    [1, 2, 3, 4]


We can use the same slicing techniques to manipulate lists as we could use on strings:


```python
print(l)

print(l[1:3])

print(l[::2])
```

    [1, 2, 3, 4]
    [2, 3]
    [1, 3]


**Heads up MATLAB users:** Indexing starts at 0!


```python
l[0]
```




    1



Elements in a list do not all have to be of the same type:


```python
l = [1, 'a', 1.0, 1-1j]

print(l)
```

    [1, 'a', 1.0, (1-1j)]


Python lists can be inhomogeneous and arbitrarily nested:


```python
nested_list = [1, [2, [3, [4, [5]]]]]

nested_list
```




    [1, [2, [3, [4, [5]]]]]



Lists play a very important role in Python. For example they are used in loops and other flow control structures (discussed below). There are a number of convenient functions for generating lists of various types, for example the `range` function:


```python
start = 10
stop = 30
step = 2

range(start, stop, step)
```




    [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]




```python
# in python 3 range generates an iterator, which can be converted to a list using 'list(...)'.
# It has no effect in python 2
list(range(start, stop, step))
```




    [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]




```python
list(range(-10, 10))
```




    [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
s
```




    'Hello world'




```python
# convert a string to a list by type casting:
s2 = list(s)

s2
```




    ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']




```python
# sorting lists
s2.sort()

print(s2)
```

    [' ', 'H', 'd', 'e', 'l', 'l', 'l', 'o', 'o', 'r', 'w']


#### Adding, inserting, modifying, and removing elements from lists


```python
# create a new empty list
l = []

# add an elements using `append`
l.append("A")
l.append("d")
l.append("d")

print(l)
```

    ['A', 'd', 'd']


We can modify lists by assigning new values to elements in the list. In technical jargon, lists are *mutable*.


```python
l[1] = "p"
l[2] = "p"

print(l)
```

    ['A', 'p', 'p']



```python
l[1:3] = ["d", "d"]

print(l)
```

    ['A', 'd', 'd']


Insert an element at an specific index using `insert`


```python
l.insert(0, "i")
l.insert(1, "n")
l.insert(2, "s")
l.insert(3, "e")
l.insert(4, "r")
l.insert(5, "t")

print(l)
```

    ['i', 'n', 's', 'e', 'r', 't', 'A', 'd', 'd']


Remove first element with specific value using 'remove'


```python
l.remove("A")

print(l)
```

    ['i', 'n', 's', 'e', 'r', 't', 'd', 'd']


Remove an element at a specific location using `del`:


```python
del l[7]
del l[6]

print(l)
```

    ['i', 'n', 's', 'e', 'r', 't']


See `help(list)` for more details, or read the online documentation 

### Tuples

Tuples are like lists, except that they cannot be modified once created, that is they are *immutable*. 

In Python, tuples are created using the syntax `(..., ..., ...)`, or even `..., ...`:


```python
point = (10, 20)

print(point, type(point))
```

    ((10, 20), <type 'tuple'>)



```python
point = 10, 20

print(point, type(point))
```

    ((10, 20), <type 'tuple'>)


We can unpack a tuple by assigning it to a comma-separated list of variables:


```python
x, y = point

print("x =", x)
print("y =", y)
```

    ('x =', 10)
    ('y =', 20)


If we try to assign a new value to an element in a tuple we get an error:


```python
point[0] = 20
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-83-ac1c641a5dca> in <module>()
    ----> 1 point[0] = 20
    

    TypeError: 'tuple' object does not support item assignment


### Dictionaries

Dictionaries are also like lists, except that each element is a key-value pair. The syntax for dictionaries is `{key1 : value1, ...}`:


```python
params = {"parameter1" : 1.0,
          "parameter2" : 2.0,
          "parameter3" : 3.0,}

print(type(params))
print(params)
```

    <type 'dict'>
    {'parameter1': 1.0, 'parameter3': 3.0, 'parameter2': 2.0}



```python
print("parameter1 = " + str(params["parameter1"]))
print("parameter2 = " + str(params["parameter2"]))
print("parameter3 = " + str(params["parameter3"]))
```

    parameter1 = 1.0
    parameter2 = 2.0
    parameter3 = 3.0



```python
params["parameter1"] = "A"
params["parameter2"] = "B"

# add a new entry
params["parameter4"] = "D"

print("parameter1 = " + str(params["parameter1"]))
print("parameter2 = " + str(params["parameter2"]))
print("parameter3 = " + str(params["parameter3"]))
print("parameter4 = " + str(params["parameter4"]))
```

    parameter1 = A
    parameter2 = B
    parameter3 = 3.0
    parameter4 = D


## Control Flow

### Conditional statements: if, elif, else

The Python syntax for conditional execution of code uses the keywords `if`, `elif` (else if), `else`:


```python
statement1 = False
statement2 = False

if statement1:
    print("statement1 is True")
    
elif statement2:
    print("statement2 is True")
    
else:
    print("statement1 and statement2 are False")
```

    statement1 and statement2 are False


For the first time, here we encounted a peculiar and unusual aspect of the Python programming language: Program blocks are defined by their indentation level. 

Compare to the equivalent C code:

    if (statement1)
    {
        printf("statement1 is True\n");
    }
    else if (statement2)
    {
        printf("statement2 is True\n");
    }
    else
    {
        printf("statement1 and statement2 are False\n");
    }

In C blocks are defined by the enclosing curly brakets `{` and `}`. And the level of indentation (white space before the code statements) does not matter (completely optional). 

But in Python, the extent of a code block is defined by the indentation level (usually a tab or say four white spaces). This means that we have to be careful to indent our code correctly, or else we will get syntax errors. 

#### Examples:


```python
statement1 = statement2 = True

if statement1:
    if statement2:
        print("both statement1 and statement2 are True")
```

    both statement1 and statement2 are True



```python
# Bad indentation!
if statement1:
    if statement2:
    print("both statement1 and statement2 are True")  # this line is not properly indented
```


      File "<ipython-input-89-78979cdecf37>", line 4
        print("both statement1 and statement2 are True")  # this line is not properly indented
            ^
    IndentationError: expected an indented block




```python
statement1 = False 

if statement1:
    print("printed if statement1 is True")
    
    print("still inside the if block")
```


```python
if statement1:
    print("printed if statement1 is True")
    
print("now outside the if block")
```

    now outside the if block


## Loops

In Python, loops can be programmed in a number of different ways. The most common is the `for` loop, which is used together with iterable objects, such as lists. The basic syntax is:

### **`for` loops**:


```python
for x in [1,2,3]:
    print(x)
```

    1
    2
    3


The `for` loop iterates over the elements of the supplied list, and executes the containing block once for each element. Any kind of list can be used in the `for` loop. For example:


```python
for x in range(4): # by default range start at 0
    print(x)
```

    0
    1
    2
    3


Note: `range(4)` does not include 4 !


```python
for x in range(-3,3):
    print(x)
```

    -3
    -2
    -1
    0
    1
    2



```python
for word in ["scientific", "computing", "with", "python"]:
    print(word)
```

    scientific
    computing
    with
    python


To iterate over key-value pairs of a dictionary:


```python
for key, value in params.items():
    print(key + " = " + str(value))
```

    parameter4 = D
    parameter1 = A
    parameter3 = 3.0
    parameter2 = B


Sometimes it is useful to have access to the indices of the values when iterating over a list. We can use the `enumerate` function for this:


```python
for idx, x in enumerate(range(-3,3)):
    print(idx, x)
```

    (0, -3)
    (1, -2)
    (2, -1)
    (3, 0)
    (4, 1)
    (5, 2)


### List comprehensions: Creating lists using `for` loops:

A convenient and compact way to initialize lists:


```python
l1 = [x**2 for x in range(0,5)]

print(l1)
```

    [0, 1, 4, 9, 16]


### `while` loops:


```python
i = 0

while i < 5:
    print(i)
    
    i = i + 1
    
print("done")
```

    0
    1
    2
    3
    4
    done


Note that the `print("done")` statement is not part of the `while` loop body because of the difference in indentation.

## Functions

A function in Python is defined using the keyword `def`, followed by a function name, a signature within parentheses `()`, and a colon `:`. The following code, with one additional level of indentation, is the function body.


```python
def func0():   
    print("test")
```


```python
func0()
```

    test


Optionally, but highly recommended, we can define a so called "docstring", which is a description of the functions purpose and behaivor. The docstring should follow directly after the function definition, before the code in the function body.


```python
def func1(s):
    """
    Print a string 's' and tell how many characters it has    
    """
    
    print(s + " has " + str(len(s)) + " characters")
```


```python
help(func1)
```

    Help on function func1 in module __main__:
    
    func1(s)
        Print a string 's' and tell how many characters it has
    



```python
func1("test")
```

    test has 4 characters


Functions that returns a value use the `return` keyword:


```python
def square(x):
    """
    Return the square of x.
    """
    return x ** 2
```


```python
square(4)
```




    16



We can return multiple values from a function using tuples (see above):


```python
def powers(x):
    """
    Return a few powers of x.
    """
    return x ** 2, x ** 3, x ** 4
```


```python
powers(3)
```




    (9, 27, 81)




```python
x2, x3, x4 = powers(3)

print(x3)
```

    27


### Default argument and keyword arguments

In a definition of a function, we can give default values to the arguments the function takes:


```python
def myfunc(x, p=2, debug=False):
    if debug:
        print("evaluating myfunc for x = " + str(x) + " using exponent p = " + str(p))
    return x**p
```

If we don't provide a value of the `debug` argument when calling the the function `myfunc` it defaults to the value provided in the function definition:


```python
myfunc(5)
```




    25




```python
myfunc(5, debug=True)
```

    evaluating myfunc for x = 5 using exponent p = 2





    25



If we explicitly list the name of the arguments in the function calls, they do not need to come in the same order as in the function definition. This is called *keyword* arguments, and is often very useful in functions that takes a lot of optional arguments.


```python
myfunc(p=3, debug=True, x=7)
```

    evaluating myfunc for x = 7 using exponent p = 3





    343



### Unnamed functions (lambda function)

In Python we can also create unnamed functions, using the `lambda` keyword:


```python
f1 = lambda x: x**2
    
# is equivalent to 

def f2(x):
    return x**2
```


```python
f1(2), f2(2)
```




    (4, 4)



This technique is useful for example when we want to pass a simple function as an argument to another function, like this:


```python
# map is a built-in python function
map(lambda x: x**2, range(-3,4))
```




    [9, 4, 1, 0, 1, 4, 9]




```python
# in python 3 we can use `list(...)` to convert the iterator to an explicit list
list(map(lambda x: x**2, range(-3,4)))
```




    [9, 4, 1, 0, 1, 4, 9]



## Classes

Classes are the key features of object-oriented programming. A class is a structure for representing an object and the operations that can be performed on the object. 

In Python a class can contain *attributes* (variables) and *methods* (functions).

A class is defined almost like a function, but using the `class` keyword, and the class definition usually contains a number of class method definitions (a function in a class).

* Each class method should have an argument `self` as its first argument. This object is a self-reference.

* Some class method names have special meaning, for example:

    * `__init__`: The name of the method that is invoked when the object is first created.
    * `__str__` : A method that is invoked when a simple string representation of the class is needed, as for example when printed.
    * There are many more, see http://docs.python.org/2/reference/datamodel.html#special-method-names


```python
class Point:
    """
    Simple class for representing a point in a Cartesian coordinate system.
    """
    
    def __init__(self, x, y):
        """
        Create a new Point at x, y.
        """
        self.x = x
        self.y = y
        
    def translate(self, dx, dy):
        """
        Translate the point by dx and dy in the x and y direction.
        """
        self.x += dx
        self.y += dy
        
    def __str__(self):
        return("Point at [%f, %f]" % (self.x, self.y))
```

To create a new instance of a class:


```python
p1 = Point(0, 0) # this will invoke the __init__ method in the Point class

print(p1)         # this will invoke the __str__ method
```

    Point at [0.000000, 0.000000]


To invoke a class method in the class instance `p`:


```python
p2 = Point(1, 1)

p1.translate(0.25, 1.5)

print(p1)
print(p2)
```

    Point at [0.250000, 1.500000]
    Point at [1.000000, 1.000000]


Note that calling class methods can modifiy the state of that particular class instance, but does not effect other class instances or any global variables.

That is one of the nice things about object-oriented design: code such as functions and related variables are grouped in separate and independent entities. 

## Modules

One of the most important concepts in good programming is to reuse code and avoid repetitions.

The idea is to write functions and classes with a well-defined purpose and scope, and reuse these instead of repeating similar code in different part of a program (modular programming). The result is usually that readability and maintainability of a program is greatly improved. What this means in practice is that our programs have fewer bugs, are easier to extend and debug/troubleshoot. 

Python supports modular programming at different levels. Functions and classes are examples of tools for low-level modular programming. Python modules are a higher-level modular programming construct, where we can collect related variables, functions and classes in a module. A python module is defined in a python file (with file-ending `.py`), and it can be made accessible to other Python modules and programs using the `import` statement. 

Consider the following example: the file `mymodule.py` contains simple example implementations of a variable, function and a class:


```python
%%file mymodule.py
"""
Example of a python module. Contains a variable called my_variable,
a function called my_function, and a class called MyClass.
"""

my_variable = 0

def my_function():
    """
    Example function
    """
    return my_variable
    
class MyClass:
    """
    Example class.
    """

    def __init__(self):
        self.variable = my_variable
        
    def set_variable(self, new_value):
        """
        Set self.variable to a new value
        """
        self.variable = new_value
        
    def get_variable(self):
        return self.variable
```

    Writing mymodule.py


We can import the module `mymodule` into our Python program using `import`:


```python
import mymodule
```

Use `help(module)` to get a summary of what the module provides:


```python
help(mymodule)
```

    Help on module mymodule:
    
    NAME
        mymodule
    
    FILE
        /Users/rob/Desktop/scientific-python-lectures/mymodule.py
    
    DESCRIPTION
        Example of a python module. Contains a variable called my_variable,
        a function called my_function, and a class called MyClass.
    
    CLASSES
        MyClass
        
        class MyClass
         |  Example class.
         |  
         |  Methods defined here:
         |  
         |  __init__(self)
         |  
         |  get_variable(self)
         |  
         |  set_variable(self, new_value)
         |      Set self.variable to a new value
    
    FUNCTIONS
        my_function()
            Example function
    
    DATA
        my_variable = 0
    
    



```python
mymodule.my_variable
```




    0




```python
mymodule.my_function() 
```




    0




```python
my_class = mymodule.MyClass() 
my_class.set_variable(10)
my_class.get_variable()
```




    10



If we make changes to the code in `mymodule.py`, we need to reload it using `reload`:


```python
reload(mymodule)  # works only in python 2
```




    <module 'mymodule' from 'mymodule.pyc'>



## Exceptions

In Python errors are managed with a special language construct called "Exceptions". When errors occur exceptions can be raised, which interrupts the normal program flow and fallback to somewhere else in the code where the closest try-except statement is defined.

To generate an exception we can use the `raise` statement, which takes an argument that must be an instance of the class `BaseException` or a class derived from it. 


```python
raise Exception("description of the error")
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-128-8f47ba831d5a> in <module>()
    ----> 1 raise Exception("description of the error")
    

    Exception: description of the error


A typical use of exceptions is to abort functions when some error condition occurs, for example:

    def my_function(arguments):
    
        if not verify(arguments):
            raise Exception("Invalid arguments")
        
        # rest of the code goes here

To gracefully catch errors that are generated by functions and class methods, or by the Python interpreter itself, use the `try` and  `except` statements:

    try:
        # normal code goes here
    except:
        # code for error handling goes here
        # this code is not executed unless the code
        # above generated an error

For example:


```python
try:
    print("test")
    # generate an error: the variable test is not defined
    print(test)
except:
    print("Caught an exception")
```

    test
    Caught an exception


To get information about the error, we can access the `Exception` class instance that describes the exception by using for example:

    except Exception as e:


```python
try:
    print("test")
    # generate an error: the variable test is not defined
    print(test)
except Exception as e:
    print("Caught an exception:" + str(e))
```

    test
    Caught an exception:name 'test' is not defined


## Further reading

* http://www.python.org - The official web page of the Python programming language.
* http://www.python.org/dev/peps/pep-0008 - Style guide for Python programming. Highly recommended. 
* http://www.greenteapress.com/thinkpython/ - A free book on Python programming.
* [Python Essential Reference](http://www.amazon.com/Python-Essential-Reference-4th-Edition/dp/0672329786) - A good reference book on Python programming.

## Versions


```python
%load_ext version_information

%version_information
```




<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.10 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>3.2.1</td></tr><tr><td>OS</td><td>Darwin 14.1.0 x86_64 i386 64bit</td></tr><tr><td colspan='2'>Sat Aug 15 10:51:55 2015 JST</td></tr></table>


