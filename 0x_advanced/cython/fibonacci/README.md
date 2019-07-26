# Fibonacci algorithm in cython

## Steps
0. (Optional) Profile the cython code:

	$ cython -3 -a fib.pyx

1. Compile cython code to c/c++ code:

	$ python setup.py build_ext --inplace

2. Import c/c++ module

	```python
	from fib import fib

	fib(100) # Call fibonacci function
	```

