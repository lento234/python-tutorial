# Load the c code

from fib import fib

def fib_python(n):
    a, b = 1, 1
    for i in range(n):
        a, b = a + b, a
    return a

# Test output
print("fib(1) =", fib(1))
print("fib(10) =", fib(10))
print("fib(0 to 15)=",[fib(i) for i in range(15)])

# Profile
%timeit 
