from math import sin, exp
def f(double x):
    return sin(x) * exp(-x)
def integrate(double a, double b, int N):
    """Integrate function sin(x) * exp(-x)
    """
    cdef double dx, s
    cdef int i
    
    dx = (b - a) / N
    s = 0.0
    for i in range(N):
        s += f(a + (i + 0.5) * dx)
    return s * dx