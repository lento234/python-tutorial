# Mandelbrot - cython version
# cython: infer_types=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np # Numpy api
cimport cython # Cython lib
from cython.parallel import prange # openmp

DTYPE_INT = np.int32 # Numpy datatype int 32

@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
def escape_time(complex[:, ::1] p, int maxtime):
    """Perform the Mantelbrot iteration until it's clear that $p$ diverges 
    or the maximum number of iterations has been reached.
    
    Parameters
    ----------
    p: complex
        point in the complex plane
    maxtime: int
        maximum number of iterations to perform
    """
    
    # Array shape
    cdef Py_ssize_t N = p.shape[0]
    cdef Py_ssize_t M = p.shape[1]
    
    # Export array
    T = np.full((N,M), maxtime, dtype=DTYPE_INT) 
    cdef int[:, ::1] T_view = T
    
    # Py_ssize_t is the proper C type for Python array indices.
    cdef Py_ssize_t i, j, k
    
    cdef complex pij, z
    
    for i in prange(N, nogil=True):
        for j in range(M):
            # Get value
            pij = p[i,j]
            # Initial value
            z = 0j
            # Iterate
            for k in range(maxtime):
                z = z ** 2 + pij
                if abs(z) > 2:
                    T_view[i,j] = k
                    break
    return T




# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[DTYPE_int_t, ndim=2] escape_time(np.ndarray[DTYPE_complex_t, ndim=2] p, int maxtime):
# def escape_time(np.ndarray[DTYPE_complex_t, ndim=2] p, int maxtime):
#     """Perform the Mantelbrot iteration until it's clear that $p$ diverges 
#     or the maximum number of iterations has been reached.
    
#     Parameters
#     ----------
#     p: complex
#         point in the complex plane
#     maxtime: int
#         maximum number of iterations to perform
#     """
#     cdef DTYPE_complex_t z, pij
#     cdef int i, j, k
#     cdef int N = p.shape[0]
#     cdef int M = p.shape[1]
    
#     cdef np.ndarray[DTYPE_int_t, ndim=2] T = np.full((N,M), maxtime, dtype=DTYPE_int) 
    
# #     cdef np.ndarray[DTYPE_int_t, ndim=2] T = np.empty((N,M),dtype=DTYPE_int) 

#     #for i in prange(N, nogil=True, schedule='static'):
#     #    for j in prange(M, nogil=True, schedule='static'):
#     for i in range(N):
#         for j in range(M):
        
#             # Get value
#             pij = p[i,j]
            
#             # Initial value
#             z = 0j
            
#             # Iterate
#             for k in range(maxtime):
#                 z = z ** 2 + pij
#                 if abs(z) > 2:
#                     T[i,j] = i
#                     break
            
#     return T