PYTHON=python
PFLAGS=build_ext --inplace

CYTHON=cython
CYTHON_FLAGS=-3 -a

SOURCE=fib.pyx

all:
	$(PYTHON) setup.py $(PFLAGS)

profile:
	$(CYTHON) $(CYTHON_FLAGS) $(SOURCE) 
	@firefox $(basename $(SOURCE)).html
	
clean:
	rm -f *.so *.c *.cpp *.html
	rm -rf build
