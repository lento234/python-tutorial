
# Exercise 1 notebook

This has just solution, or if you just want to see the finished homework.

This is a markdown cell with **bold** _italic_ inline $math$, and formulas:

$$f(x) =  ax^2+bx+c$$

$$x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$$

Triple backticks with language

```python
# I can write some code:
from IPython.display import Math
Math('\Delta = b^2-4ac')
```

Indented 4 spaces:

    # I can write some code:
    from IPython.display import Math
    Math('\Delta = b^2-4ac')


Try to get a feel of the editor (select and press TAB...)


```python
from IPython.display import Math
Math('\Delta = b^2-4ac')
```

Let's carry on with the exercise. To be efficient we'll lear to use the keyboard shortcuts. 


```python
print("Hello SciPy")
```

You will notice that `Ctrl-Enter` execute a cell in place (and keep it selected), while `Shift-Enter` execute and select next. Finally, `Alt-Enter` execute and insert below.


```python
def fibgen():
    a,b = 1,1
    for i in range(100):
        yield i, a
        a,b = b, a+b
fib = fibgen()
```


```python
# crl-enter to iter through this.
next(fib)
```

## kernel's related functionalities.


```python
import pandas as pd
```


```python
pd?
```


```python
# tab
pd.D
```


```python
#Shift-tab
pd.DataFrame(
```


```python
!ls ../data/
```


```python
# open inspector
df = pd.read_csv('../data/iris.csv')
```


```python
%matplotlib inline
```


```python
df.head()
```


```python
ax2 = df.plot.scatter(x='sepal_length',
                      y='sepal_width')
```


```python
import numpy as np
```


```python
for i in range(500):
    print(i,":",i**2)
```


```python
from IPython.display import Javascript
Javascript('element.innerHTML =  "Hello SciPy"')
```


```python

```


```python

```
