
# Dask Arrays

source: [dask examples](https://github.com/dask/dask-examples)


Dask arrays coordinate many Numpy arrays, arranged into chunks within a grid.  They support a large subset of the Numpy API.

## Start Dask Client for Dashboard

Starting the Dask Client is optional.  It will provide a dashboard which 
is useful to gain insight on the computation.  

The link to the dashboard will become visible when you create the client below.  We recommend having it open on one side of your screen while using your notebook on the other side.  This can take some effort to arrange your windows, but seeing them both at the same is very useful when learning.


```python
from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='2GB')
client
```

    /home/lento/apps/anaconda3/envs/test/lib/python3.7/site-packages/distributed/dashboard/core.py:74: UserWarning: 
    Port 8787 is already in use. 
    Perhaps you already have a cluster running?
    Hosting the diagnostics dashboard on a random port instead.
      warnings.warn("\n" + msg)





<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3>Client</h3>
<ul>
  <li><b>Scheduler: </b>inproc://152.88.86.183/22889/1
  <li><b>Dashboard: </b><a href='http://localhost:36043/status' target='_blank'>http://localhost:36043/status</a>
</ul>
</td>
<td style="vertical-align: top; border: 0px solid white">
<h3>Cluster</h3>
<ul>
  <li><b>Workers: </b>1</li>
  <li><b>Cores: </b>4</li>
  <li><b>Memory: </b>2.00 GB</li>
</ul>
</td>
</tr>
</table>



## Create Random array

This creates a 10000x10000 array of random numbers, represented as many numpy arrays of size 1000x1000 (or smaller if the array cannot be divided evenly). In this case there are 100 (10x10) numpy arrays of size 1000x1000.


```python
import dask.array as da
x = da.random.random((10000, 10000), chunks=(1000, 1000))
x
```




<table>
<tr>
<td>
<table>  <thead>    <tr><td> </td><th> Array </th><th> Chunk </th></tr>
  </thead>
  <tbody>
    <tr><th> Bytes </th><td> 800.00 MB </td> <td> 8.00 MB </td></tr>
    <tr><th> Shape </th><td> (10000, 10000) </td> <td> (1000, 1000) </td></tr>
    <tr><th> Count </th><td> 100 Tasks </td><td> 100 Chunks </td></tr>
    <tr><th> Type </th><td> float64 </td><td> numpy.ndarray </td></tr>
  </tbody></table>
</td>
<td>
<svg width="170" height="170" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="12" x2="120" y2="12" />
  <line x1="0" y1="24" x2="120" y2="24" />
  <line x1="0" y1="36" x2="120" y2="36" />
  <line x1="0" y1="48" x2="120" y2="48" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="72" x2="120" y2="72" />
  <line x1="0" y1="84" x2="120" y2="84" />
  <line x1="0" y1="96" x2="120" y2="96" />
  <line x1="0" y1="108" x2="120" y2="108" />
  <line x1="0" y1="120" x2="120" y2="120" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="120" style="stroke-width:2" />
  <line x1="12" y1="0" x2="12" y2="120" />
  <line x1="24" y1="0" x2="24" y2="120" />
  <line x1="36" y1="0" x2="36" y2="120" />
  <line x1="48" y1="0" x2="48" y2="120" />
  <line x1="60" y1="0" x2="60" y2="120" />
  <line x1="72" y1="0" x2="72" y2="120" />
  <line x1="84" y1="0" x2="84" y2="120" />
  <line x1="96" y1="0" x2="96" y2="120" />
  <line x1="108" y1="0" x2="108" y2="120" />
  <line x1="120" y1="0" x2="120" y2="120" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.000000,0.000000 120.000000,0.000000 120.000000,120.000000 0.000000,120.000000" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="140.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >10000</text>
  <text x="140.000000" y="60.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,140.000000,60.000000)">10000</text>
</svg>
</td>
</tr>
</table>



Use NumPy syntax as usual


```python
y = x + x.T
z = y[::2, 5000:].mean(axis=1)
z
```




<table>
<tr>
<td>
<table>  <thead>    <tr><td> </td><th> Array </th><th> Chunk </th></tr>
  </thead>
  <tbody>
    <tr><th> Bytes </th><td> 40.00 kB </td> <td> 4.00 kB </td></tr>
    <tr><th> Shape </th><td> (5000,) </td> <td> (500,) </td></tr>
    <tr><th> Count </th><td> 430 Tasks </td><td> 10 Chunks </td></tr>
    <tr><th> Type </th><td> float64 </td><td> numpy.ndarray </td></tr>
  </tbody></table>
</td>
<td>
<svg width="170" height="75" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="25" x2="120" y2="25" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
  <line x1="12" y1="0" x2="12" y2="25" />
  <line x1="24" y1="0" x2="24" y2="25" />
  <line x1="36" y1="0" x2="36" y2="25" />
  <line x1="48" y1="0" x2="48" y2="25" />
  <line x1="60" y1="0" x2="60" y2="25" />
  <line x1="72" y1="0" x2="72" y2="25" />
  <line x1="84" y1="0" x2="84" y2="25" />
  <line x1="96" y1="0" x2="96" y2="25" />
  <line x1="108" y1="0" x2="108" y2="25" />
  <line x1="120" y1="0" x2="120" y2="25" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.000000,0.000000 120.000000,0.000000 120.000000,25.412617 0.000000,25.412617" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >5000</text>
  <text x="140.000000" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,12.706308)">1</text>
</svg>
</td>
</tr>
</table>



Call `.compute()` when you want your result as a NumPy array.

If you started `Client()` above then you may want to watch the status page during computation.


```python
z.compute()
```




    array([0.99189748, 1.00585354, 1.01467388, ..., 0.99612958, 0.99998405,
           1.00643283])



## Persist data in memory

If you have the available RAM for your dataset then you can persist data in memory.  

This allows future computations to be much faster.


```python
y = y.persist()
```


```python
%time y[0, 0].compute()
```

    CPU times: user 1.9 s, sys: 201 ms, total: 2.1 s
    Wall time: 502 ms





    0.5761779735705479




```python
%time y.sum().compute()
```

    CPU times: user 330 ms, sys: 9.02 ms, total: 339 ms
    Wall time: 184 ms





    99999241.24948291



## Further Reading 

A more in-depth guide to working with Dask arrays can be found in the [dask tutorial](https://github.com/dask/dask-tutorial), notebook 03.
