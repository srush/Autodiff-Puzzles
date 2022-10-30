# Autodiff Puzzles
- by [Sasha Rush](http://rush-nlp.com) - [srush_nlp](https://twitter.com/srush_nlp)


This notebook contains a series of self-contained puzzles for learning about derivatives in tensor libraries. It is the 3rd puzzle set in a series of puzzles about deep learning programming ([Tensor Puzzles](https://github.com/srush/Tensor-Puzzles), [GPU Puzzles](https://github.com/srush/GPU-Puzzles) ) . While related in spirit, the puzzles are all pretty seperate and can be done on their own. 


![](https://github.com/srush/autodiff-puzzles/raw/main/image.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/autodiff-puzzles/blob/main/autodiff_puzzlers.ipynb)



## Introduction

Deep learning libraries like PyTorch utilize autodifferentiation of tensors to compute the parameter updates necessary to learn complex models from data. This technique is central to understanding why deep learning has become so widely used and effective. The autodifferentiation process is a neat trick that builds up a computational graph and then uses that graph to provide derivatives for user-constructed mathematical functions. At heart, this process is just an instantiation of the chain-rule based on implementations of every function and its derivative. 

However, a library *needs* to have efficient implementations of derivatives for its key building blocks. This sounds trivial -> just implement high school calculus. However, this is a bit more tricky than it sounds. Tensor-to-tensor functions are pretty complex and require keeping careful track of indexing on both the input and the output side. 

Your goal in these puzzles is to implement the derivatives for each function below. In each case the function takes in a tensor x and returns a tensor f(x), so your job is to compute $\frac{d f(x)_o}{dx_i}$ for all indices $o$ and $i$. If you get discouraged, just remember, you did this in high school (it just had way less indexing).

## Rules and Tips

* Every answer is 1 line of 80-column code. 
* Everything in these puzzles should be done with standard Python numbers. (No need for torch or anything else.)
* Recall the basic multivariate calculus identities, most importantly: 
$$f(x_1, x_2, x_3) = x_1 + x_2 + x_3 \Rightarrow \frac{d f(x)_1}{dx_1} = 1, \frac{d f(x)_1}{dx_2} = 0$$

* Python booleans auto-cast with python numbers. So you can use them as indicator functions, i.e. $$\mathbf{1}(3=3) \cdot (25-3)$$



```python
(3==3) * (25- 3)
```




    22




```python

```


```python
# Library code - SKIP ME
!pip install -qqq git+https://github.com/chalk-diagrams/chalk

import chalk
from chalk import *
import torch
import random
from colour import Color
import sys
import pandas as pd
sys.setrecursionlimit(10000)
set_svg_height(400)
set_svg_draw_height(400)
from IPython.display import display, HTML

def line(y, rev=False):
    return [(i + j, -2 * float(y[i])) 
            for i in range(y.shape[0])
            for j in range(2)]

def graph(y, name, splits=[]):
    top = make_path([(0, -2), (0, 2)])
    top += make_path([(0, 0), (y.shape[0],0)]).line_width(0.05).line_color(Color("grey"))    
    top += make_path(line(y)).line_width(0.2)
    for s in splits:
        top += make_path([(s, -2), (s, 2)])
    top = top.named(Name(name))
    top = frame(name, top, Color("lightgrey"), name, h=1.5) + top
    top = top.scale_uniform_to_x((y.shape[0] + 15)/ 100).align_tl()
    return top

def overgraph(d, name, y, y2, color):
    one = torch.tensor(1)
    y = torch.maximum(torch.minimum(one, y), -one)
    y2 = torch.maximum(torch.minimum(one, y2), -one)
    
    bot2 =  make_path([(0, -2), (0, 2)])

    pts = line(y2) + list(reversed(line(y)))
    bot2 += make_path(pts, True)
    bot2 = bot2.line_color(color).fill_opacity(0.5).fill_color(color).line_width(0.1)
    scale, trans = get_transform(name, d, y)
    old = make_path(line(y))
    return (bot2 + old).scale_x(scale[0]).scale_y(scale[1]).translate_by(trans)

def frame(name, d, c, l = "", w=15, h=1.5):
    s = d.get_subdiagram(Name(name))
    env = s.get_envelope()
    r = rectangle(env.width + w, env.height * h).fill_color(c)
    if l:
        r = r.align_l() + (hstrut(w/4) | text(l, 3).fill_color(ORANGE)).align_l()
    return r.center_xy().translate_by(env.center) + d

def get_transform(name, d, v):
    s = d.get_subdiagram(Name(name))
    e = s.get_envelope()
    x = e.width / v.shape[0]
    y = e.height / (2 * 2)
    l = s.get_location()
    return (x, y), l  

def get_locations(name, d, x, y, v):
    (x_step, y_step), l = get_transform(name, d, v) 
    return (l[0] + x_step * x, 
            l[1] - y_step * 2 * y )
    
ORANGE = Color("orange")

def connect(d, x, x_name, f_x, f_x_name, q,
            i_s=None, 
            color=ORANGE, to_line=False, bad={}):
    offset = 0.1
    x1, y1 = get_locations(x_name, d, 
                           torch.arange(x.shape[0]+1), 
                           torch.zeros(x.shape[0]+1) - offset, x)
    
    if to_line:
        land = f_x
    else:
        land = f_x / f_x
    land = torch.cat([land, torch.tensor([0, 0])], 0)
    x2, y2 = get_locations(f_x_name, d, 
                           torch.arange(f_x.shape[0]+2), 
                           land + offset, f_x)
    c = color
    if i_s is None:
        i_s = range(x.shape[0])
    m = q.abs().max()
    #q = torch.where(q.abs() / m > 0.1, q, 0)
    for i in i_s: 
        for j in q[:, i].nonzero()[:, 0]:
            if q[j, i] != 0 and i < x1.shape[0] -1  and j < y2.shape[0]-2:
                p = make_path([((x1[i] + x1[i+1])/2, y1[i]),
                                (x2[j], (y2[j] + y2[j-1]) / 2),
                                (x2[j+1], (y2[j] + y2[j+1]) / 2)], True)
                p = p.line_width(0).fill_color(c).fill_color(c if j not in bad.get(i, []) else Color("red"))\
                     .fill_opacity(0.3 * abs(q[j, i] / m))

                d += p
    return d

def diff_graph(d, f_x, f_x_name, d_f_x, c, amount=5):
    updated = f_x + 5* d_f_x
    overgraph(d, f_x_name, "", f_x, updated, c)

def double(y, f_y):
    d = vcat([graph(y, "x").center_xy(),
              graph(f_y, "f(x)").center_xy()], 0.1)
    return d

def two_arg(x, y, f_xy, gaps=None):
    return vcat([hcat([graph(x, "x", gaps), graph(y, "y")], 0.03).center_xy(),
                    graph(f_xy, "f(x)").center_xy()], 0.1)
def outer_frame(d):
    return frame("full", d.named(Name("full")), Color("white"), w=0.1, h=1.2) + d 
    

def m(inp, out, f):
    ret = torch.zeros(inp.shape[0], out.shape[0])
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
          ret[i, j] = f(i, j)
    return ret.T

def m2(inp, out, f):
    ret = torch.zeros(inp.shape[0], out.shape[0], out.shape[1])
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            for k in range(ret.shape[2]):
                ret[i, j, k] = f(i, j, k)
    return ret.permute(1, 2, 0)

def m3(inp, out, f):
    ret = torch.zeros(inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            for k in range(ret.shape[2]):
                for l in range(ret.shape[3]):
                    ret[i, j, k, l] = f(i, j, k, l)
    return ret.permute(2, 3, 0, 1)

def v(s, f):
    ret = torch.zeros(s)
    for i in range(s):
        ret[i] = f(i)
    return ret

def v2(s, f):
    ret = torch.zeros(*s)
    for i in range(s[0]):
        for j in range(s[1]):
           ret[i, j] = f(i, j)
    return ret

def numerical_deriv(fb, x, out):
    s = None
    if s is None:
        s = x.shape[0]
    dx2 = torch.zeros(out.shape[0], s)
    for i in range(s):
        up = x + 1e-5 * torch.eye(s)[i]            
        f = fb(up)
        for j in range(out.shape[0]):
            dx2[j, i] = (f(j) - out[j]) / 1e-5
    return dx2

def two_argf(fb, x, y, out_shape):
    f, dx, dy = fb(x, y)
    out = v(out_shape, f)
    dx = m(x, out, dx)
    dy = m(y, out, dy)
    dx2 = numerical_deriv(lambda a: fb(a, y)[0], x, out)
    dy2 = numerical_deriv(lambda b: fb(x, b)[0], y, out)
    return out, dx, dy, dx2, dy2

def one_argf(fb, x, out_shape):
    f, dx = fb(x)
    out = v(out_shape, f)
    dx = m(x, out, dx)
    dx2 = numerical_deriv(lambda a: fb(a)[0], x, out)
    return out, dx, dx2

# def two_mat_argf(fb, x, y, out_shape, in_shape):
#     f, dx, dy = fb(x, y2)
#     out = v2(out_shape, f)
#     dx = m3(x, out, dx)
#     dy = m2(y2, out, dy)
#     dx2 = []
#     dx2 = numerical_deriv(lambda a: lambda v: fb(a, y)[0](v // out_shape[0], v % out_shape[0]), x.view(-1), out.view(-1), in_shape=in_shape)
#     dy2 = numerical_deriv(lambda b: fb(x, b)[0], y.view(-1), out.view(-1))
#     return out, dx, dy, dx2, dy2

def check(dx, dx2):
    bad = {}
    df = []
    for j, i in (~torch.isclose(dx, dx2, atol=1e-2)).nonzero():
        #print(i.item(), j.item(), dx[i,j].item(), dx2[i,j].item())
        bad.setdefault(i.item(), [])
        bad[i.item()].append(j.item())
        df.append({"In Index": i.item(), "Out Index": j.item()})
    return bad, pd.DataFrame(df)

gy = torch.tensor([math.sin(x/20) * 0.5 + (random.random() - 0.5)
                  for x in range(50)])

def in_out(fb, pos=None, overlap=False, diff=1, out_shape=50, y=gy):
    "For functions with point samples"
    out, dx, dx2 = one_argf(fb, y, out_shape)
    bad, df = check(dx, dx2)

    if pos is None:
        pos = range(y.shape[0])
    set_svg_height(300)
    d = double(y, out)
    if overlap:
        for p in pos:
            d += overgraph(d, "f(x)", out, out + diff * dx[:, p], Color("red"))
            d += overgraph(d, "f(x)", out, out + diff * dx2[:, p], Color("green"))

    else:
        d += overgraph(d, "f(x)", out, out + diff * dx[:, pos].sum(-1), Color("red"))
        d += overgraph(d, "f(x)", out, out + diff * dx2[:, pos].sum(-1), Color("lightyellow"))

    d = connect(d, y, "x", out, "f(x)", dx, 
                pos, to_line=True, bad=bad)
    set_svg_height(300)
    if bad:
        print("Errors")
        display(df[:10])
    return outer_frame(d)

def zip(fb, split=25, pos1 = None, pos2=None, out_shape=25, diff=1, overlap=False, gaps=[0], y=gy):
    x, y2 = y[:split], y[split:]
    out, dx, dy, dx2, dy2= two_argf(fb, x, y2, out_shape)
    bad_x, df_x = check(dx, dx2)
    bad_y, df_y = check(dy, dy2)
    if pos1 is None:
        pos1 = range(x.shape[0])
    if pos2 is None:
        pos2 = range(y2.shape[0])    
    d = two_arg(x, y2, out, gaps)
    gaps = gaps + [x.shape[0]]
    print(len(gaps))
    if len(gaps) == 2:
        colors = [ORANGE]
    else:         
        colors = list(Color("yellow").range_to("darkorange", len(gaps)-1))
    for k, c in enumerate(colors):
        d = connect(d, x, "x", out, "f(x)", dx, [p for p in  pos1 if gaps[k] < p < gaps[k+1]], 
                    bad=bad_x, color=c)

    d = connect(d, y2, "y", out, "f(x)", dy, color=Color("lightblue"), i_s=pos2, bad =bad_y)
   
    if overlap:
        for p in pos1:
            d += overgraph(d, "f(x)", out, out + diff * dx2[:, p], Color("lightyellow"))
            d += overgraph(d, "f(x)", out, out + diff * dx[:, p], Color("darkorange"))
        for p in pos2:
            d += overgraph(d, "f(x)", out, out + diff * dy2[:, p], Color("lightblue"))
            d += overgraph(d, "f(x)", out, out + diff * dy[:, p], Color("blue"))

    else:
        d += overgraph(d, "f(x)", out, out + diff * dx2.sum(-1), Color("lightyellow"))
        d += overgraph(d, "f(x)", out, out + diff * dy2.sum(-1), Color("lightblue"))
        d += overgraph(d, "f(x)", out, out + diff * dx.sum(-1), Color("darkorange"))
        d += overgraph(d, "f(x)", out, out + diff * dy.sum(-1), Color("blue"))
    set_svg_height(300)
    if bad_x:
        print("x Errors")
        display(df_x[:10])
    if bad_y:
        print("y Errors")
        display(df_y[:10]) 
    return outer_frame(d)
# def fb_index(x):
#     f = lambda o: x[o+5]
#     dx = lambda i, o: (o + 25) == i
#     return f, dx
# in_out(fb_index, overlap=False, out_shape=25)
# def mat(fb, split, in_shape, out_shape):
#     x, y2 = y[:split], y[split:]
#     x = x.view(*in_shape)
#     f, dx, dy, dx2, dy2 = two_mat_argf(fb, x, y2, out_shape)
#     bad_y, df_y = check(dy, dy2)

#     d = vcat([graph(x[i], f"x{i}") for i in range(x.shape[0])], 0.0)
#     d = hcat([d.center_xy(), graph(y2, "y")], 0.2)
#     d = vcat([d.center_xy(), 
#              vcat([graph(out[i], f"f(x){i}").center_xy() for i in range(out.shape[0])])], 0.15)
#     s = d
#     for j in range(out.shape[0]):
#         for i in range(x.shape[0]):
#             d = connect(d, x[i], f"x{i}", out[j], f"f(x){j}", dx[j, :, i], 
#                         range(x.shape[1]), 
#                         list(Color("red").range_to("orange", x.shape[0]))[i])
#         d = connect(d, y2, "y", out[j], f"f(x){j}", dy[j], 
#                     range(y2.shape[0]), 
#                      Color("lightblue"), bad=bad_y)
#     if bad_y:
#         print("y Errors")
#         display(df_y[:10])
#     set_svg_height(800)
#     return outer_frame(d)
def make_mat(fb, in_shape, out_shape):
    def nf(x, y):
        f, d_x, d_y = fb(x.view(in_shape), y)
        def f2(o):
            return f(o // out_shape[1], o % out_shape[1])
        def d_x2(i, o):
            return d_x(i // in_shape[1], i % in_shape[1], o // out_shape[1], o % out_shape[1])
        def d_y2(j, o):
            return d_y(j, o // out_shape[1], o % out_shape[1])
        return f2, d_x2, d_y2
    return nf

def make_mat2(fb, in_shape, in_shape2, out_shape):
    def nf(x, y):
        f, d_x, d_y = fb(x.view(in_shape), y.view(in_shape2))
        def f2(o):
            return f(o // out_shape[1], o % out_shape[1])
        def d_x2(i, o):
            return d_x(i // in_shape[1], i % in_shape[1], o // out_shape[1], o % out_shape[1])
        def d_y2(j, o):
            return d_y(j // in_shape2[1], j % in_shape2[1], o // out_shape[1], o % out_shape[1])
        return f2, d_x2, d_y2
    return nf
```


For each of the problems, a function $f$ is provided. Your job is to compute the derivative $\frac{df(x)_o} {dx_i}$. This is done by filling in the function for the `dx` variable. If you get any of the derivatives wrong, it will print out the values of $o$ and $i$ that you need to fix. Correct derivatives will be shown in orange, whereas incorrect (non-zero) derivatives will show up as red. The target differential is shown as a light grey on the graphic.

### Problem 1: Id

Warmup: $f(x_1) = x_1$


```python
def fb_id(x):
    f = lambda o: x
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_id, overlap=False, y=gy[1:2], out_shape=1)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_6_2.svg)
    



### Problem 2: Cosine

Warmup: $f(x_1) = \cos(x_1)$


```python
def fb_cos(x):
    f = lambda o: math.cos(x)
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_cos, overlap=False, y=gy[1:2], out_shape=1)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_8_2.svg)
    



### Problem 3: Mean

$f(x_1, x_2, \ldots, x_N) = (x_1 + x_2 + \ldots + x_N) / N$


```python
def fb_mean(x):
    N = x.shape[0]
    f = lambda o: sum(x[i] for i in range(N)) / N
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_mean, overlap=False,  out_shape=1)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_10_2.svg)
    



### Problem 4: Product

$f(x_1, x_2, \ldots, x_N) = x_1  x_2  \ldots  x_N$


```python
def fb_prod(x):
    pr = torch.prod(x)
    f = lambda o: pr
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_prod, overlap=False,  out_shape=1)
```




    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_12_0.svg)
    



### Problem 5: Repeat

$f(x_1) = [x_1, x_1,  x_1  \ldots  x_1]$


```python
def fb_repeat(x):
    f = lambda o: x
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_repeat, overlap=False, y=gy[1:2], out_shape=50)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_14_2.svg)
    



### Problem 6: Repeat and Scale

$$f(x_1) = [x_1, x_1 * 2/N,  x_1 * 3/N,  \ldots  x_N * N/N]$$


```python
def fb_repeat_scale(x):
    N = 50
    f = lambda o: x * (o / N)
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_repeat_scale, overlap=False, y=gy[1:2], out_shape=50)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_16_2.svg)
    



## Problem 7: Negation

$$f(x_1, x_2, \ldots) = [-x_1, -x_2, \ldots]$$

(Hint: remember the indicator trick, i.e. 

```python 
(a == b) * 27 # 27 if a == b else 0
```


```python
def fb_neg(x):
    f = lambda o: -x[o]
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_neg)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_18_2.svg)
    



### Problem 8: ReLU

$$f(x_1, x_2, \ldots) = [\text{relu}(x_1), \text{relu}(x_2), \ldots]$$

Recall 

$$
\text{relu}(x) = \begin{cases}
0 & x < 0 \\
x & x >= 0
\end{cases}
$$

(Note: you can ignore the not of non-differentiability at 0.)


```python
def fb_relu(x):
    f = lambda o: (x[o] > 0) * x[o] 
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_relu)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_20_2.svg)
    



### Problem 9: Index

$$f(x_1, x_2, \ldots, x_{25}) = [x_{10}, x_{11}, \ldots, x_{25}]$$




```python
def fb_index(x):
    f = lambda o: x[o+10]
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_index, overlap=False, out_shape=25)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>18</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>19</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_22_2.svg)
    



Note: for the next several problems the visualization changes and only shows the derivatives for some points $i$ for graphical simplicity.

### Problem 9: Cumsum

$$f(x_1, x_2, \ldots) = [\sum^1_{i=1} x_{i}, \sum^2_{i=1} x_{i}, \sum^3_{i=1} x_{i}, \ldots, ] / 20$$




```python
def fb_cumsum(x):
    out = torch.cumsum(x, 0)
    f = lambda o: out[o] / 20
    dx = lambda o, i: 0 # Fill in this line
    return f, dx
in_out(fb_cumsum, [20, 35, 40], overlap=True, diff=20)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_24_2.svg)
    



### Problem 10: Running Mean


$$f(x_1, x_2, \ldots)_o = \frac{\displaystyle \sum^o_{i=\max(o-W, 1)} x_i}{ W}$$


```python
def fb_running(x):
    W = 10
    f = lambda o: sum([x[o-do] for do in range(W) if o-do >= 0]) / W
    dx = lambda i, o: 0 # Fill in this line
    return f, dx

in_out(fb_running, [0, 20, 35], diff=4)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_26_2.svg)
    



### Problem 11: Sort


$$f(x_1, x_2, \ldots) = \text{x's in sorted order}$$

(This one is a bit counterintuitive! Note that we are not asking you to differentiate the sorting function it self.)


```python
def fb_sort(x):
    sort, argsort = torch.sort(x, 0)
    f = lambda o: sort[o]
    dx = lambda i, o: 0 # Fill in this line
    return f, dx
in_out(fb_sort, overlap=False)
```

    Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>33</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>25</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_28_2.svg)
    



Next we move on to functions of two arguments. For these you will produce two derivatives: $$\frac{df(x, y)_o}{x_i}, \frac{df(x, y)_o}{y_j}$$. Everything else is the same.

### Problem 12: Elementwise mean

$$f(x, y)_o = (x_o + y_o) /2 $$


```python
def fb_emean(x, y):
    f = lambda o: (x[o] + y[o]) / 2
    dx = lambda i, o: 0 # Fill in this line
    dy = lambda j, o: 0 # Fill in this line
    return f, dx, dy
zip(fb_emean)
```

    2
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>


    y Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_30_4.svg)
    



### Problem 13: Elementwise mul

$$f(x, y)_o = x_o * y_o $$


```python
def fb_mul(x, y):
    f = lambda o: x[o] * y[o]
    dx = lambda i, o: 0 # Fill in this line
    dy = lambda j, o: 0 # Fill in this line
    return f, dx, dy

zip(fb_mul)
```

    2
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>


    y Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_32_4.svg)
    



### Problem 14: 1D Convolution

This is the standard 1D convolution used in deep learning. There is no wrap-around.

$$f(x, y)_o = \sum_{j=1}^K x_{o+j} * y_{j} / K $$

Note: This is probably the hardest one. The answer is short but tricky.


```python
def fb_conv(x, y):
    W = 5
    f = lambda o: sum((x[o + j] * y[j]) / W for j in range(W))
    dx = lambda i, o: 0 # Fill in this line
    dy = lambda j, o: 0 # Fill in this line
    return f, dx, dy

SHOW_KERNEL = False
zip(fb_conv, split=45, out_shape=39, pos1=[10, 20, 30], pos2=[5] if SHOW_KERNEL else [], 
    diff=5, overlap=True)
```

    2
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    y Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_34_4.svg)
    



For these next problems, the input is a matrix and an optional vector, and the output is a matrix.

$$\frac{df(x, y)_{o, p}}{x_{i, j}}, \frac{df(x, y)_{o,p}}{y_j}$$

For visual simplicity results are shown on the flattened version of these matrices.

## Problem 15: View

Compute the identity function for all $o,p$. $Y$ is ignored.

$$f(X, Y)_{o, p} = X_{o, p}$$


```python
def fb_view(x, y):
    f = lambda o, p: x[o, p]
    dx = lambda i, j, o, p: 0 # Fill in this line
    dy = lambda j, o, p: 0 # Fill in this line
    return f, dx, dy

zip(make_mat(fb_view, (4, 10), (4, 10)), split=40, out_shape=40, gaps=[10 * i for i in range(4)])
```

    5
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_37_2.svg)
    



### Problem 16: Transpose

Transpose row and columns

$$f(X, Y)_{o, p} = X_{p, o}$$



```python
def fb_trans(x, y):
    f = lambda o, p: x[p, o]
    dx = lambda i, j, o, p: 0 # Fill in this line
    dy = lambda j, o, p: 0 # Fill in this line
    return f, dx, dy
zip(make_mat(fb_trans,  in_shape=(4, 10), out_shape=(10, 4)), split=40, out_shape=40, gaps=[10 * i for i in range(4)])
```

    5
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>21</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>31</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_39_2.svg)
    



### Problem 17: Broadcast 

Broadcast a matrix with a vector

$$f(X, y)_{o, p} = X_{o, p} \cdot y_p$$


```python
def fb_broad(x, y):
    f = lambda o, p: x[o, p] * y[p]
    dx = lambda i, j, o, p: 0 # Fill in this line
    dy = lambda j, o, p: 0 # Fill in this line
    return f, dx, dy
zip(make_mat(fb_broad,  in_shape=(4, 10), out_shape=(4, 10)), split=40, out_shape=40, gaps=[10 * i for i in range(4)])
```

    5
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>


    y Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_41_4.svg)
    



### Problem 18: Mean Reduce

Compute the mean over rows

$$f(X, y)_{o, p} = \sum_{i} X_{i, p} / R$$



```python
def fb_mean(x, y):
    R = x.shape[0]
    f = lambda o, p: sum(x[di, p] for di in range(R)) / R
    dx = lambda i, j, o, p: 0 # Fill in this line
    dy = lambda j, o, p: 0 # Fill in this line
    return f, dx, dy
zip(make_mat(fb_mean,  in_shape=(4, 10), out_shape=(1, 10)), 
    split=40, out_shape=10, gaps=[10 * i for i in range(4)])
```

    5
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_43_2.svg)
    



### Problem 19: Matmul

Standard matrix multiplication

$$f(X, Y)_{o,p} = \sum_j X_{o, j} Y_{j,p}$$



```python
def fb_mm(x, y):
    _, M = x.shape
    f = lambda o, p: sum(x[o, d] * y[d, p] for d in range(M)) / M 
    dx = lambda i, j, o, p: 0 # Fill in this line
    dy = lambda i, j, o, p: 0 # Fill in this line
    return f, dx, dy

zip(make_mat2(fb_mm,  in_shape=(5, 5), in_shape2=(5, 5), out_shape=(5, 5)), 
    split=25, out_shape=25, gaps=[5 * i for i in range(5)],  
    pos1=None, pos2=None)
```

    6
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    y Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_45_4.svg)
    



### Problem 20: 2D Conv

Two Dimensional convolution

$$f(X, Y)_{o,p} = \sum_{dh}\sum_{dw} X_{o+dh, p+dw} Y_{dh,dw}$$




```python
def fb_conv(x, y):
    kh, kw = y.shape
    f = lambda o, p: sum(x[o + di, p + dj] * y[di, dj] / (kh * kw)
                         for di in range(kh) for dj in range(kw)) 
    dx = lambda i, j, o, p: 0 # Fill in this line
    dy = lambda i, j, o, p: 0 # Fill in this line
    return f, dx, dy

zip(make_mat2(fb_conv,  in_shape=(7, 7), in_shape2=(3, 3), out_shape=(5, 5)), 
    split=49, out_shape=25, gaps=[7 * i for i in range(7)], 
    y=torch.cat(2*[gy.abs() + 0.2],0)[:58], 
    pos1=[23, 36], pos2=[])
```

    8
    x Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    y Errors



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>In Index</th>
      <th>Out Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





    
![svg](autodiff_puzzlers_files/autodiff_puzzlers_47_4.svg)
    


