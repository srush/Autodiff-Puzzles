import chalk
from chalk import *
import torch
import random
from colour import Color

import pandas as pd
import sys
sys.setrecursionlimit(10000)
set_svg_height(400)
set_svg_draw_height(400)
from IPython.display import display, HTML

def show_dog():
    print("Passed Tests!")
    pups = [
    "2m78jPG",
    "pn1e9TO",
    "MQCIwzT",
    "udLK6FS",
    "ZNem5o3",
    "DS2IZ6K",
    "aydRUz8",
    "MVUdQYK",
    "kLvno0p",
    "wScLiVz",
    "Z0TII8i",
    "F1SChho",
    "9hRi2jN",
    "lvzRF3W",
    "fqHxOGI",
    "1xeUYme",
    "6tVqKyM",
    "CCxZ6Wr",
    "lMW0OPQ",
    "wHVpHVG",
    "Wj2PGRl",
    "HlaTE8H",
    "k5jALH0",
    "3V37Hqr",
    "Eq2uMTA",
    "Vy9JShx",
    "g9I2ZmK",
    "Nu4RH7f",
    "sWp0Dqd",
    "bRKfspn",
    "qawCMl5",
    "2F6j2B4",
    "fiJxCVA",
    "pCAIlxD",
    "zJx2skh",
    "2Gdl1u7",
    "aJJAY4c",
    "ros6RLC",
    "DKLBJh7",
    "eyxH0Wc",
    "rJEkEw4"]
    display(HTML("""
    <video alt="test" controls autoplay=1>
        <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
    </video>
    """%(random.sample(pups, 1)[0])))



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
    top = frame(name, top, Color("#EEEEEE"), name, h=1.5) + top
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
    r = rectangle(env.width + w, env.height * h).fill_color(c).line_width(0)
    if l:
        r = r.align_l() + (hstrut(w/4) | text(l, 2.5).fill_color(Color("black")).line_width(0)).align_l()
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
        up = x + 1e-5 * torch.eye(s).double()[i]            
        f1 = fb(up)
        up = x - 1e-5 * torch.eye(s).double()[i]            
        f2 = fb(up)        
        for j in range(out.shape[0]):
            dx2[j, i] = (f1(j) - f2(j)) / (2 * 1e-5)
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
    for j, i in (~torch.isclose(dx, dx2, atol=1e-4)).nonzero():
        #print(i.item(), j.item(), dx[i,j].item(), dx2[i,j].item())
        bad.setdefault(i.item(), [])
        bad[i.item()].append(j.item())
        df.append({"In Index": i.item(), "Out Index": j.item()})
    return bad, pd.DataFrame(df)

gy = torch.tensor([math.sin(x/20) * 0.5 + (random.random() - 0.5)
                  for x in range(50)]).double()

def fb_demo(x):
    f = lambda o: x[o]
    dx = lambda i, o: (abs(o-i) < 4) * (abs(o-i) % 2) # Fill in this line
    return f, dx

def in_out2(fb, fb2, pos=None, overlap=False, diff=1, out_shape=50, y=gy):
    "For functions with point samples"
    set_svg_height(500)
    f_y, dx, _ = one_argf(fb, y, out_shape)
    g_f_y, dxg, _ = one_argf(fb2, f_y, out_shape)

    if pos is None:
        pos = range(y.shape[0])
    d = vcat([graph(y, "x").center_xy(),
              graph(f_y, "f(x)").center_xy(),
              graph(g_f_y, "g(f(x))").center_xy(),
    ], 0.1)

    d += overgraph(d, "f(x)", f_y, f_y + diff * dx[:, pos].sum(-1), Color("red"))
    d += overgraph(d, "g(f(x))", g_f_y, g_f_y + diff * (dxg @ dx)[:, pos].sum(-1), Color("green"))
    
    d = connect(d, y, "x", f_y, "f(x)", dx, pos, to_line=True)
    d = connect(d, f_y, "f(x)", g_f_y, "g(f(x))", dxg @ dx,
                [i.item() for i in  dx[:, pos].sum(-1).nonzero()], to_line=True, color=Color("lightgreen"))
    
    return outer_frame(d)


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
    else:
        show_dog()
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
    if not bad_x and not bad_y:
        show_dog()
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
