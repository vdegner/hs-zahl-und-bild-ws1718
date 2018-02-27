import itertools
import numpy as np
import json

r5 = np.sqrt(5)
tau = .5 + .5 * np.sqrt(5)

# vertices
vertices = {}
i = 0

def add_permutations(pt):
    global vertices, i
    for x in set(list(itertools.permutations(pt))):
        vertices[i] = list(x)
        i += 1 


pts = [
    ( 2, 2, 0, 0),
    (-2, 2, 0, 0),
    (-2,-2, 0, 0),
    
    
    ( r5,  1,  1,  1),
    (-r5,  1,  1,  1),
    ( r5, -1,  1,  1),
    (-r5, -1,  1,  1),
    ( r5, -1, -1,  1),
    (-r5, -1, -1,  1),
    ( r5, -1, -1, -1),
    (-r5, -1, -1, -1),
    
    ( tau,  tau,  tau,  tau**(-2)),
    ( tau,  tau, -tau,  tau**(-2)),
    ( tau,  tau,  tau, -tau**(-2)),
    ( tau, -tau, -tau,  tau**(-2)),
    ( tau,  tau, -tau, -tau**(-2)),
    (-tau, -tau, -tau,  tau**(-2)),
    ( tau, -tau, -tau, -tau**(-2)),
    (-tau, -tau, -tau, -tau**(-2)),
    
    ( 1/tau,  1/tau,  1/tau,  tau**(2)),
    ( 1/tau,  1/tau, -1/tau,  tau**(2)),
    ( 1/tau,  1/tau,  1/tau, -tau**(2)),
    ( 1/tau, -1/tau, -1/tau,  tau**(2)),
    ( 1/tau,  1/tau, -1/tau, -tau**(2)),
    (-1/tau, -1/tau, -1/tau,  tau**(2)),
    ( 1/tau, -1/tau, -1/tau, -tau**(2)),
    (-1/tau, -1/tau, -1/tau, -tau**(2)),
]

for pt in pts:
    add_permutations(pt)
    
def add_even_permutations(pt):
    global vertices, i
    
    def parity(p):
        return sum(
            1 for (x,px) in enumerate(p)
              for (y,py) in enumerate(p)
              if x<y and px>py
            )%2==0
    
    for x in set(list(itertools.permutations([0,1,2,3]))):
        if parity(x) is True:
            vertices[i] = [pt[i] for i in x]
            i += 1

pts = [
    ( tau**2,  1,  tau**(-2), 0),
    (-tau**2,  1,  tau**(-2), 0),
    ( tau**2, -1,  tau**(-2), 0),
    ( tau**2,  1, -tau**(-2), 0),
    (-tau**2, -1,  tau**(-2), 0),
    ( tau**2, -1, -tau**(-2), 0),
    (-tau**2,  1, -tau**(-2), 0),
    (-tau**2, -1, -tau**(-2), 0),
    
    ( r5,  tau,  1/tau, 0),
    (-r5,  tau,  1/tau, 0),
    ( r5, -tau,  1/tau, 0),
    ( r5,  tau, -1/tau, 0),
    (-r5, -tau,  1/tau, 0),
    ( r5, -tau, -1/tau, 0),
    (-r5,  tau, -1/tau, 0),
    (-r5, -tau, -1/tau, 0),
    
    ( 2,  tau,  1,  1/tau),
    (-2,  tau,  1,  1/tau),
    ( 2, -tau,  1,  1/tau),
    ( 2,  tau, -1,  1/tau),
    ( 2,  tau,  1, -1/tau),
    (-2, -tau,  1,  1/tau),
    ( 2, -tau, -1,  1/tau),
    ( 2,  tau, -1, -1/tau),
    (-2,  tau,  1, -1/tau),
    ( 2, -tau,  1, -1/tau),
    (-2,  tau, -1,  1/tau),
    (-2, -tau, -1,  1/tau),
    ( 2, -tau, -1, -1/tau),
    (-2,  tau, -1, -1/tau),
    (-2, -tau,  1, -1/tau),
    (-2, -tau, -1, -1/tau),
]

for pt in pts:
    add_even_permutations(pt)
    
with open("vertices", "w") as f:
    json.dump(vertices, f)
    
    
# edges
d = 2 / tau**2
edges = {i: [] for i in range(600)}

for i in range(600):
    x = vertices[i]    
    for j in range(600):
        dist = np.asarray(vertices[j]) - np.asarray(x)
        dist = np.sqrt(np.dot(dist, dist))
        if not j in edges[i]:
            if (abs(dist - d) < 1e-2):
                edges[i].append(j)
                edges[j].append(i)
                
with open("edges", "w") as f:
    json.dump(edges, f)
    
    
def crossproduct4D(a, b, c):
    """
    Calculate the generalized cross product of 3 vectors in 4 dimensions.
    """
    a0, a1, a2, a3 = a
    b0,b1,b2,b3 = b
    c0,c1,c2,c3 = c
    x = np.identity(4)
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    
    return -a0*b1*c2*x3 + a0*b1*c3*x2 + a0*b2*c1*x3 - a0*b2*c3*x1 - a0*b3*c1*x2 + a0*b3*c2*x1 + a1*b0*c2*x3 - a1*b0*c3*x2 - a1*b2*c0*x3 + a1*b2*c3*x0 + a1*b3*c0*x2 - a1*b3*c2*x0 - a2*b0*c1*x3 + a2*b0*c3*x1 + a2*b1*c0*x3 - a2*b1*c3*x0 - a2*b3*c0*x1 + a2*b3*c1*x0 + a3*b0*c1*x2 - a3*b0*c2*x1 - a3*b1*c0*x2 + a3*b1*c2*x0 + a3*b2*c0*x1 - a3*b2*c1*x0 

def stereographic_projection(x, dim=3, N=None):
    """
    Stereographic projection in arbitrary dimension and with arbitrary north pole. 
    """
    # input data
    x = np.asarray(x).copy()
    if not x.shape[-1] == dim:
        raise ValueError("x contains invalid points.")
    if len(x.shape) == 1:
        x = np.array([x, ])
    if N is None:
        N = np.zeros(dim)
        N[-1] = 1
    if len(N) != dim:
        raise ValueError("North pole does not have the proper dimension.")
    else:
        d =  np.linalg.norm(N)
        p = N / d
        
    # basis
    B = np.identity(dim)
    B[:, 0] = 1
    B[:,-1] = p    
    
    for i in range(2,dim+1):
        B[:,-i] = B[:,-i] 
        for j in range(1,i):
            B[:,-i] -= np.dot(B[:,-i], B[:,-j]) * B[:,-j]
            B[:,-i] /= np.linalg.norm(B[:,-i])
    B = np.matrix(B)
    
    # projection
    for i in range(len(x)):
        pt = x[i]
        pt = np.asarray(B.I.dot(pt)).flatten()

        for j in range(dim-1):
            pt[j] = pt[j] / (1 - (pt[-1])) 
        pt[-1] = 0
        x[i] = pt
    return x, B
