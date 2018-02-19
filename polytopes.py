import numpy as np

class HyperCell:
    """
    Simple (n-1)-dimensional cell of a hypercube.
    """
    def __init__(self, parent, idx, value):
        self.dimension = parent.dimension - 1
        self.coordinates = np.array([pt for pt in parent.coordinates if pt[idx] == value])
        self.full_coordinates = self.coordinates.copy()
        self.coordinates = np.delete(self.coordinates, idx, 1)
        
        if self.dimension > 1:
            self.calculate_subcells()
        
    def calculate_subcells(self):
        """
        Calculate subcells of hypercube.
        """
        self.childs = []

        for value in [0,1]:
            for idx in range(self.dimension):
                child = HyperCell(self, idx, value)
                self.childs.append(child)
        self.childs = np.array(self.childs)

class Hypercube:
    """
    Simple n-dimensional hypercube.
    """
    def __init__(self, dimension):
        """
        Create hypercube in n-dimension.
        
        params
        ------
        dimension: int, dimension of hypercube
        
        returns
        -------
        hypercube class instance
        """
        self.dimension = dimension
        
        self.compute_coordinates(self.dimension)
        
        self.calculate_subcells()
        
        self.calculate_edges()
            
    def compute_coordinates(self, dimension):
        """
        Compute coordinates of (n+1)-dimensional hypercube from n-dimensional base.
        """
        n, coords = 1, [[0], [1]]
        while n < dimension:
            newcoords = []
            for p in coords:
                pp = p.copy()
                p.insert(0, 0)
                pp.insert(0, 1)
                newcoords.append(p)
                newcoords.append(pp)
            coords = newcoords
            n += 1
        self.coordinates = np.array(coords)
        
    def calculate_subcells(self):
        """
        Calculate subcells of hypercube.
        """
        self.childs = []

        for value in [0,1]:
            for idx in range(self.dimension):
                child = HyperCell(self, idx, value)
                self.childs.append(child)
        self.childs = np.array(self.childs)
        
    def calculate_edges(self):
        edges = []
        pts = self.coordinates
        for i in range(len(pts)):
            pt = pts[i]
            j = 0
            for j in range(i, len(pts)):
                PT = pts[j]
                if (np.linalg.norm(pt - PT) == 1):
                    edges.append([pt, PT])
        self.edges = np.array(edges)
