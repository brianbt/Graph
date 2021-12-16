import numpy as np
import pprint as pp
from queue import PriorityQueue

# Adjacency Matrix representation in Python
class Graph(object):

    # Initialize the matrix
    def __init__(self, size):
        self.adjMatrix = np.ones((size,size))*np.inf
        self.size = size
    def resize(self, size):
        """
        Change the size of graph. New size must be larger than old size
        """
        old = self.size
        new = size
        assert old<new
        t1 = np.ones((new-old, old))*np.inf
        t2 = np.ones((new, new-old))*np.inf
        self.adjMatrix = np.hstack((np.vstack((self.adjMatrix, t1)), t2))
        self.size = size
    # Add edges
    def add_edge(self, v1, v2, w=1):
        """
        add edge from v1 to v2 and v2 to v1
    
        Args:
            v1, v2 (int): vertex
            w (float): weight of edge. Defaults=1
        
        """
        if v1 == v2:
            print(f"Same vertex {v1} and {v2}, Notice: adding self-loop-edge")
        if np.not_equal([self.adjMatrix[v1,v2], self.adjMatrix[v2,v1]], [np.inf, np.inf]).any():
            raise Exception("edge already existed")
        self.adjMatrix[v1,v2] = w
        self.adjMatrix[v2,v1] = w

    # Add edges
    def add_Dedge(self, v1, v2, w=1):
        """
        add edge from v1 to v2, one direction only

        Args:
            v1, v2: vertex (int)
            w: weight of edge. Defaults=1
        """
        if v1 == v2:
            print(f"Same vertex {v1} and {v2}, Notice: adding self-loop-edge")
        if np.not_equal([self.adjMatrix[v1,v2]], [np.inf]).any():
            raise Exception("edge already existed")
        self.adjMatrix[v1,v2] = w

    # Remove edges
    def remove_edge(self, v1, v2):
        """
        Remove edge from v1 to v2.

        Args:
            v1, v2: vertex (int)
        """
        self.adjMatrix[v1,v2] = np.inf
    
    def get_adj_vertices(self, v):
        """
        Return list of vertices that is one step reachable from v.
        Args:
            v: vertex (list)
        """
        return np.argwhere(~np.isinf(self.adjMatrix[v])).flatten()
    
    def get_edge_weight(self, v1, v2):
        """
        Return edge weight from v1 to v2.
        This function DOES NOT search the path. Thus, 1 and v2 need to be Adjcent. 
        
        Args:
            v1, v2: vertex (int)
        Returns:
            edge from v1 to v2, v1 and v2 need to be adjacent node, else return np.inf
        """
        return self.adjMatrix[v1,v2]

    def get_edges(self):
        """
        Retruns:
            list of edge in this graph.
        """
        return [(i, j) for i, l in enumerate(self.adjMatrix) for j, v in enumerate(l) if v]

    def _topological_sort_util(self, v, visited, stack):
        visited[v] = True
        for i in self.get_adj_vertices(v):
            if visited[i] == False:
                self._topological_sort_util(i, visited, stack)
        stack.append(v)

    def topological_sort(self):
        """
        Perform Topological Sort on the graph. 
        Return Topological List.
        [v1, v2, v3, ...], v_i come before v_{i+1}
        """
        visited = [False]*self.size
        stack = []
        for i in range(self.size):
            if visited[i] == False:
                self._topological_sort_util(i, visited, stack)
        return(stack[::-1])

    def _is_cyclic_util(self, v, visited, recStack):
        visited[v] = True
        recStack[v] = True
        for neighbour in self.get_adj_vertices(v):
            if visited[neighbour] == False:
                if self._is_cyclic_util(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True
        recStack[v] = False
        return False
 
    def is_cyclic(self):
        """
        Check if the graph has cycle, return (bool). ONLY for directed graph
        """
        visited = [False] * (self.size + 1)
        recStack = [False] * (self.size + 1)
        for node in range(self.size):
            if visited[node] == False:
                if self._is_cyclic_util(node,visited,recStack) == True:
                    return True
        return False

    def get_entries(self, hasPath=False):
        """
        Return list of vertices that is the entry point. Vertices that no one point to.
        Args:
            hasPath: If True -> only the vertices that are in a path will be return
        """
        if hasPath:
            out = set(self.get_entries())
            side = set(self.get_dests())
            return list(out-side)
        return np.where(np.apply_along_axis(lambda x: all(np.isinf(x)), 0, self.adjMatrix))[0]

    def get_dests(self, hasPath=False):
        """
        Return list of vertices that is the destination. Vertices that point to nothing.
        Args:
            hasPath: If True -> only the vertices that are in a path will be return
        """
        if hasPath:
            out = set(self.get_dests())
            side = set(self.get_entries())
            return list(out-side)
        return np.where(np.apply_along_axis(lambda x: all(np.isinf(x)), 1, self.adjMatrix))[0]

    def get_isolated(self):
        """
        Return list of vertices that is both entries and destination.
        """
        return list(set(self.get_dests()).intersection(set(self.get_entries())))

    def get_adjMatrix(self):
        return self.adjMatrix

    def __len__(self):
        return self.size
    
    def __str__(self): # pretty print(g)
        return pp.pformat(self.adjMatrix)


def BFS(graph, s, unweight=False, get_dict=False):
    '''
    Start from s, travel through entire graph, using BFS.

    Args:
        unweight (bool): Assume the graph is unweighted (weight=1) if True.
        get_dict (bool): Return dictionary[vertex:val] if True
    Returns: 
        (p,d)
        p: list/dict of parent_node, p[i] means the parent of vertex i
           -1 means the parent of s, -2 means not reachable
        d: list/dict of distance, d[i] means the distance needed from s to i
           Yield shortest path if unweight=True.
    '''
    visited = [False for _ in range(len(graph))]
    p = [-2 for _ in range(len(graph))]
    d = [np.inf for _ in range(len(graph))]
    queue = [s]
    visited[s] = True
    p[s] = -1
    d[s] = 0
    while(len(queue) != 0):
        curr_v = queue.pop(0)
        adj_list = graph.get_adj_vertices(curr_v)
        for next_v in adj_list:
            if visited[next_v] is False:
                visited[next_v] = True
                p[next_v] = curr_v
                d[next_v] = d[curr_v] + 1
                if unweight:
                    d[next_v] = d[curr_v] + 1
                else:
                    d[next_v] = d[curr_v] + graph.get_edge_weight(curr_v, next_v)
                queue.append(next_v)
    if get_dict:
        return ({i: p[i] for i in range(len(p))}, {i: d[i] for i in range(len(d))})
    else:
        return (p, d)


def DFS(graph, s, unweight=False, get_dict=False):
    '''
    Start from s, travel through entire graph, using DFS.

    Args:
        s (int): starting vertex
        unweight (bool): Assume the graph is unweighted (weight=1) if True.
        get_dict (bool): Return dictionary[vertex:val] if True
    Returns: 
        (p,d)
        p: list/dict of parent_node, p[i] means the parent of vertex i
           -1 means the parent of s, -2 means not reachable
        d: list/dict of distance, d[i] means the distance needed from s to i
           NOT necessarily yield shortest path if unweight=True
    '''
    visited = [False for _ in range(len(graph))]
    p = [-2 for _ in range(len(graph))]
    d = [np.inf for _ in range(len(graph))]
    stack = [s]
    visited[s] = True
    p[s] = -1
    d[s] = 0
    while(len(stack) != 0):
        curr_v = stack.pop()
        adj_list = graph.get_adj_vertices(curr_v)
        for next_v in adj_list:
            if visited[next_v] is False:
                visited[next_v] = True
                p[next_v] = curr_v
                if unweight:
                    d[next_v] = d[curr_v] + 1
                else:
                    d[next_v] = d[curr_v] + graph.get_edge_weight(curr_v, next_v)
                stack.append(next_v)
    if get_dict:
        return ({i: p[i] for i in range(len(p))}, {i: d[i] for i in range(len(d))})
    else:
        return (p, d)


def DFS_recursive_inner(curr_v, graph, visited, p, d, unweight):
    adj_list = graph.get_adj_vertices(curr_v)
    for next_v in adj_list:
        if visited[next_v] is False:
            visited[next_v] = True
            p[next_v] = curr_v
            if unweight:
                d[next_v] = d[curr_v] + 1
            else:
                d[next_v] = d[curr_v] + graph.get_edge_weight(curr_v, next_v)
            DFS_recursive_inner(next_v, graph, visited, p, d, unweight)

def DFS_recursive(graph, s, unweight=False, get_dict=False):
    '''
    Start from s, travel through entire graph, using DFS.

    Args:
        s (int): starting vertex
        unweight (bool): Assume the graph is unweighted (weight=1) if True.
        get_dict (bool): Return dictionary[vertex:val] if True
    Returns: 
        (p,d)
        p: list/dict of parent_node, p[i] means the parent of vertex i
           -1 means the parent of s, -2 means not reachable
        d: list/dict of distance, d[i] means the distance needed from s to i
           NOT necessarily yield shortest path if unweight=True
    '''
    visited = [False for _ in range(len(graph))]
    p = [-2 for _ in range(len(graph))]
    d = [np.inf for _ in range(len(graph))]
    visited[s] = True
    p[s] = -1
    d[s] = 0
    DFS_recursive_inner(s, graph, visited, p, d, unweight)
    if get_dict:
        return ({i: p[i] for i in range(len(p))}, {i: d[i] for i in range(len(d))})
    else:
        return (p, d)


def get_order_inner(p, curr_v, output_list):
    if p[curr_v] == -2:
        return
    if p[curr_v] == -1:
        output_list.append(curr_v)
        return

    get_order_inner(p, p[curr_v], output_list)
    output_list.append(curr_v)

def get_order(graph, s, u, method="BFS"):
    '''
    This function give the path from s to u

    Args:
        graph (Graph): an Graph() object 
        method (str): either "BFS", "BFS" or "bellmanFord". Return shortest-weight-path if "BellmanFord"
        p (list/dict): parent_node (return from BFS, DFS)
        u (int): ending node
    Returns:
        path from s to u, if u is unreachable from s -> return []
    '''
    output = []
    if method == "BFS":
        p,_ = BFS(graph, s)
    elif method == "DFS":
        p, _ = DFS(graph, s)
    elif method == "bellmanFord":
        Fp, _ = bellman_ford(graph)
        p = Fp[s]
    get_order_inner(p, u, output)
    return output



def relax(u, v, duv, d, p):
    '''
    relax function used in Bellman Ford algorithm
    Args:
        u, v (int): vertices
        duv (float): weight(u, v)
        d (list/dict): distance vector from 0
        p (list/dict): parent_node vector, -1 means the parent of 0, -2 means not reachable
    '''
    if d[v] > d[u] + duv:
        d[v] = d[u] + duv
        p[v] = u

def bellman_ford(graph):
    """
    Run Bellman Ford algorithm on the entire graph
    
    Args:
        graph (Graph): an Graph() object 
    Retruns:
        dict of (p,d), d[vertiex] = (p,d)
        p: dict of parent_node, p[i] means the parent of vertex i
           -1 means the parent of s, -2 means not reachable
        d: dict of distance, d[i] means the distance needed from s to i
           Shortest path if true_dist(heuristic function) is correct
    """
    p = {}
    d = {}
    for s in range(len(graph)):
        p[s] = [-2 for i in range(len(graph))]
        d[s] = [np.inf for i in range(len(graph))]
        p[s][s] = -1
        d[s][s] = 0
        for i in range(len(graph)):
            for edge in graph.get_edges():
                u = edge[0]
                v = edge[1]
                relax(u, v, graph.get_edge_weight(u, v), d[s], p[s])
        for edge in graph.get_edges():
            u = edge[0]
            v = edge[1]
            if(np.not_equal([d[s][u]], [np.inf]).any() and
               d[s][v] > d[s][u] + graph.get_edge_weight(u, v)):
               print("There is negative cycle for starting node = ", s)
    return (p, d)


def a_star(graph, true_dist, start, end):
    """"
    Using a_star algorithm to find the shortest path from start to end.  
    f(n) = g(n) + h(n). 
    f(n) is total distance  
    g(n) is distance taken from start to n  
    h(n) is the heuristic(est. straight list dist) from n to end  

    Args:
        graph (Graph): an Graph() object 
        true_dist (List[List]): heuristic function, 2d array, true_dist[from][to]
        start, end (int): vertices
    Retruns:
        (p,d)
        p: dict of parent_node, p[i] means the parent of vertex i
           -1 means the parent of s, -2 means not reachable
        d: dict of distance, d[i] means the distance needed from s to i
           Shortest path if true_dist(heuristic function) is correct
    """
    p = {}
    d = {}      ##without heuristic
    visited = {}
    q = PriorityQueue()     ##with heuristic
    p[start] = -1
    d[start] = 0
    visited[start] = True
    q.put((0, start))
    while not q.empty():
        (_, u) = q.get()
        if u == end:
            break
        for v in graph.get_adj_vertices(u):
            curr_dist = d[u] + graph.get_edge_weight(u,v)
            if v not in visited.keys() or curr_dist < d[v]:
                q.put((curr_dist + true_dist[v][end], v))
                p[v] = u
                d[v] = curr_dist 
                visited[v] = True
    return p,d


g = Graph(5)
g.add_Dedge(0, 1)
g.add_Dedge(0, 2, 10)
g.add_Dedge(1, 2)
g.add_Dedge(2, 3)
g.add_Dedge(2, 4, 3)
g.add_Dedge(3, 4)
print(g)
print(g.get_adj_vertices(0))
print(BFS(g, 0, unweight=True))
print(DFS(g, 0, unweight=True))
print(DFS_recursive(g, 0))
print(get_order(g, 0, 2, "bellmanFord"))
true_dist = [[0,1,10,11,12],[np.inf,0,1,2,3],[np.inf,np.inf,0,1,2],[np.inf,np.inf,np.inf,0,1],[np.inf,np.inf,np.inf,np.inf,0]]
a_star_d = a_star(g, true_dist, 0, 4)
print(a_star_d)
print(bellman_ford(g)[0])
print(g.topological_sort())
print(g.get_entries())

# g = Graph(4)
# print(g)
# g.add_Dedge(2,0)
# g.add_Dedge(1,0)
# g.add_Dedge(0,3)
# print(g.get_entries())
# print(g.get_dests())
# g.resize(20)
# print(g.get_entries())
# print(g.get_dests())
# print(g.get_entries(True))
# print(g.get_dests(True))
# print(g.get_isolated())
# print(g)



# g= Graph(6)
# g.add_Dedge(0, 1, 3)
# g.add_Dedge(1, 0)
# g.add_Dedge(1, 2)
# g.add_Dedge(2, 1)
# g.add_Dedge(2, 5, 15)
# g.add_Dedge(0, 3, 5)
# g.add_Dedge(1, 4, 8)
# g.add_edge(3, 4)
# g.add_edge(4, 5)
# print(g)
# print(np.isinf(g.adjMatrix[5]))
# print(~np.isinf(g.adjMatrix[5]))
# print(np.argwhere(~np.isinf(g.adjMatrix[5])))
# print(np.argwhere(~np.isinf(g.adjMatrix[5])).flatten())
# print(g.get_adj_vertices(4))
# true_dist = [[0,3,4,5,6,7],[1,0,1,6,7,8],[2,1,0,7,8,9],[np.inf,np.inf,np.inf,0,1,2],[np.inf,np.inf,np.inf,1,0,1],[np.inf,np.inf,np.inf,2,1,0]]
# a_star_d = a_star(g, 0, 5)
# print(a_star_d)

