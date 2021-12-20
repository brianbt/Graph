import numpy as np
from queue import PriorityQueue

def BFS(graph, s, unweight=False, get_dict=False):
    '''
    Start from s, travel through entire graph, using BFS.  

    Args:  
        unweight (bool): Assume the graph is unweighted (weight=1) if True.  
        get_dict (bool): Return dictionary[vertex:val] if True  

    Returns: 
        (p,d):
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
        (p,d):
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
        (p,d):
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
    else:
        raise ValueError(f"Not support for method={method}")
    get_order_inner(p, u, output)
    return output



def relax(u, v, duv, d, p):
    '''relax function used in Bellman Ford algorithm

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
    """Run Bellman Ford algorithm on the entire graph
    
    Args:
        graph (Graph): an Graph() object 

    Returns:  
        tuple(dict of (p), dict of (d)):  
        where p[vertiex]=[...]
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


def a_star(graph, true_dist, start, end, get_dict=False):
    """Using a_star algorithm to find the shortest path from start to end.  

    |  f(n) = g(n) + h(n). 
    |  f(n) is total distance  
    |  g(n) is distance taken from start to n  
    |  h(n) is the heuristic(est. straight list dist) from n to end  

    Args:  
        graph (Graph): an Graph() object   
        true_dist (List[List]): heuristic function, 2d array, true_dist[from][to]  
        start, end (int): vertices  
        get_dict (bool): Return dictionary[vertex:val] if True  

    Returns:  
        (p,d):  
        p: list/dict of parent_node, p[i] means the parent of vertex i
           -1 means the parent of s, -2 means not reachable
        d: list/dict of distance, d[i] means the distance needed from s to i
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
    if get_dict:
        return p,d
    else:
        return (list(p.values()), list(d.values()))

