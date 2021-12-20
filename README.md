# Graph  
[![Documentation Status](https://readthedocs.org/projects/graph-brianbt/badge/?version=latest)](https://graph-brianbt.readthedocs.io/en/latest/?badge=latest)  
This is a python library for graph (data structure and algorithm) operations.  
Support both directed and undirected graph.  Implement by Adjacent List.  
It includes many basic graph algorithm like:
 - BFS
 - DFS
 - is_cyclic
 - Topological Sort
 - Bellman-Ford
 - A-star


### Install
`! pip install git+git://github.com/brianbt/Graph.git`
### Usages
```
from graph import Graph
from graph import graph_algo
import numpy as np

# Construct Graph
g = Graph(5)
g.add_Dedge(0, 1)
g.add_Dedge(0, 2, 10)
g.add_Dedge(1, 2)
g.add_Dedge(2, 3)
g.add_Dedge(2, 4, 3)
g.add_Dedge(3, 4)
g.add_Dedge(4, 0)
print(g)    # for printing the graph in a neat way
print(g.get_adj_vertices(0))    #get all vertices directly reachable from node 0

# Graph Traversal
print(graph_algo.BFS(g, 0, unweight=True))
print(graph_algo.DFS(g, 0, unweight=True))
print(graph_algo.DFS_recursive(g, 0))
print(graph_algo.get_order(g, 0, 2, "bellmanFord"))

# Shortest Path
true_dist = [[0,1,10,11,12],[np.inf,0,1,2,3],[np.inf,np.inf,0,1,2],[np.inf,np.inf,np.inf,0,1],[np.inf,np.inf,np.inf,np.inf,0]]
a_star_d = graph_algo.a_star(g, true_dist, 0, 4)
print(a_star_d)
print(graph_algo.bellman_ford(g))
print(g.topological_sort())
print(g.get_entries())
print(graph_algo.get_order(g,0,4))
```