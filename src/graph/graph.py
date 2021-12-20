import numpy as np
import warnings
class Graph:
    """
    A graph class implemented by adjacent list using dict of list.  

    Args:
        num (int): number of nodes in the graph  

    Instance Attributes:  
        self.size (int): number of nodes in the graph  
        self.graph (dict): dict of list

    Usages:  
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
    """
    
    def __init__(self, num):
        self.size = num
        self.graph = dict()

    # Add edges
    def add_edge(self, v1, v2, w=1):
        """add edge from v1 to v2 and v2 to v1
    
        Args:
            v1 (int): vertex
            v2 (int): vertex
            w (float): weight of edge. Defaults=1
        
        """
        if v1 >= self.size or v2 >= self.size:
            raise Exception("vertex index should be smaller than graph size.")
        self.add_Dedge(v1, v2, w)
        if v1 != v2:
            self.add_Dedge(v2, v1, w)

    def add_Dedge(self, v1, v2, w=1):
        """
        add edge from v1 to v2, one direction only

        Args:
            v1 (int): vertex
            v2 (int): vertex
            w: weight of edge. Defaults=1
        """
        if v1 >= self.size or v2 >= self.size:
            raise Exception("vertex index should be smaller than graph size.")
        if v1 == v2:
            print(f"Same vertex {v1} and {v2}, Notice: adding self-loop-edge")
        if v1 not in self.graph:
            self.graph[v1] = []
        if v2 not in self.graph:
            self.graph[v2] = []
        self.graph[v1].append((v2, w))

    def remove_edge(self, v1, v2, all=False):
        """
        Remove edge from v1 to v2 and v2 to v1.

        Args:
            v1, v2: vertex (int)
            all (bool): if False -> remove the first added edge from v1 to v2
                        if True -> remove all edge from v1 to v2
        """
        self.remove_Dedge(v1, v2, all)
        if v1 != v2:
            self.remove_Dedge(v2, v1, all)

    def remove_Dedge(self, v1, v2, all=False):
        """
        Remove edge from v1 to v2, one direction only

        Args:
            v1, v2: vertex (int)
            all (bool): if False -> remove the first added edge from v1 to v2
                        if True -> remove all edge from v1 to v2
        """
        if all:
            self.graph[v1] = [x for x in self.graph[v1] if x[0] != v2]
        else:
            for i in range(len(self.graph[v1])):
                if self.graph[v1][i][0] == v2:
                    break
            if self.graph[v1][i][0] == v2:
                del self.graph[v1][i]

    def get_adj_vertices(self, v, self_loop=True):
        """Return list of vertices that is one step reachable from v.

        Args:
            v (int): vertex
            self_loop (bool): if True -> include self vertex, else -> Exclude
        """
        if v in self.graph:
            if self_loop:
                return [x[0] for x in self.graph[v]]
            else:
                return [x[0] for x in self.graph[v] if v!=x[0]]
        else:
            return []

    def get_edge_weight(self, v1, v2, all=False):
        """Return all edge weight from v1 to v2. 

        This function DOES NOT search the path. Thus, 1 and v2 need to be Adjcent.  
        
        Args:
            v1, v2: vertex (int)
            all (bool): if True -> return all weight if multi edge on two node
                        else -> return the first edge's weight

        Returns:
            edge from v1 to v2, v1 and v2 need to be adjacent node, else return np.inf
        """
        out = []
        if v1 in self.graph:
            for k, v in self.graph[v1]:
                if k == v2:
                    out.append(v)
        if all:
            return out
        else:
            if out:
                return out[0]
            else:
                return np.inf

    def get_edges(self):
        """
        Retruns:
            list of edge in this graph.
        """
        return [(i, j[0]) for i, v in self.graph.items() for j in v]

    def _topological_sort_util(self, v, visited, stack):
        visited[v] = True
        for i in self.get_adj_vertices(v):
            if visited[i] == False:
                self._topological_sort_util(i, visited, stack)
        stack.append(v)

    def topological_sort(self):
        """Perform Topological Sort on the graph. 

        Returns:
          Topological List. For example: [v1, v2, v3, ...] where v_i come before v_{i+1}
        """
        if self.is_cyclic():
            warnings.warn("WARNING: this graph maybe cyclic (unless existing self-loop).\nTopological sort has no meaning")
        visited = [False]*self.size
        stack = []
        for i in range(self.size):
            if visited[i] == False:
                self._topological_sort_util(i, visited, stack)
        if len(stack) > self.size:
            print("have cycle")
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
        """Check if the graph has cycle, return (bool). ONLY for directed graph
        """
        visited = [False] * (self.size + 1)
        recStack = [False] * (self.size + 1)
        for node in range(self.size):
            if visited[node] == False:
                if self._is_cyclic_util(node, visited, recStack) == True:
                    return True
        return False

    def get_entries(self, hasPath=False):
        """ Return list of vertices that is the entry point. Vertices that no one point to.

        Args:
            hasPath: If True -> only the vertices that are in a path will be return
        """
        out = []
        topo = self.topological_sort()
        out.append(topo[0])
        for i in range(1, len(topo)):
            if not topo[i] in self.get_adj_vertices(out[-1]):
                if hasPath and self.get_adj_vertices(topo[i], self_loop=False):
                    out.append(topo[i])
                elif not hasPath:
                    out.append(topo[i])
            else:
                break
        if hasPath and self.get_adj_vertices(topo[0], self_loop=False):
            return out
        elif hasPath and not self.get_adj_vertices(topo[0], self_loop=False):
            return out[1:]
        return out


    def get_dests(self, hasPath=False):
        """Return list of vertices that is the destination. Vertices that point to nothing.

        Args:
            hasPath: If True -> only the vertices that are in a path will be return
        """
        combined = [x for x in self.graph if not self.get_adj_vertices(x, self_loop=False) or not self.graph[x]]
        havePath = [x for x in self.graph if not self.get_adj_vertices(x)]
        if hasPath:
            return havePath
        else:
            return list(set(combined).union(set(havePath)))

    def get_isolated(self):
        """Return list of vertices that is both entries and destination.
        """
        return list(set(self.get_dests()).intersection(set(self.get_entries())))

    # Print the graph
    def print_graph(self, store=False):
        """
        format: 
            Vertex u: -> (v1, w1) -> (v2, w2) -> (v3, w3)  
            Arg store is internal use
        """
        if store:
            output = ""
            for k,v in self.graph.items():
                if v:
                    output += ("Vertex " + str(k) + ":")
                    for i in v:
                        output += (f" -> {i}")
                    output += "\n"
            return output
        else:
            print(self, end="")
        
    def get_graph(self):
        return self.graph

    def __len__(self):
        return self.size

    def __str__(self):
        return self.print_graph(store=True)


