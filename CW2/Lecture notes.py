"""

Lecture 6

Network: N nodes, L edges
Can represent with an adjacency matrix or an edge list

Degree of a node: number of edges attached to it (sum of that row/column)
Degree distribution P: P(q) is the fraction of nodes with degree q

Directed network: adjacency matrix may not be symmetric
Weighted network: adjacency matrix entries can be other than 0 or 1

networkx package has good tools for large/complex networks.
"Avoid writing your own code whenever possible" - use efficient libraries

import networkx as nx (use version 2.x, e.g. 2.2)
G = nx.Graph()

See/add edges or nodes:
G.add_edge(1,2)
G.edges()
G.nodes() not necessarily in order

Add many edges at once
e = [(1,5),(2,5),(2,3),(3,4),(4,5),(4,6)]
G.add_edges_from(e)

Plot the network:
nx.draw(G, with_labels=True, font_weight='bold')

Neighbours of a node
G.adj[x].keys() or something like it

Adjacency matrix
A = nx.adjacency_matrix(G)
A.todense() to print it out

Adjacency list; ith entry is a list of nodes connected to node i
G.adjacency_list() doesn't work but can look up an equivalent if needed
Much more efficient for sparse networks

Degree histogram as a list
nx.degree_histogram(G)

Erdos-Renyi graph
Grandom = nx.gnp_random_graph(1000,0.05) nodes, probability of edge
nx.draw(Grandom,node_shape='.')
Degree distribution is binomial
Not really a good model for large complex networks.
There should be large-degree nodes, and power-law degree distribution
Clustering coefficient should be large, mean degree small
Another model is the Barabasi-Albert model

Clustering coefficient: fraction of links between neighbours out of max possible
nx.clustering(G,500)
In G_Np, expect this to be p

Shortest path (fewest nodes)
nx.shortest_path(G,source=0,target=500)



Lecture 2

Which nodes have a path between them?
Given a node s, we can progressively find nodes in its 'chunk' until no unexplored nodes are left
Can go breadth-first or depth-first

Code for breadth-first search
O(N+M) (N nodes, M edges) if adding/removing from queue is O(1). Use dequeue (deque)

The collections module has a dequeue datatype; like a list but fast append/pop at either end



Lecture 8

Code for depth-first search
O(N+M) (N nodes, M edges) if adding/removing from queue is O(1). Use dequeue (deque)

Shortest path on a weighted network (least sum of edges)?
Dijkstra's algorithm does it for positive weights
    Outputs the length of the shortest path from a given source to all nodes

Code for Dijkstra's using dictionaries
O(N^2) for naive approach ('closest' unexplored node considered)
    O(M) for edges (each considered at most twice). O(N) for finding closest edge... Can we do better?



Lecture 9

A binary heap can find the closest node and update the unexplored distances in O(log_2(N)) time
Binary heap: nodes are in a list L ordered like a tree in some sense
Once the closest node n* is popped, the list can be reordered in O(log_2(N)) time

Other algorithms exist for dealing with negative weights

---Next section---

networkx: must understand strengths/weaknesses and cost of underlying algorithms, especially for large networks

Chat about 1D diffusion, with some equations

One way to model flows around a graph with flux. An ODE



Lecture 10

Simplifying the ODE for undirected graphs, in terms of the Laplacian matrix

(Chat about relation to the Laplacian operator on a grid)

Once initial conditions are specified, we have a system of linear, constant-coefficient ODEs, i.e. an eigenvalue problem
Can use np.linalg.eig or scipy.sparse.linalg.eig

What if we have nonlinear dynamics? In general, can only find numerical solutions

Can discretize time and use Taylor series to find each step. Smaller steps gives more accuracy

Can use Runge-Kutta methods for more accuracy and stability; uses half-step calculations for better approximations
Variable time-step R-K also available (adjusting timestep to match specified accuracy). Available in scipy.integrate
But 'explicit' time-step methods like this can struggle with systems of non-linear ODEs
Implicit Euler method more stable perhaps. Requires solution of system of equations at each step
Variable-time implicit methods also available. scipy.integrate.odeint



Lecture 11

Model equation: dy/dt = ay
Explicit Euler: conditionally stable at Re(a) = 0
Implicit Euler: unconditionally stable
RK4: conditionally stable, error term to the 4

Explicit time-step (e.g. RK45): good for linear dynamics and one time scale
Implicit time-step (odeint, BDF): good for non-linear systems and multiple time scales, and 'stiff' systems (variable)

Use odeint (or something similar) for systems of ODEs, and single PDEs
Use something like RK4 for systems of PDEs

---Data science---

Finding x with |x|=1 s.t. |Ax| is maximised

Solving using evalues



Lecture 12

Can do the above with the 'first eigenvector method'
Seek evalues, evectors of A^tA
Variants of QR method: O(N^3). Can we do better?
SVD method: something not very well-explained here

Should use scipy.sparse.linalg when working with large sparse matrices

Dimensionality reduction: PCA














"""