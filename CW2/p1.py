"""M345SC Homework 2, part 1
Christophe Jefferies 01202145
"""

from collections import deque

#1.1

def scheduler(L):
    """
    Question 1.1
    Schedule tasks using dependency list provided as input

    Input:
    L: Dependency list for tasks. L contains N sub-lists, and L[i] is a sub-list
    containing integers (the sub-list my also be empty). An integer, j, in this
    sub-list indicates that task j must be completed before task i can be started.

    Output:
    S: A list of integers corresponding to the schedule of tasks. S[i] indicates
    the day on which task i should be carried out. Days are numbered starting
    from 0.
    
    
    ---Discussion---
    
    Algorithm description:
    
    We use a recursive approach. For each task, we see on what day all its
    dependencies have been completed, and do the task on the next day.
    If the task has no dependencies, we perform it on day 0.
    If any dependency's completion day has not been found, we find it then
    (i.e. implement recursion here).

    
    Runtime analysis:
    
    Initialising the output S is O(N).
    We ignore defining the function 'getday' here (its calls are what contribute
    to running time).
    
    'getday' is explicitly called N times. Counting operations in recursive
    approaches can be tricky, but we can simplify matters by considering L as
    a network.
    
    Let N nodes represent the tasks, and a (directed) edge represent a
    dependency. Then the algorithm is just a depth-first approach: given a
    starting node, it will find all completion days required to calculate that
    starting node's completion day.
    The starting nodes are in no particular order, but in any case, each node
    and each edge are checked exactly once.
    Hence (from the notes) this algorithm runs in O(N+E) time, where E is the
    number of edges in the network.
        
    Given only N and M as in the question, we cannot know E. At the very worst,
    all (N-M) nodes without dependencies are needed for all the others, and the
    M nodes with dependencies rely on each other as much as possible, meaning
    there are (M choose 2) edges between them. So the largest E for given N and
    M is NM + M(M-1)/2, giving worst-case runtime O((N+M)M).
    However, if we assume that E is O(N) or put a sensible limit on the maximum
    degree of a node, then the runtime is effectively O(N).
    
    
    Efficiency analysis:
    
    The algorithm finds the first day possible on which each task can be
    completed, so its output is efficient in that sense.
    In terms of running efficiency, recursion is typically not a good idea; the
    recursive 'call stack frame' can quickly become large, which slows things
    down.
    However, in this case, the recursion will never be too deep; at worst, when
    the tasks are 'in a line', the recursion goes N deep, but for typical L the
    depth will likely be negligible relative to N.
    There are no nested 'for' or 'while' loops in the code, no iterations
    through L, and no need for large memory usage. As whole, this algorithm
    could be seen as efficient.
    """
    
    S = len(L)*[-1] #ith entry will be the day to complete task i
    
    def getday(i):
        
        #if i's completion day has already been calculated, return it
        if S[i] != -1:
            return S[i]
        
        #if i has no dependencies, do it on day 0
        if L[i] == []:
            S[i] = 0
            return 0
        
        #otherwise return the first day all its dependencies are completed, plus 1
        val = max(getday(j) for j in L[i]) + 1
        S[i] = val
        return val
    
    #Apply the above to all tasks, WLOG in node number order
    for i in range(len(L)):
        getday(i)
    
    return S
    
    



#2.1

def findPath(A,a0,amin,J1,J2):
    """
    Question 1.2 i)
    Search for feasible path for successful propagation of signal
    from node J1 to J2

    Input:
    A: Adjacency list for graph. A[i] is a sub-list containing two-element tuples (the
    sub-list may also be empty) of the form (j,Lij). The integer, j, indicates that there is a link
    between nodes i and j and Lij is the loss parameter for the link.

    a0: Initial amplitude of signal at node J1

    amin: If a>=amin when the signal reaches a junction, it is boosted to a0.
    Otherwise, the signal is discarded and has not successfully
    reached the junction.

    J1: Signal starts at node J1 with amplitude, a0
    J2: Function should determine if the signal can successfully reach node J2 from node J1

    Output:
    L: A list of integers corresponding to a feasible path from J1 to J2.
    
    
    ---Discussion---
    
    Algorithm description
    
    We use a breadth-first search starting at J1.
    We only 'explore' from node i to j if Lij*a0 is greater than amin, i.e. if
    the signal will be strong enough to reach j.
    As soon as J2 is found, we trigger a break and calculate the path from back
    to front; this is easily done by keeping track of which node first led to
    each other node.
    
    
    Runtime analysis
    
    We use collections.deque for the queue of nodes, which allows for quick
    pops and appends on either end. With this, as seen in the notes, a breadth-
    first search runs in O(N+M) time for N nodes and M edges.
    
    
    Efficiency analysis

    A linear leading-order runtime is often a good sign of an efficient algorithm.
    A breadth-first search is also generally seen as a good way to perform searches
    through random finite graphs.
    
    Using deque to quickly pop an element from the front of the queue, and
    to quickly build the path from back to front using appendleft, is much
    quicker than the list equivalent (which requires shifting all elements
    over by one each time). So quite a few operations are cut out here relative
    to a list-based approach.
    
    The break is triggered as soon as possible, potentially cutting out a few
    unnecessary operations.
    
    With no extra assumptions about the graph, we cannot do better than linear
    time; as the target node could be anywhere in the graph, in the worst case,
    we will have to iterate through all the nodes and edges at least once. So
    in that sense this algorithm and its runtime can be seen as efficient.
    """
    
    Q = deque([J1]) #queue
    E = deque([J1]) #explored nodes
    firstnodes = dict() #entry at i is the first node to 'find' i
    targetfound = 0 #triggers a break when target is found
    
    while len(Q)>0:
        
        node = Q.popleft() #Pick out front of queue
        
        for (i, Lij) in A[node]: #for each of node's neighbours:
            
            if (Lij*a0 >= amin) and not(i in E): #if it's unexplored and L is high enough 
                Q.append(i) #add it to queue
                E.append(i) #mark it as explored
                firstnodes[i] = node #save 'node' as the first one to discover i
                
                #If i is the target, trigger a break
                if i == J2:
                    targetfound = 1
                    break
        
        if targetfound == 1:
            break
    
    L = deque([]) #Will hold a path from J1 to J2 if one exists
    
    if targetfound == 1:
        #build path backwards using firstnodes
        L.append(J2)
        while not L[0] == J1:
            L.appendleft(firstnodes[L[0]])

    return list(L)





#2.2

def a0min(A,amin,J1,J2):
    """
    Question 1.2 ii)
    Find minimum initial amplitude needed for signal to be able to
    successfully propagate from node J1 to J2 in network (defined by adjacency list, A)

    Input:
    A: Adjacency list for graph. A[i] is a sub-list containing two-element tuples (the
    sub-list my also be empty) of the form (j,Lij). The integer, j, indicates that there is a link
    between nodes i and j and Lij is the loss parameter for the link.

    amin: Threshold for signal boost
    If a>=amin when the signal reaches a junction, it is boosted to a0.
    Otherwise, the signal is discarded and has not successfully
    reached the junction.

    J1: Signal starts at node J1 with amplitude, a0
    J2: Function should determine min(a0) needed so the signal can successfully
    reach node J2 from node J1

    Output:
    (a0min,L) a two element tuple containing:
    a0min: minimum initial amplitude needed for signal to successfully reach J2 from J1
    L: A list of integers corresponding to a feasible path from J1 to J2 with
    a0=a0min
    If no feasible path exists for any a0, return output as shown below.

    ---Discussion---
    
    
    Algorithm description:
    
    We use a modified Dijkstra's algorithm.
    The queue initially contains only J1. We use 'maxvals' to keep track of the
    highest Lij we can 'stay above' in travelling to each node. We then repeatedly
    choose the node in the queue with the highest maxval, mark it as done, and
    update all its neighbours with new maxvals if possible.
    
    This is repeated until J2 has the highest maxval of nodes in the queue, or
    until the queue is empty.
    If the queue became empty, J2 was never found and we return an empty list.
    Otherwise we build a path from back to front, similarly to the previous part.
    
    
    Runtime and efficiency analysis:
    
    Setting up the queue, maxvals, etc. is all O(1) or O(N).
    Each node is popped out of the queue at most once, so each edge is considered
    at most twice. Hence the second 'for' loop below contributes O(M) operations.
    
    The only part here with non-linear runtime is finding the node in Q with
    the highest maxval. We could use a binary heap to achieve this in
    O(log_2(N)) operations, giving a final runtime of O((N+M)log_2(N)) as seen
    in the notes.
    Without the use of heaps, the first 'for' loop below contributes O(N^2)
    operations, and this term dominates the leading-order runtime.
    
    The use of deque again speeds up queue management and path building, as in
    the previous part.
    Here 'bestnodes' is a dictionary whose entry at 'i' is the node that provided
    i with its maxval. With this it is easy to build a path that maximises the
    signal at the end.
    """
    
    Q = deque([J1]) #Queue
    maxvals = deque(len(A)*[0]) #greatest of all the minimum L's of paths leading to i
    maxvals[J1] = 1
    done = deque([]) #Nodes whose maxvals are finalised
    bestnodes = dict() #entry at i is the node that gave i its current maxval
    
    while len(Q)>0:
        
        #Pick out node in Q with the highest maxval
        topval = 0
        for i in Q:
            if maxvals[i]>=topval:
                topval = maxvals[i]
                node = i
        
        #If it's J2, we're done
        if node == J2:
            break
        
        #Else remove it from Q and mark it as done
        Q.remove(node)
        done.append(node)
        
        for (i, Lij) in A[node]: #For each of node's neighbours:
            
            if not(i in done):
                #Improve its maxval if possible
                if min(maxvals[node],Lij)>maxvals[i]:
                    maxvals[i] = min(maxvals[node],Lij)
                    bestnodes[i] = node #save 'node' as a potential step in the path
                
                #Add it to the queue if it's not there already
                if not(i in Q):
                    Q.append(i)
                
    #If there are no paths to J2 that don't go through a zero Lij, return -1, []
    if maxvals[J2] == 0:
        return (-1, [])
    
    #Else return the least value needed to stay above amin the whole time, and a path from J1 to J2 that does this
    else:
        path = deque([J2])
        while not path[0] == J1:
            path.appendleft(bestnodes[path[0]])
        
        #Need a0*maxvals[J2] == amin for the least possible a0. So a0min = amin/maxvals[J2]
        return (amin/maxvals[J2], path) 





if __name__=='__main__':
    #add code here if/as desired
    L=None #modify as needed