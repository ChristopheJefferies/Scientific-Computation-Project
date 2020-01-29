"""M345SC Homework 2, part 2
Christophe Jefferies 01202145
"""

import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#2.1

def model1(G,x=0,params=(50,80,105,71,1,0),tf=6,Nt=400,display=False):
    """
    Question 2.1
    Simulate model with tau=0

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    tf,Nt: Solutions Nt time steps from t=0 to t=tf (see code below)
    display: A plot of S(t) for the infected node is generated when true

    x: node which is initially infected

    Output:
    S: Array containing S(t) for infected node
    """
    a,theta0,theta1,g,k,tau=params
    tarray = np.linspace(0,tf,Nt+1)

    def RHSlinear(y, t):
        S, I, V = y
        theta = theta0 + theta1*(1-np.sin(2*np.pi*t))
        return [a*I - (g+k)*S, theta*S*V - (k+a)*I, k*(1-V) - theta*S*V]
    
    y0 = [0.1, 0.05, 0.05]
    sol = odeint(RHSlinear, y0, tarray)
    S = sol[:,0]
    
    if display:
        plt.plot(tarray, S)
        plt.title('Fraction of spreaders at node ' + str(x) + ' against time \n Christophe Jefferies \n Plot by modelN')
        plt.xlabel('Time')
        plt.ylabel('Fraction of cells which are spreaders')
        plt.show()

    return S





#2.2

from scipy import sparse

def modelN(G,x=0,params=(50,80,105,71,1,0.01),tf=6,Nt=400,display=False):
    """
    Question 2.2
    Simulate model with tau != 0

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    tf,Nt: Solutions Nt time steps from t=0 to t=tf (see code below)
    display: A plot of S(t) for the infected node is generated when true

    x: node which is initially infected

    Output:
    Smean,Svar: Array containing mean and variance of S across network nodes at
                each time step.
    
    ---Discussion---
    
    #Assume nodes are in order, so adjacency matrix is in node order
    #Would surely be more meaningful to ignore origin node from the statistics
    """
    
    a,theta0,theta1,g,k,tau=params
    tarray = np.linspace(0,tf,Nt+1)
    
    #Do the hard work here
    N = len(G.nodes())
    A = nx.adjacency_matrix(G).todense() #adjacency matrix
    B = np.multiply(A, sum(A)).transpose() #ij entry is qi*Aij
    F = tau*np.nan_to_num(np.divide(B, sum(B))) #ij entry is as needed for F. nan_to_num fixed any division by zero
    bigmat = sparse.block_diag((F, F, F)).toarray()
    
    #To compute RHS of equations
    def RHS(y,t):
        """Compute RHS of model at time t
        input: y should be a 3N x 1 array containing with
        y[:N],y[N:2*N],y[2*N:3*N] corresponding to
        S on nodes 0 to N-1, I on nodes 0 to N-1, and
        V on nodes 0 to N-1, respectively.
        output: dy: also a 3N x 1 array corresponding to dy/dt

        ---Discussion---
        
        Once the right-hand side of the equations are written in vector form,
        they can be greatly simplified.
        For instance, the Fji's in the summation terms all become tau, so here
        these terms are added in one go at the end.
        Similarly 'bigmat' is a (pre-calculated) block-diagonal matrix of three
        copies of F, which allows us to quickly add the first term from each
        summation in one go.
        All that remains are the non-summation terms, which can be quickly
        formed with some vector additions and subtractions.
        
        dSi/dt operation count estimation:
            Extract S from y: N operations
            Preallocate dy: N
            Find theta and theta*S*V: these do not contribute to finding dSi/dt
            a*I: N
            (g+k): 1
            (g+k)*S: N
            a*I - (g+k)*S: N
            tau*y: N
            bigmat[:N,:].dot(S): 3N*N = 3N^2
            Add results in 'return' line: 2N
            
            Total: 3N^2 + 8N + 1

        We could also include some non-summation terms in bigmat, such as the
        (g+k)Si term in dSi/dt. This would remove 2N+1 operations from the above,
        but would require many more (O(N^2)) operations to form the resulting
        bigmat in each call to the function (even if using scipy.sparse).
        In the approach below, as much as possible is put in to the matrix before
        the function call, so this takes fewer operations.
        """
        
        S, I, V = y[:N], y[N:2*N], y[2*N:] #vectors S, I, V
        theta = theta0 + theta1*(1-np.sin(2*np.pi*t))
        dy = np.zeros(3*N) #output ready to be filled in
        tSV = theta*S*V #this term appears twice, calculate it here once
        
        dy[:N] = a*I - (g+k)*S
        dy[N:2*N] = tSV - (k+a)*I
        dy[2*N:] = k*(1-V) - tSV
        
        return dy - tau*y + bigmat.dot(y)
    
    #Set initial conditions and solve
    y0 = 3*N*[0]
    y0[x], y0[x+N], y0[x+2*N] = 0.1, 0.05, 0.05
    sol = odeint(RHS, y0, tarray)
    
    #Find mean and variance of S across all nodes, for all times
    Smean = [sol[i][:N].mean() for i in range(Nt+1)]
    Svar = [sol[i][:N].var() for i in range(Nt+1)]
    
    #Plots
    if display:
        for (stat, name) in [(Smean, 'Mean value'), (Svar, 'Variance')]:
            plt.figure(figsize=(8,8))
            plt.plot(tarray, stat)
            plt.title(name + ' of S across all nodes against time \n Christophe Jefferies \n Plot by modelN')
            plt.xlabel('Time')
            plt.ylabel(name + ' of S across all nodes')
            plt.show()

    return Smean, Svar





#2.3

def diffusion():
    """Analyze similarities and differences
    between simplified infection model and linear diffusion on
    Barabasi-Albert networks.
    Modify input and output as needed.

    ---Discussion---
    
    The key points we will discuss here are:
        - The mean of S, I, and V across all nodes, for various tau and D
        - How long it takes for nodes to become 'infected'
        - The effect of inital conditions on behaviour
    
    We will compare how these behave on Barabasi-Albert graphs, for both a
    simplified version of the infection model from part 2.2 (call this model 1),
    and a linear diffusion model (call this model 2).
    
    
    KEY POINT 1: mean of S, I, and V across all nodes, for various tau and D
    (and simple starting conditions).
    
    The parameter tau controls the strength of the flux matrix in model 1, whilst
    the parameter D controls the strength of the linear diffusion in model 2.
    Therefore it seems sensible to compare how these two parameters affect the
    models' behaviour.
    (We have chosen not to investigate the effect of varying theta0 in model 1,
    as there is no clear equivalent in model 2 with which to compare it.)
    
    Small values of tau or D: very little flux or diffusion
    
        Figure 0 shows the mean value of S, I, and V in model 1 for some small
        values of tau. The initial conditions are one infected node at (0.1, 0.05, 0.05)
        as in part 2.1, the rest zero.
        #plotmeans(RHS1, y01, [0.05, 0.1, 1], 100, 0)
        (We ignore the source node itself when calculating the means, as this
        tends to skew the outcome, and in any case we are more interested in the
        spread of infected cells to the rest of the network than how the source
        node behaves.)
        
        We see that the numbers of spreaders and infected cells grow together,
        whilst the number of vulnerable cells stays at or close to zero.
        Looking at the simplified infection model's equations, this makes sense;
        the summation terms will have no effect of interest, as they are independent
        of one another and so just cause identical growth of each cell type
        based on degree.
        The nonlinear theta0*Si*Vi term is what triggers the behaviour - it is
        negative in dVi/dt, meaning <V> will tend to shrink, whilst it is positive
        in dIi/dt which will cause <I> to grow.
        In particular, the bigger S is, the more <I> will grow, meaning its growth
        can keep up with that of <S>. The exact ratio of growth speeds is decided
        by theta=theta0, which does not depend on time (and is fixed at 80 here).
        
        Figure 1 shows the same statistics, but for model 2.
        #plotmeans(RHS2, y01, [0.01, 0.05, 0.1], 100, 1)
        This time <S>, <I>, and <V> all follow the same behaviour, except <S> is
        consistently twice as large as <I> and <V>. This makes sense considering
        the linear diffusion model; growth of S, I, and V do not depend on their
        current states, but only on the Laplacian matrix L (and the constant D).
        Since <S> starts twice as high as the other two, this remains the case
        under linear growth.
        More precisely, we are repeatedly applying a constant matrix to the system,
        so the system is tending to an equilibrium state at which the means do
        not change. This corresponds to a distribution of S, I, and V which is
        invariant under applying linear growth, i.e. corresponds to eigenvectors
        of the system.
        The lines in the plot flatten out over time as the values converge to
        that of the equilibrium state.
        (Note: below, model 2 is implemented with odeint rather than explicitly
        finding eigenvectors, as it leads to nicer plot-making code for both models)
    
    Medium and high values of tau or D: stronger flux or diffusion
        
        Figures 2 and 3 show the same behaviour but with larger values of tau and D.
        #plotmeans(RHS1, y01, [5, 30, 50], 100, 2)
        #plotmeans(RHS2, y01, [1, 5, 10], 100, 3)
        In model 1, a larger tau inevitably exaggerates the fast growth of <S>,
        and hence of I. However, as tau becomes very large, we get a sudden burst
        of vulnerable cells before they die out. This is most likely because at
        first, the huge flux dominates any of the system's non-linear terms, meaning
        high-degree nodes far from the infected source can spawn lots of cells
        of all kinds. As soon as spreaders start to reach these nodes, however,
        the vulnerable cells are smothered by the theta*S*V term and quickly
        die out.
        
        In model 2, a larger value of D simply speeds up the linear diffusion
        process, which on the plots corresponds to quickly converging to the
        equilibrium state. This helps confirm the analysis above.
    
    
    #KEY POINT 2: how long it takes for nodes to become infected
    (for simple starting conditions and a few tau or D)
    
    Figures 4 and 5 (for models 1 and 2) show the fraction of all nodes which
    are over K% infected, for various K. (This is over a larger timeframe than above.)
    #plotfracs(RHS1, y01, 0.05, 4)
    #plotfracs(RHS2, y01, 0.005, 5)
    (tau and D control similar behaviour, but have different scaling factors;
    they have been chosen here to display typical non-degenerate behaviours.)
    
    In model 1, we see an initial burst in which many cells immediately become
    more that 0.01% infected. After that, cells' infected percentages seem to
    increase roughly linearly, with eventually all cells reaching 0.1% infection.
    Model 2 shows almost identical behaviour, though it takes a bit longer
    for the cells to become infected to each percentage.
    
    The fact that both models show fairly steady increase of infected nodes is
    not surprising; nodes tend to become infected when their neighbours are, which
    will propagate through the graph.
    A closer look shows that each line tends to sit still for a fraction of a second
    before jumping up. This suggests that nodes tend to reach the next percentage
    of infection in batches, i.e. that a connected cluster of nodes will all become
    more infected at once (in both models).
    In particular, since Barabasi-Albert models are scale-free, we would expect
    some nodes with high degrees; once one of these becomes infeced, its many direct
    neighbours will likely be infected soon as well, explaining the jumps on the plots.


    #KEY POINT 3: effect of initial condition on behaviour
    
    We will consider:
        1) the case when five nodes are initially infected
        2) the case when only the highest-degree node is initially infected.
    
    Case 1: Figures 6 and 7 show the means of S, I, and V with five initially
    infected nodes, in models 1 and 2 respectively.
    #plotmeans(RHS1, y02, [0.1, 1, 5], 100, 6)
    #plotmeans(RHS2, y02, [0.01, 0.1, 1], 100, 7)
    These plots are most meaningful when compared to figures 0 and 1. However,
    the values of the initially-infected nodes were not excluded this time when
    calculating the means, so the graphs must be interpreted carefully.
    
    In essence, we see the same behaviour (in terms of growth and ratios of the
    three means), but occurring much faster. This is to be expected, as a greater
    initial infection in five random nodes will inevitably be able to spread more
    quickly than from one node.
    In particular, in model 2 (figure 7), the snap to the equilibrium state is
    very fast even for quite small parameter values.
    
    The effect is more easily seen by observing the fraction of infected nodes;
    figures 8 and 9 (in comparison to 4 and 5) show a much faster infection rate
    than before.
    #plotfracs(RHS1, y02, 0.05, 8)
    #plotfracs(RHS2, y02, 0.005, 9)
    
    
    Case 2: highest-degree node only is infected
    As mentioned, the scale-free property of Barabasi-Albert models means they
    are likely to have outlier nodes of very high degree. We would expect the
    infection to happen faster than when a random node is initially infected.
    Figures 10 to 13 are as above, with this initialy condition.
    #plotmeans(RHS1, y03, [0.1, 1, 5], 100, 10)
    #plotmeans(RHS2, y03, [0.01, 0.1, 1], 100, 11)
    #plotfracs(RHS1, y03, 0.05, 12)
    #plotfracs(RHS2, y03, 0.005, 13)
    
    Plots 10 and 11 do show what we might expect; the same behaviour as in plots
    0 to 3 (comparing the same tau) but faster and more exaggerated.
    
    Interestingly, plots 12 and 13 show that the time it takes for nodes to become
    infected is similar to that for a low-degree node initially infected, but
    slower than for five initially infected nodes.
    
    This might indicate that it is not the degree of infected nodes that affects
    how quickly the illness spreads, but how many nodes are infected to start with.
    (This effect could also happen if we were unlucky and randomly chose a high-
    degree node earlier, but by taking several runs this was ensured not to be
    the case.)
    """

    #Barabasi-Albert graph and timestep array
    N, M = 100, 5
    BA = nx.barabasi_albert_graph(N, M)
    tf, Nt = 40, 400
    tarray = np.linspace(0,tf,Nt+1)
    
    #Setup for simplified infection model
    theta0 = 80
    A = nx.adjacency_matrix(BA).todense() #adjacency matrix
    B = np.multiply(A, sum(A)).transpose() #ij entry is qi*Aij
    F = np.nan_to_num(np.divide(B, sum(B))) #not multiplied by tau yet as we will vary it
    bigmat = sparse.block_diag((F, F, F)).toarray()
    
    #Outputs RHS of simplified infection model, with given tau
    def RHS1(y, t, tau):
        S, V = y[:N], y[2*N:] #don't need I to find the simplified model
        dy = np.zeros(3*N)
        dy[N:2*N] = theta0*S*V
        dy[2*N:] = - theta0*S*V
        return dy - tau*y + tau*bigmat.dot(y)
    
    #Outputs RHS of linear diffusion model
    L = np.diag([BA.degree(i) for i in range(N)]) - A #Laplacian matrix
    bigL = sparse.block_diag((L,L,L))
    def RHS2(y, t, D):
        return -D*bigL*y
    
    #Solve specified system with given y0, and tau or D. Return things I want to plot
    def solve(func, y0, tD, Ifrac):
        #func is one of RHS1, RHS2
        #tD is either tau (RHS1) or D (RHS2)
        sol = odeint(func, y0, tarray, args=(tD,))
        
        #Find mean of each cell type at all times, not including the source node
        Smean = [sol[i][1:N].mean() for i in range(Nt+1)]
        Imean = [sol[i][N+1:2*N].mean() for i in range(Nt+1)]
        Vmean = [sol[i][2*N+1:].mean() for i in range(Nt+1)]
        
        #Find fraction of nodes whose fraction of infected cells is over Ifrac
        fracreached = [sum(i>Ifrac for i in sublist)/float(N) for sublist in sol[:,N:2*N]]
        
        return (Smean, Imean, Vmean, fracreached)
    
    #Plot means of S, I, V for specified model, y0, cutoff time, and tau or D. Save result
    def plotmeans(func, y0, tDs, cutoff, fignum):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        
        sols = [[] for i in range(9)]
        sols[0], sols[1], sols[2], _ = solve(func, y0, tDs[0], 0.1)
        sols[3], sols[4], sols[5], _ = solve(func, y0, tDs[1], 0.1)
        sols[6], sols[7], sols[8], _ = solve(func, y0, tDs[2], 0.1)
        sols = [sublist[:cutoff] for sublist in sols]
        x = tarray[:cutoff]
        
        l1, l2, l3 = axs[0].plot(x, sols[0], x, sols[1], '.', x, sols[2])
        l4, l5, l6 = axs[1].plot(x, sols[3], x, sols[4], '.', x, sols[5])
        l7, l8, l9 = axs[2].plot(x, sols[6], x, sols[7], '.', x, sols[8])
        
        axs[0].set_ylabel('Mean over all nodes in the network')
        for i in range(3):
            axs[i].set_xlabel('Time')
            if func == RHS1:
                axs[i].set_title("tau = " + str(tDs[i]))
            if func == RHS2:
                axs[i].set_title("D = " + str(tDs[i]))
        
        fig.legend((l1, l2, l3), ('S', 'I', 'V'), 'center right')
        if func==RHS1:
            fig.suptitle('Simplified infection model: mean of S, I, and V over all nodes \n Christophe Jefferies \n Plot by diffusion: plotmeans')
        if func==RHS2:
            fig.suptitle('Linear diffusion model: mean of S, I, and V over all nodes \n Christophe Jefferies \n Plot by diffusion: plotmeans')
        plt.savefig("fig" + str(fignum) + ".png", bbox_inches='tight')
    
    
    #Plot fraction of infected nodes for specified model, y0, and tau or D. Save result
    def plotfracs(func, y0, tD, fignum):
        
        #Simplified infection model
        fracs = [[] for i in range(10)] #will store output
        for value in range(10):
            (_, _, _, fracs[value]) = solve(func, y0, tD, (value+1)/float(10000))
        
        #Make and save plot
        plt.figure(figsize=(8,8))
        for i in range(10):
            plt.plot(tarray, fracs[i], label=str((i+1)/float(100)))
        if func==RHS1:
            plt.title('Simplified infection model: fraction of nodes over K% infected (tau = ' +str(tD)+ ')\n Christophe Jefferies \n Plot by diffusion: plotfracs')
        if func==RHS2:
            plt.title('Linear diffusion model: fraction of nodes over K% infected (D = ' +str(tD)+ ')\n Christophe Jefferies \n Plot by diffusion: plotfracs')
        plt.xlabel('Time')
        plt.ylabel('Fraction of nodes over K% infected')
        plt.legend(title='K')
        plt.savefig("fig" + str(fignum) + ".png", bbox_inches='tight')
        
    
    #Define initial conditions
    
    #Basic initial condition as in 2.1
    piece1 = [1]+ (N-1)*[0.0]
    y01 = piece1 + 2*([i/2 for i in piece1])
    
    #k nodes infected
    k = 5
    piece2 = [0.0 for i in range(N)]
    for i in range(k):
        piece2[int(i*N/float(k))] = 1
    y02 = piece2 + 2*([i/2 for i in piece2])
    
    #Infect a node with highest degree
    maxdeg = max([deg for (node, deg) in list(BA.degree())])
    topnode = [node for node in range(N) if BA.degree(node)==maxdeg][0]
    piece3 = [0.0 for i in range(N)]
    piece3[topnode] = 1
    y03 = piece3 + 2*([i/2 for i in piece3])


    plotmeans(RHS1, y01, [0.05, 0.1, 1], 100, 0)
    plotmeans(RHS2, y01, [0.01, 0.05, 0.1], 100, 1)
    plotmeans(RHS1, y01, [5, 30, 50], 100, 2)
    plotmeans(RHS2, y01, [1, 5, 10], 100, 3)
    plotfracs(RHS1, y01, 0.05, 4)
    plotfracs(RHS2, y01, 0.005, 5)
    plotmeans(RHS1, y02, [0.1, 1, 5], 100, 6)
    plotmeans(RHS2, y02, [0.01, 0.1, 1], 100, 7)
    plotfracs(RHS1, y02, 0.05, 8)
    plotfracs(RHS2, y02, 0.005, 9)
    plotmeans(RHS1, y03, [0.1, 1, 5], 100, 10)
    plotmeans(RHS2, y03, [0.01, 0.1, 1], 100, 11)
    plotfracs(RHS1, y03, 0.05, 12)
    plotfracs(RHS2, y03, 0.005, 13)


if __name__=='__main__':
    diffusion()