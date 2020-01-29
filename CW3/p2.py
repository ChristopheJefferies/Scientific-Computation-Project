"""M345SC Homework 3, part 2
Christophe Jefferies 01202145
"""

import numpy as np
import networkx as nx
import scipy.linalg
import scipy.integrate



def growth1(G,params=(0.02,6,1,0.1,0.1),T=6):
    
    """
    
    Question 2.1
    Find maximum possible growth, G=e(t=T)/e(t=0) and corresponding initial
    condition.
    
    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth
    
    Output:
    G: Maximum growth
    y: 3N-element array containing computed initial condition where N is the
    number of nodes in G and y[:N], y[N:2*N], y[2*N:] correspond to S,I,V
    
    
    
    ---Discussion---
    
    The system is entirely linear, so we can represent it as y' = My for
    M a 3Nx3N matrix. This has general solution y(t) = Ay0 with A = exp(Mt),
    and the perturbation energy is then e(t)=||y(t)||^2.
    
    From lectures, for a fixed ||y0||^2, we can maximise e(T) by choosing
    y0 as the eigenvector of (A^t)(A) with the most positive eigenvalue.
    The growth is then e(T)/e(0) where e(0)=||y0||^2.
    
    Since the system is entirely linear, multiplying y0 through by a constant
    will have the same effect on y(t), which cancels out in the growth. So we
    can assume WLOG that y0 is unit, and that e(0) = 1. Hence the growth is
    also maximised by such an eigenvector.
    
    In this case the growth will simply be the most positive eigenvalue L of 
    A^tA: e(0) = 1, so
    
    e(T)/e(0) = ||Ay0|| = (Ay0)^t(Ay0) = y0^t(A^tA)y0 = y0^tLy0 = L||y0||^2 = L
    
    as y0 is unit.    
    
    """
    
    a,theta,g,k,tau=params
    N = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]
    Pden = np.zeros(N)
    Pden_total = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)
    #-------------------------------------
    G=0
    y = np.zeros(3*N)
    
    #Construct M
    eye = np.identity(N)
    Z = np.zeros((N,N))
    M = np.block([[F - (g+k+tau)*eye, a*eye, Z],
            [theta*eye, F - (a+k+tau)*eye, Z],
            [-theta*eye, Z, F + (k-tau)*eye]])
    
    #Set A = exp(MT)
    A = scipy.linalg.expm(M*T)
    
    #Singular value decomposition
    _, S, V = np.linalg.svd(A)
    
    #First row of V is the leading eigenvector of A^tA
    y = V[0] #unit by default
    
    #The leading eigenvalue (and hence growth) is the leading singular
    #value squared
    G = S[0]**2
    
    return G, y





def growth2(G,params=(0.02,6,1,0.1,0.1),T=6):
    
    """
    
    Question 2.2
    Find maximum possible growth, G=sum(Ii^2)/e(t=0) and corresponding initial
    condition.

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    G: Maximum growth
    y: 3N-element array containing computed initial condition where N is the
    number of nodes in G and y[:N], y[N:2*N], y[2*N:] correspond to S,I,V
    
    

    ---Discussion---
    
    The numerator in this part is ||I||^2 where I is the 'middle third' of y.
    With y = Ay0 as above, if we let B be the matrix obtained by discarding the
    top and bottom thirds of A, we have I = By0. The norm of this can then be
    maximised using the same theory as above.
    
    In fact V does not affect the change of S or I at all. We could also
    discard the rightmost third of B, find its 2N-long leading eigenvector, and
    append zeros at the end to find the initial condition. However
    np.linalg.svd finds these entries to be effectively zero (<=1e-16 in
    most cases), so for simplicity we leave B in its Nx3N state here.
    
    """
    
    a,theta,g,k,tau=params
    N = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    Pden_total = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)
    #-------------------------------------
    G=0
    y = np.zeros(3*N)
    
    #Construct M
    eye = np.identity(N)
    Z = np.zeros((N,N))
    M = np.block([[F - (a+k+g)*eye, a*eye, Z],
            [theta*eye, F - (a+k+tau)*eye, Z],
            [-theta*eye, Z, F + (k-tau)*eye]])
    
    #Set A = exp(MT)
    A = scipy.linalg.expm(M*T)
    
    #Discard top and bottom thirds of A
    B = A[N:2*N]
    
    #SVD and leading eigenvector of B^tB
    _, S, V = np.linalg.svd(B)
    y = V[0]
    
    #Growth is again the leading eigenvalue
    G = S[0]**2

    return G,y





def growth3(G,params=(2,2.8,1,1.0,0.5),T=6):
    
    """
    
    Question 2.3
    Find maximum possible growth, G=sum(Si Vi)/e(t=0)
    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    G: Maximum growth
    
    

    ---Discussion---
    
    The sum over nodes i of SiVi is not a sum of squares, so is not the norm of
    a vector. Hence we cannot directly apply the same theory as above.
    However, it is a difference of squares: (Si+Vi)^2 - (Si-Vi)^2 = 4SiVi,
    so divide through this by 4 = 2**2.
    
    Letting S, I, V be the thirds of the vector y, if we can write S+V and
    S-V as B1y, B2y for some matrices B1, B2, then we have
    4.Σ(SiVi) = ||S+V||-||S-V|| = (B1y)^t(B1y)-(B2y)^t(B2y)
    = y^t(B1^tB1 - B2^tB2)y.
    
    There are such B1 and B2: let y = Ay0 be the full system solution as above.
    Write A as three Nx3N matrices K1, K2, K3 on top of each other. Then
    (K1+K3)y = S+V, and (K1-K3)y = S-V.
    
    Note that B=(B1^tB1 - B2^tB2) is real and symmetric, so can be written C^tC
    for some matrix C (this is also clear from its SVD and its evalues being
    real and nonnegative). So we can apply the theory above and find the
    y0 that maximises ||Cy0||, and hence the growth.
    (In fact the expression for B simplifies down to 2(K1^tK3 + K3^tK1).)
    
    """
    
    a,theta,g,k,tau=params
    N = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    Pden_total = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)
    #-------------------------------------
    G=0

    #Construct M
    eye = np.identity(N)
    Z = np.zeros((N,N))
    M = np.block([[F - (g+k+tau)*eye, a*eye, Z],
            [theta*eye, F - (a+k+tau)*eye, Z],
            [-theta*eye, Z, F + (k-tau)*eye]])
    
    #Set A = exp(MT)
    A = scipy.linalg.expm(M*T)
    
    #Construct K1, K3, and B as outlined above
    K1, K3 = A[:N,:], A[2*N:,:]
    B = 2*(K1.T@K3 + K3.T@K1)
        
    #Find the growth (leading eigenvalue scaled by 4)
    evals, _ = np.linalg.eig(B)
    index = np.argmax(evals)
    G = evals[index]/4
    
    return G





def Inew(D):
    
    """
    
    Question 2.4

    Input:
    D: N x M array, each column contains I for an N-node network

    Output:
    I: N-element array, approximation to D containing "large-variance"
    behavior
    
    

    ---Discussion---
    
    We can regard our NxM data as N data points in M-dimensional space. The
    question is effectively asking us to project this down to a 1-dimensional
    space, whilst preserving as much of the data's variance as possible.
    
    This is exactly the purpose of PCA. The principal component is the leading
    eigenvector of the variance-covariance matrix DD^t. By projecting the data
    down on to this component, we have a 1-dimensional representation of
    the data whose total variance is as high as possible (out of all linear
    projections in to one dimension).
    
    Below, we first normalise each column of the data so that different
    scaling between organisms does not skew the data. Then we find the leading
    eigenvector of DD^t and project the data on to it with a dot product.
    
    """
    
    N,M = D.shape
    
    #Shift data column mean to zero
    D -= np.outer(np.ones((N,1)),D.mean(axis=0))
    
    #Find eigenvectors of DD^t
    U, _, _ = np.linalg.svd(D.T)
    
    #Project data on to leading eigenvector
    I = np.dot(D, U[:,0]).T

    return I



if __name__=='__main__':
    z=0
    #add/modify code here if/as desired
    
    #N,M = 100,5
    #G = nx.barabasi_albert_graph(N, M, seed=1)
    
    #D = np.loadtxt(u'q22test.txt')



