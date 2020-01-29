"""M345SC Homework 3, part 1
Christophe Jefferies 01202145
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from scipy.signal import hann
import time



def nwave(alpha,beta,Nx=256,Nt=801,T=200,display=True):
    """
    Question 1.1
    Simulate nonlinear wave model

    Input:
    alpha, beta: complex model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of |g| when true

    Output:
    g: Complex Nt x Nx array containing solution
    """

    #generate grid
    L = 100
    x = np.linspace(0,L,Nx+1)
    x = x[:-1]
    
    #generate wavenumbers, shifted to match the output of fft
    n = np.fft.fftshift(np.arange(-Nx/2, Nx/2))
    k = 2*np.pi*n/L

    def RHS(f,t,alpha,beta):
        """Computes dg/dt for model eqn.,
        f[:N] = Real(g), f[N:] = Imag(g)
        Called by odeint below
        """
        
        g = f[:Nx]+1j*f[Nx:]
        
        #Find Fourier coefficients, multiply by -(1j*k)**2, and inverse transform
        c = np.fft.fft(g) #No need to multiply/divide by Nx as we don't need the coefficients
        d2g = np.fft.ifft(-k**2*c)
        
        dgdt = alpha*d2g + g -beta*g*g*g.conj()
        df = np.zeros(2*Nx)
        df[:Nx] = dgdt.real
        df[Nx:] = dgdt.imag
        return df

    #set initial condition
    g0 = np.random.rand(Nx)*0.1*hann(Nx)
    f0=np.zeros(2*Nx)
    f0[:Nx]=g0
    t = np.linspace(0,T,Nt)

    #compute solution
    f = odeint(RHS,f0,t,args=(alpha,beta))
    g = f[:,:Nx] + 1j*f[:,Nx:]

    if display:
        plt.figure()
        plt.contour(x,t,g.real)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of Real(g)')

    return g





def analyze():
    """
    Question 1.2
    Add input/output as needed

    ---Discussion---
    
    Figures 0 and 1 show the contour of the real part of g, for all x and up to
    T=200, for cases A and B respectively. At first glance, both look quite
    chaotic; there are unpredictable variations at all times, and no signs of
    global periodicity. They do however each have somewhat consistent global
    behaviour.
    
    Upon closer inspection, case B's behaviour contains many periodic patches.
    (We ignore the presence of this behaviour in the initial transient, t<50,
    which depends on initial conditions.) In other words, the real part of g
    often locally fluctuates back and forth, though exactly when this happens
    still seems very random. Case A shows a very weak version of this
    behaviour, but nothing worthy of note.
    
    Both cases continue to behave as seen here for larger times, so we don't
    show plots for larger T here as they are effectively the same.
    Similarly, the imaginary part of g has visually identical contour behaviour
    to its real part, so there is no new information there.
    
    For comparing cases A and B, varying Nx and Nt is not especially
    informative; these only affect the 'resolution' of the resulting plots, but
    do not display any new behaviour. (Lower Nx and Nt sometimes cause strange
    behaviour, but this is likely a side-effect of the behaviour of odeint.)
    
    Figure 2 shows the autospectral density of the two cases (i.e. the energy
    of each wavenumber) at t=200, well after the transient. Both cases
    have very high energies for lower wavenumbers, and effectively negligible
    energies for high wavenumbers. Case B's energies seem to die away faster,
    suggesting that the signal can more easily be broken down in to fewer
    waves, and is in some sense less chaotic. However, case B's values are much
    more scattered at the extremities than case A's, suggesting that case A's
    autospectral density might fit some well-behaved distribution.
    
    Figures 3 and 4 are one representation of how the a wavenumber's energy
    varies through time.
    Figure 3 shows that the leading-term energy varies quite smoothly for
    both cases, i.e. if the energy of the leading wavenumber is high at time t,
    then it is likely to be high at time t+1. The correlation for case B is
    especially 'tight', perhaps relating to the periodic patches seen before;
    if these relate to, or are captured by, the leading frequency, then the
    correlation will be made clear on a plot like this.
    The maximum energies for case B are also higher, suggesting that lots of
    case B's behaviour can be summarised by a smaller number of waves.
    Figure 4 makes the same comparison for a higher wavenumber (50); this time
    the energies are about the same size, but case B's correlation is still
    visibly 'tighter', suggesting that even the less important frequencies
    vary more smoothly for this case.
    
    Figure 5 shows how two similar initial conditions diverge from each other.
    In this case a small amount of random noise was added to the previous
    initial condition, and then the L1-norm difference (i.e. component-wise
    difference) between the two resulting waves. We would expect faster
    divergence from a more chaotic case.
    Ignoring the initial transient (say t<80), we can see that there is very
    fast divergence for case A, and slightly slower for case B. If we
    take behaviour for t>400 as a 'standard' for the initial condition being
    completely lost, then case A has done this by t=100, whereas case B takes
    up to t=200. This might again suggest that B is slightly less chaotic,
    though in the long run dependence on initial conditions is still
    completely lost.
    
    Finally, figure 6 shows the correlation sum for both cases, a measure of
    the fractal dimension of the data. Here we find it for g's values at x=50,
    across all t>50. The outcome depends on our choice of epsilon; the middle
    portion of the plot can be taken as a good indicator of the true behaviour.
    Both cases have almost identical gradient of around 1.20, which corresponds
    to having very similar fractal dimension. Case A's is still a little
    higher, suggesting again that it is the slightly more chaotic case.

    In summary, both cases are quite chaotic, but case B shows hints of
    periodicity, breaks down slightly better in to waves, and is overall
    slightly less chaotic than case A.
    
    """
    
    #Make plots
    
    #Setup
    L = 100
    Nx = 256 #Fix Nx and Nt as changing them is not informative for comparing cases A and B
    Nt = 801
    n = np.fft.fftshift(np.arange(-Nx/2, Nx/2))
    k = 2*np.pi*n/L
    x = np.linspace(0,L,Nx+1)[:-1]
    
    def RHS(f,t,alpha,beta):
        g = f[:Nx]+1j*f[Nx:]
        c = np.fft.fft(g)
        d2g = np.fft.ifft(-k**2*c)
        dgdt = alpha*d2g + g -beta*g*g*g.conj()
        df = np.zeros(2*Nx)
        df[:Nx] = dgdt.real
        df[Nx:] = dgdt.imag
        return df
    
    #Set an initial condition, compute both solutions
    g0 = np.random.rand(Nx)*0.1*hann(Nx)
    f0 = np.zeros(2*Nx)
    f0[:Nx] = g0    
    t = np.linspace(0,200,Nt)
    fA = odeint(RHS,f0,t,args=(1-2j,1+2j))
    fB = odeint(RHS,f0,t,args=(1-1j,1+2j))
    gA = fA[:,:Nx] + 1j*fA[:,Nx:]
    gB = fB[:,:Nx] + 1j*fB[:,Nx:]
    coefficientsA = np.fft.fft(gA)/Nt
    coefficientsB = np.fft.fft(gB)/Nt
    
    #fig0
    plt.figure()
    plt.contour(x,t,gA.real)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Contours of Real(g) for case A \n Nx = 256, Nt = 801, T = 200 \n Christophe Jefferies \n Plot by analyze')
    plt.savefig("fig0.png", bbox_inches='tight')
    
    #fig1
    plt.figure()
    plt.contour(x,t,gB.real)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Contours of Real(g) for case B \n Nx = 256, Nt = 801, T = 200 \n Christophe Jefferies \n Plot by analyze')
    plt.savefig("fig1.png", bbox_inches='tight')

    #fig2
    plt.figure()
    timeindex = 200 #Will pick out a time after the initial transient
    from scipy.signal import welch #For nice windowing
    wA, PA = welch(gA[timeindex])
    wB, PB = welch(gB[timeindex])
    wA, wB = wA*Nx/200, wB*Nx/200
    line1, = plt.semilogy(wA,PA,'r.',label='Case A')
    line2, = plt.semilogy(wB,PB,'g.',label='Case B')
    plt.xlabel('Frequency')
    plt.ylabel('Power spectral density')
    plt.title('Autospectral density at time ' + str(timeindex) + ' for cases A and B \n Christophe Jefferies \n Plot by analyze')
    plt.legend(handles=[line1, line2])
    plt.savefig("fig2.png", bbox_inches='tight')
    
    #fig 3-4
    for wavenum in [0, 50]:
        energiesA = np.abs(coefficientsA[50:][:,wavenum]) #discard initial transient, and only track one wavenumber
        energiesB = np.abs(coefficientsB[50:][:,wavenum])

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(energiesA[:-1], energiesA[1:], '.', label='Case A')
        plt.xlabel('$|c_{%i}|^2$ at time $t_n$' % wavenum)
        plt.ylabel('$|c_{%i}|^2$ at time $t_{n+1}$' % wavenum)
        plt.title('Case A')
        
        plt.subplot(1, 2, 2)
        plt.plot(energiesB[:-1], energiesB[1:], 'g.', label='Case B')
        plt.xlabel('$|c_{%i}|^2$ at time $t_n$' % wavenum)
        plt.ylabel('$|c_{%i}|^2$ at time $t_{n+1}$' % wavenum)
        plt.title('Case B')
        
        plt.suptitle('Iterative plot of $|c_{%i}|^2$' % wavenum + '\n Christophe Jefferies \n Plot by analyze')
        plt.savefig('fig' + str(int((wavenum/50)+3)) + '.png', bbox_inches='tight')
    
    #fig5
    g02 = g0 + np.random.rand(Nx)*0.01*hann(Nx)
    f02 = np.zeros(2*Nx)
    f02[:Nx] = g02
    fA2 = odeint(RHS,f02,t,args=(1-2j,1+2j))
    fB2 = odeint(RHS,f02,t,args=(1-1j,1+2j))
    gA2 = fA2[:,:Nx] + 1j*fA2[:,Nx:]
    gB2 = fB2[:,:Nx] + 1j*fB2[:,Nx:]
    plt.figure()
    plt.semilogy(sum(((gA.real-gA2.real)**2).transpose()), label='Case A')
    plt.semilogy(sum(((gB.real-gB2.real)**2).transpose()), label='Case B')
    plt.xlabel('t')
    plt.ylabel('Squared component-wise difference')
    plt.title('Squared component-wise difference in outcome for similar starting conditions')
    plt.legend()
    plt.savefig('fig5.png', bbox_inches='tight')
    
    #fig6
    #Setup
    epsnum = 100 #number of points to try
    eps = np.exp(np.linspace(-5, np.log(3), epsnum)[::-1])
    N = len(gA[:,0])
    
    #Calculate correlation sums and plot them
    plt.figure()
    for (vals, case) in [(gA, 'A'), (gB, 'B')]:
        vals = vals[50:, 50] #ignore initial transient and pick out x=50
        real, imag = vals.real, vals.imag
        A = np.vstack([real, imag]).T
        D = scipy.spatial.distance.pdist(A) #All pairwise distances in the data
        C = epsnum*[0] #Ready to be filled in
        
        for i in range(epsnum):
            C[i] = D[D<eps[i]].size
        
        C = [2*k/(N**2 - N) for k in C]
        plt.loglog(eps, C, label=case)
    
    #Finish plot
    plt.title('Correlation sum $C(\epsilon)$ for cases A and B \n Christophe Jefferies \n Plot by analyze')
    plt.xlabel('$\epsilon$')
    plt.ylabel('$C(\epsilon)$')
    plt.legend()
    plt.savefig('fig6.png', bbox_inches='tight')





def wavediff():
    
    L=100
    alpha, beta = 1-1j, 1+2j #Only using case B in this part
    
    #g as found by nwave
    g = nwave(alpha, beta, display=False)
    g = g[401] #Will only use the wave at t=100
    
    #FOURIER DIFFERENTIATION
    def Fourier(Nx=256):
        n = np.fft.fftshift(np.arange(-Nx/2, Nx/2))
        k = 2*np.pi*n/L #wavenumbers
        c = np.fft.fft(g[::256//Nx]) #coefficients
        dg = np.fft.ifft(1j*k*c).real #dg/dx for all times
        return dg
    
    #FINITE DIFFERENCE DIFFERENTIATION
    def FD(Nx=256):
            
        #Generate a matrix A s.t. Ag' is the LHS of the finite difference system
        #Needs to be in matrix ordered diagonal form, ready for sp.linalg.solve_banded
        A = np.array([[3/8, 1, 3/8],]*Nx).transpose()
        A[0][0], A[0][1], A[-1][-2], A[-1][-1] = 0, 3, 3, 0 #Edge cases

        #Generate a matrix C s.t. Cg is the RHS of the finite difference system    
        h = L/Nx #Timestep
        a2h, b4h, c6h = (25/16)/(2*h), (1/5)/(4*h), (-1/80)/(6*h) #a, b, c as specified, already divided through as necessary
        entrylist = [a2h, b4h, c6h, -c6h, -b4h, -a2h, -b4h, -c6h, c6h, b4h] #negative entries are for edge cases, which use periodicity of g
        indexlist = [1, 2, 3, Nx-3, Nx-2, -1, -2, -3, -(Nx-3), -(Nx-2)] #corresponding diagonals of C
        C = scipy.sparse.diags(entrylist, indexlist, shape=(Nx, Nx), format='lil')
        #Correct edge cases
        edgelist = [-17/(6*h), 3/(2*h), 3/(2*h), -1/(6*h)]
        for i, entry in enumerate(edgelist):
            C[0,i], C[Nx-1, Nx-1-i] = entry, -entry
        
        #Calculate dg/dx at time t=100
        right = C@g[::256//Nx]
        dg = scipy.linalg.solve_banded((1,1), A, right).real
        return dg
    
    g = g.real #Only need real part for plots
    dg = Fourier() #For comparisons
    
    #fig7, fig8
    for index, Nx in enumerate([256, 64]):
        #Real parts of g and dg (from Fourier and FD) at t=100, and Nx=256, 128, 64 respectively
        dg1 = Fourier(Nx)
        dg2 = FD(Nx)
        x = np.linspace(0, L, Nx+1)[:-1]
        plt.figure()
        if index==0:
            line1, = plt.plot(np.linspace(0, L, 256+1)[:-1], g, 'b', label='g')
        line2, = plt.plot(x, dg1, 'r', label='Fourier differentiation')
        line3, = plt.plot(x, dg2, 'g', label='Finite difference')
        if index==1:
            line4, = plt.plot(np.linspace(0, L, 256+1)[:-1], dg, label='Fourier differentiation (Nx=256)')
        plt.xlabel('t')
        plt.ylabel('Real part')
        plt.legend()
        plt.title('Real parts of g and $\partial g / \partial x$ with Fourier differentiation and a compact finite difference method \n Nx = ' + str(Nx) + ' \n Christophe Jefferies \n Plot by wavediff')
        plt.savefig('fig' + str(index+7) + '.png', bbox_inches='tight')
    
    #fig9
    plt.figure()
    for index, Nx in enumerate([256, 128, 64]):
        smooth = dg[0::256//Nx] #Pick out entries to allow element-wise comparison
        dg1 = Fourier(Nx)
        dg2 = FD(Nx)
        x = np.linspace(0, L, Nx+1)[:-1]
        plt.subplot(1, 3, index+1)
        plt.plot(x, (smooth-dg1), label = 'Fourier differentiation')
        plt.plot(x, (smooth-dg2), label = 'Finite difference')
        plt.xlabel('t')
        plt.ylabel('Squared difference')
        plt.title('Nx = ' + str(Nx))
        plt.legend()
    plt.suptitle('Deviation from "smooth" Fourier differentiation of "rough" $\partial g / \partial x$ from both methods \n Christophe Jefferies \n Plot by wavediff')
    plt.savefig('fig9.png', bbox_inches='tight')
    
    #fig10
    Nxvals = [2**i for i in range(3, 9)]
    N = 10000
    data = np.array(2*[len(Nxvals)*[0]])
    
    for i, Nx in enumerate(Nxvals):
        
        #Time Fourier() for various Nx
        total=0
        for j in range(N):
            start = time.time()
            Fourier(Nx=Nx)
            end = time.time()
            total += end-start
        data[0, i] = total
        
        #Time FD() for various Nx()
        total=0
        for j in range(N):
            start = time.time()
            FD(Nx=Nx)
            end = time.time()
            total += end-start
        data[1, i] = total
    
    #Make figure
    plt.figure()
    plt.semilogx(Nxvals, data[0,:], 'x-', label = 'Fourier()')
    plt.semilogx(Nxvals, data[1,:], 'x-', label = 'FD()')
    plt.xlabel('Nx')
    plt.ylabel('Time for ' + str(N) + ' runs (seconds)')
    plt.title('Execution time for ' + str(N) + ' runs')
    plt.legend()
    plt.savefig('fig10.png', bbox_inches='tight')
        
    
    """
    ---Discussion---
    
    We consider the test wave generated at t=100, for case B as above, and
    compare Fourier differentiation and a compact finite difference (FD) method
    for finding its first derivative.
    
    Figure 7 shows the real part of the testwave, and the real part of its
    first derivative as found by both methods with Nx=256. The two results
    almost entirely coincide, with a small disagreement at each end (discussed
    more below). In the middle, both results seems reasonable; the derivative
    passes through zero whenever the wave has a turning point, and also has the
    correct sign.
    
    We can compare how the two approaches fare when presented with fewer data
    points, i.e. how well they 'interpolate' when using a smaller Nx.
    The plot as above but with Nx=128 is almost identical; figure 8 shows the
    results with Nx=64, at which point the two differ by a more noticeable
    amount. The derivative found by Fourier differentiation with Nx=256 is also
    superimposed as a standard (which we assume to be quite accurate since
    the two methods agreed so closely for Nx=256).
    
    This time the two methods disagree almost every time the derivative has a
    turning point. Fourier differentiation tends to overshoot during rapid
    fluctuations, whilst the FD method tends to track these more precisely.
    Whenever the derivative sharply changes sign, neither method is really
    able to resolve this behaviour; this is perhaps just a result of being
    provided with too few data points to gain accuracy at certain points. That
    said, Fourier differentiation does tend to reach slightly 'further in' to
    these peaks and crevices, so assuming g is reasonably smooth, Fourier
    differentiation seems to handle using fewer data points slightly better.
    
    We can more concretely measure this difference by plotting how 'far' each
    result is from the Nx=256 standard. Figure 9 demonstrates this, where the
    'distance' is taken to be the L1-norm difference, i.e. the sum of the
    squares of the component-wise differences bewteen the two arrays.
    
    At Nx=256, the Fourier method will trivially not deviate at all from
    itself; the leftmost subplot just shows by how much the two approaches
    disagree, in particular at the edges.
    With Nx=128, the FD method tends to either coincide well with the smoother
    curve, or have a 'spike' of inaccuracy. This might represent inability of
    the method to handle sharp changes in the derivative at this resolution,
    whilst it handles not-too-severe behaviour well.
    The Fourier method more uniformly has small error, but no spikes of large
    magnitude. So it seems the Fourier method is less variable (i.e. less
    affected by quick fluctuations in g), but the finite difference method
    is best when the derivative is not too sharply changing.
    At Nx=64, both methods disagree on the order of 10e-1, suggesting that in
    fact this many data points is not enough to expect any consistent accuracy
    from either method.
    In each plot, the endpoints of the FD method seem to fluctuate away from
    the desired answer, despite the use of a one-sided method close to the
    edges. Perhaps this adjustment is still not enough to resolve detail from
    little data, again making the Fourier method slighty superior.
    
    We can also compare the computational efficiency of the two approaches.
    Figure 10 shows the execution time for 10000 runs of each method, for
    various Nx. The FD method's runtime seems to increase with Nx, whilst the
    Fourier method actually happens too quickly at this level to produce any
    non-negligible measurement.
    
    The implementations above are not perfectly optimised, but arguably
    sufficiently so to make this a fair comparison of the methods. The Fourier
    differentiation boils down to calling np.fft twice (whose speed is beyond
    our control), whilst the FD method comes down to quickly building some
    matrices and solving a banded system (which is handled here with scipy).
    Assuming that these modules are well-built for purpose, we could conclude
    that Fourier differentiation is the more computationally efficient method
    in this case.
         
    """

    return data





if __name__=='__main__':
    #Add code here to call functions above and
    #generate figures you are submitting
    
    analyze()
    wavediff()


