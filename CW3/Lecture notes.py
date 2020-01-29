"""

Lecture 10

Solving non-linear ODEs numerically

Can discretize time and march forward, computing f at each step using the last

Explicit Euler method: linear Taylor series bit
Smaller time steps gives more accuracy
For dy/dt = y : this is unconditionally unstable, i.e. always blows up

Runge-Kutta: better accuracy and stability. Uses evaluations at substeps
Still 'explicit'
Given example of 4th-order RK
Conditionally stable for small enough steps
Requires four evaluations of the function for each step
Error is deltat^4, rather than deltat for explicit/implicit EM

Variable-time methods adjust timestep to ensure a certain accuracy
Can still be explicit or implicit
Explicit variable timestep: good for linear dynamics (single time scale)
Implicit variable timestep e.g. odeint: good for strongly nonlinear systems, stiff, with multiple time scales (?)

Variable-time RK: explicit
Stiffness' is of concern rather than stability
Available in scipy.integrate (as ODE23 or ODE45?)

Explicit time-marching methods can struggle with nonlinear ODE systems
Implicit methods can be better

Implicit Euler method: like explicit EM but RHS 'contains' LHS, see end. Still fixed time-step
Smaller time step gives more accuracy
For dy/dt = y: unconditionally stable
Requires solution of system of equations at each step

scipy.integrate.odeint:
    Must specify initial conditions, time span, RHS function
    Can specify error tolerance, times to return, much more



Lecture 11

Stability of dy/dt = ay, Re(a)<0 for RK and ex/im EM

Explanation of how variable-time methods choose their timestep

Use odeint for systems of ODEs and single PDEs
Use RK45 for systems of PDEs

---Data science---

Given a matrix A, find x with |x|=1 s.t. |Ax| is maximised
Evector of A^tA with largest evalue is best

A faster approach to finding the evectors and evalues of A^tA

Should use scipy.sparse.linalg when working with large sparse matrices

PCA intuition, maths, and implementation

Low-rank approximations of matrices via SVD

Can we fill in missing data from a dataset?
E.g. recommendation systems: can capture the information in a ratings matrix, with missing entries
Can try to minimise the rank of the filled-in version, only making "small" changes to existing entries
Tackled here via SVD, singular values, cost function, Lagrange multipliers, regularization, gradient iteration



Lecture 15, 16

Data arranged in 1D arrays
We can look for trends by decomposing the 'signal' in to waves, i.e. Fourier transforming
Assume the function and all derivatives are continuous on the domain
Continuous periodic function: coefficients a_n are o(n^(-k))

Can find Fourier coefficients for 1D data with the Discrete Fourier Transform
Available in numpy.fft, scipy.fftpack
Given explicit truncated sums for transform and inverse transform at the jth point in time

Finding Fourier coefficients for Y (discrete, i.e. a list)
c = np.fft.fft(Y)
c = np.fft.fftshift(c)/Nt #puts coefficients in a sensible order
n = np.arange(-Nt/2,Nt/2)

Use rfft to find only for n>=0

Plot them
plt.figure()
plt.plot(n,np.abs(c),'x')
plt.xlabel('mode number, n')
plt.ylabel('$|c_n|$')
plt.grid()

The square of the nth Fourier coefficient is the 'energy' of the mode with frequency n/T, T the timespan of the signal (total length)
Need the signal timespan to be an integer multiple of each frequency for a clean outcome
The highest frequency that can be 'resolved' is (Nt/2 - 1)/T
Rule of thumb: >2 points per period of the highest-frequency component

Signals generally aren't periodic... can use 'windowing' (not really explained). Makes it look more like a simple wave, though we lose energy
No need to do this ourselves... scipy.signal.welch breaks the signal in to overlapping segments, windows each bit, and takes averages
This produces an estimate of the 'autospectral density'
In [201]: w2,Pxx2 = sig.welch(Y2)
In [203]: w2 = w2*Nt/T
In [206]: plt.semilogy(w2,Pxx2)

So we need Δt < 2/fmin to resolve the highest-frequency components
We also need T >> 1/fmax to make sure slow components are captured. Slow components often have the largest amplitude
FFT has better runtime than DFT: O(Nlog_2(N)) vs O(N^2)



Lecture 17

Explicit finite differences method for an exponential. Less accurate and needs more points than Fourier analysis
Implicit finite difference: tackled here by a general 'stencil', represented in a matrix. Many free variables
Gives better accuracies, higher orders are possible. Can also impose constraints on the modified wavenumber, giving even better
But which is fastest for a given accuracy? It boils down to Ax=b for A a pentadiagonal matrix.
Use scipy.diags - see slides 17-18
Trickier but better is scipy.linalg.solve_banded - see slides 19-20

These quasi-spectral finite difference methods: can optimise efficiency for a desired accuracy, and lower memory usage than
explicit FD and Fourier. However it's global (like spectral/Fourier) and tricky to apply to complex geometries, whatever that means



Lecture 18

More linear data analysis with fancy methods (logistic map), finding fractal dimension of data, Lorenz systems and attractors

"""