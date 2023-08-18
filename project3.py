import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy.linalg import solve_banded
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from time import time
from scipy.signal import welch
from scipy.spatial.distance import pdist
from scipy.stats import linregress
#use scipy as needed
#----------------------
# Code for part 1
#----------------------

def ufield(r,th,u,levels=50):
    """Displays velocity field stored in 2D array, u, at
    one instant in time. Uses polar grid data stored in 1D arrays r and th.
    Use/modify/ignore as needed.
    """
    Nth = th.size
    Nr = r.size
    thn = np.zeros(Nth+1)
    thn[:-1]=th;thn[-1]=th[0]
    un = np.zeros((Nr,Nth+1))
    un[:,:-1] = u;un[:,-1]=un[:,0]
    thg,rg = np.meshgrid(thn,r)
    xg = rg*np.cos(thg)
    yg = rg*np.sin(thg)
    plt.figure()
    plt.contourf(xg,yg,un,levels)
    plt.axis('equal')
    return None

def part1q1():
    """
    Question 1.1
    Add input/output as needed
    """

    #-------------------------
    #Load data
    #t: times
    #r: radial positions
    #th: angular positions
    #U: velocity field, u(r,th,t)
    temp = np.load('data1.npz')
    t = temp['t'];r = temp['r'];th = temp['theta'];U = temp['U']
    temp = None
    #-------------------------


    r_inds = np.array([np.argmin(np.abs(r-0.1)), np.argmin(np.abs(r-0.5))]) # indices for r=0.1 and r=0.5

    # Plot 2D plots for r=0.1 and r=0.5
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0, 0].plot(t, U[r_inds[0], :, :].T)
    ax[0, 0].set_title('r = 0.1, time vs velocity')
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('Velocity')
    ax[0, 1].plot(t, U[r_inds[1], :, :].T)
    ax[0, 1].set_title('r = 0.5, time vs velocity')
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('Velocity')

    ax[1, 0].plot(th, U[r_inds[0], :, :])
    ax[1, 0].set_title('r = 0.1, theta vs velocity')
    ax[1,0].set_xlabel('Theta (rad)')
    ax[1,0].set_ylabel('Velocity')

    ax[1, 1].plot(th, U[r_inds[1], :, :])
    ax[1, 1].set_title('r = 0.5, theta vs velocity')
    ax[1,1].set_xlabel('Theta (rad)')
    ax[1,1].set_ylabel('Velocity')

    plt.tight_layout()
    plt.show()

    # Plot 3D plots for r=0.1 and r=0.5

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

    # Compute the minimum and maximum values of U for both values of r
    vmin = np.min(U[r_inds])
    vmax = np.max(U[r_inds])

    # Create the meshgrid for the 3D plots
    T, TH = np.meshgrid(t, th)

    # plot first surface plot
    surf1 = axs[0].plot_surface(T, TH, U[r_inds[0],:,:], cmap='plasma', edgecolor='none', vmin=vmin, vmax=vmax)
    axs[0].set_title(f'r = {r[r_inds[0]]}')
    #set labels
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Theta (rad)')
    axs[0].set_zlabel('Velocity')
    axs[0].view_init(elev=30, azim=45)

    # plot second surface plot
    surf2 = axs[1].plot_surface(T, TH, U[r_inds[1],:,:], cmap='plasma', edgecolor='none', vmin=vmin, vmax=vmax)
    axs[1].set_title(f'r = {r[r_inds[1]]}')
    #set labels
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Theta (rad)')
    axs[1].set_zlabel('Velocity')
    axs[1].view_init(elev=30, azim=45)
    # Add a colorbar
    cbar_ax = fig.add_axes([0.25, -0.1, 0.5, 0.05])

    fig.colorbar(surf2, ax=axs, shrink=0.6, aspect=10, cax=cbar_ax, orientation='horizontal')
    #common title
    fig.suptitle('Velocity field', fontsize=16)

    fig.tight_layout()

    return None

def part1q2C(U,inputs=()):
    """
    Question 1.2: Construct one or more arrays from U
    that can be used by part1q2E
    Input:
        U: 3-D data array
        inputs: can be used to provide other input as needed
    Output:
        arrays: a tuple containing the arrays produced from U
    """
    #load data
    temp = np.load('data1.npz')
    t = temp['t'];r = temp['r'];th = temp['theta'];U = temp['U']
    temp = None

    # calculate dimensions of data array
    Nt = len(t)
    Nr = len(r)
    Nth = len(th)

    # reshape data array
    A = U.reshape((Nr*Nth,Nt)).astype(float)

    # calculate dimensions of reshaped data array
    M,N = np.shape(A)

    # subtract mean along columns
    Am = np.mean(A, axis=1)
    A = A - Am[:, np.newaxis]

    # perform truncated SVD on centered data array
    U,S,WT = np.linalg.svd(A)

    # print rank of SVD
    print("rank(A):", S[S>1e-11].size)

    # calculate variance of new variables and plot cumulative variance explained
    rank = S[S>1e-11].size
    vars = S**2
    cum_var_prop = np.cumsum(vars) / sum(vars) * 100
    plt.plot(cum_var_prop[:rank], 'x--')
    plt.xlabel('Number of modes')
    plt.ylabel('Variance (%)')
    plt.title('Cumulative Variance Explained of new variables')
    plt.show()

    # plot singular values 
    plt.plot(vars[:rank], 'x--')
    plt.xlabel('Number of modes')
    plt.ylabel('Singular value')
    plt.title('Singular values of new variables')
    plt.show()

    arrays = (U, S, WT, Am, A, (Nr,Nth,Nt)) 
    return arrays


def part1q2E(arrays,n_modes=4):
    """
    Question 1.2: Generate numpy array with same shape as U (see part1q2E above)
    that has some meaningful correspondence to U
    Input:
        arrays: tuple generated by part1q2C
        inputs: can be used to provide other input as needed
    Output:
        Unew: a numpy array with the same shape as U
    """

    U, S, WT, Am, A, dims = arrays
    Nr, Nth, Nt = dims

    # Keep only the first four columns of U and first four rows of WT
    U_trunc = U[:, :n_modes]
    WT_trunc = WT[:n_modes, :]

    # Keep only the first four singular values and form a diagonal matrix
    S_trunc = np.diag(S[:n_modes])

    # Reconstruct the centered data using the truncated SVD components
    A_approx = np.dot(U_trunc, np.dot(S_trunc, WT_trunc))

    # Add back the row means to get the approximate original data
    A_approx += Am[:, np.newaxis]

    Unew = A_approx.reshape((Nr,Nth,Nt))


    # Calculate the memory usage of the original data array
    memory_usage_A = A.nbytes

    # Calculate the memory usage of the compressed data array
    memory_usage_A_approx = (U_trunc.nbytes 
                            + S_trunc.nbytes 
                            + WT_trunc.nbytes 
                            + Am.nbytes 
                            + np.prod(S_trunc.shape)*sys.getsizeof(np.float64(0)))

    # Calculate the percentage reduction in memory usage
    percent_reduction = 100 * (1 - memory_usage_A_approx / memory_usage_A)

    # Print the results
    print(f"Memory usage of original data array: {memory_usage_A} bytes")
    print(f"Memory usage of compressed data array: {memory_usage_A_approx} bytes")
    print(f"Percentage reduction in memory usage: {percent_reduction:.2f}%")

    return Unew

def part1q3():
    """
    Question 1.3
    Add input/output as needed
    """

    #-------------------------
    #Load data
    #U: matrix with missing data
    #R1,R2,R3: three "repaired" matrices
    temp = np.load('data2.npz')
    U = temp['U'];R1 = temp['R1'];R2=temp['R2'];R3=temp['R3']
    temp = None
    #-------------------------

    # Compute ranks using SVD
    s1 = np.linalg.svd(R1, compute_uv=False)
    s2 = np.linalg.svd(R2, compute_uv=False)
    s3 = np.linalg.svd(R3, compute_uv=False)
    rank_R1 = np.sum(s1 > 1e-10)
    rank_R2 = np.sum(s2 > 1e-10)
    rank_R3 = np.sum(s3 > 1e-10)

    mse_R1 = np.mean((U[U!=-1000] - R1[U!=-1000])**2)
    mse_R2 = np.mean((U[U!=-1000] - R2[U!=-1000])**2)
    mse_R3 = np.mean((U[U!=-1000] - R3[U!=-1000])**2)

    bool = [U!=-1000]
    U = U[bool]
    R1 = R1[bool]
    R2 = R2[bool]
    R3 = R3[bool]

    ssim_R1 = compare_ssim(U, R1, data_range=U.max()-U.min())
    ssim_R2 = compare_ssim(U, R2, data_range=U.max()-U.min())
    ssim_R3 = compare_ssim(U, R3, data_range=U.max()-U.min())

    psnr_R1 = compare_psnr(U, R1, data_range=U.max()-U.min())
    psnr_R2 = compare_psnr(U, R2, data_range=U.max()-U.min())
    psnr_R3 = compare_psnr(U, R3, data_range=U.max()-U.min())

    print('Rank(R1):', rank_R1, 'Rank(R2):', rank_R2, 'Rank(R3):', rank_R3)
    print('MSE(R1):', mse_R1, 'MSE(R2):', mse_R2, 'MSE(R3):', mse_R3)
    print('SSIM(R1):', ssim_R1, 'SSIM(R2):', ssim_R2, 'SSIM(R3):', ssim_R3)
    print('PSNR(R1):', psnr_R1, 'PSNR(R2):', psnr_R2, 'PSNR(R3):', psnr_R3)

    return None #modify as needed

#----------------------
# Code for part 2
#----------------------
def model1d(a=0.028,b=0.053,L = 5,Nx=256,Nt=8001,T=4000,display=False,method='RK45',bc=0):
    """
    Question 2.1
    Simulate 2-species chemical reaction model

    Input:
    a,b: model parameters
    L: domain size
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of f when true
    method: method used by solve_ivp
    bc:
        bc=0, homogeneous Neumann boundary conditions
        bc =/= 0, periodic boundary conditions

    Output:
    f,g: Nt x Nx arrays containing solution
    """

    #generate grid
    x = np.linspace(0,L,Nx)
    dx = x[1]-x[0]
    dx2inv = 1/dx**2

    #model constants
    d1 = 2e-5
    d2 = 1e-5

    def RHS(t,y):
        """
        RHS of model equations used by solve_ivp
        homogeneous Neumann boundary conditions
        """
        n = y.size//2
        f = y[:n]
        g = y[n:]

        #Compute 2nd derivatives
        d2f = (f[2:]-2*f[1:-1]+f[:-2])*dx2inv
        d2g = (g[2:]-2*g[1:-1]+g[:-2])*dx2inv

        #Construct RHS
        fg2 = f*g**2
        dfdt = d1*d2f - fg2[1:-1] - f[1:-1]*a + a
        dgdt = d2*d2g + fg2[1:-1] - (a+b)*g[1:-1]
        dy = np.zeros(2*n)
        dy[1:n-1] = dfdt
        dy[n+1:-1] = dgdt

        #Enforce boundary conditions
        a1,a2 = 4/3,-1/3
        dy[0] = a1*dy[1]+a2*dy[2]
        dy[n-1] = a1*dy[n-2]+a2*dy[n-3]
        dy[n] = a1*dy[n+1]+a2*dy[n+2]
        dy[-1] = a1*dy[-2]+a2*dy[-3]

        return dy

    def RHS2(t,y):
        """
        RHS of model equations used by solve_ivp,
        periodic boundary conditions
        """
        n = y.size//2
        f = y[:n]
        g = y[n:]

        #Compute 2nd derivatives
        d2f = np.zeros_like(f)
        d2g = np.zeros_like(g)
        d2f[1:-1] = (f[2:]-2*f[1:-1]+f[:-2])*dx2inv
        d2g[1:-1] = (g[2:]-2*g[1:-1]+g[:-2])*dx2inv

        d2f[0] = (f[1]-2*f[0]+f[-1])*dx2inv
        d2g[0] = (g[1]-2*g[0]+g[-1])*dx2inv
        d2f[-1] = d2f[0]
        d2g[-1] = d2g[0]

        fg2 = f*g**2
        dfdt = d1*d2f - fg2 - f*a + b
        dgdt = d2*d2g + fg2 - (a+b)*g
        dy = np.zeros(2*n)
        dy[:n] = dfdt
        dy[n:] = dgdt
        return dy

    #initial condition
    d = 1-4*(a+b)**2/a
    f0 = 0.5*(1+np.sqrt(d))
    g0 = 0.5*a/(a+b)*(1-np.sqrt(d))
    y0 = np.zeros(2*Nx)
    y0[:Nx] = f0 +0.1*np.cos(4*np.pi/L*x) + 1*np.cos(8*np.pi/L*x)
    y0[Nx:] = g0 +0.1*np.cos(2*np.pi/L*x) + 1*np.cos(6*np.pi/L*x)

    t = np.linspace(0,T,Nt)

    #compute solution
    print("running simulation...")
    if bc==0:
        out = solve_ivp(RHS,[t[0],t[-1]],y0,t_eval = t,rtol=1e-6,method=method)
    else:
        out = solve_ivp(RHS2,[t[0],t[-1]],y0,t_eval = t,rtol=1e-6,method=method)

    print(out.message)
    y =out.y
    f = y[:Nx,:]
    g = y[Nx:,:]
    print("finished simulation")
    if display:
        plt.figure()
        plt.contour(x,t,g.T)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of f')

    return t,x,f,g

def part2q1analyze():
    """
    Question 2.1
    Add input/output as needed
    """
    t, x, f, g = model1d(T=20000,display=True)
    #cut off transient part
    transient = 500
    f = f[:,transient:]
    t = t[transient:]

    
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    #plot f for every x is 0


    ax[0,0].plot(t,f[np.where(x==1)[0][0],:],label='x=1')


    ax[0,0].set_xlabel('t')
    ax[0,0].set_ylabel('f')

    ax[0,0].set_title('x=1')


    im = ax[0,1].imshow(f, cmap='coolwarm', origin='lower', aspect='auto')
    ax[0,1].set_xlabel('t')
    ax[0,1].set_ylabel('x')
    cbar = fig.colorbar(im)
    cbar.set_label('Concentration')
    #set correct t and x labels
    ax[0,1].set_xticks(np.linspace(0, f.shape[1], 5))
    ax[0,1].set_xticklabels(np.linspace(t[0], t[-1], 5))
    ax[0,1].set_yticks(np.linspace(0, f.shape[0], 5))
    ax[0,1].set_yticklabels(np.linspace(x[0], x[-1], 5))


    N= len(x)
    c = np.fft.fft(f[:,np.where(t==6000)[0]].T)/N
    n = np.arange(-N/2,N/2)

    ax[1,0].semilogy(n,np.fft.fftshift(np.abs(c)).T, 'x-')
    ax[1,0].set_ylabel("$|f_x|$")
    ax[1,0].set_xlabel('x')
    ax[1,0].set_xlim(-10e-5,126)
    ax[1,0].set_ylim(10e-7,1)
    ax[1,0].set_title('t=3000')


    #plot f_n vs f_n+1 scatter
    ax[1,1].scatter(f[np.where(x==0)[0][0],:-1],f[np.where(x==0)[0][0],1:], s =5, label='x=0', alpha=0.1)
    ax[1,1].scatter(f[np.where(x==1)[0][0],:-1],f[np.where(x==1)[0][0],1:], s =5, label='x=1', alpha=0.1)
    ax[1,1].scatter(f[np.where(x==2)[0][0],:-1],f[np.where(x==2)[0][0],1:], s =5, label='x=2', alpha=0.1)
    ax[1,1].set_xlabel('$f^*_n$')
    ax[1,1].set_ylabel('$f^*_{n+1}$')
    leg = ax[1,1].legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)


    dt = t[1] #timestep
    fxx,Pxx = welch(f,fs=1/dt)
    #plot fxx against pxx for x in 0,1, 2

    ax[2,0].plot(fxx,Pxx[np.where(x==0)[0][0],:],label='x=0')
    ax[2,0].plot(fxx,Pxx[np.where(x==1)[0][0],:],label='x=1')
    ax[2,0].plot(fxx,Pxx[np.where(x==2)[0][0],:],label='x=2')
    ax[2,0].legend()
    #log scale 
    ax[2,0].set_yscale('log')
    # set labels
    #set x ticks
    ax[2,0].set_xticks(np.linspace(0, fxx[-1], 5))
    #round x ticks to 4 decimal places
    ax[2,0].set_xticklabels(np.round(np.linspace(fxx[0], fxx[-1], 5),4))
    ax[2,0].set_xlabel('Frequency')
    ax[2,0].set_ylabel('$P_{xx}$')


    y1 = f[:,:-1:2]
    y2 = f[:,1::2]
    A = np.vstack([y1,y2]).T
    D = pdist(A)
    eps = np.logspace(-2, 1, num=50)
    C = np.zeros(len(eps))
    for i in range(len(eps)):
        C[i] = D[D<eps[i]].size


    ax[2,1].loglog(eps,C)
    ax[2,1].set_xlabel('$\epsilon$')
    ax[2,1].set_ylabel('$C(\epsilon)$')

    # Estimate the best-fit slope using linear regression for epsilon between 0.1 and 10
    slope, intercept, r_value, p_value, std_err = linregress(np.log10(eps[20:45]), np.log10(C[20:45]))
    #plot line of best fit between 0.1 and 5
    ax[2,1].plot(eps[20:45],10**(intercept)*eps[20:45]**slope, label='best fit, slope = '+str(slope), color='black', linestyle='--')
    ax[2,1].legend()
    #xlim between 0.1 and 10
    ax[2,1].set_xlim(0.08,9.9)
    plt.tight_layout()
    #save figure
    plt.show()
    return None #modify as needed

def part2q2(f,N,L):
    """
    Compute d2f/dx2 and d2f/dy2 of N x N array f on square equispaced grid
    with N points in each direction.
    f[i,j] corresponds to data at x[j] and y[i]

    Output:
    fxx: N x N numpy array; 2nd-derivative with respect to x computed using Fourier transforms
    assume underlying function is periodic in x, f(x+L)=f(x)

    fyy: N x N numpy array; 2nd-derivative with respect to y computed using implicit Finite
    difference scheme
    """

    #Generate grid (x,y are not explicitly needed here)
    x = np.linspace(0,L,N+1)
    x = x[:-1]
    y = x.copy()
    h = x[1]-x[0]
    h2inv = 1/h**2
    xg,yg = np.meshgrid(x,y)


    def diffx(f,L):
        """
        Question 2.2)
        Input:
            f: real array whose 2nd derivative will be computed along each row
            L: domain size
        Output:
            d2f: (real) second derivative of f computed with discrete Fourier transforms
        """
        n = np.arange(f.shape[0]/2 + 1)
        k = 2*np.pi*n
        k2m = -k**2/L**2
        fk = np.fft.rfft(f,axis=1)
        d2f = np.fft.irfft(k2m*fk,axis=1)
        return d2f #modify as needed
    

    def diffy(f,h2inv):
        """
        Question 2.2)
        Input:
            f: real array whose 2nd derivative will be computed along each column
            h2inv: 1/h^2 where h is the grid spacing
        Output:
            d2f: second derivative of f computed with implicit FD scheme
        """
        #coefficients for interior points
        alpha = 2/11
        a = 12/11
        b = 3/11 

        #coefficients for near-boundary points
        alpha_bc = 10
        a_bc,b_bc,c_bc,d_bc,e_bc = (145/12,-76/3,29/2,-4/3,1/12)
        

        m, n = f.shape
        """
        mid_diag = np.ones(n)
        up_diag = alpha * np.ones(n-1)
        up_diag[:2] = alpha_bc * np.ones(2)
        up_diag[-1] = 0
        low_diag = alpha * np.ones(n-1)
        low_diag[0] = 0
        low_diag[-2:] = alpha_bc * np.ones(2)
        #make tridiagonal matrix using numpy
        diagonals = [low_diag,mid_diag,up_diag]
        offsets = [-1,0,1]
        ab = np.diag(diagonals[0], offsets[0]) + np.diag(diagonals[1], offsets[1]) + np.diag(diagonals[2], offsets[2])
        """

        ab = np.zeros((3,n))
        #main diagonal
        ab[1,:] = 1.0
        #upper diagonal
        ab[0,1:3] = alpha_bc
        ab[0,3:-1] = alpha
        #lower diagonal
        ab[2,1:-2] = alpha
        ab[2,-3:-1] = alpha_bc


        #right-hand-side
        R = np.zeros((m,n))
        # add code to make R based on implicit FD scheme
        R[2:-2] = h2inv*((b/4)*(f[4:]-2*f[2:-2]+f[:-4])+a*(f[3:-1]-2*f[2:-2]+f[1:-3]))
        R[0:2] = h2inv*(a_bc * f[0:2] + b_bc * f[1:3] + c_bc * f[2:4] + d_bc * f[3:5] + e_bc * f[4:6])
        R[-2:] = h2inv*(a_bc * f[-2:] + b_bc * f[-3:-1] + c_bc * f[-4:-2] + d_bc * f[-5:-3] + e_bc * f[-6:-4])
        d2f = solve_banded((1,1),ab, R, check_finite=False)


        return d2f



    fxx = diffx(f,L)
    fyy = diffy(f,h2inv)


    return x,y,fxx,fyy

def part2q2analyze():
    """
    Question 2.2(b)
    Add input/output as needed

    """
    N = 1000
    L = 2 * np.pi
    x = np.linspace(0, L, N+1)[:-1]
    y = x.copy()
    h = x[1] - x[0]
    h2inv = 1 / h**2
    xg, yg = np.meshgrid(x, y)
    f = np.cos(xg) * np.sin(yg)
    def diffx(f,L):
        """
        Question 2.2)
        Input:
            f: real array whose 2nd derivative will be computed along each row
            L: domain size
        Output:
            d2f: (real) second derivative of f computed with discrete Fourier transforms
        """
        n = np.arange(f.shape[0]/2 + 1)
        k = 2*np.pi*n
        k2m = -k**2/L**2
        fk = np.fft.rfft(f,axis=1)
        d2f = np.fft.irfft(k2m*fk,axis=1)
        return d2f #modify as needed

    def diff2x(f, h):
        """
        Compute the second derivative of f with respect to x using
        2nd-order centered finite differences.

        Input:
            f: N x N numpy array
            h: float; grid spacing

        Output:
            d2f: N x N numpy array; second derivative of f with respect to x
        """
        N = f.shape[0]
        d2f = np.zeros_like(f)
        for i in range(N):
            if i == 0:
                d2f[i, :] = (f[i+2, :] - 2*f[i+1, :] + f[i, :]) / h**2
            elif i == 1:
                d2f[i, :] = (f[i+1, :] - 2*f[i, :] + f[i-1, :]) / h**2
            elif i == N-2:
                d2f[i, :] = (f[i+1, :] - 2*f[i, :] + f[i-1, :]) / h**2
            elif i == N-1:
                d2f[i, :] = (f[i, :] - 2*f[i-1, :] + f[i-2, :]) / h**2
            else:
                d2f[i, :] = (f[i+1, :] - 2*f[i, :] + f[i-1, :]) / h**2
        return d2f

    def diffy(f,h2inv):
            """
            Question 2.2)
            Input:
                f: real array whose 2nd derivative will be computed along each column
                h2inv: 1/h^2 where h is the grid spacing
            Output:
                d2f: second derivative of f computed with implicit FD scheme
            """
            #coefficients for interior points
            alpha = 2/11
            a = 12/11
            b = 3/11 

            #coefficients for near-boundary points
            alpha_bc = 10
            a_bc,b_bc,c_bc,d_bc,e_bc = (145/12,-76/3,29/2,-4/3,1/12)
            

            m, n = f.shape
            """
            mid_diag = np.ones(n)
            up_diag = alpha * np.ones(n-1)
            up_diag[:2] = alpha_bc * np.ones(2)
            up_diag[-1] = 0
            low_diag = alpha * np.ones(n-1)
            low_diag[0] = 0
            low_diag[-2:] = alpha_bc * np.ones(2)
            #make tridiagonal matrix using numpy
            diagonals = [low_diag,mid_diag,up_diag]
            offsets = [-1,0,1]
            ab = np.diag(diagonals[0], offsets[0]) + np.diag(diagonals[1], offsets[1]) + np.diag(diagonals[2], offsets[2])
            """

            ab = np.zeros((3,n))
            #main diagonal
            ab[1,:] = 1.0
            #upper diagonal
            ab[0,1:3] = alpha_bc
            ab[0,3:-1] = alpha
            #lower diagonal
            ab[2,1:-2] = alpha
            ab[2,-3:-1] = alpha_bc


            #right-hand-side
            R = np.zeros((m,n))
            # add code to make R based on implicit FD scheme
            R[2:-2] = h2inv*((b/4)*(f[4:]-2*f[2:-2]+f[:-4])+a*(f[3:-1]-2*f[2:-2]+f[1:-3]))
            R[0:2] = h2inv*(a_bc * f[0:2] + b_bc * f[1:3] + c_bc * f[2:4] + d_bc * f[3:5] + e_bc * f[4:6])
            R[-2:] = h2inv*(a_bc * f[-2:] + b_bc * f[-3:-1] + c_bc * f[-4:-2] + d_bc * f[-5:-3] + e_bc * f[-6:-4])
            d2f = solve_banded((1,1),ab, R, check_finite=False)


            return d2f
    hlist = []
    err1 = []
    err2 = []
    err3 = []
    w1 = []
    w2 = []
    w3 = []

    for N in range(100,1000,100):
        L = 20 * np.pi
        x = np.linspace(0, L, N+1)[:-1]
        y = x.copy()
        h = x[1] - x[0]
        hlist.append(h)
        h2inv = 1 / h**2
        xg, yg = np.meshgrid(x, y)
        f = np.cos(xg) * np.sin(yg)
        start = time()
        fxx_DFT = diffx(f, L)
        w1.append(time() - start)
        start = time()
        fxx_exp = diff2x(f, h)
        w2.append(time() - start)
        start = time()
        fxx_imp = diffy(f, h2inv) #same as diffx due to f being symmetric
        w3.append(time() - start)
        d2fdx2 = -np.cos(xg) * np.sin(yg)

    #get max absolute error
    err1.append(np.mean(np.abs(fxx_DFT - d2fdx2)))
    err2.append(np.mean(np.abs(fxx_exp - d2fdx2)))
    err3.append(np.mean(np.abs(fxx_imp - d2fdx2)))
    #have a 2 by 2 subplot
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0,0].loglog(hlist, err1, 's--', label='DFT')
    ax[0,0].loglog(hlist, err2, '*--', label='Explicit FD')
    ax[0,0].loglog(hlist, err3, 'x--', label='Implicit FD')
    #plot a line of h^4
    hlist = np.array(hlist)
    ax[0,0].loglog(hlist, hlist**4, label='$h^4$', c='black', linestyle = '--')
    ax[0,0].legend()
    ax[0,0].set_xlabel('h')
    ax[0,0].set_ylabel('Mean Absolute Error')
    #set title
    ax[0,0].set_title('Mean Absolute Error vs. Step Size')

    #plot walltimes
    ax[1,0].loglog(range(100,1000,100), w1, 's--', label='DFT')
    ax[1,0].loglog(range(100,1000,100), w2, '*--', label='Explicit FD')
    ax[1,0].loglog(range(100,1000,100), w3, 'x--', label='Implicit FD')
    ax[1,0].legend()
    ax[1,0].set_xlabel('N')
    ax[1,0].set_ylabel('Walltime (s)')
    #set title
    ax[1,0].set_title('Walltime vs. Number of Points')

    kh = np.linspace(0, 3, 100)
    alpha = 2/11
    a = 12/11
    b = 3/11 
    ax[0,1].plot(kh, np.sqrt(-2*(np.cos(kh) - 1)), label='Explicit FD', linestyle='--')
    ax[0,1].plot(kh, np.sqrt(((b/2)*(1-np.cos(2*kh))+2*a*(1-np.cos(kh)))/(1+2*alpha*np.cos(kh))), label='Implicit FD', linestyle='--')
    ax[0,1].plot(kh,kh, label='Exact', linestyle='--', c='black')
    ax[0,1].legend()
    #set labels
    ax[0,1].set_xlabel('wavenumber')
    ax[0,1].set_ylabel('modified wavenumber')
    #set title
    ax[0,1].set_title('Modified Wavenumber vs. Wavenumber')

    #plot ratio of walltimes
    ax[1,1].loglog(range(100,1000,100), np.array(w1)/np.array(w2), 's--', label='DFT/Explicit')
    ax[1,1].loglog(range(100,1000,100), np.array(w1)/np.array(w3), '*--', label='DFT/Implicit')
    ax[1,1].loglog(range(100,1000,100), np.array(w2)/np.array(w3), 'x--', label='Explicit/Implicit')
    ax[1,1].legend()
    ax[1,1].set_xlabel('N')
    ax[1,1].set_ylabel('Walltime Ratio')
    #set title
    ax[1,1].set_title('Walltime Ratio vs. Number of Points')


    plt.tight_layout()
    #save figure
    plt.show()
    #plot mae vs walltime
    plt.loglog(w1, err1, 's--', label='DFT')
    plt.loglog(w2, err2, '*--', label='Explicit FD')
    plt.loglog(w3, err3, 'x--', label='Implicit FD')
    plt.legend()
    plt.xlabel('Walltime (s)')
    plt.ylabel('Mean Absolute Error')
    #set title
    plt.title('MAE vs Walltime')
    plt.show()

#----------------------
if __name__=='__main__':
    pass
    #if you would like to call your functions, please call them from within
    #this if block
