"""
Code for Scientific Computation Project 2
Please add college id here
CID: 01724711
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx

#===== Codes for Part 1=====#
def part1q1n(Hlist, Hdict={}, option=0, x=[]):
    """
    Code for part 1, question 1
    Hlist should be a list of 2-element lists.
    The first element of each of these 2-element lists
    should be an integer. The second elements should be distinct and >-10000 prior to using
    option=0.
    Sample input for option=0: Hlist = [[8,0],[2,1],[4,2],[3,3],[6,4]]
    x: a 2-element list whose 1st element is an integer and x[1]>-10000
    """
    if option == 0:
        print("=== Option 0 ===")
        print("Original Hlist=", Hlist)
        heapq.heapify(Hlist)
        print("Final Hlist=", Hlist)
        Hdict = {}
        for l in Hlist:
            Hdict[l[1]] = l
        print("Final Hdict=", Hdict)
        return Hlist, Hdict
    elif option == 1:
        while len(Hlist)>0:
            wpop, npop = heapq.heappop(Hlist)
            if npop != -10000:
                del Hdict[npop]
                return Hlist, Hdict, wpop, npop
    elif option == 2:
        if x[1] in Hdict:
            l = Hdict.pop(x[1])
            l[1] = -10000
            Hdict[x[1]] = x
            heapq.heappush(Hlist, x)
            return Hlist, Hdict
        else:
            heapq.heappush(Hlist, x)
            Hdict[x[1]] = x
            return Hlist, Hdict


def part1q2(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """

    dinit = np.inf
    Fdict = {}
    Mdict = {}
    n = len(G)
    Plist = [[] for l in range(n)]

    Mdict[s]=1
    Plist[s] = [s]

    while len(Mdict)>0:
        dmin = dinit
        for n,delta in Mdict.items():
            if delta<dmin:
                dmin=delta
                nmin=n
        if nmin == x:
            return dmin, Plist[nmin]
        Fdict[nmin] = Mdict.pop(nmin)
        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = dmin*wn
                if dcomp<Mdict[en]:
                    Mdict[en]=dcomp
                    Plist[en] = Plist[nmin].copy()
                    Plist[en].append(en)
            else:
                dcomp = dmin*wn
                Mdict[en] = dcomp
                Plist[en].extend(Plist[nmin])
                Plist[en].append(en)
    return Fdict


def part1q3(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """

    # Initialize dictionaries and lists
    Fdict = {}   # Finalized nodes
    Mdict = {}   # Non-finalized nodes
    n = len(G)
    pred = {i: [] for i in range(n)}  # Predecessor dictionary
    Mlist = [[1, s]]  # Non-finalized nodes heap
    Mdict[s] = 1

    # Loop until all nodes are finalized
    while len(Mdict) > 0:

        # Find the node with the smallest distance
        Mlist, Mdict, dmin, nmin = part1q1n(Mlist, Mdict, option=1)

        # If the smallest distance is the target node, return the distance and the path
        if nmin == x:
            path = [nmin]
            while path[-1] != s:
                path.append(pred[path[-1]])
            path.reverse()
            return dmin, path

        # Mark the node as finalized
        Fdict[nmin] = dmin

        # Loop over the edges of the finalized node to update the distances of the nodes at the other end of the edges
        for m, en, wn in G.edges(nmin, data='weight'):
            
            # If the node at the other end of the edge is already finalized, do nothing
            if en in Fdict:
                pass

            # If the node at the other end of the edge is not finalized, but is already in the heap, update the distance if necessary
            elif en in Mdict:
                dcomp = dmin * wn

                # If the distance is updated, update the path as well
                if dcomp < Mdict[en][0]:
                    Mlist, Mdict = part1q1n(Mlist, Mdict, option=2, x=[dcomp, en])
                    pred[en] = nmin
            # If the node at the other end of the edge is not finalized and is not in the heap, add it to the heap
            else:
                dcomp = dmin * wn
                Mlist, Mdict = part1q1n(Mlist, Mdict, option=2, x=[dcomp, en])
                pred[en] = nmin

    # Return the finalized nodes
    return Fdict



#===== Code for Part 2=====#
def part2q1(n=50,tf=100,Nt=4000,seed=1):
    """
    Part 2, question 1
    Simulate n-individual opinion model

    Input:

    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    xarray: n x Nt+1 array containing x for the n individuals at
            each time step including the initial condition.
    """
    tarray = np.linspace(0,tf,Nt+1)
    xarray = np.zeros((Nt+1,n))

    def RHS(t, y):
        """
        Compute RHS of model
        """
        dydt = -np.mean((y[:, np.newaxis] - y) * np.exp(-(y[:, np.newaxis] - y)**2), axis=1)
        return dydt

    #Initial condition
    np.random.seed(seed)
    x0 = n*(np.random.rand(n)-0.5)

    #Compute solution
    out = solve_ivp(RHS,[0,tf],x0,t_eval=tarray,rtol=1e-8)
    xarray = out.y

    return tarray,xarray

def part2q1copy(n=50,tf=100,Nt=4000,seed=1,x0=None):
    """
    Part 2, question 1
    Simulate n-individual opinion model

    Input:

    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    xarray: n x Nt+1 array containing x for the n individuals at
            each time step including the initial condition.
    """
    tarray = np.linspace(0,tf,Nt+1)
    xarray = np.zeros((Nt+1,n))

    def RHS(t, y):
        """
        Compute RHS of model
        """
        dydt = -np.mean((y[:, np.newaxis] - y) * np.exp(-(y[:, np.newaxis] - y)**2), axis=1)
        return dydt
    
    if x0 is None:
        #Initial condition
        np.random.seed(seed)
        x0 = n*(np.random.rand(n)-0.5)

    #Compute solution
    out = solve_ivp(RHS,[0,tf],x0,t_eval=tarray,rtol=1e-8)
    xarray = out.y

    return tarray,xarray

def part2q2(n=50,tf=100,Nt=4000,seed=1,x0=None):
    """
    Add code used for part 2 question 2.
    Code to save your equilibirium solution is included below
    """


    if 'seed' not in globals():
        seed = 1
    else:
        seed += 1
    while True:
        tarray,xarray = part2q1copy(n,tf,Nt,seed,x0)
        
        x = xarray[:,-1]
        if np.all(np.abs(x) > 0) and np.all(np.abs(x) <= 1000) and len(np.unique(np.round(x,1))) >= n/2:
            plt.figure(figsize=(15,10))
            plt.plot(tarray,xarray.T)
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title('n = {}, tf = {}, Nt = {}, seed = {}'.format(n,tf,Nt,seed))
            plt.show()
            print("success")
            break
        seed += 1

    np.savetxt('xeq.txt',x) #saves xeq in file xeq.txt
    # I have submitted 2 equilibrium solutions named xeq1.txt and xeq2.txt (seed report for details)

    return x

def part2q3():
    """
    Add code used for part 2 question 3.
    Code to load your equilibirium solution is included below
    """
    #load saved equilibrium solution
    xeq = np.loadtxt('xeq.txt') 

    def jacobian(x):
        n = len(x)
        J = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    J[i,j] = (1/n) * (1-2*(x[i]-x[j])**2) * np.exp(-(x[i]-x[j])**2)
                else:
                    s = 0
                    for k in range(n):
                        if k != i:
                            s += (-1+2*(x[i]-x[k])**2) * np.exp(-(x[i]-x[k])**2)
                    J[i,j] = (1/n) * s
        return J
    
    J = jacobian(xeq)
    eigvals = np.linalg.eigvals(J)
    return eigvals


def part2q4(n=50,m=100,tf=40,Nt=10000,mu=0.2,seed=1):
    """
    Simulate stochastic opinion model using E-M method
    Input:
    n: number of individuals
    m: number of simulations
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf
    mu: model parameter
    seed: ensures same intial condition is generated with each simulation

    Output:
    tarray: size Nt+1 array
    Xave: size n x Nt+1 array containing average over m simulations
    Xstdev: size n x Nt+1 array containing standard deviation across m simulations
    """

    #Set initial condition
    np.random.seed(seed)
    x0 = n*(np.random.rand(1,n)-0.5)
    X = np.zeros((m,n,Nt+1)) #may require substantial memory if Nt, m, and n are all very large
    X[:,:,0] = np.ones((m,1)).dot(x0)


    Dt = tf/Nt
    tarray = np.linspace(0,tf,Nt+1)
    dW= np.sqrt(Dt)*np.random.normal(size=(m,n,Nt))

    #Iterate over Nt time steps
    for j in range(Nt):
        #Iterate over m simulations
        for i in range(m):
            #Compute RHS
            RHS = -(X[i,:,j] - X[i,:,j]) * np.exp(-(X[i,:,j] - X[i,:,j])**2) / n
            #Update solution
            X[i,:,j+1] = X[i,:,j] + Dt*RHS + mu*dW[i,:,j]
    
    #compute statistics
    Xave = X.mean(axis=0)
    Xstdev = X.std(axis=0)

    return tarray,Xave,Xstdev


def part2Analyze():
    """
    Code for part 2, question 4(b)
    """
    #Add code here to generate figures included in your report
    # Run deterministic model
    t_det, Xave_det = part2q1(n=10,tf=50)
    Xstd_det = np.zeros(Xave_det.shape)

    # Run stochastic model
    t_sto, Xave_sto, Xstd_sto = part2q4(n=10,tf=50)

    # Plot average opinion values over time for both models
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    for i in range(Xave_det.shape[0]):
        if i == 0:
            axes[0].plot(t_det, Xave_det[i], c='red', label='Deterministic')
            axes[0].plot(t_sto, Xave_sto[i], c='blue', label='Stochastic')
        else:
            axes[0].plot(t_det, Xave_det[i], c='red')
            axes[0].plot(t_sto, Xave_sto[i], c='blue')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Average opinion')
    axes[0].legend()


    # Plot standard deviation of opinion values over time for both models
    for i in range(Xstd_det.shape[0]):
        if i == 0:
            axes[1].plot(t_det, Xstd_det[i], c='red', label='Deterministic')
            axes[1].plot(t_sto, Xstd_sto[i], c='blue', label='Stochastic')
        else:
            axes[1].plot(t_det, Xstd_det[i], c='red')
            axes[1].plot(t_sto, Xstd_sto[i], c='blue')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Standard deviation of opinion')
    axes[1].legend()
    fig.suptitle('$\mu=0.2$', fontsize=10)
    plt.tight_layout()

    # Run deterministic model
    t_det, Xave_det = part2q1(n=10,tf=50)
    Xstd_det = np.zeros(Xave_det.shape)

    # Run stochastic model
    t_sto, Xave_sto, Xstd_sto = part2q4(n=10,tf=50,mu=2)

    # Plot average opinion values over time for both models
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    for i in range(Xave_det.shape[0]):
        if i == 0:
            axes[0].plot(t_det, Xave_det[i], c='red', label='Deterministic')
            axes[0].plot(t_sto, Xave_sto[i], c='blue', label='Stochastic')
        else:
            axes[0].plot(t_det, Xave_det[i], c='red')
            axes[0].plot(t_sto, Xave_sto[i], c='blue')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Average opinion')
    axes[0].legend()


    # Plot standard deviation of opinion values over time for both models
    for i in range(Xstd_det.shape[0]):
        if i == 0:
            axes[1].plot(t_det, Xstd_det[i], c='red', label='Deterministic')
            axes[1].plot(t_sto, Xstd_sto[i], c='blue', label='Stochastic')
        else:
            axes[1].plot(t_det, Xstd_det[i], c='red')
            axes[1].plot(t_sto, Xstd_sto[i], c='blue')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Standard deviation of opinion')
    axes[1].legend()
    fig.suptitle('$\mu=2$', fontsize=10)
    plt.tight_layout()


    return None
