"""
Code for Scientific Computation Project 1
Please add college id here
CID: 01724711
"""


#===== Code for Part 1=====#
from part1_utilities import * #do not modify

def method1(L_all,x):
    """
    First method for finding location of target in list containing
    M length-N sorted lists
    Input: L_all: A list of M length-N lists. Each element of
    L_all is a list of integers sorted in non-decreasing order.
    Example input for M=3, N=2: L_all = [[1,3],[2,4],[6,7]]

    """
    M = len(L_all)
    for i in range(M):
        ind = bsearch1(L_all[i],x)
        if ind != -1000:
            return((i,ind))

    return (-1000,-1000)




def method2(L_all,x,L_new = []):
    """Second method for finding location of target in list containing
    M length-N sorted lists
    Input: L_all: A list of M length-N lists. Each element of
    L_all is a list of integers sorted in non-decreasing order.
    Example input for M=3, N=2: L_all = [[1,3],[2,4],[6,7]]
    """

    if len(L_new)==0:
        M = len(L_all)
        N = len(L_all[0])
        L_temp = []
        for i in range(M):
            L_temp.append([])
            for j in range(N):
                L_temp[i].append((L_all[i][j],(i,j)))

        def func1(L_temp):
            M = len(L_temp)
            if M==1:
                return L_temp[0]
            elif M==2:
                return merge(L_temp[0],L_temp[1])
            else:
                return merge(func1(L_temp[:M//2]),func1(L_temp[M//2:]))

        L_new = func1(L_temp)

    ind = bsearch2(L_new,x)
    if ind==-1000:
        return (-1000,-1000),L_new
    else:
        return L_new[ind][1],L_new


def time_test():
    """
    Timing test for two methods, method1 and method2, with dependence on the dimensions of the input matrices M and N
    and the number of target values P.

    The function plots the average wall times for both methods against the number of targets, P, for three different combinations
    of M and N. Least-squares fits for the wall time with respect to P, M, N, and MN are also included on the plots.

    Returns:
    A matplotlib plot with three subplots showing the dependence of wall time on P, M, N and MN.
    """
    import time
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 3, figsize=(13, 13))
    # Loop over three different combinations of M and N
    for plot in range(1,4):
        M, N = [[4, 64],[64,4],[64,64]][plot-1] # set M and N for each iteration
        wall1, wall2 = [], [] # initialize lists to store wall times for method1 and method2
        
        # Averaging wall times over 50 runs
        for k in range(50):
            wall1.append([])
            wall2.append([])
            L = np.random.randint(low=0,high=100000,size=(M,N)) # generate a random matrix L with shape (M, N)
            P_vals = np.logspace(start=0,stop=10,num=11,base=2) # create logarithmically spaced values for P
            
            # Loop over the values of P
            for length in P_vals:
                P = np.random.randint(low=0,high=100000,size=int(length)) # generate a random array with length P
                
                # Calculate wall time for method1
                start = time.time()
                for p in P:
                    method1(L,p)
                end = time.time()
                wall1[-1].append(end-start)
                
                # Calculate wall time for method2
                start = time.time()
                L_new = []
                for p in P:
                    tar_pos, L_new = method2(L, p, L_new)
                end = time.time()
                wall2[-1].append(end-start)
        
        # Create subplot for the current iteration
        plt.subplot(1,3,plot)
        plt.title(f"M,N={M,N}")
        
        # Plot average wall times for method1 and method2
        t1 = np.mean(wall1,axis=0)
        t2 = np.mean(wall2,axis=0)
        plt.loglog(P_vals, t1, 'x--', label='Method 1')
        plt.loglog(P_vals, t2, 'x--', label='Method 2')
        
        # Plot least-squares fits for method1
        a0, b0 = np.polyfit(P_vals*M*np.log2(N),t1,1)
        plt.loglog(P_vals,a0*P_vals*M*np.log2(N)+b0,'k-.',label=r'least-squares fit, $t \sim PMlog_2 N$')
        # Plot least-squares fits for method2
        a1, b1 = np.polyfit((P_vals+M*N)*np.log2(M*N),t2,1) #least squares fit for method2
        plt.loglog(P_vals,a1*(P_vals+M*N)*np.log2(M*N)+b1,'k:',label=r'least-squares fit, $t \sim (P+MN)log_2 MN$')
        plt.xlabel("Number of Targets: P")
        plt.ylabel("Walltimes (s)")
    plt.tight_layout()
    plt.legend(loc='lower right')
    
    plt.savefig("SC23_project1_plot1")
    















#===== Code for Part 2=====#

def findGene(L_in,L_p):
    """
    Find locations within adjacent strings (contained in input list,L_in)
    that contain patterns in input list L_p
    
    Input:
    L_in: A list containing two length-n strings
    L_p: A list containing p length-m strings
    
    Output:
    L_out: A length-p list whose ith element is a list of locations where the
    ith pattern has been found (see project description for further details)
    """
    
    #Size parameters
    n = len(L_in[0]) #length of a sequence
    p = len(L_p) #number of patterns
    m = len(L_p[0]) #length of pattern
    
    L_out = [[] for i in range(p)] # Initialize the output list of lists

    # Helper function to check if two strings are the same
    def match(X,Y):
        """
        Return True if X, Y are the same and False otherwise
        """
        for i in range(len(X)):
            if X[i] != Y[i]:
                return False
        return True

    # Helper function to convert string to list of integers
    def char2base4(S):
        """
        Convert gene test_sequence string to list of ints
        """
        c2b = {}
        c2b['A']=0
        c2b['C']=1
        c2b['G']=2
        c2b['T']=3
        L=[]
        for s in S:
            L.append(c2b[s])
        return L

    # Helper function to convert list to number
    def heval(L,Base):
            """
            Convert list L to base-10 number
            where Base specifies the base of L
            """
            f = 0
            for l in L[:-1]:
                f = Base*(l+f)
            h = (f + (L[-1]))
            return h

    S1, S2 = L_in # extract the two adjacent strings
    X1 = char2base4(S1) # convert the first string to list of integers
    d = 4 # base for converting list to number
    bm = 4**m# precompute value 
    #Pre compute hashes for S1
    ind = 0 # initialize index
    hi_dic_1 = dict()
    hi = heval(X1[:m],d)
    hi_dic_1[hi] = [ind] # compute hash value for first m characters of string
    for ind in range(1,n-m+1):
        #Update rolling hash
        hi = (4*hi - int(X1[ind-1])*bm + int(X1[ind-1+m]))
        if hi in hi_dic_1.keys():
            hi_dic_1[hi] = hi_dic_1[hi] + [ind]
        else:
            hi_dic_1[hi] = [ind]
    X2 = char2base4(S2) # convert the first string to list of integers
    #Pre compute hashes for S2
    ind = 0 # initialize index
    hi_dic_2 = dict()
    hi = heval(X1[:m],d)
    hi_dic_2[hi] = [ind] # compute hash value for first m characters of string
    for ind in range(1,n-m+1):
        #Update rolling hash
        hi = (4*hi - int(X1[ind-1])*bm + int(X1[ind-1+m]))
        if hi in hi_dic_2.keys():
            hi_dic_2[hi] = hi_dic_2[hi] + [ind]
        else:
            hi_dic_2[hi] = [ind]
    #Find intersection of two dictionaries
    hi_dic = dict()
    for hi in set(hi_dic_1.keys()).union(set(hi_dic_2.keys())):
        if hi not in hi_dic_1:
            hi_dic[hi] = hi_dic_2[hi]
        if hi not in hi_dic_2:
            hi_dic[hi] = hi_dic_1[hi]
        if hi in hi_dic_1 and hi in hi_dic_2:
            hi_dic[hi] = list(set(hi_dic_1[hi]) & set(hi_dic_2[hi]))
    
    # Loop through each pattern in L_p
    for i, pattern in enumerate(L_p):
        Y = char2base4(pattern) # convert pattern to list of integers
        hp = heval(Y,d) # compute hash value for pattern
        if hp in hi_dic:
            L_out[i] = hi_dic[hp]
    return L_out


if __name__=='__main__':
    #Small example for part 2
    S1 = 'ATCGTACTAGTTATC'
    S2 = 'ATCTTAGTAGTCGTC'
    L_in = [S1,S2]
    L_p = ['ATC','AGT']
    out = findGene(L_in,L_p)

    #Large gene sequences
    infile1,infile2 = open("S1example.txt"), open("S2example.txt")
    S1,S2 = infile1.read(), infile2.read()
    infile1.close()
    infile2.close()
