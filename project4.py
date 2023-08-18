import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

#-----------------------
# Code for Question 2(b)
#-----------------------
def part2(x0,y0):
    """
    Question 2.(b)
    Solve given optimization problem
    Input:
    x0,y0: lower bounds for x and y, x>=x0, y>=y0
    Output:
        res: OptimizeResult variable generated by linprog
    """
    c = [1, 1] #coefficients of the objective function to be minimized
    A = [[-2, -1], [-1, -2]] #coefficients of the linear inequalities
    b = [-6, -6] #constants on the right-hand side of the inequalities
    x_bounds = (x0, None) #bounds on x
    y_bounds = (y0, None) #bounds on y
    res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds])
    return res 


#-------------------------
# Code for Question 3. (a)
#-------------------------
def part3(M,fname='data4.txt',display=False,seed=1):
    """
    Question 3. (a)
    Solve compressive sensing signal reconstruction problem

    Input:
        M: number of measurements
        fname: filename for given image file
        display: when true, image corresonding to data is displayed
        seed: can be used to regenerate same measurement matrix with multiple
            calls to function

    Output:
    xtilde: N-element reconstructed signal vector (Numpy array)
    y: M-element measurement vector (Numpy array)
    """

    #load file and construct signal vector, x
    C = np.loadtxt(fname,dtype=float)
    N = C.size
    x = C.reshape(N,)


    #display image
    if display:
        plt.figure()
        plt.imshow(C,'gray')


    #Generate measurement matrix and measurement vector
    np.random.seed(seed) #set seed for reproducibility
    Phi = np.random.normal(0,1/np.sqrt(N),(M,N)) #measurement matrix
    y = np.dot(Phi,x) #measurement vector


    #Solve reconstruction problem
    c = np.ones(N) #l1 norm objective function
    res = linprog(c, A_eq=Phi, b_eq=y) #solve l1 norm minimization problem
    xtilde = res.x #reconstructed signal vector


    return xtilde,y




#------------------------
# Code for Question 3.(b)
#------------------------
def part3analyze(fname='data4.txt', seed=1, num_seeds=10):
    """
    Question 3.(b)
    """

    #define range of M values to test
    M_values = list(np.arange(10, 401, 10)) + list(np.arange(410,431,1)) + [440, 450, 460, 470, 480, 490, 500]

    #load file
    C = np.loadtxt(fname,dtype=float)

    #initialize array to store error values
    errors = []

    #calculate error for each M value
    for M in M_values:
        #get reconstructed signal vector
        xtilde, y = part3(M, fname=fname, display=False, seed=seed)
        #reconstruct image from xtilde
        Ctilde = xtilde.reshape(C.shape)
        #calculate error using Frobenius norm
        error = np.linalg.norm(Ctilde - C, ord='fro')
        errors.append(error)
    #plot error vs M
    plt.plot(M_values, errors)
    plt.xlabel('Number of measurements (M)')
    plt.ylabel('Reconstruction error')
    #calculate cutoff value for M where error is less than 1e-6
    cutoff = M_values[np.where(np.array(errors)<1e-6)[0][0]]
    K = C[C > 1e-6].size
    N = C.size
    c = cutoff / (K*np.log(N/K))
    plt.title(f'lo')
    plt.axvline(c * K*np.log(N/K), c='red', linestyle='--', label=f"cKlog(N/K), c={c:.3f}")
    plt.legend()
    plt.show()

    #find average c value for n different seeds
    c_list = []
    
    for s in range(1,num_seeds+1):
    #initialize array to store error values
        errors = []

        #calculate error for each M value
        for M in M_values:
            xtilde, y = part3(M, fname='data4.txt', display=False, seed=s)
            #reconstruct image from xtilde
            Ctilde = xtilde.reshape(C.shape)
            #calculate error using Frobenius norm
            error = np.linalg.norm(Ctilde - C, ord='fro')
            errors.append(error)
        cutoff = M_values[np.where(np.array(errors)<1e-6)[0][0]]
        c = cutoff / (K*np.log(N/K))
        c_list.append(c)
    return c_list, np.mean(c_list), np.std(c_list)



#----------------------
if __name__=='__main__':
    pass
    #if you would like to call your functions, please call them from within
    #this if block

