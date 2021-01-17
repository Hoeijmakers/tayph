
def test_running_median_2D():
    """This tests running median, mean and std functions in 2D, and implicitly tests
    strided-window."""
    import numpy as np
    from ..functions import running_mean_2D,running_std_2D,running_median_2D,strided_window

    def slow_running_median_v2(D,w):
        """This is the way I'd want a running median to behave, with a truncated kernel at the edges."""
        nx=D.shape[1]
        s=np.arange(0,nx,dtype=float)
        dx1=int(0.5*w)
        dx2=int(0.5*w)+(w%2)
        for i in range(0,nx):
            minx = max([0,i-dx1])
            maxx = min([nx,i+dx2])#This here is only a 3% slowdown.
            s[i]=np.median(D[:,minx:maxx])#This is what takes 97% of the time.
        return(s)
    def slow_running_mean_v2(D,w):
        """This is the way I'd want a running mean to behave, with a truncated kernel at the edges."""
        nx=D.shape[1]
        s=np.arange(0,nx,dtype=float)
        dx1=int(0.5*w)
        dx2=int(0.5*w)+(w%2)
        for i in range(0,nx):
            minx = max([0,i-dx1])
            maxx = min([nx,i+dx2])#This here is only a 3% slowdown.
            s[i]=np.mean(D[:,minx:maxx])#This is what takes 97% of the time.
        return(s)
    def slow_running_std_v2(D,w):
        """This is the way I'd want a running std to behave, with a truncated kernel at the edges."""
        nx=D.shape[1]
        s=np.arange(0,nx,dtype=float)
        dx1=int(0.5*w)
        dx2=int(0.5*w)+(w%2)
        for i in range(0,nx):
            minx = max([0,i-dx1])
            maxx = min([nx,i+dx2])#This here is only a 3% slowdown.
            s[i]=np.mean(D[:,minx:maxx])#This is what takes 97% of the time.
        return(s)




    Ny=5
    Nx1=17
    Nx2=300
    w1=5
    w2=10
    D1=np.tile(np.arange(0,int(Nx1),dtype=float),(Ny,1))#Create a data array.
    D2=np.tile(np.arange(0,int(Nx2),dtype=float),(Ny,1))#Create a data array.
    m1=slow_running_median_v2(D1,w1)
    m2=slow_running_median_v2(D2,w2)
    a1=slow_running_mean_v2(D1,w1)
    a2=slow_running_mean_v2(D2,w2)
    s1=slow_running_std_v2(D1,w1)
    s2=slow_running_std_v2(D2,w2)

    strided_m1=slow_running_median_v2(D1,w1)
    strided_m2=slow_running_median_v2(D2,w2)
    strided_a1=slow_running_mean_v2(D1,w1)
    strided_a2=slow_running_mean_v2(D2,w2)
    strided_s1=slow_running_std_v2(D1,w1)
    strided_s2=slow_running_std_v2(D2,w2)

    assert(np.sum(np.abs(m1-strided_m1))==0)
    assert(np.sum(np.abs(m2-strided_m2))==0)
    assert(np.sum(np.abs(a1-strided_a1))==0)
    assert(np.sum(np.abs(a2-strided_a2))==0)
    assert(np.sum(np.abs(s1-strided_s1))==0)
    assert(np.sum(np.abs(s2-strided_s2))==0)

def test_findgen():
    import pkg_resources
    import pathlib
    from ..functions import findgen
    import math
    import numpy as np
    a=findgen(8)
    b=[0,1,2,3,4,5,6,7]
    assert np.sum(np.abs(a-b)) == 0

def test_gaussfit():
    from ..functions import gaussian,gaussfit,findgen
    from lmfit import Model
    import numpy as np
    from numpy import random
    import matplotlib.pyplot as plt
    import math


    def do_a_gaussfit(plot=False):
        x=findgen(100)
        ex=x*0.0+0.05
        for i in range(0,len(x),2):
            ex[i]*=4
        err = random.normal(0,1.0, x.size)*ex
        y=gaussian(x,1.0,50.0,5.0)+err
        r,e=gaussfit(x,y,nparams=3,yerr=ex)

        if plot:
            print(r,e)
            plt.errorbar(x,y,ex,fmt='.')
            plt.plot(x,fun.gaussian(x,*r))
            plt.show()
        return(r,e)


    As = []
    Aes = []
    # t1 = ut.start()
    for i in range(2000):
        a,b = do_a_gaussfit()
        As.append(a[0])#Append the gaussian amplitudes.
        Aes.append(b[0])
    # ut.end(t1)

    assert math.isclose(np.std(As),np.mean(Aes),rel_tol=5e-2)
