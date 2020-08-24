


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
    for i in range(1000):
        a,b = do_a_gaussfit()
        As.append(a[0])#Append the gaussian amplitudes.
        Aes.append(b[0])
    # ut.end(t1)

    assert math.isclose(np.std(As),np.mean(Aes),rel_tol=5e-2)
