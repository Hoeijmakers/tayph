def test_dimtest():
    from ..vartests import dimtest
    import numpy as np
    a=[[1,2,3],[4,3,9]]
    dimtest(a,[2,3])
    dimtest(a,(2,3))
    dimtest(a,np.array([2,3]))
    dimtest(a,np.shape(a))
    trigger = 0
    try:
        dimtest(a,[2,2])
    except ValueError:
        trigger+=1
    try:
        dimtest(a,'a')
    except TypeError:
        trigger+=1
    assert(trigger == 2)
