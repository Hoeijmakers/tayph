__all__ = [
    "nantest",
    "postest",
    "notnegativetest",
    "minlength",
    "typetest",
    "typetest_array",
    "dimtest",
    "lentest"
]

def nantest(var,varname=''):
    """This function tests for the presence of NaNs and infinites.

    Parameters
    ----------
    var : int, float or array-like
        The variable that needs to be tested.

    varname : str, optional
        Name or description of the variable to assist in debugging.

    """
    import numpy as np
    if np.isnan(var).any()  == True:
        raise ValueError(f"Variable {varname} contains NaNs but is not allowed to.")
    if np.isinf(var).any()  == True:
        raise ValueError(f"Variable {varname} contains in-finite values but is not allowed to.")




def postest(a,varname=''):
    """This function tests that all elements in the input variable are strictly positive.

    Parameters
    ----------
    var : int, float or array-like
        The variable that needs to be tested.

    varname : str, optional
        Name or description of the variable to assist in debugging.

    """
    import numpy as np
    if np.min(a) <= 0:
        raise ValueError(f'Variable {varname} is only allowed to be strictly positive ({np.min(a)}).')



def notnegativetest(a,varname=''):
    """This function tests that all elements in the input variable are zero or positive.

    Parameters
    ----------
    var : int, float or array-like
        The variable that needs to be tested.

    varname : str, optional
        Name or description of the variable to assist in debugging.
    """
    import numpy as np
    if np.min(a) < 0:
        raise ValueError(f'Variable {varname} is not allowed to be negative {np.min(a)}.')

def minlength(var,n,varname='',warning_only=False):
    import warnings
    """
    This function tests if a variable has a certain minimum length. An error can be converted into a warning if the warning_only keyword is set.
    """
    trigger = 0
    if len(var) < n:
        trigger = 1
    if warning_only and trigger == 1:
        warnings.warn(f'Variable {varname} has a length that is shorter than expected ({len(var)} < {n}.', RuntimeWarning)
    elif trigger == 1:
        raise ValueError(f'Variable {varname} has a length that is too short ({len(var)} < {n}).')


def typetest(var,vartype,varname='var'):
    """This function tests the type of var against a requested variable type and raises an exception if either varname is not a string,
    or if type(var) is not equal to vartype. A list of vartypes can be supplied to test whether the type is any of a provided set (i.e. OR logic).

    Parameters
    ----------
    var : any variable
        The variable that needs to be tested.

    vartype : type, list
        A python class that would be identified with type(); or a list of classes.

    varname : str, optional
        Name or description of the variable to assist in debugging.
    """
    if isinstance(varname,str) != True:
        raise TypeError(f"Variable should be of type string in typetest().")

    if type(vartype) == list:
        trigger = 0
        for t in vartype:
            trigger+=isinstance(var,t)
        if trigger == 0:
            errormsg="Variable type of %s should be equal to any of " % varname
            for t in vartype:
                errormsg+=('%s,'%t)
            errormsg=errormsg[0:-1]+(' (%s)'%type(var))
            raise TypeError(errormsg)
    else:
        if isinstance(var,vartype) != True:
            raise TypeError(f"Variable type of {varname} should be equal to {vartype} ({type(var)}).")



def typetest_array(var,vartype,varname='var'):
    """This function tests the types of the elements in the array or list var against a requested variable type and raises an exception if either
    varname is not a string, type(var) is not equal to numpy.array or list, or the elements of
    var are not ALL of a type equal to vartype. A list of vartypes can be supplied to test whether the type is any of a provided set (i.e. OR logic)..

    Parameters
    ----------
    var : list, np.array
        A list of variables each of which will be tested against vartype.

    vartype : type, list
        A python class that would be identified with type(); or a list of classes.

    varname : str, optional
        Name or description of the variable to assist in debugging.
    """
    import numpy as np
    if isinstance(varname,str) != True:
        raise TypeError("Varname should be of type string in typetest_array().")
    typetest(var,[list,tuple,np.ndarray])
    # if (isinstance(var,list) != True) and (isinstance(var,np.ndarray) != True):
    #     raise Exception("Input error in typetest_array: %s should be of class list or numpy array." % varname)
    for i in range(0,len(var)):
        typetest(var[i],vartype,varname=f'element {i} of {varname}')


def lentest(var,length,varname='var'):
    """
    This function tests the length of the input list.

    Parameters
    ----------
    var : list, np.ndarray, array-like
        An array with certain dimensions.

    length : int
        Target length to test var against.

    varname : str, optional
        Name or description of the variable to assist in debugging.

    Example
    -------
    >>> import numpy as np
    >>> a=[3,5,2.0,np.array([4,3,9]),'abc']
    >>> lentest(a,5)
    """
    import numpy as np
    typetest(length,[int])
    typetest(varname,str)
    typetest(var,list)
    if len(var) != length:
        raise ValueError(f"Wrong length of {varname}:  len = {len(var)} but was required to be {ndim}.")



def dimtest(var,sizes,varname='var'):
    """
    This function tests the dimensions and shape of the input array var.
    Sizes is the number of elements on each axis.

    Parameters
    ----------
    var : list, np.ndarray, array-like
        An array with certain dimensions.

    sizes : list, tuple, np.ndarray
        A list of dimensions (integers) to check var against.

    varname : str, optional
        Name or description of the variable to assist in debugging.

    Example
    -------
    >>> import numpy as np
    >>> a=[[1,2,3],[4,3,9]]
    >>> b=np.array(a)
    >>> dimtest(a,[2,3])
    >>> dimtest(a,np.shape(a))
    """
    import numpy as np
    typetest(sizes,[list,tuple,np.ndarray])
    typetest_array(sizes,[int,np.int32,np.int64],varname='sizes in dimtest')
    typetest(varname,str)

    ndim=len(sizes)

    dimerror=0.0
    sizeerror=0.0
    if np.ndim(var) != ndim:
        raise ValueError(f"Wrong dimension of {varname}:  ndim = {np.ndim(var)} but was required to be {ndim}.")

    sizes_var=np.shape(var)

    for i in range(0,len(sizes)):
        if sizes[i] < 0:
            raise ValueError(f"Sizes was not set correctly in {varname}. It contains negative values. ({sizes(i)})")
        if sizes[i] > 0:
            if sizes[i] != sizes_var[i]:
                raise ValueError(f"{varname} has wrong shape: Axis {i} contains {sizes_var[i]} elements, but {sizes[i]} were required.")
