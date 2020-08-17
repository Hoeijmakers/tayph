__all__ = [
    "t_eff",
    "paramget"
]


def paramget(keyword,dp,full_path=False):
    """This code queries a planet system parameter from a config file located in the folder
    specified by the path dp; or run configuration parameters from a file speciefied by the full
    path dp, if full_path is set to True.

    Parameters
    ----------
    keyword : str
        A keyword present in the cofig file.

    dp : str, Path
        Output filename/path.

    full_path: bool
        If set, dp refers to the actual file, not the location of a folder with a config.dat;
        but the actual file itself.

    Returns
    -------
    value : int, float, bool, str
        The value corresponding to the requested keyword.

    """
    from tayph.vartests import typetest
    from tayph.util import check_path
    import pathlib
    import distutils.util
    import pdb

    dp=check_path(dp)
    typetest(keyword,str,'keyword in paramget()')

    if isinstance(dp,str) == True:
        dp=pathlib.Path(dp)
    try:
        if full_path == False:
            dp = dp/'config'
        f = open(dp, 'r')
    except FileNotFoundError:
        raise FileNotFoundError('parameter file does not exist at %s' % str(dp)) from None
    x = f.read().splitlines()
    f.close()
    n_lines=len(x)
    keywords={}
    for i in range(0,n_lines):
        line=x[i].split()
        if len(line) > 1:
            try:
                value=float(line[1])
            except ValueError:
                try:
                    value=bool(distutils.util.strtobool(line[1]))
                except ValueError:
                    value=(line[1])
            keywords[line[0]] = value
    try:
        return(keywords[keyword])
    except KeyError:
        # print(keywords)
        raise Exception('Keyword %s is not present in parameter file at %s' % (keyword,dp)) from None

def t_eff(M,R):
    """This function computes the mass and radius of a star given its mass and radius relative to solar."""
    from tayph.vartests import typetest
    import numpy as np
    import astropy.constants as const

    typetest(M,[int,float],'M in t_eff()')
    typetest(R,[int,float],'R in t_eff()')
    M=float(M)
    R=float(R)

    Ms = const.M_sun
    Rs = const.R_sun
    Ls = const.L_sun
    sb = const.sigma_sb

    if M < 0.43:
        a = 0.23
        b = 2.3
    elif M < 2:
        a = 1.0
        b = 4.0
    elif M < 55:
        a = 1.4
        b = 3.5
    else:
        a = 32000.0
        b = 1.0

    T4 = a*M**b * Ls / (4*np.pi*R**2*Rs**2*sb)
    return(T4**0.25)
