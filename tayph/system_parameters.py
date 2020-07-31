def paramget(keyword,dp):
    """This code queries a planet system parameter from a config file located in the folder
    specified by the path dp.

    Parameters
    ----------
    keyword : str
        A keyword present in the cofig file.

    dp : str, Path
        Output filename/path.


    Returns
    -------
    value : int, float, bool, str
        The value corresponding to the requested keyword.

    """
    from tayph.vartests import typetest
    from tayph.util import check_path
    import pathlib

    dp=check_path(dp)
    typetest(keyword,str,'keyword in paramget()')

    if isinstance(dp,str) == True:
        dp=pathlib.Path(dp)
    try:
        f = open(dp/'config.dat', 'r')
    except FileNotFoundError:
        raise FileNotFoundError('config.dat does not exist at %s' % str(dp)) from None
    x = f.read().splitlines()
    f.close()
    n_lines=len(x)
    keywords={}
    for i in range(0,n_lines):
        line=x[i].split()
        try:
            value=float(line[1])
        except ValueError:
            value=(line[1])
        keywords[line[0]] = value
    try:
        return(keywords[keyword])
    except KeyError:
        raise Exception('Keyword %s is not present in configfile at %s' % (keyword,dp)) from None
