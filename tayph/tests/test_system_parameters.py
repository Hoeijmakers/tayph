
def test_paramget():
    import pkg_resources
    import pathlib
    from ..system_parameters import paramget
    configfolder=pathlib.Path(pkg_resources.resource_filename('tayph', 'data/'))
    assert(paramget('P',configfolder)==1.27492485)#Test that the configfile can be read.
    assert(paramget('air',configfolder)==True)
    assert(paramget('name',configfolder)=='WASP-121')
    trigger = 0
    try:#Test that wrong pathfiles raise a FileNotFoundError
        paramget('r',configfolder/'doesntexist')
    except FileNotFoundError:
        trigger = 1
    assert(trigger==1)


def test_v_orb():
    import pkg_resources
    import pathlib
    from ..system_parameters import v_orb
    import math
    configfolder=pathlib.Path(pkg_resources.resource_filename('tayph', 'data/'))
    v = v_orb(configfolder)
    assert math.isclose(v,221.09268033226203, rel_tol=1e-5)


def test_astropy_berv():
    import pkg_resources
    import pathlib
    from ..system_parameters import astropyberv
    import math
    configfolder=pathlib.Path(pkg_resources.resource_filename('tayph', 'data/'))
    v = astropyberv(configfolder)
    assert math.isclose(v[0],4.9, rel_tol=2e-2)
