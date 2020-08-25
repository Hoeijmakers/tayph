def test_get_model():
    import pkg_resources
    import pathlib
    from ..models import get_model
    import numpy as np
    libfile1=pathlib.Path(pkg_resources.resource_filename('tayph', str(pathlib.Path('data')/pathlib.Path('library'))))#This is to be able to point
    #tox to the path into which a temporary deployment of Tayph is copied to.
    libfile2=pathlib.Path(pkg_resources.resource_filename('tayph', str(pathlib.Path('data')/pathlib.Path('no_library_no_no_no'))))

    root =pathlib.Path(pkg_resources.resource_filename('tayph', 'data'))


    #Check that the test model can be read.
    wl,fx=get_model('m1',library=libfile1,root=root)

    #Check that a model is loaded correctly.
    assert(np.sum((wl-np.array([0,1,2,3]))**2)==0)
    assert(np.sum((fx-np.array([9,9,8,9]))**2)==0)


    #Test that giving a wrong model yields a KeyError.
    trigger = 0
    try:
        wl,fx = get_model('m2',library=libfile1,root=root)
    except KeyError:
        trigger = 1
    assert(trigger==1)

    #Test that a wrong library file gives a FileNotFoundError.
    trigger = 0
    try:
        wl,fx = get_model('m1',library=libfile2,root=root)
    except FileNotFoundError:
        trigger = 2
    assert(trigger==2)
