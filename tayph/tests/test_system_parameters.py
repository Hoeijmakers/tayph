
def test_paramget():
    import pkg_resources
    import pathlib
    from ..system_parameters import paramget
    configfolder=pathlib.Path(pkg_resources.resource_filename('tayph', 'data/'))
    assert(paramget('lala',configfolder)==123)#Test that the configfile can be read.
    trigger = 0
    try:#Test that wrong pathfiles raise a FileNotFoundError
        paramget('lala',configfolder/'doesntexist')
    except FileNotFoundError:
        trigger = 1
    assert(trigger==1)
