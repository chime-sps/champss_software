def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import imp

    import pkg_resources

    __file__ = pkg_resources.resource_filename(__name__, "huffman.so")
    __loader__ = None
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)


__bootstrap__()
