from __future__ import print_function
import re
import os
import sys
import ctypes
import platform

try:
    from numba.misc.findlib import find_lib, find_file  # for numba > 0.50.1
except Exception as e:
    from numba.findlib import find_lib, find_file

if sys.platform == 'win32':
    _dllopener = ctypes.WinDLL
elif sys.platform == 'darwin':
    _dllopener = ctypes.CDLL
else:
    _dllopener = ctypes.CDLL


def get_nccllib(lib, platform=None, path=None):
    if not path:
        libdir = os.environ.get('NUMBA_NCCLLIB')
    else:
        libdir = path
    candidates = find_lib(lib, libdir, platform)
    return max(candidates) if candidates else None


def open_nccllib(lib, ccc=False, path=None):
    path = get_nccllib(lib, path=path)
    if path is None:
        raise OSError('library %s not found' % lib)
    if ccc:
        return ctypes.CDLL(path)
    return _dllopener(path)


# from /pyculib/pyculib/utils/libutils.py

class ctype_function(object):
    def __init__(self, restype=None, *argtypes):
        self.restype = restype
        self.argtypes = argtypes


class PLib(object):
    __singleton = None
    lib = None

    def __new__(cls):
        # Check if we already have opened the dll
        if cls.__singleton is None:
            try:
                dll = open_nccllib(cls.lib)
            except OSError as e:
                raise Exception("Cannot open library for %s:\n%s" % (cls.lib,
                                                                     e))
            # Create new instance
            inst = object.__new__(cls)
            cls.__singleton = inst
            inst.dll = dll
            inst._initialize()
        else:
            inst = cls.__singleton
        return inst

    def _initialize(self):
        # Populate the instance with the functions
        for name, obj in vars(type(self)).items():
            if isinstance(obj, ctype_function):
                fn = getattr(self.dll, name)
                fn.restype = obj.restype
                fn.argtypes = obj.argtypes
                setattr(self, name, self._auto_checking_wrapper(fn, name=name))

    def _auto_checking_wrapper(self, fn, name):
        def wrapped(*args, **kws):
            nargs = len(args) + len(kws)
            expected = len(fn.argtypes)
            if nargs != expected:
                msg = "expecting {expected} arguments but got {nargs}: {fname}"
                raise TypeError(msg.format(expected=expected, nargs=nargs,
                                           fname=name))
            status = fn(*args, **kws)
            self.check_error(status)
            return status
        return wrapped

    def check_error(self, status):
        if status != 0:
            raise self.ErrorType(status)
