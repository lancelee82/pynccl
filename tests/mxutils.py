import ctypes

import mxnet as mx
from mxnet import base
from mxnet import context

from mxnet.base import _LIB
from mxnet.base import NDArrayHandle, check_call


'''
# /incubator-mxnet-v1.1.0/src/c_api/c_api.cc
int MXNDArrayGetData(NDArrayHandle handle,
                     void **out_pdata) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    *out_pdata = arr->data().dptr_;
  } else {
    *out_pdata = nullptr;
  }
  API_END();
}
'''

def get_data_p(self):
    """Get data memory address of ndarray."""
    arr_data_dptr = ctypes.c_void_p()
    check_call(_LIB.MXNDArrayGetData(
        self.handle, ctypes.byref(arr_data_dptr)))
    return arr_data_dptr

def patch_ndarray():
    mx.ndarray.ndarray.NDArray.get_data_p = get_data_p


# -------------------------------------------------------------------
def test_1():
    patch_ndarray()

    a = mx.nd.zeros((256, 3, 128, 128))

    print(a.handle)
    print(a.handle.value)

    p = a.get_data_p()
    print(p)


    d = a.as_in_context(context.gpu(0))

    print(d.handle)
    print(d.handle.value)

    p = d.get_data_p()
    print(p)


    '''
    c_void_p(66178784)
    66178784
    c_void_p(139726040330256)
    c_void_p(43279120)
    43279120
    c_void_p(139725184696320)
    '''
