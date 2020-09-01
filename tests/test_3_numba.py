import os
import sys
import time

import numpy as np

os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
# for: \numba\examples\cudajit\matmul.py
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
from numba import cuda



'''
  //managing 4 devices
  int nDev = 4;
  int size = 32*1024*1024;
  int devs[4] = { 0, 1, 2, 3 };

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }
'''

def test_3_001():

    arr = np.arange(1000)
    d_arr = cuda.to_device(arr)

    #my_kernel[100, 100](d_arr)
    time.sleep(10)

    r_arr = d_arr.copy_to_host()
    print(r_arr)

'''
(Pdb) p type(d_arr)
<class 'numba.cuda.cudadrv.devicearray.DeviceNDArray'>
(Pdb) p dir(d_arr)
['T', '_DeviceNDArrayBase__writeback', '__array__', '__class__', '__cuda_memory__',
'__cuda_ndarray__', '__delattr__', '__dict__', '__doc__', '__format__',
'__getattribute__', '__getitem__', '__hash__', '__init__', '__module__',
'__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
'__setitem__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
'_default_stream', '_do_getitem', '_do_setitem', '_dummy', '_numba_type_',
'alloc_size', 'as_cuda_arg', 'bind', 'copy_to_device', 'copy_to_host',
'device_ctypes_pointer', 'dtype', 'flags', 'get_ipc_handle', 'getitem',
'gpu_data', 'is_c_contiguous', 'is_f_contiguous', 'ndim', 'ravel', 
'reshape', 'setitem', 'shape', 'size', 'split', 'stream', 'strides', 'to_host', 'transpose']
(Pdb) p d_arr.device_ctypes_pointer
c_ulong(139823801171968L)
(Pdb) n
'''


def test_3_002():  # create ndarray on gpus

    gpu_n = 4

    for i in range(gpu_n):
        cuda.select_device(i)

        arr = np.arange(1000)
        d_arr = cuda.to_device(arr)

        #my_kernel[100, 100](d_arr)
        time.sleep(5)

        r_arr = d_arr.copy_to_host()
        print(r_arr)

        cuda.close()

    time.sleep(10)


def test_3_003():  # create ndarray on gpus

    gpu_n = 4

    arr_a = np.array(np.random.random((gpu_n, 10000)), dtype=np.float32)

    for i in range(gpu_n):
        cuda.select_device(i)

        arr = arr_a[i]
        d_arr = cuda.to_device(arr)

        #my_kernel[100, 100](d_arr)
        time.sleep(5)

        r_arr = d_arr.copy_to_host()
        print(r_arr)

        cuda.close()

    time.sleep(10)





if __name__ == '__main__':
    #test_3_001()
    #test_3_002()
    test_3_003()
