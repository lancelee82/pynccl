import os
import sys
import time
import random

import multiprocessing as mp
import multiprocessing.queues

import numpy as np

os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
# for: \numba\examples\cudajit\matmul.py
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
from numba import cuda
from numba.cuda import driver as cuda_driver

#from ctypes import c_void_p, c_int, c_char, POINTER, byref
from ctypes import *

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, ROOT_DIR)

import pynccl

# mxnet patch
import mxutils
mxutils.patch_ndarray()


'''
(vmmx) likun@gpu44:/data/likun$ python
Python 2.7.12 (default, Nov 20 2017, 18:23:56)
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import mxnet as mx
>>> a = mx.nd.array(range(9))
>>> a

[ 0.  1.  2.  3.  4.  5.  6.  7.  8.]
<NDArray 9 @cpu(0)>
>>> aa = a.reshape((3,3))
>>> aa

[[ 0.  1.  2.]
 [ 3.  4.  5.]
 [ 6.  7.  8.]]
<NDArray 3x3 @cpu(0)>
>>> id(a)
140132774638736
>>> id(aa)
140132773920464
>>> a[0] = 22
>>> aa

[[ 22.   1.   2.]
 [  3.   4.   5.]
 [  6.   7.   8.]]
<NDArray 3x3 @cpu(0)>
>>>
>>> b = mx.nd.array(range(20, 29))
>>> b

[ 20.  21.  22.  23.  24.  25.  26.  27.  28.]
<NDArray 9 @cpu(0)>
>>> bb = b.reshape((3,3))
>>> bb

[[ 20.  21.  22.]
 [ 23.  24.  25.]
 [ 26.  27.  28.]]
<NDArray 3x3 @cpu(0)>
>>> c = mx.nd.stack(aa,bb, axis=1)
>>> c

[[[ 22.   1.   2.]
  [ 20.  21.  22.]]

 [[  3.   4.   5.]
  [ 23.  24.  25.]]

 [[  6.   7.   8.]
  [ 26.  27.  28.]]]
<NDArray 3x2x3 @cpu(0)>
>>> c.shape
(3L, 2L, 3L)
>>> cc = c.reshape(3,6)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: reshape() takes exactly 2 arguments (3 given)
>>> cc = c.reshape((3,6))
>>> cc

[[ 22.   1.   2.  20.  21.  22.]
 [  3.   4.   5.  23.  24.  25.]
 [  6.   7.   8.  26.  27.  28.]]
<NDArray 3x6 @cpu(0)>
>>> c[0][0][0] = 1
>>> cc

[[  1.   1.   2.  20.  21.  22.]
 [  3.   4.   5.  23.  24.  25.]
 [  6.   7.   8.  26.  27.  28.]]
<NDArray 3x6 @cpu(0)>
>>>
>>>
>>>
>>>
>>>
>>> cc.T

[[  1.   3.   6.]
 [  1.   4.   7.]
 [  2.   5.   8.]
 [ 20.  23.  26.]
 [ 21.  24.  27.]
 [ 22.  25.  28.]]
<NDArray 6x3 @cpu(0)>
>>> cc.T.T == cc

[[ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.]]
<NDArray 3x6 @cpu(0)>
>>>
>>>
>>>
>>>
>>>
>>>
>>> e = mx.nd.array(range(24))
>>> e

[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
  15.  16.  17.  18.  19.  20.  21.  22.  23.]
<NDArray 24 @cpu(0)>
>>> ee = e.reshape((4,3,2))
>>> e

[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
  15.  16.  17.  18.  19.  20.  21.  22.  23.]
<NDArray 24 @cpu(0)>
>>> ee

[[[  0.   1.]
  [  2.   3.]
  [  4.   5.]]

 [[  6.   7.]
  [  8.   9.]
  [ 10.  11.]]

 [[ 12.  13.]
  [ 14.  15.]
  [ 16.  17.]]

 [[ 18.  19.]
  [ 20.  21.]
  [ 22.  23.]]]
<NDArray 4x3x2 @cpu(0)>
>>>
>>> e3 = ee.reshape((12,2))
>>> e3

[[  0.   1.]
 [  2.   3.]
 [  4.   5.]
 [  6.   7.]
 [  8.   9.]
 [ 10.  11.]
 [ 12.  13.]
 [ 14.  15.]
 [ 16.  17.]
 [ 18.  19.]
 [ 20.  21.]
 [ 22.  23.]]
<NDArray 12x2 @cpu(0)>
>>>
>>>
>>> e4 = ee.reshape((4,6))
>>> e4

[[  0.   1.   2.   3.   4.   5.]
 [  6.   7.   8.   9.  10.  11.]
 [ 12.  13.  14.  15.  16.  17.]
 [ 18.  19.  20.  21.  22.  23.]]
<NDArray 4x6 @cpu(0)>
>>>
>>> e44 = e.reshape((4,6))
>>> e44

[[  0.   1.   2.   3.   4.   5.]
 [  6.   7.   8.   9.  10.  11.]
 [ 12.  13.  14.  15.  16.  17.]
 [ 18.  19.  20.  21.  22.  23.]]
<NDArray 4x6 @cpu(0)>
>>>
>>>
>>> e2 = e.reshape((2,12))
>>> e2

[[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.]
 [ 12.  13.  14.  15.  16.  17.  18.  19.  20.  21.  22.  23.]]
<NDArray 2x12 @cpu(0)>
>>> e2[:, 0:3]

[[  0.   1.   2.]
 [ 12.  13.  14.]]
<NDArray 2x3 @cpu(0)>
>>>

'''





def test_5_005():
    #nk = pynccl.Nccl()
    #nc = nk._nccl  # cuNccl
    #api = nc._api  # libnccl

    nrank = 4#8

    # -------------------------------------

    procs = []

    q = mp.queues.Queue()

    # -------------------------------------

    for i in range(nrank):
        worker = mp.Process(
            #target=gpu_worker_proc_5,
            target=gpu_worker_proc_5_2,
            args=(None, nrank, i, i, q))
        worker.daemon = True
        worker.start()
        procs.append(worker)


    for worker in procs:
        worker.join()



def gpu_worker_proc_5_2(api, kn, rank, gpu_i, q):

    # NOTE: do this at first of all
    cuda.select_device(gpu_i)

    nk = pynccl.Nccl()
    #nc = nk._nccl  # cuNccl
    #api = nc._api  # libnccl

    if rank == 0:

        nuid = nk.get_unique_id()

        #w = mxutils.mx.nd.array(np.random.random((kn, 5, 10)), dtype=np.float32)  # w
        w = mxutils.mx.nd.array(np.random.random((5, kn * 10)), dtype=np.float32)  # w
        print('w', w)

        for j in range(kn - 1):
            q.put((nuid, w))

    else:
        nuid, w = q.get()
    # -------------------------------------

    x = mxutils.mx.nd.array(np.random.random((7, 5)), dtype=np.float32)

    #arr_send = mxutils.mx.nd.array(np.random.random(5, 40), dtype=np.float32)  # w
    #arr_send = w[rank]
    arr_send = w[:, rank*10:(rank+1)*10]

    #arr_recv = arr_send.zeros_like()
    #arr_recv = mxutils.mx.nd.zeros((kn, 5, 10), dtype=np.float32)  # recv
    #arr_recv = mxutils.mx.nd.zeros((5, kn * 10), dtype=np.float32)  # recv
    arr_recv = mxutils.mx.nd.zeros((kn * 10, 5), dtype=np.float32)  # recv.T

    #arr_send[1][1] = random.random()
    print(arr_send[1][1])

    #x#sz = 32 * 1000 * 10000
    sz = arr_send.size

    d_arr_send = arr_send.as_in_context(mxutils.context.gpu(gpu_i))
    d_arr_recv = arr_recv.as_in_context(mxutils.context.gpu(gpu_i))


    comm_i = nk.get_comm()
    nRanks = int(kn)  #2
    myRank = int(rank)  #0
    r = nk.comm_init_rank(byref(comm_i), nRanks, nuid, myRank)
    print('>>> ncclCommInitRank ', r)

    stream_i = nk.get_stream()

    # for test: rank-0 's sleep will block the others allreduce
    if rank == 0:
        print('-x' * 40, rank)
        time.sleep(10)
        print('=x' * 40, rank)

    r = nk.group_start()
    print('>>> ncclGroupStart ', r)

    ############
    t_arr_send = d_arr_send.T

    #p_arr_send = d_arr_send.get_data_p()
    p_arr_send = t_arr_send.get_data_p()
    p_arr_recv = d_arr_recv.get_data_p()

    '''
    r = nk.all_reduce(p_arr_send, p_arr_recv,
                      sz,
                      pynccl.binding.ncclFloat, pynccl.binding.ncclSum,
                      comm_i, stream_i.handle)  # NOTE:
                      #comm_i, c_void_p(0))  # NOTE:
    print('>>> ncclAllReduce ', rank, r)
    '''

    r = nk.all_gather(p_arr_send, p_arr_recv,
                      sz,
                      pynccl.binding.ncclFloat,
                      comm_i, stream_i.handle)
    print('>>> ncclAllGather ', r)


    r = nk.group_end()
    print('>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    mxutils.mx.ndarray.ndarray.waitall() ###

    r_arr = d_arr_recv.asnumpy()
    #print(r_arr.T)  # ############ the results is r_arr.T
    print(rank, w.asnumpy() == r_arr.T)


    r = nk.comm_destroy(comm_i)
    print('>>> ncclCommDestroy ', r)






if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
    os.environ['NUMBA_NCCLLIB'] = '/usr/lib/x86_64-linux-gnu/'

    os.environ['NCCL_SOCKET_IFNAME'] = 'enp11s0'  # TODO: for IB

    test_5_005()
