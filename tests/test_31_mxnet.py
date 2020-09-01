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


def test_2_005():
    #nk = pynccl.Nccl()
    #nc = nk._nccl  # cuNccl
    #api = nc._api  # libnccl

    nrank = 8#4

    # -------------------------------------

    procs = []

    q = mp.queues.Queue()

    # -------------------------------------

    for i in range(nrank):
        worker = mp.Process(
            #target=gpu_worker_proc_5,
            #target=gpu_worker_proc_5_2,
            target=gpu_worker_proc_5_3,
            args=(None, nrank, i, i, q))
        worker.daemon = True
        worker.start()
        procs.append(worker)


    for worker in procs:
        worker.join()


def gpu_worker_proc_5(api, kn, rank, gpu_i, q):

    # NOTE: do this at first of all
    cuda.select_device(gpu_i)

    nk = pynccl.Nccl()
    nc = nk._nccl  # cuNccl
    api = nc._api  # libnccl

    if rank == 0:

        nuid = pynccl.binding.ncclUniqueId()

        intnl_buf = chr(0) + chr(2) + 'nccl-%d-%d' % (os.getpid(), 0)
        intnl_buf += chr(0) * (pynccl.binding.NCCL_UNIQUE_ID_BYTES - len(intnl_buf))
        nuid.internal = intnl_buf

        r = api.ncclGetUniqueId(byref(nuid))  # TODO:
        print('>>> ncclGetUniqueId ', r)

        for j in range(kn - 1):
            q.put(nuid)

    else:
        nuid = q.get()
    # -------------------------------------

    #arr_send = np.array(np.random.random((1000, 10000)), dtype=np.float32)
    #arr_recv = np.empty_like(arr_send)
    #print(arr_send[1][1])
    arr_send = mxutils.mx.nd.zeros((1000, 10000), dtype=np.float32)
    arr_recv = arr_send.zeros_like()
    arr_send[1][1] = random.random()
    print(arr_send[1][1])

    #x#sz = 32 * 1000 * 10000
    sz = arr_send.size


    ####cuda.select_device(gpu_i)

    #d_arr_send = cuda.to_device(arr_send)
    #d_arr_recv = cuda.to_device(arr_recv)
    d_arr_send = arr_send.as_in_context(mxutils.context.gpu(gpu_i))
    d_arr_recv = arr_recv.as_in_context(mxutils.context.gpu(gpu_i))


    comm_i = c_void_p(0)
    nRanks = int(kn)  #2
    myRank = int(rank)  #0
    r = api.ncclCommInitRank(byref(comm_i), nRanks, nuid, myRank)
    #x#r = api.ncclCommInitRank(byref(comm_i), nRanks, byref(nuid), myRank)
    print('>>> ncclCommInitRank ', r)

    stream_i = cuda.stream()


    r = api.ncclGroupStart()
    print('>>> ncclGroupStart ', r)

    #ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
    #    comms[i], s[i])

    #x#p_arr_send = cast(d_arr_send.device_ctypes_pointer, c_void_p)
    #x#p_arr_recv = cast(d_arr_recv.device_ctypes_pointer, c_void_p)

    #p_arr_send = c_void_p(d_arr_send.device_ctypes_pointer.value)  # NOTE:
    #p_arr_recv = c_void_p(d_arr_recv.device_ctypes_pointer.value)
    p_arr_send = d_arr_send.get_data_p()
    p_arr_recv = d_arr_recv.get_data_p()

    #pd_arr_send = cuda_driver.device_pointer(p_arr_send)  # int
    #pd_arr_recv = cuda_driver.device_pointer(d_arr_recv)  # int

    #r = api.ncclAllReduce(d_arr_send.device_ctypes_pointer, d_arr_recv.device_ctypes_pointer,
    r = api.ncclAllReduce(p_arr_send, p_arr_recv,
                          sz, pynccl.binding.ncclFloat, pynccl.binding.ncclSum,
                          comm_i, stream_i.handle)  # NOTE:
                          #comm_i, c_void_p(0))  # NOTE:
    print('>>> ncclAllReduce ', r)

    r = api.ncclGroupEnd()
    print('>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    r_arr = d_arr_recv.asnumpy()
    #print(r_arr)
    print(r_arr[1][1])

    #cuda.close()


    r = api.ncclCommDestroy(comm_i)
    print('>>> ncclCommDestroy ', r)


def gpu_worker_proc_5_2(api, kn, rank, gpu_i, q):

    # NOTE: do this at first of all
    cuda.select_device(gpu_i)

    nk = pynccl.Nccl()
    #nc = nk._nccl  # cuNccl
    #api = nc._api  # libnccl

    if rank == 0:

        nuid = nk.get_unique_id()

        for j in range(kn - 1):
            q.put(nuid)

    else:
        nuid = q.get()
    # -------------------------------------

    arr_send = mxutils.mx.nd.zeros((1000, 10000), dtype=np.float32)
    arr_recv = arr_send.zeros_like()
    arr_send[1][1] = random.random()
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

    p_arr_send = d_arr_send.get_data_p()
    p_arr_recv = d_arr_recv.get_data_p()

    r = nk.all_reduce(p_arr_send, p_arr_recv,
                          sz, pynccl.binding.ncclFloat, pynccl.binding.ncclSum,
                          comm_i, stream_i.handle)  # NOTE:
                          #comm_i, c_void_p(0))  # NOTE:
    print('>>> ncclAllReduce ', rank, r)

    r = nk.group_end()
    print('>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    r_arr = d_arr_recv.asnumpy()
    #print(r_arr)
    print(r_arr[1][1])

    #cuda.close()

    r = nk.comm_destroy(comm_i)
    print('>>> ncclCommDestroy ', r)


# test: two nccl comms in one process
def gpu_worker_proc_5_3(api, kn, rank, gpu_i, q):

    # NOTE: do this at first of all
    cuda.select_device(gpu_i)

    nk = pynccl.Nccl()
    #nc = nk._nccl  # cuNccl
    #api = nc._api  # libnccl

    if rank == 0:

        nuid0 = nk.get_unique_id()

        for j in range(kn - 1):
            q.put(nuid0)

    else:
        nuid0 = q.get()
    # -------------------------------------

    time.sleep(10)
    # -------------------------------------

    if rank == 1:

        nuid1 = nk.get_unique_id()

        for j in range(kn - 1):
            q.put(nuid1)

    else:
        if rank % 2 == 1:
            nuid1 = q.get()

    # -------------------------------------

    arr_send = mxutils.mx.nd.zeros((1000, 10000), dtype=np.float32)
    arr_recv = arr_send.zeros_like()
    arr_send[1][1] = random.random()
    print(arr_send[1][1])

    #x#sz = 32 * 1000 * 10000
    sz = arr_send.size

    d_arr_send = arr_send.as_in_context(mxutils.context.gpu(gpu_i))
    d_arr_recv = arr_recv.as_in_context(mxutils.context.gpu(gpu_i))

    # -------------------------------------

    comm_i0 = nk.get_comm()
    nRanks = int(kn)  #2
    myRank = int(rank)  #0
    r = nk.comm_init_rank(byref(comm_i0), nRanks, nuid0, myRank)
    print('>>> ncclCommInitRank ', r)

    stream_i0 = nk.get_stream()

    # for test: rank-0 's sleep will block the others allreduce
    if rank == 0:
        print('-x' * 40, rank)
        time.sleep(10)
        print('=x' * 40, rank)

    r = nk.group_start()
    print('>>> ncclGroupStart ', r)

    p_arr_send = d_arr_send.get_data_p()
    p_arr_recv = d_arr_recv.get_data_p()

    r = nk.all_reduce(p_arr_send, p_arr_recv,
                          sz, pynccl.binding.ncclFloat, pynccl.binding.ncclSum,
                          comm_i0, stream_i0.handle)  # NOTE:
                          #comm_i, c_void_p(0))  # NOTE:
    print('>>> ncclAllReduce ', rank, r)

    r = nk.group_end()
    print('>>> ncclGroupEnd ', r)

    stream_i0.synchronize()

    r_arr = d_arr_recv.asnumpy()
    #print(r_arr)
    print(r_arr[1][1])

    #cuda.close()

    # -------------------------------------

    if rank % 2 == 1:
        comm_i1 = nk.get_comm()
        nRanks = int(kn / 2)  #2
        myRank = int(rank / 2)  # ####################################
        r = nk.comm_init_rank(byref(comm_i1), nRanks, nuid1, myRank)
        print('>>> 1 ncclCommInitRank ', r)

        stream_i1 = nk.get_stream()

        # for test: rank-0 's sleep will block the others allreduce
        if rank == 0:
            print('-x' * 40, rank)
            time.sleep(10)
            print('=x' * 40, rank)

        r = nk.group_start()
        print('>>> 1 ncclGroupStart ', r)

        p_arr_send = d_arr_send.get_data_p()
        p_arr_recv = d_arr_recv.get_data_p()

        r = nk.all_reduce(p_arr_send, p_arr_recv,
                              sz, pynccl.binding.ncclFloat, pynccl.binding.ncclSum,
                              comm_i1, stream_i1.handle)  # NOTE:
                              #comm_i, c_void_p(0))  # NOTE:
        print('>>> 1 ncclAllReduce ', rank, r)

        r = nk.group_end()
        print('>>> 1 ncclGroupEnd ', r)

        stream_i1.synchronize()

        r_arr = d_arr_recv.asnumpy()
        #print(r_arr)
        print(r_arr[1][1])

        #cuda.close()

    else:
        time.sleep(15)

    # -------------------------------------

    r = nk.comm_destroy(comm_i0)
    if rank % 2 == 1:
        r = nk.comm_destroy(comm_i1)
    print('>>> ncclCommDestroy ', r)







if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
    os.environ['NUMBA_NCCLLIB'] = '/usr/lib/x86_64-linux-gnu/'

    #os.environ['NCCL_SOCKET_IFNAME'] = 'enp11s0'  # TODO: for IB

    #test_1_001()
    #test_2_001()

    #argv = sys.argv
    #if len(argv) < 3:
    #    print 'Usage: test.py nranks myrank'
    #test_2_002(argv)

    #x#test_2_003()
    #x#test_2_004()

    test_2_005()


'''
('>>> ncclGetUniqueId ', 0)

[ 0.]
<NDArray 1 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>
NCCL version 2.1.2+cuda9.0
('>>> ncclCommInitRank ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
0.0
('>>> ncclCommDestroy ', 0)
0.0
('>>> ncclCommDestroy ', 0)
0.0
0.0
0.0
0.0
0.0
0.0
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
'''
