""""""
import os
import sys
import time
import random

import multiprocessing as mp
import multiprocessing.queues

import numpy as np

#os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
# for: \numba\examples\cudajit\matmul.py
#os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
#os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
#from numba import cuda
#from numba.cuda import driver as cuda_driver

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, ROOT_DIR)

import pynccl

import torch


def test_7_001():

    nrank = 2#4#8
    gpu_is = [1, 3]

    # -------------------------------------

    procs = []

    q = mp.Queue()

    # -------------------------------------

    for i in range(nrank):
        worker = mp.Process(
            target=gpu_worker_proc_7_1,
            args=(nrank, i, gpu_is[i], q))
        worker.daemon = True
        worker.start()
        procs.append(worker)


    for worker in procs:
        worker.join()


def gpu_worker_proc_7_1(kn, rank, gpu_i, q):

    nk = pynccl.NcclWrp(kn, rank, gpu_i)

    if rank == 0:

        nuid = nk.get_nuid()

        w = torch.Tensor(np.random.random((kn * 10, 5)))  # w
        print('w', w.shape, w)

        for j in range(kn - 1):
            q.put((nuid, w))

    else:
        nuid, w = q.get()

    nk.set_nuid(nuid)

    nk.init_comm()

    # -------------------------------------

    arr_send = w[rank*10:(rank+1)*10, :]  # send

    arr_recv = torch.zeros((kn * 10, 5))  # recv

    print(arr_send[1][1])

    sz = np.prod(arr_send.size()) #* arr_send.element_size()

    d_arr_send = arr_send.cuda(gpu_i)
    d_arr_recv = arr_recv.cuda(gpu_i)

    p_arr_send = d_arr_send.data_ptr()
    p_arr_recv = d_arr_recv.data_ptr()

    # for test: rank-0 's sleep will block the others allreduce
    if rank == 0:
        print('-x' * 40, rank)
        time.sleep(3)
        print('=x' * 40, rank)

    nk.do_all_gather(p_arr_send, p_arr_recv, sz,
                     datatype=pynccl.binding.ncclFloat)

    nk.stream_sync()

    r_arr = d_arr_recv.cpu().numpy()
    time.sleep(rank)
    print(rank, 'r_arr', r_arr)
    #print(rank, w.cpu().numpy() == r_arr)
    print(rank, (w.cpu().numpy() - r_arr) < 1e-6)

    # -------------------------------------

    nk.abort_comm()


if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NUMBA_NCCLLIB'] = '/usr/lib/x86_64-linux-gnu/'

    #os.environ['NCCL_SOCKET_IFNAME'] = 'enp11s0'  # TODO: for IB

    test_7_001()
