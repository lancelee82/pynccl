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


def test_6_001():

    nrank = 2#4#8
    gpu_is = [1, 3]

    # -------------------------------------

    procs = []

    q = mp.Queue()

    # -------------------------------------

    for i in range(nrank):
        worker = mp.Process(
            target=gpu_worker_proc_6_1,
            args=(nrank, i, gpu_is[i], q))
        worker.daemon = True
        worker.start()
        procs.append(worker)


    for worker in procs:
        worker.join()


def gpu_worker_proc_6_1(kn, rank, gpu_i, q):

    nk = pynccl.NcclWrp(kn, rank, gpu_i)

    if rank == 0:

        nuid = nk.get_nuid()

        w = torch.Tensor(np.random.random((kn * 10, 5)))  # w
        w = w.cuda(gpu_i)  # =====> gpu
        print('w', w.shape, w)

        for j in range(kn - 1):
            q.put((nuid, w))

    else:
        nuid, w = q.get()

    nk.set_nuid(nuid)

    nk.init_comm()

    # -------------------------------------

    rank_send = 0
    rank_recv = 1


    sz = np.prod(w.size())

    if rank == rank_send:  # send
        d_arr_send = w.data_ptr()
        d_arr_recv = None
        nk.do_pp_send_recv(d_arr_send, d_arr_recv, sz,
                           rank_send, rank_recv,)

        nk.stream_sync()

    elif rank == rank_recv:  # recv
        arr_recv = torch.zeros((kn * 10, 5))  # recv
        arr_recv = arr_recv.cuda(gpu_i)  # =====> gpu

        d_arr_send = None
        d_arr_recv = arr_recv.data_ptr()
        nk.do_pp_send_recv(d_arr_send, d_arr_recv, sz,
                           rank_send, rank_recv,)

        nk.stream_sync()

        print('arr_recv', arr_recv.shape, arr_recv)

        print((arr_recv.cpu() - w.cpu()) < 1e-6)

    # -------------------------------------

    nk.abort_comm()


if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'WARN'
    #os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
    os.environ['NUMBA_NCCLLIB'] = '/usr/lib/x86_64-linux-gnu/'

    #os.environ['NCCL_SOCKET_IFNAME'] = 'enp11s0'  # TODO: for IB

    test_6_001()
