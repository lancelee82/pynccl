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
from numba import cuda
from numba.cuda import driver as cuda_driver

from ctypes import c_void_p, c_int, c_char, POINTER, byref

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, ROOT_DIR)

import pynccl

import torch


def test_41_002():

    nrank = 4#8

    # -------------------------------------

    procs = []

    q = mp.Queue()

    # -------------------------------------

    for i in range(nrank):
        worker = mp.Process(
            target=gpu_worker_proc_41_2,
            args=(None, nrank, i, i, q))
        worker.daemon = True
        worker.start()
        procs.append(worker)


    for worker in procs:
        worker.join()


def gpu_worker_proc_41_2(api, kn, rank, gpu_i, q):

    # NOTE: do this at first of all
    cuda.select_device(gpu_i)

    nk = pynccl.Nccl()
    #nc = nk._nccl  # cuNccl
    #api = nc._api  # libnccl

    pg0 = list(range(kn))
    comm_0 = cre_nccl_comm_fn(nk, q, rank, pg0)

    pg1 = list(range(kn))[1:]
    comm_1 = cre_nccl_comm_fn(nk, q, rank, pg1)

    pg2 = list(range(kn))[2:]
    comm_2 = cre_nccl_comm_fn(nk, q, rank, pg2)

    time.sleep(2)

    nccl_fn_on_comm(nk, comm_0, q, rank, pg0, gpu_i)
    time.sleep(1)

    nccl_fn_on_comm(nk, comm_1, q, rank, pg1, gpu_i)
    time.sleep(1)

    nccl_fn_on_comm(nk, comm_2, q, rank, pg2, gpu_i)
    time.sleep(1)

    r = nk.comm_destroy(comm_0)
    print(rank, '>>> ncclCommDestroy ', r)

    r = nk.comm_destroy(comm_1)
    print(rank, '>>> ncclCommDestroy ', r)

    r = nk.comm_destroy(comm_2)
    print(rank, '>>> ncclCommDestroy ', r)


def cre_nccl_comm_fn(nk, q, pg_rank, pg_ranks):
    if pg_rank not in pg_ranks:
        return  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ranks = sorted(pg_ranks)
    kn = len(ranks)

    if pg_rank == ranks[0]:
        nuid = nk.get_unique_id()

        for j in range(kn - 1):
            q.put(nuid)

    else:
        nuid = q.get()


    comm_i = nk.get_comm()
    nRanks = kn
    myRank = ranks.index(pg_rank)
    r = nk.comm_init_rank(byref(comm_i), nRanks, nuid, myRank)
    print(pg_rank, '>>> ncclCommInitRank ', r)

    return comm_i


def nccl_fn_on_comm(nk, comm_i, q, pg_rank, pg_ranks, gpu_i):
    if pg_rank not in pg_ranks:
        return  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ranks = sorted(pg_ranks)
    kn = len(ranks)

    if pg_rank == ranks[0]:
        w = torch.Tensor(np.random.random((kn * 10, 5)))  # w

        for j in range(kn - 1):
            q.put(w)

    else:
        w = q.get()


    rank = ranks.index(pg_rank)


    #arr_send = w[rank]
    arr_send = w[rank*10:(rank+1)*10, :]

    arr_recv = torch.zeros((kn * 10, 5))  # recv

    #arr_send[1][1] = random.random()
    #print(arr_send[1][1])

    #x#sz = arr_send.size
    sz = np.prod(arr_send.size()) #* arr_send.element_size()

    d_arr_send = arr_send.cuda(gpu_i)
    d_arr_recv = arr_recv.cuda(gpu_i)


    stream_i = nk.get_stream()

    # for test: rank-0 's sleep will block the others allreduce
    if rank == 0:
        print('-=' * 40, rank)
        time.sleep(3)
        print('==' * 40, rank)

    r = nk.group_start()
    print(rank, '>>> ncclGroupStart ', r)

    # NOTE: in pytorch, the t() function of Tensor does NOT change the
    # original memory, so here we should create a new Tensor to nccl
    #t_arr_send = d_arr_send
    t_arr_send = torch.Tensor(d_arr_send.cpu()).cuda()

    #p_arr_send = d_arr_send.data_ptr()
    p_arr_send = t_arr_send.data_ptr()
    p_arr_recv = d_arr_recv.data_ptr()

    '''
    r = nk.all_reduce(p_arr_send, p_arr_recv,
                      sz,
                      pynccl.binding.ncclFloat, pynccl.binding.ncclSum,
                      comm_i, stream_i.handle)  # NOTE:
                      #comm_i, c_void_p(0))  # NOTE:
    print(rank, '>>> ncclAllReduce ', rank, r)
    '''

    r = nk.all_gather(p_arr_send, p_arr_recv,
                      sz,
                      pynccl.binding.ncclFloat,
                      comm_i, stream_i.handle)
    print(rank, '>>> ncclAllGather ', r)


    r = nk.group_end()
    print(rank, '>>> ncclGroupEnd ', r)

    stream_i.synchronize()


    r_arr = d_arr_recv.cpu().numpy()
    time.sleep(rank)
    print(rank, 'r_arr', r_arr)
    #print(rank, w.cpu().numpy() == r_arr)
    print(rank, (w.cpu().numpy() - r_arr) < 1e-6)


    #r = nk.comm_destroy(comm_i)
    #print(rank, '>>> ncclCommDestroy ', r)


if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'WARN'
    #os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
    os.environ['NUMBA_NCCLLIB'] = '/usr/lib/x86_64-linux-gnu/'

    #os.environ['NCCL_SOCKET_IFNAME'] = 'enp11s0'  # TODO: for IB

    test_41_002()
