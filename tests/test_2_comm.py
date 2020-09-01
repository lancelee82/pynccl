import os
import sys
import time

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


def test_1_001():
    nk = pynccl.Nccl()
    print(dir(nk))


# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html
'''
ncclComm_t comms[4];
int devs[4] = { 0, 1, 2, 3 };
ncclCommInitAll(comms, 4, devs);


for (int i=0; i<4; i++)
   ncclCommDestroy(comms[i]);
'''

def test_2_001():
    nk = pynccl.Nccl()
    nc = nk._nccl  # cuNccl
    api = nc._api  # libnccl


    comms_a = c_void_p * 4
    int_a = c_int * 4

    # <1>
    comms = comms_a(0, 0, 0, 0)
    comms_p = cast(comms, POINTER(c_void_p))

    # <2>
    #comms = c_void_p(0)

    devs = int_a(0, 1, 2, 3)
    # <3>
    #devs_p = byref(devs)
    # <4>
    devs_p = cast(devs, POINTER(c_int))

    # <1>
    r = api.ncclCommInitAll(comms_p, 4, devs_p)
    # <2>
    #r = api.ncclCommInitAll(byref(comms), 4, devs_p)

    print(r)

    for i in range(4):
        print(comms[i])
        r = api.ncclCommDestroy(comms[i])
        print(r)



'''
First, we retrieve MPI information about processes:
int myRank, nRanks;
MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

Next, a single rank will create a unique ID and send it to
all other ranks to make sure everyone has it:
ncclUniqueId id;
if (myRank == 0) ncclGetUniqueId(&id);
MPI_Bcast(id, sizeof(id), MPI_BYTE, 0, 0, MPI_COMM_WORLD);

Finally, we create the communicator:
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);
We can now call the NCCL collective operations using the communicator.

Finally, we destroy the communicator object:
ncclCommDestroy(comm);
'''

def test_2_002(argv):
    nk = pynccl.Nccl()
    nc = nk._nccl  # cuNccl
    api = nc._api  # libnccl

    nuid = pynccl.binding.ncclUniqueId()
    print(nuid)
    print(type(nuid.internal))

    #nuid.internal = pynccl.binding.NcclUniqueId_data_t()
    #nuid.internal = str(create_string_buffer('', 128))
    ##nuid.internal = create_string_buffer('', 128)

    '''
    intnl_str = create_string_buffer('', 128)
    print(intnl_str)

    for i in range(128):
        intnl_str[i] = chr(0)
    for i in range(12):
        intnl_str[i] = 'a'

    intnl_char_p = cast(intnl_str, c_char_p)
    #intnl_char_p = cast(intnl_str, pynccl.binding.NcclUniqueId_data_t)
    nuid.internal = intnl_char_p
    '''

    #for i in range(128):
    #    nuid.internal[i] = 44


    # socket.AF_INET = 2  # 0x0002  ==>  chr(0) + chr(2)
    #nuid.internal = chr(0) + chr(2) + 'nccl-%d-%d' % (os.getpid(), 0)  # TODO: global counter

    intnl_buf = chr(0) + chr(2) + 'nccl-%d-%d' % (os.getpid(), 0)
    intnl_buf += chr(0) * (pynccl.binding.NCCL_UNIQUE_ID_BYTES - len(intnl_buf))
    nuid.internal = intnl_buf

    print(nuid.internal)
    print(len(nuid.internal))
    #print(dir(nuid.internal))

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    r = api.ncclGetUniqueId(byref(nuid))  # TODO:
    print('>>> ncclGetUniqueId ', r)
    #print(nuid)
    '''
    print(id(nuid.internal))
    print(type(nuid.internal))
    #print(len(nuid.internal))
    print(nuid.internal)
    '''

    comm = c_void_p(0)
    nRanks = int(argv[1])  #2
    myRank = int(argv[2])  #0

    r = api.ncclCommInitRank(byref(comm), nRanks, nuid, myRank)
    #x#r = api.ncclCommInitRank(byref(comm), nRanks, byref(nuid), myRank)
    print('>>> ncclCommInitRank ', r)




def test_2_003():
    nk = pynccl.Nccl()
    nc = nk._nccl  # cuNccl
    api = nc._api  # libnccl

    # -------------------------------------

    procs = []

    # -------------------------------------

    comms_a = c_void_p * 4
    int_a = c_int * 4

    # <1>
    comms = comms_a(0, 0, 0, 0)
    comms_p = cast(comms, POINTER(c_void_p))

    # <2>
    #comms = c_void_p(0)

    devs = int_a(0, 1, 2, 3)
    # <3>
    #devs_p = byref(devs)
    # <4>
    devs_p = cast(devs, POINTER(c_int))

    # <1>
    r = api.ncclCommInitAll(comms_p, 4, devs_p)
    # <2>
    #r = api.ncclCommInitAll(byref(comms), 4, devs_p)

    print('>>> ncclCommInitAll ', r)

    #x#cuda.close()

    # -------------------------------------

    for i in range(4):
        worker = mp.Process(
            target=gpu_worker_proc,
            args=(api, i, i, comms[i]))
        worker.daemon = True
        worker.start()
        procs.append(worker)


    for worker in procs:
        worker.join()

    # -------------------------------------

    for i in range(4):
        #print(comms[i])
        r = api.ncclCommDestroy(comms[i])
        print('>>> ncclCommDestroy ', r)



'''

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
'''
def gpu_worker_proc(api, rank, gpu_i, comm_i):

    arr_send = np.array(np.random.random((1000, 10000)), dtype=np.float32)
    arr_recv = np.empty_like(arr_send)
    print(arr_send[1][1])

    sz = 32 * 1000 * 10000

    cuda.close()

    cuda.select_device(gpu_i)

    d_arr_send = cuda.to_device(arr_send)
    d_arr_recv = cuda.to_device(arr_recv)

    stream_i = cuda.stream()

    r = api.ncclGroupStart()
    print('>>> ncclGroupStart ', r)

    #ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
    #    comms[i], s[i])

    r = api.ncclAllReduce(d_arr_send.device_ctypes_pointer, d_arr_recv.device_ctypes_pointer,
                          sz, pynccl.binding.ncclFloat, pynccl.binding.ncclSum,
                          comm_i, stream_i)
    print('>>> ncclAllReduce ', r)

    r = api.ncclGroupEnd()
    print('>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    r_arr = d_arr_recv.copy_to_host()
    #print(r_arr)
    print(r_arr[1][1])

    cuda.close()



def test_2_004():
    nk = pynccl.Nccl()
    nc = nk._nccl  # cuNccl
    api = nc._api  # libnccl

    # -------------------------------------

    procs = []

    q = mp.queues.Queue()

    # -------------------------------------

    for i in range(4):
        worker = mp.Process(
            target=gpu_worker_proc_4,
            args=(api, i, i, q))
        worker.daemon = True
        worker.start()
        procs.append(worker)


    for worker in procs:
        worker.join()


def gpu_worker_proc_4(api, rank, gpu_i, q):
    nk = pynccl.Nccl()
    nc = nk._nccl  # cuNccl
    api = nc._api  # libnccl

    if rank == 0:
        comms_a = c_void_p * 4
        int_a = c_int * 4

        # <1>
        comms = comms_a(0, 0, 0, 0)
        comms_p = cast(comms, POINTER(c_void_p))

        # <2>
        #comms = c_void_p(0)

        devs = int_a(0, 1, 2, 3)
        # <3>
        #devs_p = byref(devs)
        # <4>
        devs_p = cast(devs, POINTER(c_int))

        # <1>
        r = api.ncclCommInitAll(comms_p, 4, devs_p)
        # <2>
        #r = api.ncclCommInitAll(byref(comms), 4, devs_p)

        print('>>> ncclCommInitAll ', r)

        #x#cuda.close()
        q.put(comms)

    else:
        comms = q.get()
    # -------------------------------------

    arr_send = np.array(np.random.random((1000, 10000)), dtype=np.float32)
    arr_recv = np.empty_like(arr_send)
    print(arr_send[1][1])

    sz = 32 * 1000 * 10000

    cuda.select_device(gpu_i)

    d_arr_send = cuda.to_device(arr_send)
    d_arr_recv = cuda.to_device(arr_recv)

    comm_i = comms[i]
    stream_i = cuda.stream()

    r = api.ncclGroupStart()
    print('>>> ncclGroupStart ', r)

    #ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
    #    comms[i], s[i])

    r = api.ncclAllReduce(d_arr_send.device_ctypes_pointer, d_arr_recv.device_ctypes_pointer,
                          sz, pynccl.binding.ncclFloat, pynccl.binding.ncclSum,
                          comm_i, stream_i)
    print('>>> ncclAllReduce ', r)

    r = api.ncclGroupEnd()
    print('>>> ncclGroupEnd ', r)

    stream_i.synchronize()

    r_arr = d_arr_recv.copy_to_host()
    #print(r_arr)
    print(r_arr[1][1])

    #cuda.close()

    # -------------------------------------
    if rank == 0:
        for i in range(4):
            #print(comms[i])
            r = api.ncclCommDestroy(comms[i])
            print('>>> ncclCommDestroy ', r)






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
            target=gpu_worker_proc_5,
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

    arr_send = np.array(np.random.random((1000, 10000)), dtype=np.float32)
    arr_recv = np.empty_like(arr_send)
    print(arr_send[1][1])

    #x#sz = 32 * 1000 * 10000
    sz = arr_send.size


    ####cuda.select_device(gpu_i)

    d_arr_send = cuda.to_device(arr_send)
    d_arr_recv = cuda.to_device(arr_recv)


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

    p_arr_send = c_void_p(d_arr_send.device_ctypes_pointer.value)  # NOTE:
    p_arr_recv = c_void_p(d_arr_recv.device_ctypes_pointer.value)

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

    r_arr = d_arr_recv.copy_to_host()
    #print(r_arr)
    print(r_arr[1][1])

    #cuda.close()


    r = api.ncclCommDestroy(comm_i)
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
0.795402
0.795402
0.795402
0.795402
0.795402
0.795402
0.795402
0.795402
NCCL version 2.1.2+cuda9.0
('>>> ncclCommInitRank ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclCommInitRank ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
('>>> ncclGroupStart ', 0)
('>>> ncclAllReduce ', 0)
('>>> ncclGroupEnd ', 0)
6.363216
('>>> ncclCommDestroy ', 0)
6.363216
6.363216
6.363216
6.363216
6.363216
6.363216
6.363216
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
('>>> ncclCommDestroy ', 0)
'''
