import os
import sys

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
    #nuid.internal = intnl_buf
    nuid.internal = bytes(intnl_buf, 'utf-8')

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




if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NUMBAPRO_CUDALIB'] = '/usr/local/cuda/lib64/'
    os.environ['NUMBA_NCCLLIB'] = '/usr/lib/x86_64-linux-gnu/'

    #os.environ['NCCL_SOCKET_IFNAME'] = 'enp11s0'  # TODO: for IB

    #test_1_001()
    #test_2_001()

    argv = sys.argv
    if len(argv) < 3:
        print('Usage: test.py nranks myrank')
        sys.exit(0)
    test_2_002(argv)
