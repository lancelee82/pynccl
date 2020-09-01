import ctypes
from ctypes import c_void_p, c_int, c_char, c_char_p, POINTER, byref

from numba.cuda.cudadrv.drvapi import cu_stream  # c_void_p
from numba.cuda.cudadrv.driver import device_pointer

from .utils import (ctype_function, finalizer, PLib,
                    c_double_complex, c_complex, memalign)


NCCL_MAJOR = 2
NCCL_MINOR = 7
NCCL_PATCH = 8
NCCL_SUFFIX = ""

NCCL_VERSION_CODE = 2708
#define NCCL_VERSION(X,Y,Z) ((X) * 1000 + (Y) * 100 + (Z))

# Error type
STATUS = {
    0x0: 'ncclSuccess',
    0x1: 'ncclUnhandledCudaError',
    0x2: 'ncclSystemError',
    0x3: 'ncclInternalError',
    0x4: 'ncclInvalidArgument',
    0x5: 'ncclInvalidUsage',
    0x6: 'ncclNumResults',
}

ncclResult_t = c_int

# Opaque handle to communicator
# nccl/src/core.h  :  struct ncclComm {...}
ncclComm_t = c_void_p # opaque handle

NCCL_UNIQUE_ID_BYTES = 128 * 4  # NOTE: from dbg nccl2

NcclUniqueId_data_t = c_char * NCCL_UNIQUE_ID_BYTES

class ncclUniqueId(ctypes.Structure):
    _fields_ = [('internal', NcclUniqueId_data_t)]
    #_fields_ = [('internal', c_char_p)]
    #_fields_ = [('internal', POINTER(ctypes.c_char))]

# Reduction operation selector
ncclSum        = 0
ncclProd       = 1
ncclMax        = 2
ncclMin        = 3
ncclNumOps     = 4

ncclRedOp_t = c_int

# Data types
ncclInt8       = 0; ncclChar       = 0
ncclUint8      = 1
ncclInt32      = 2; ncclInt        = 2
ncclUint32     = 3
ncclInt64      = 4
ncclUint64     = 5
ncclFloat16    = 6; ncclHalf       = 6
ncclFloat32    = 7; ncclFloat      = 7
ncclFloat64    = 8; ncclDouble     = 8
ncclNumTypes   = 9

ncclDataType_t = c_int


class NcclV2Error(Exception):
    def __init__(self, code):
        super(NcclV2Error, self).__init__(STATUS[code])


class libnccl(PLib):
    lib = 'nccl'
    ErrorType = NcclV2Error

    @property
    def version(self):
        return NCCL_MAJOR


    #ncclResult_t  ncclGetVersion(int *version);
    ncclGetVersion = ctype_function(ncclResult_t,
                                    POINTER(c_int),
    )
    #pncclGetVersion = ncclGetVersion

    #ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId);
    ncclGetUniqueId = ctype_function(ncclResult_t,
                                     #c_void_p,
                                     POINTER(ncclUniqueId),
    )
    #pncclGetUniqueId = ncclGetUniqueId

    #ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
    ncclCommInitRank = ctype_function(ncclResult_t,
                                      POINTER(ncclComm_t),  # comm
                                      c_int,  # nranks
                                      ncclUniqueId,  # commId
                                      c_int,  # rank
    )
    #pncclCommInitRank = ncclCommInitRank

    #ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
    ncclCommInitAll = ctype_function(ncclResult_t,
                                     POINTER(ncclComm_t),  # *comm
                                     c_int,  # ndev
                                     POINTER(c_int),  # devlist
                                     #c_void_p,  # devlist  # TODO: #########
    )

    #ncclResult_t  ncclCommDestroy(ncclComm_t comm);
    ncclCommDestroy = ctype_function(ncclResult_t,
                                     ncclComm_t,  # comm
    )

    #ncclResult_t  ncclCommAbort(ncclComm_t comm);
    ncclCommAbort = ctype_function(ncclResult_t,
                                   ncclComm_t,  # comm
    )

    #const char*  ncclGetErrorString(ncclResult_t result);
    ncclGetErrorString = ctype_function(POINTER(c_char),
                                        ncclResult_t,  # result
    )

    #ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
    ncclCommGetAsyncError = ctype_function(ncclResult_t,
                                           ncclComm_t,  # comm
                                           POINTER(ncclResult_t),  # asyncError
    )

    #ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count);
    ncclCommCount = ctype_function(ncclResult_t,
                                   ncclComm_t,  # comm
                                   POINTER(c_int),  # count
    )

    #ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device);
    ncclCommCuDevice = ctype_function(ncclResult_t,
                                      ncclComm_t,  # comm
                                      POINTER(c_int),  # device
    )

    #ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank);
    ncclCommUserRank = ctype_function(ncclResult_t,
                                      ncclComm_t,  # comm
                                      POINTER(c_int),  # rank
    )


    # Collective communication operations


    #ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    #    ncclDataType_t datatype,
    #    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
    ncclReduce = ctype_function(ncclResult_t,
                                c_void_p,  # sendbuff
                                c_void_p,  # recvbuff
                                c_int,  # size_t count
                                ncclDataType_t,  # datatype
                                ncclRedOp_t,  # op
                                c_int,  # root
                                ncclComm_t,  # comm
                                cu_stream,  # stream
    )

    #ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    #    ncclComm_t comm, cudaStream_t stream);
    ncclBcast = ctype_function(ncclResult_t,
                               c_void_p,  # buff
                               c_int,  # size_t count
                               ncclDataType_t,  # datatype
                               c_int,  # root
                               ncclComm_t,  # comm
                               cu_stream,  # stream
    )

    #ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    #    ncclComm_t comm, cudaStream_t stream);
    ncclBroadcast = ctype_function(ncclResult_t,
                                   c_void_p,  # sendbuff
                                   c_void_p,  # recvbuff
                                   c_int,  # size_t count
                                   ncclDataType_t,  # datatype
                                   c_int,  # root
                                   ncclComm_t,  # comm
                                   cu_stream,  # stream
    )

    #ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    #    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
    ncclAllReduce = ctype_function(ncclResult_t,
                                   c_void_p,  # sendbuff
                                   c_void_p,  # recvbuff
                                   c_int,  # size_t count
                                   ncclDataType_t,  # datatype
                                   ncclRedOp_t,  # op
                                   ncclComm_t,  # comm
                                   cu_stream,  # stream
    )

    #ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
    #    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    #    cudaStream_t stream);
    ncclReduceScatter = ctype_function(ncclResult_t,
                                       c_void_p,  # sendbuff
                                       c_void_p,  # recvbuff
                                       c_int,  # size_t recvcount
                                       ncclDataType_t,  # datatype
                                       ncclRedOp_t,  # op
                                       ncclComm_t,  # comm
                                       cu_stream,  # stream
    )

    #ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    #    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
    ncclAllGather = ctype_function(ncclResult_t,
                                   c_void_p,  # sendbuff
                                   c_void_p,  # recvbuff
                                   c_int,  # size_t sendcount
                                   ncclDataType_t,  # datatype
                                   ncclComm_t,  # comm
                                   cu_stream,  # stream
    )

    #ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    #    ncclComm_t comm, cudaStream_t stream);
    ncclSend = ctype_function(ncclResult_t,
                              c_void_p,  # sendbuff
                              c_int,  # size_t count
                              ncclDataType_t,  # datatype
                              c_int,  # int peer
                              ncclComm_t,  # comm
                              cu_stream,  # stream
    )

    #ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    #    ncclComm_t comm, cudaStream_t stream);
    ncclRecv = ctype_function(ncclResult_t,
                              c_void_p,  # recvbuff
                              c_int,  # size_t count
                              ncclDataType_t,  # datatype
                              c_int,  # int peer
                              ncclComm_t,  # comm
                              cu_stream,  # stream
    )

    #ncclResult_t ncclGroupStart();
    ncclGroupStart = ctype_function(ncclResult_t)

    #ncclResult_t ncclGroupEnd();
    ncclGroupEnd = ctype_function(ncclResult_t)


class cuNccl(finalizer.OwnerMixin):
    def __init__(self):
        self._api = libnccl()
