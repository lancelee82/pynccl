import os
from contextlib import contextmanager

from ctypes import c_void_p, c_int, c_char, POINTER, byref

from numba import cuda

from . import binding as pynccl_binding


class Nccl(object):
    """NCCL2 python API by ctypes"""

    @cuda.require_context
    def __init__(self):
        self._nccl = pynccl_binding.cuNccl()
        self.api = self._nccl._api  # libnccl

    def get_version(self, *args, **kwargs):
        r = self.api.ncclGetVersion(*args)
        return r

    def get_comm(self, p=0):
        comm_i = c_void_p(p)
        return comm_i

    def get_stream(self):
        stream_i = cuda.stream()
        return stream_i

    def get_unique_id(self, k=0):
        nuid = pynccl_binding.ncclUniqueId()

        intnl_buf = chr(0) + chr(2) + 'nccl-%d-%d' % (os.getpid(), k)
        intnl_buf += chr(0) * (pynccl_binding.NCCL_UNIQUE_ID_BYTES - len(intnl_buf))

        # FIXED: TypeError: expected bytes, str found
        # str to bytes (python 2 / 3 all ok)
        nuid.internal = intnl_buf.encode('utf-8')

        r = self.api.ncclGetUniqueId(byref(nuid))

        return nuid


    def comm_init_rank(self, *args, **kwargs):
        r = self.api.ncclCommInitRank(*args)
        return r

    def comm_init_all(self, *args, **kwargs):
        r = self.api.ncclCommInitAll(*args)
        return r

    def comm_destroy(self, *args, **kwargs):
        r = self.api.ncclCommDestroy(*args)
        return r

    def comm_abort(self, *args, **kwargs):
        r = self.api.ncclCommAbort(*args)
        return r

    def get_error_string(self, *args, **kwargs):
        r = self.api.ncclGetErrorString(*args)
        return r

    def comm_get_async_error(self, *args, **kwargs):
        r = self.api.ncclCommGetAsyncError(*args)
        return r

    def comm_count(self, *args, **kwargs):
        r = self.api.ncclCommCount(*args)
        return r

    def comm_cu_device(self, *args, **kwargs):
        r = self.api.ncclCommCuDevice(*args)
        return r

    def comm_user_rank(self, *args, **kwargs):
        r = self.api.ncclCommUserRank(*args)
        return r


    def reduce(self, *args, **kwargs):
        r = self.api.ncclReduce(*args)
        return r

    def bcast(self, *args, **kwargs):
        r = self.api.ncclBcast(*args)
        return r

    def broadcast(self, *args, **kwargs):
        r = self.api.ncclBroadcast(*args)
        return r

    def all_reduce(self, *args, **kwargs):
        r = self.api.ncclAllReduce(*args)
        return r

    def all_gather(self, *args, **kwargs):
        r = self.api.ncclAllGather(*args)
        return r

    def reduce_scatter(self, *args, **kwargs):
        r = self.api.ncclReduceScatter(*args)
        return r

    def pp_send(self, *args, **kwargs):
        r = self.api.ncclSend(*args)
        return r

    def pp_recv(self, *args, **kwargs):
        r = self.api.ncclRecv(*args)
        return r

    def group_start(self, *args, **kwargs):
        r = self.api.ncclGroupStart(*args)
        return r

    def group_end(self, *args, **kwargs):
        r = self.api.ncclGroupEnd(*args)
        return r


class NcclWrp(object):
    def __init__(self, kn, rank, gpu_i=None):
        self.kn = kn
        self.rank = rank
        self.gpu_i = gpu_i
        self.init()

    def init(self):

        # NOTE: do this at first of all
        if self.gpu_i is not None:
            cuda.select_device(self.gpu_i)

        self.nk = Nccl()

    def get_nuid(self):
        nuid = self.nk.get_unique_id()
        return nuid

    def set_nuid(self, nuid):
        self.nuid = nuid

    def init_comm(self):
        self.comm_i = self.nk.get_comm()

        r = self.nk.comm_init_rank(
            byref(self.comm_i), int(self.kn), self.nuid, int(self.rank))

        # NOTE: cuda stream
        self.stream_i = self.nk.get_stream()


    def do_all_reduce(self, d_arr_send, d_arr_recv, sz,
                      datatype=pynccl_binding.ncclFloat,
                      op=pynccl_binding.ncclSum):

        r = self.nk.group_start()

        p_arr_send = d_arr_send
        p_arr_recv = d_arr_recv

        r = self.nk.all_reduce(p_arr_send, p_arr_recv,
                               sz, datatype, op,
                               self.comm_i,
                               self.stream_i.handle)
                               #self.comm_i, c_void_p(0))

        r = self.nk.group_end()

        return r

    def do_all_gather(self, d_arr_send, d_arr_recv, sz,
                      datatype=pynccl_binding.ncclUint32):
        r = self.nk.group_start()

        p_arr_send = d_arr_send
        p_arr_recv = d_arr_recv

        r = self.nk.all_gather(p_arr_send, p_arr_recv,
                               sz, datatype,
                               self.comm_i,
                               self.stream_i.handle)

        r = self.nk.group_end()

        return r

    def do_reduce(self, d_arr_send, d_arr_recv, sz, root,
                  datatype=pynccl_binding.ncclFloat,
                  op=pynccl_binding.ncclSum):
        r = self.nk.group_start()

        p_arr_send = d_arr_send
        p_arr_recv = d_arr_recv

        r = self.nk.reduce(p_arr_send, p_arr_recv,
                           sz, datatype, op, root,
                           self.comm_i,
                           self.stream_i.handle)

        r = self.nk.group_end()

        return r

    def do_bcast(self, d_arr_send, sz, root,
                 datatype=pynccl_binding.ncclFloat):
        r = self.nk.group_start()

        p_arr_send = d_arr_send

        r = self.nk.bcast(p_arr_send,
                          sz, datatype, root,
                          self.comm_i,
                          self.stream_i.handle)

        r = self.nk.group_end()

        return r

    def do_broadcast(self, d_arr_send, d_arr_recv, sz, root,
                     datatype=pynccl_binding.ncclFloat):
        r = self.nk.group_start()

        p_arr_send = d_arr_send
        p_arr_recv = d_arr_recv

        r = self.nk.broadcast(p_arr_send, p_arr_recv,
                              sz, datatype, root,
                              self.comm_i,
                              self.stream_i.handle)

        r = self.nk.group_end()

        return r

    def do_reduce_scatter(self, d_arr_send, d_arr_recv, sz,
                          datatype=pynccl_binding.ncclFloat,
                          op=pynccl_binding.ncclSum):
        r = self.nk.group_start()

        p_arr_send = d_arr_send
        p_arr_recv = d_arr_recv

        r = self.nk.reduce_scatter(p_arr_send, p_arr_recv,
                                   sz, datatype, op,
                                   self.comm_i,
                                   self.stream_i.handle)

        r = self.nk.group_end()

        return r

    def do_pp_send_recv(self, d_arr_send, d_arr_recv, sz,
                        rank_send, rank_recv,
                        datatype=pynccl_binding.ncclFloat):
        r = self.nk.group_start()

        p_arr_send = d_arr_send
        p_arr_recv = d_arr_recv

        # NOTE: when calling this func, one of d_arr_send, d_arr_recv
        #       should be None and the another should be not None
        if p_arr_send is not None and p_arr_recv is None:
            r = self.nk.pp_send(p_arr_send, sz, datatype,
                                rank_recv,
                                self.comm_i,
                                self.stream_i.handle)
        elif p_arr_recv is not None and p_arr_send is None:
            r = self.nk.pp_recv(p_arr_recv, sz, datatype,
                                rank_send,
                                self.comm_i,
                                self.stream_i.handle)
        else:
            raise Exception('one of d_arr_send, d_arr_recv should be None')

        r = self.nk.group_end()

        return r


    def stream_sync(self):
        self.stream_i.synchronize()

    def del_comm(self):
        r = self.nk.comm_destroy(self.comm_i)

    def abort_comm(self):
        r = self.nk.comm_abort(self.comm_i)
