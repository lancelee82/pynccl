# pynccl

Nvidia NCCL2 Python bindings using ctypes and numba.

Many codes and ideas of this project come from the project [pyculib](https://github.com/numba/pyculib).
It is originally as part of the distributed deep learning project called 
[necklace](https://github.com/lancelee82/necklace).


## Install

* NCCL

Please follow the Nvidia doc [here](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html) to 
install [NCCL](https://github.com/nvidia/nccl).

* pynccl

from source,

```
python setup.py install
```

or just,

```
pip install pynccl
```


## Usage

### Environments

* for numba

```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

(the following may be no need)
```
export NUMBAPRO_CUDALIB=/usr/local/cuda/lib64
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/
```

* for NCCL

```
export NUMBA_NCCLLIB=/usr/lib/x86_64-linux-gnu/

export NCCL_DEBUG=INFO

export NCCL_SOCKET_IFNAME=<your-ifname-like-ens11>
```


### Examples

* pynccl.NcclWrp

This piece of code is an example of NcclWrp with multiprocessing for dispatching
the ncclUniqueId to all processes. See the complete code [here](https://github.com/lancelee82/pynccl/blob/master/tests/test_22_pt_ncclwrp.py)

```python
    nc = pynccl.NcclWrp(kn, rank, gpu_i)

    if rank == 0:
        nuid = nc.get_nuid()

        for j in range(kn - 1):
            q.put((nuid, w))
    else:
        nuid, w = q.get()

    nc.set_nuid(nuid)
    nc.init_comm()
```

* pynccl.Nccl

You also can use the original functions of pynccl.Nccl, see the code [here](https://github.com/lancelee82/pynccl/blob/master/tests/test_21_pt_tensor.py)

```python
    # NOTE: do this at first of all
    cuda.select_device(gpu_i)

    nk = pynccl.Nccl()

    comm_i = nk.get_comm()
    r = nk.comm_init_rank(byref(comm_i), world_size, nuid, rank)

    stream_i = nk.get_stream()

    r = nk.group_start()

    ......

    r = nk.all_gather(p_arr_send, p_arr_recv,
                      sz,
                      pynccl.binding.ncclFloat,
                      comm_i, stream_i.handle)

    r = nk.group_end()

    stream_i.synchronize()
```

* multi comms

You can create multiple NCCL communicators with different world_size and ranks list, which is something like the process group and 
important for distributed deep learning framework, see the code [here](https://github.com/lancelee82/pynccl/blob/master/tests/test_41_multi_comms.py)
