# pynccl

Nvidia NCCL2 Python bindings using ctypes and numba.

Many codes and ideas of this project come from the project [pyculib](https://github.com/numba/pyculib).
The main goal of this project is to use Nvidia NCCL with only python code and without any other compiled
language code like C++. It is originally as part of the distributed deep learning project called 
[necklace](https://github.com/lancelee82/necklace), and now it could be used at other places.


## Install

* NCCL

Please follow the Nvidia doc [here](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html) to install NCCL.

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

```
    nk = pynccl.NcclWrp(kn, rank, gpu_i)

    if rank == 0:

        nuid = nk.get_nuid()

        for j in range(kn - 1):
            q.put((nuid, w))

    else:
        nuid, w = q.get()

    nk.set_nuid(nuid)

    nk.init_comm()
```

* pynccl.Nccl

You also can use the original functions of pynccl.Nccl, see the code [here](https://github.com/lancelee82/pynccl/blob/master/tests/test_21_pt_tensor.py)
