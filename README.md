
This is my first attempt to train a LoRA.
I'm trying to augment a coder model to improve completion accuracy for a specific python library called [build123d](https://github.com/gumyr/build123d)



The original example has some lint I wanted to clean up.  This project will serve as my template for creating other trainers that take advantage of FSDP2. 

I use AMD GPUs. The requirements.txt will install the ROCm specific PyTorch from AMD's wheels.</br>
[Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html#install-pytorch-via-pip)

- PyTorch 2.6
- ROCm 6.4.1


I used [https://github.com/pytorch/examples/tree/main/distributed/FSDP2](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)
as my starting point, but it's probably unrecognizable now that Claude has had her way with it.
Original README below
***
***

## FSDP2
To run FSDP2 on transformer model:
```
cd distributed/FSDP2
torchrun --nproc_per_node 2 train.py
```
* For 1st time, it creates a "checkpoints" folder and saves state dicts there
* For 2nd time, it loads from previous checkpoints

To enable explicit prefetching
```
torchrun --nproc_per_node 2 train.py --explicit-prefetch
```

To enable mixed precision
```
torchrun --nproc_per_node 2 train.py --mixed-precision
```

To showcase DCP API
```
torchrun --nproc_per_node 2 train.py --dcp-api
```

## Ensure you are running a recent version of PyTorch:
see https://pytorch.org/get-started/locally/ to install at least 2.5 and ideally a current nightly build.
