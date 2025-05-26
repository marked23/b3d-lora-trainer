
This is my take on the official PyTorch FSDP2 example found here:</br>
[https://github.com/pytorch/examples/tree/main/distributed/FSDP2](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)

The original example has some lint I wanted to clean up.  This project will serve as my template for creating other trainers that take advantage of FSDP2. 

I use AMD GPUs. The requirements.txt will install the ROCm specific PyTorch from AMD's wheels.</br>
[Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html#install-pytorch-via-pip)

- PyTorch 2.6
- ROCm 6.4.1

What does it do when you run it?</br>
It creates a useless model out of thin air (random inputs).
```bash
(.venv) mark@wide:~/prog/fsdp2-minimal-rocm$ ./dotrain.sh 
FSDPTransformer(
  (tok_embeddings): Embedding(1024, 16)
  (pos_embeddings): Embedding(64, 16)
  (dropout): Dropout(p=0, inplace=False)
  (layers): ModuleList(
    (0-9): 10 x FSDPTransformerBlock(
      (attention_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
      (attention): Attention(
        (resid_dropout): Dropout(p=0, inplace=False)
        (wq): Linear(in_features=16, out_features=16, bias=False)
        (wk): Linear(in_features=16, out_features=16, bias=False)
        (wv): Linear(in_features=16, out_features=16, bias=False)
        (wo): Linear(in_features=16, out_features=16, bias=False)
      )
      (ffn_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
      (feed_forward): FeedForward(
        (w1): Linear(in_features=16, out_features=64, bias=True)
        (gelu): GELU(approximate='none')
        (w2): Linear(in_features=64, out_features=16, bias=True)
        (resid_dropout): Dropout(p=0, inplace=False)
      )
    )
  )
  (norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
  (output): Linear(in_features=16, out_features=1024, bias=False)
)
... then a bunch of warnings about AOTriton, which you can ignore.
```

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
