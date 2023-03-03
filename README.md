This repository presents the 1. the results of the transformers benchmark on AMD GPU. 2. Hands-on step to run it on a HPC. 3. Appendix like readme from original fork.

# 1. results of the transformers benchmark on AMD GPU.
Code: myBenchmark_AMDGPU.py
## 1.1.1 Overview
|                                        | A100      | A6000    | V100     | 3090 Ti   | AMD MI250x |
|----------------------------------------|-----------|----------|----------|-----------|------------|
| Theory TF32(FP32) / FP16               | 156 / 312 | 75 / 150 | 16 / 125 | 80 / 160  | ?           |
| Memory (GB) / Bandwidth (GB/s)         | 80 / 2039 | 48 / 768 | 32 / 900 | 24 / 1008 | 128/TBC     |
| Approximate Price $                    | 16,000    | 4,000    | 3,500    | 1,500     |  ?          |
| Matrix Multiplication FP32 / FP16      | 116 / 230 | 60 / 95  | 14 / 95  | 42 / 81   |   31/115         |
| Vector Multiplication                  | 0.202     | 0.082    | 0.098    | 0.107     |      0.151      |
| Bert Layer Forward / Forward+Backward  | 110 / 136 | 60 / 70  | 53 / 64  | 56 / 62   |       52/63     |
| GPT-2 Layer Forward / Forward+Backward | 45 / 53   | 35 / 38  | 32 / 36  | 37 / 39   | 37/42      |
| T5 Encoder Forward / Forward+Backward  | 44 / 56   | 34 / 41  | 31 / 38  | 36 / 41   |    38/45        |
| T5 Decoder Forward / Forward+Backward  | 38 / 47   | 28 / 34  | 26 / 32  | 30 / 36   |  32/39          |
## 1.2 AMD Instinct MI250X
Pytorch version : 1.13.1+rocm5.2 
AMD Instinct MI250X
COMMAND: srun  -n  1  python3 newBench.py 
### 1.2.1 Matrix Multiplication FP32 / FP16
|               | n=128 | n=512  | n=2048 | n=8192  |                       |
|---------------|-------|--------|--------|---------|-----------------------|
| torch.float32 | 0.176 | 15.929 | 28.375 | 31.362  |                       |
| torch.float16 | 0.470 | 21.110 | 98.133 | 114.675 | Matrix Multiplication |
### 1.2.2 Vector Multiplication
|        | 65536  | 262144  | 1048576 | 4194304  |                       |
|--------|--------|---------|---------|----------|-----------------------|
| TFLOPS | 0.011  | 0.041   | 0.121   | 0.151    |                       |
| GB/s   | 90.822 | 329.992 | 965.916 | 1209.090 | Vector Multiplication |
### 1.2.3 Bert Layer Forward / Forward+Backward 
| | batch=2              | batch=4        | batch=8 | ...    | batch=32 | batch=64 | batch=128 |        
|----------------------|----------------|---------|--------|----------|----------|-----------|--------|
| fwd seq_len=128      | 11.117         | 22.134  | 32.548 | ...      | 46.224   | 50.008    | 52.025 |
| fwd+bwd seq_len=128  | 13.792         | 25.871  | 38.004 | ...      | 54.693   | 60.958    | 63.396 |
| fwd seq_len=512      | 32.054         | 38.514  | 43.833 | ...      | 49.100   | 49.119    | 47.890 |
| fwd+bwd seq_len=512  | 36.356         | 44.793  | 51.100 | ...      | 58.355   | 59.249    | 59.731 |
| [4 rows x 7 columns] | bert benchmark |         |        |          |          |           |        |
### 1.2.4 GPT-2 Layer Forward / Forward+Backward
|  result gpt2  Pytorch version : 1.13.1+rocm5.2                         |                       |         |         |          |          |          |
|----------------------------------------------------------|-----------------------|---------|---------|----------|----------|----------|
| result gpt2                                              | batch=2               | batch=4 | batch=8 | batch=16 | batch=32 | batch=64 |
| fwd seq_len=512                                          | 24.346                | 30.047  | 33.711  | 36.273   | 37.478   | 37.101   |
| fwd+bwd seq_len=512                                      | 29.239                | 33.928  | 37.970  | 41.511   | 42.780   | 41.853   |
| fwd seq_len=1024                                         | 26.463                | 29.082  | 30.773  | 31.664   | 31.401   | 31.159   |
| fwd+bwd seq_len=1024                                     | 30.484                | 33.417  | 35.838  | 36.770   | 36.120   | 35.580   |                  
### 1.2.5  T5  
encoder
|                     | batch=2 | batch=4 | batch=8 | ... | batch=32 | batch=64 | batch=128 |
|---------------------|---------|---------|---------|-----|----------|----------|-----------|
| fwd seq_len=512     | 20.944  | 28.044  | 32.714  | ... | 37.298   | 38.201   | 37.630    |
| fwd+bwd seq_len=512 | 27.512  | 34.565  | 39.564  | ... | 44.908   | 45.397   | 45.157    |

decoder (TODO: markdown format)
                     batch=2  batch=4  batch=8  ...  batch=32  batch=64  batch=128
fwd seq_len=512       17.860   23.558   27.574  ...    31.154    31.917     31.544
fwd+bwd seq_len=512   23.758   29.552   33.733  ...    37.869    38.756     38.631

[2 rows x 7 columns] t5 decoder

## 1.3 Run two projects on AMD Instinct MI250X  simultaneously 
Pytorch version : 1.13.1+rocm5.2 
AMD Instinct MI250X
COMMAND: srun  -n  2  python3 newBench.py 

It seems that the codes will run on the GPU simultaneously and the performance halved for GPT-2.
### 1.3.1 Matrix Multiplication FP32 / FP16
|               |               | n=128 | n=512  | n=2048 | n=8192  |
|---------------|---------------|-------|--------|--------|---------|
| torch.float32 | torch.float32 | 0.177 | 13.529 | 22.548 | 31.332  |
| torch.float16 | torch.float16 | 0.135 | 14.332 | 83.004 | 113.965 |
### 1.3.2 Vector Multiplication
|        | 65536  | 262144  | 1048576 | 4194304  |
|--------|--------|---------|---------|----------|
| TFLOPS | 0.005  | 0.021   | 0.060   | 0.151    |
| GB/s   | 43.634 | 170.351 | 482.503 | 1210.360 |
### 1.3.3 Bert Layer Forward / Forward+Backward
|                      | batch=2        | batch=4 | batch=8 | ... | batch=32 | batch=64 | batch=128 |
|----------------------|----------------|---------|---------|-----|----------|----------|-----------|
| fwd seq_len=128      | 10.438         | 10.515  | 20.551  | ... | 24.746   | 29.128   | 26.344    |
| fwd+bwd seq_len=128  | 12.359         | 12.929  | 22.756  | ... | 28.312   | 34.913   | 33.110    |
| fwd seq_len=512      | 11.630         | 17.457  | 23.578  | ... | 23.265   | 27.053   | 22.908    |
| fwd+bwd seq_len=512  | 17.850         | 23.490  | 28.509  | ... | 30.718   | 29.609   | 31.129    |
| [4 rows x 7 columns] | bert benchmark |         |         |     |          |          |           |
### 1.3.4 GPT-2 Layer Forward / Forward+Backward
|                      | batch=2 | batch=4 | batch=8 | batch=16 | batch=32 | batch=64 |             |
|----------------------|---------|---------|---------|----------|----------|----------|-------------|
| fwd seq_len=512      | 24.239  | 16.053  | 17.437  | 18.431   | 18.942   | 22.207   |             |
| fwd+bwd seq_len=512  | 13.958  | 19.066  | 19.602  | 21.672   | 23.938   | 22.010   |             |
| fwd seq_len=1024     | 15.328  | 13.767  | 15.640  | 16.816   | 15.441   | 17.479   |             |
| fwd+bwd seq_len=1024 | 16.832  | 17.845  | 19.782  | 21.105   | 18.723   | 17.505   | result gpt2 |

### 1.3.5 T5
|                         | batch=2 | batch=4 | batch=8 | ... | batch=32 | batch=64 | batch=128 |
|-------------------------|---------|---------|---------|-----|----------|----------|-----------|
| fwd seq_len=512         | 20.680  | 15.510  | 16.952  | ... | 18.930   | 18.928   | 18.620    |
| fwd+bwd seq_len=512     | 26.867  | 18.907  | 20.129  | ... | 22.338   | 23.109   | 22.703    |
| [2 rows x 7 columns] t5 |         |         |         |     |          |          |           |


# 2. Hands-on step to run it on a HPC.
## 2.1 login the HPC, install conda packages, solve the enviroment issues, salloc, etc
You may need to 'conda install packages' like install pandas, requests, etc.
## 2.2 Run the code
```bash
wget https://github.com/798253405/transformers-benchmarks-AMDGPU/blob/main/myBenchmark_AMDGPU.py
```
```bash
srun  -n  1  python3 myBenchmark_AMDGPU.py
```
You should be able to see the outputs like:
![2023-03-03_15-32](https://user-images.githubusercontent.com/7099084/222747051-c8fc66d6-59b7-4c95-8a94-8652ed8a8565.png)


# 3. Appendix
************************************below is the original readme file from https://github.com/mli/transformers-benchmarks **********************************
# Transformers Benchmarks

We benchmark real [TeraFLOPS](https://en.wikipedia.org/wiki/FLOPS) that training Transformer models can achieve on various GPUs, including single GPU, multi-GPUs, and multi-machines. It helps you to estimate how many machine times you need to train your large-scale Transformer models. 

The real performance depends on multiple factors, including your hardware, cooling, CUDA version, transformer models, hyper-parameters such as batch sizes, and implementations. We list the numbers we got on both personal PC and cloud instances. We also provide Jupyter notebooks for you to benchmark on your machines and workloads:

- [Understanding Transformer layer performance](micro_bench.ipynb)
- [Training BERT and GPT-2 with (multi-)GPUs](transformers.ipynb)

## Micro-Benchmarking Summary

Measure the TFLOPS for various micro-benchmarkings. Results are from running [micro_bench.ipynb](micro_bench.ipynb).

|                                        | A100      |  A6000   | V100      | 3090 Ti  |
| -------------------------------------- | :-------: | :------: | :-------: | :------: |
| Theory TF32(FP32) / FP16               | 156 / 312 | 75 / 150 | 16 / 125  | 80 / 160 |
| Memory (GB) / Bandwidth (GB/s)         | 80 / 2039 | 48 / 768 | 32 / 900  | 24 / 1008 |
| Approximate Price $                    |  16,000   |  4,000   |   3,500   |  1,500   |
| Matrix Multiplication FP32 / FP16      | 116 / 230 | 60 / 95  |  14 / 95  | 42 / 81  |
| Vector Multiplication                  |   0.202   |  0.082   |   0.098   |  0.107   |
| Bert Layer Forward / Forward+Backward  | 110 / 136 | 60 / 70  |  53 / 64  | 56 / 62  |
| GPT-2 Layer Forward / Forward+Backward |  45 / 53  | 35 / 38  |  32 / 36  | 37 / 39  |
| T5 Encoder Forward / Forward+Backward  |  44 / 56  | 34 / 41  |  31 / 38  | 36 / 41  |
| T5 Decoder Forward / Forward+Backward  |  38 / 47  | 28 / 34  |  26 / 32  | 30 / 36  |



## Set Up

You need a CUDA-enabled pytorch to run workloads. We recommend you to use the latest version CUDA and pytorch for better performance. One easy way is using [nvidia docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Once installed, you can find latest tag of the [pytorch image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), for exmaple, `22.07-py3`, then run 

```bash
sudo docker run --gpus all -it --rm -p 8888:8888 -v ~:/workspace \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	nvcr.io/nvidia/pytorch:22.07-py3
```

After the docker is running, execute  `jupyter notebook` in the docker's shell to open this notebook.
