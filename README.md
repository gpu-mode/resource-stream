# CUDA MODE Resource Stream

Here you find a collection of CUDA related material (books, papers, blog-post, youtube videos, tweets, implementations etc.). We also collect information to higher level tools for performance optimization and kernel development like [Triton](https://triton-lang.org) and `torch.compile()` ... whatever makes the GPUs go brrrr. 

You know a great resource we should add? Please see [How to contribute](#how-to-contribute).

## 1st Contact with CUDA
- [An Easy Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [CUDA Toolkit Documentation ](https://docs.nvidia.com/cuda/)
- Basic terminology: Thread block, Warp, Streaming Multiprocessor: [Wiki: Thread Block](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)), [A tour of CUDA](https://tbetcke.github.io/hpc_lecture_notes/cuda_introduction.html)
- [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [OLCF NVIDIA CUDA Training Series](https://www.olcf.ornl.gov/cuda-training-series/), talk recordings can be found under the presentation footer for each lecture; [exercises](https://github.com/olcf/cuda-training-series)
- [GTC 2022 - CUDA: New Features and Beyond - Stephen Jones](https://www.youtube.com/watch?v=SAm4gwkj2Ko)
- Intro video: [Writing Code That Runs FAST on a GPU](https://youtu.be/8sDg-lD1fZQ)


## 2nd Contact
- [CUDA Refresher](https://developer.nvidia.com/blog/tag/cuda-refresher/)


## Papers, Case Studies
- [A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library](https://arxiv.org/abs/2312.11918)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Anatomy of high-performance matrix multiplication](https://dl.acm.org/doi/10.1145/1356052.1356053)


## Books
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311)
- [Cuda by Example: An Introduction to General-Purpose Gpu Programming](https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf); [code](https://github.com/tpn/cuda-by-example)
- [The CUDA Handbook](https://www.cudahandbook.com/)


## CUDA Grandmasters

### Tri Dao
- x: [@tri_dao](https://twitter.com/tri_dao)
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention), [paper](https://arxiv.org/abs/2205.14135)
- [state-spaces/mamba](https://github.com/state-spaces/mamba), paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), minimal impl: [mamba-minimal](https://github.com/johnma2006/mamba-minimal)


### Tim Dettmers
- x: [@Tim_Dettmers](https://twitter.com/Tim_Dettmers)
- [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes), docs: [docs](https://bitsandbytes.readthedocs.io/en/latest/)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)


## Practice
- [Sasha Rush's GPU Puzzles](https://github.com/srush/GPU-Puzzles), dshah3's [CUDA C++ version](https://github.com/dshah3/GPU-Puzzles) & [walkthrough video](https://www.youtube.com/watch?v=3frRR6fycgM)


## PyTorch Performance Optimization
- [Accelerating Generative AI with PyTorch: Segment Anything, Fast](https://pytorch.org/blog/accelerating-generative-ai/)
- [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/)
- [Speed, Python: Pick Two. How CUDA Graphs Enable Fast Python Code for Deep Learning](https://blog.fireworks.ai/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning-353bf6241248)
- [Performance Debugging of Production PyTorch Models at Meta](https://pytorch.org/blog/performance-debugging-of-production-pytorch-models-at-meta/)


## PyTorch Internals & Debugging
- [PyTorch Compiler Troubleshooting](https://github.com/pytorch/pytorch/blob/main/docs/source/torch.compiler_troubleshooting.rst)
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [Pytorch 2 internals](https://drive.google.com/file/d/1XBox0G3FI-71efQQjmqGh0-VkCd-AHPL/view)
- Understanding GPU memory: [1: Visualizing All Allocations over Time](https://pytorch.org/blog/understanding-gpu-memory-1/), [2: Finding and Removing Reference Cycles](https://pytorch.org/blog/understanding-gpu-memory-2/) 
- Debugging memory using snapshots: [Debugging PyTorch memory use with snapshots](https://zdevito.github.io/2022/08/16/memory-snapshots.html)
- Trace Analyzer:  [PyTorch Trace Analysis for the Masses](https://pytorch.org/blog/trace-analysis-for-masses/)



## Code / Libs
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)


## Essentials
- [Triton compiler tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch: Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html), Code: [pytorch/extension-cpp](https://github.com/pytorch/extension-cpp/tree/master)
- [PyTorch C++ API](https://pytorch.org/cppdocs/index.html)
- [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/)
- [NVIDIA Tensor Core Programming](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
- [GPU Programming: When, Why and How?](https://enccs.github.io/gpu-programming/#)
- [How GPU Computing Works | GTC 2021](https://youtu.be/3l10o0DYJXg?si=t5FHswnibAbo3s0t) (more basic than the 2022 version)
- [How CUDA Programming Works | GTC 2022](https://youtu.be/n6M8R8-PlnE?si=cJ4dWtpYaPoIuJ0q)


## Profiling
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [mcarilli/nsight.sh](https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223) - Favorite nsight systems profiling commands for PyTorch scripts
- [Profiling GPU Applications with Nsight Systems](https://www.youtube.com/watch?v=kKANP0kL_hk) 


## News
- [SemiAnalysis](https://www.semianalysis.com/)


## Technical Blog Posts
- [Cooperative Groups: Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/cooperative-groups/)


## Hardware Architecture
- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [NVIDIA GH200 Whitepaper](https://resources.nvidia.com/en-us-grace-cpu/nvidia-grace-hopper)
- [AMD CDNA 3 Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD MI300X Data Sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)


## pscan project
- GPU Gems: [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda), [PDF version (2007)](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf), impl: [stack overflow](https://stackoverflow.com/a/30835030/387870), nicer impl: [mattdean1/cuda](https://github.com/mattdean1/cuda)
- [Accelerating Reduction and Scan Using Tensor Core Units](https://arxiv.org/abs/1811.09736)
- Thrust: [Prefix Sums](https://docs.nvidia.com/cuda/thrust/index.html#prefix-sums), Reference: [scan variants](https://thrust.github.io/doc/group__prefixsums.html)
- [CUB](https://nvlabs.github.io/cub/), part of cccl: [NVIDIA/cccl/tree/main/cub](https://github.com/NVIDIA/cccl/tree/main/cub)
- SAM Algorithm: [Higher-Order and Tuple-Based Massively-Parallel Prefix Sums](https://userweb.cs.txstate.edu/~mb92/papers/pldi16.pdf) (licensed for non commercial use only)
- CUB Algorithm: [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
- Group Experiments: [johnryan465/pscan](https://github.com/johnryan465/pscan), [andreaskoepf/pscan_kernel](https://github.com/andreaskoepf/pscan_kernel)


## Triton Kernels / Examples

- [`unsloth`](https://github.com/unslothai/unsloth) that implements custom kernels in Triton for faster QLoRA training
- Custom implementation of relative position attention ([link](https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/flash_4.py))
- Tri Dao's Triton implementation of Flash Attention: [flash_attn_triton.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)


## How to contribute
To share interesting CUDA related links please create a pull request for this file. See [editing files](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files) in the github documentation.

Or contact us on the **CUDA MODE** discord server: [https://discord.gg/XsdDHGtk9N](https://discord.gg/XsdDHGtk9N)
