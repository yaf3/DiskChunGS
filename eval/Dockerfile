FROM lsgs-slam:latest

RUN pip uninstall torch torchvision
RUN pip cache purge  # Clear pip cache
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN pip install evo numpy scipy scikit-image lpips pillow tqdm plyfile pytorch-msssim opencv-python

RUN pip install numpy==1.24.4

WORKDIR /installation
COPY submodules /installation/submodules

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV FORCE_CUDA=1

RUN pip install /installation/submodules/simple-knn/ 
RUN pip install /installation/submodules/diff-gaussian-rasterization/

WORKDIR /large_scale_gaussian_slam/eval

