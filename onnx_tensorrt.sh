wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.18.1.tgz
rm onnxruntime-linux-x64-gpu-1.18.1.tgz
sudo cp -r onnxruntime-linux-x64-gpu-1.18.1/* /usr/local/
rm -rf onnxruntime-linux-x64-gpu-1.18.1
sudo ldconfig

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.10.0/local_repo/nv-tensorrt-local-repo-ubuntu2004-10.10.0-cuda-11.8_1.0-1_amd64.deb
os="ubuntu2004"
tag="10.10.0-cuda-11.8"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
echo "Package: *
Pin: origin \"\"
Pin-Priority: 1001" | sudo tee /etc/apt/preferences.d/tensorrt-local
sudo apt-get update
sudo apt-get install tensorrt=10.10.0.31-1+cuda11.8 libnvinfer10=10.10.0.31-1+cuda11.8 libnvinfer-plugin10=10.10.0.31-1+cuda11.8 libnvinfer-vc-plugin10=10.10.0.31-1+cuda11.8 libnvinfer-lean10=10.10.0.31-1+cuda11.8 libnvinfer-dispatch10=10.10.0.31-1+cuda11.8 libnvonnxparsers10=10.10.0.31-1+cuda11.8 libnvinfer-bin=10.10.0.31-1+cuda11.8 libnvinfer-dev=10.10.0.31-1+cuda11.8 libnvinfer-lean-dev=10.10.0.31-1+cuda11.8 libnvinfer-plugin-dev=10.10.0.31-1+cuda11.8 libnvinfer-vc-plugin-dev=10.10.0.31-1+cuda11.8 libnvinfer-dispatch-dev=10.10.0.31-1+cuda11.8 libnvonnxparsers-dev=10.10.0.31-1+cuda11.8 libnvinfer-samples=10.10.0.31-1+cuda11.8 python3-libnvinfer-dev=10.10.0.31-1+cuda11.8 libnvinfer-win-builder-resource10=10.10.0.31-1+cuda11.8 libnvinfer-headers-python-plugin-dev=10.10.0.31-1+cuda11.8 libnvinfer-headers-dev=10.10.0.31-1+cuda11.8 libnvinfer-headers-plugin-dev=10.10.0.31-1+cuda11.8 python3-libnvinfer=10.10.0.31-1+cuda11.8 python3-libnvinfer-lean=10.10.0.31-1+cuda11.8 python3-libnvinfer-dispatch=10.10.0.31-1+cuda11.8
rm -rf nv-tensorrt-local-repo-ubuntu2004-10.10.0-cuda-11.8_1.0-1_amd64.deb



wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-11.8_1.0-1_amd64.deb
os="ubuntu2004"
tag="8.6.1-cuda-11.8"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
echo "Package: *
Pin: origin \"\"
Pin-Priority: 1001" | sudo tee /etc/apt/preferences.d/tensorrt-local
sudo apt-get update
sudo apt-get install -y \
    tensorrt=8.6.1.6-1+cuda11.8 \
    libnvinfer8=8.6.1.6-1+cuda11.8 \
    libnvinfer-plugin8=8.6.1.6-1+cuda11.8 \
    libnvonnxparsers8=8.6.1.6-1+cuda11.8 \
    libnvinfer-bin=8.6.1.6-1+cuda11.8 \
    libnvinfer-dev=8.6.1.6-1+cuda11.8 \
    libnvinfer-plugin-dev=8.6.1.6-1+cuda11.8 \
    libnvonnxparsers-dev=8.6.1.6-1+cuda11.8 \
    libnvinfer-samples=8.6.1.6-1+cuda11.8 \
    python3-libnvinfer-dev=8.6.1.6-1+cuda11.8 \
    python3-libnvinfer=8.6.1.6-1+cuda11.8