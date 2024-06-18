# TensorRT-LLM Backend
한국어 버전의 설명은 [여기](./docs/README_ko.md)를 참고하시기 바랍니다.

## Introduction
Here's a brief summary of the process of converting an LLM model to TensorRT and serving the converted model with Triton.
This tutorial uses the [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend) repositories, both based on version v0.10.0.
The entire process, from LLM conversion to Triton serving, can be carried out in a Linux environment.
The following tutorial uses Microsoft's [Phi3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) as an example and explains the process from installing the NVIDIA Container Toolkit to deploying Triton.
<br><br><br>

## Environment Settings
### 1. [NVIDIA Container Toolkit Settings](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
#### 1.1. Configure the production repository:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### 1.2. Configure the repository to use experimental packages:
```bash
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 1.3. Install the NVIDIA Container Toolkit packages:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
<br><br>


### 2. TensorRT-LLM Environment Settings
#### 2.1. Clone the TensorRT-LLM Repository
```bash
# submodule update
git submodule update --init --recursive

# checkout to v0.10.0
cd tensorrt_llm
git fetch origin refs/tags/v0.10.0
git checkout tags/v0.10.0
```

### 2.2. Docker Environment Settings
We will set up the TensorRT-LLM execution environment using [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags)'s docker image.
* PATH_OF_THIS_REPO: Path of `tensorrtllm_backend` repository.
* PATH_OF_LOCAL_CACHE: Path of cache folder. Generally, the cache folder is created in the home directory (e.g. `~/.cache/`).
```bash
docker run --name tensorrt-llm --runtime=nvidia --gpus all --entrypoint /bin/bash -it -d -v ${PATH_OF_THIS_REPO}:/tensorrtllm_backend -v ${PATH_OF_LOCAL_CACHE}:/root/.cache nvidia/cuda:12.4.0-devel-ubuntu22.04
```
After executing the above command, run the container named tensorrt-llm.
```bash
docker exec -it tensorrt-llm /bin/bash
```

#### 2.3. Setting up a Python Environment
We will now install Python 3.10 inside the Docker container.
```bash
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs
```
Then, we will install TensorRT-LLM package.
```bash
cd /tensorrtllm_backend
git lfs install
pip3 install tensorrt-llm==0.10.0
```

#### 2.4. Verification of the installation.
If executing each of the following commands does not result in errors, then the installation is considered successful.
```bash
nvidia-smi
```
```bash
# it will print version of tensorrt_llm
python3 -c "import tensorrt_llm"
```
```bash
python -c "import torch; assert torch.cuda.is_available()"
```
<br><br>


### 3. Convert LLM to TensorRT
#### 2.1. Clone the TensorRT-LLM Repository
```bash
# submodule update
git submodule update --init --recursive

# checkout to v0.10.0
cd tensorrt_llm
git fetch origin refs/tags/v0.10.0
git checkout tags/v0.10.0
```
