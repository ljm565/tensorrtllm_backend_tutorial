# TensorRT-LLM Backend Tutorial

## Introduction
본 레포지토리는 LLM 모델을 TensorRT로 변환하고, 변환된 모델을 Triton으로 serving 하는 과정을 간략히 정리합니다.
여기서는 TensorRT-LLM, TensorRT-LLM Backend 레포지토리를 사용하며, 각각 v0.10.0 버전을 기준으로 tutorial을 진행합니다.
아래 LLM 변환부터 Triton 서빙까지의 과정은 Linux 환경에서 진행할 수 있습니다.
아래 튜토리얼은 Microsoft의 [Phi3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)를 예제로 사용하며, Nvidia Container Toolkit 설치부터, Triton 올리는 방법까지 과정을 설명합니다.

<br><br><br>

## Environment Settings
### 1. [NVIDIA Container Toolkit Settings](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
#### 1.1. Configure the production repository
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### 1.2. Configure the repository to use experimental packages
```bash
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### 1.3. Install the NVIDIA Container Toolkit packages
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
<br><br>


### 2. TensorRT-LLM Environment Settings
#### 2.1. Clone the TensorRT-LLM repository
```bash
# Submodule update
git submodule update --init --recursive

# Checkout to v0.10.0
cd tensorrt_llm
git fetch origin refs/tags/v0.10.0
git checkout tags/v0.10.0
```

#### 2.2. Docker environment settings
여기서 TensorRT-LLM 실행 환경을 구축하기 위해서 [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags)의 Docker image를 사용합니다.
* `PATH_OF_THIS_REPO`: Path of `tensorrtllm_backend` repository.
* `PATH_OF_LOCAL_CACHE`: Path of cache folder. Generally, the cache folder is created in the home directory (e.g. `~/.cache/`).
```bash
docker run --name tensorrt-llm --runtime=nvidia --gpus all --entrypoint /bin/bash -it -d -v ${PATH_OF_THIS_REPO}:/tensorrtllm_backend -v ${PATH_OF_LOCAL_CACHE}:/root/.cache nvidia/cuda:12.4.0-devel-ubuntu22.04
```
위 명령어를 통해 Docker container를 만든 후, tensorrt-llm이라는 이름의 container를 실행할 수 있습니다. 
```bash
docker exec -it tensorrt-llm /bin/bash
```

#### 2.3. Setting up a Python environment
여기서는 Docker container 안에서 Python 3.10을 설치합니다.
```bash
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs
```
그리고 TensorRT-LLM 패키지를 설치합니다.
```bash
cd /tensorrtllm_backend
git lfs install
pip3 install tensorrt-llm==0.10.0
```

#### 2.4. Verification of the installation.
모든 환경이 성공적으로 설치 되었다면 아래 명령어를 실행 했을 때 오류가 나타나지 않아야 합니다.
```bash
nvidia-smi
```
```bash
# It will print version of tensorrt_llm
python3 -c "import tensorrt_llm"
```
```bash
python -c "import torch; assert torch.cuda.is_available()"
```
<br><br>


### 3. Convert LLM to TensorRT
#### 3.1. Hugging Face login
```bash
cd /tensorrtllm_backend/tensorrt_llm
git lfs install

# If you're using the LLaMA models, qualifications might be required.
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token *****
```

#### 3.2. Install requirements
이제부터 [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) 모델을 가지고 튜토리얼을 진행합니다.
```bash
cd examples/phi
pip3 install -r requirements.txt
```

#### 3.3. Convert LLM weights to TensorRT-LLM checkpoint format
```bash
# You can use the `--model_dir` option to specify a Hugging Face model repository.
python3 ./convert_checkpoint.py --model_type "Phi-3-mini-128k-instruct" --model_dir "microsoft/Phi-3-mini-128k-instruct" --output_dir ./phi-checkpoint --dtype float16
```

#### 3.4. Compile model
```bash
# Build a float16 engine using a single GPU and HF weights.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.
# --tp_size and --pp_size are the model shard size
trtllm-build \
    --checkpoint_dir ./phi-checkpoint \
    --output_dir ./phi-engine \
    --gemm_plugin float16 \
    --max_batch_size 8 \
    --max_input_len 1024 \
    --max_output_len 1024 \
    --tp_size 1 \
    --pp_size 1
```

#### 3.5. Compiled model test
```bash
python3 ../run.py --engine_dir phi-engine/ --max_output_len 1024 --no_add_special_tokens --tokenizer_dir microsoft/Phi-3-mini-128k-instruct --input_text "What is the color of the sky?"
```
<br><br>


### 4. Deploying a Model through Triton
#### 4.1.  Docker image build for Triton serving
모델 서빙을 위한 트리톤 backend 도커 이미지를 만들기 위해서 몇 가지 방법이 있습니다.
첫 번째는 [NGC Triton image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)를 사용하는 것입니다.
하지만 `24.06.16`를 기준으로 최신 버전의 이미지는 TensorRT-LLM과 TensorRT-LLM backend 버전 v0.7.0까지 밖에 지원하지 않습니다.
따라서 Phi3, LLaMA3 등의 최신 모델을 사용하기 위해서는 v0.10.0을 사용해아하며, 이 버전을 위해서는 아래 명령어로 직접 이미지를 빌드해야합니다.
이 과정은 시간이 많이 소요되며, 완료 되면 `triton_trt_llm`의 72GB 가량 크기의 Docker image가 생성 됩니다.
```bash
cd /tensorrtllm_backend
git lfs install
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .
```

#### 4.2. Execute the docker container
* `PATH_OF_THIS_REPO`: Path of `tensorrtllm_backend` repository.
```bash
docker run -d -it --name tensorrtllm-backend --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -p8000:8000 -p8001:8001 -p8002:8002 -v ${PATH_OF_THIS_REPO}:/tensorrtllm_backend triton_trt_llm 
docker exec -it tensorrtllm-backend /bin/bash
```

#### 4.3. Triton serving
3.4의 과정에서 생성된 engine 데이터들을 모두 `triton_model_repo`로 복사합니다.
```bash
cd /tensorrtllm_backend
mkdir triton_model_repo

# Copy the example models to the model repository
cp -r all_models/inflight_batcher_llm/tensorrt_llm triton_model_repo/

# Copy the TRT engine to triton_model_repo/tensorrt_llm/1/
cp tensorrt_llm/examples/phi/phi-engine/* triton_model_repo/tensorrt_llm/1/
```
그리고 아래 명령어 중 하나로 Triton에 모델을 띄웁니다.
```bash
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py --model_repo=/tensorrtllm_backend/triton_model_repo --multi-model

# (Optional) You can also directly run Triton Server using the tritonserver command.
tritonserver --model-repository=/tensorrtllm_backend/triton_model_repo --model-control-mode=explicit --load-model=*
```

#### 4.4. Model's config.pbtxt
<details>
<summary><code>config.pbtxt</code> of the model</summary>

```python
 # Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 4

model_transaction_policy {
  decoupled: false
}

dynamic_batching {
    preferred_batch_size: [ 1 ]
    max_queue_delay_microseconds: 1000
}

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 , 1 ]
    allow_ragged_batch: false
  },
  {
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
  },
  {
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "draft_input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
    optional: true
    allow_ragged_batch: true
  },
  {
    name: "draft_logits"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
    optional: true
    allow_ragged_batch: true
  },
  {
    name: "draft_acceptance_threshold"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "end_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "pad_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "stop_words_list"
    data_type: TYPE_INT32
    dims: [ 2, -1 ]
    optional: true
    allow_ragged_batch: true
  },
  {
    name: "bad_words_list"
    data_type: TYPE_INT32
    dims: [ 2, -1 ]
    optional: true
    allow_ragged_batch: true
  },
  {
    name: "embedding_bias"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
    allow_ragged_batch: true
  },
  {
    name: "beam_width"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "runtime_top_k"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "runtime_top_p"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "runtime_top_p_min"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "runtime_top_p_decay"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "runtime_top_p_reset_ids"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "len_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "early_stopping"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "repetition_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "min_length"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "beam_search_diversity_rate"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "presence_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "frequency_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "random_seed"
    data_type: TYPE_UINT64
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "return_log_probs"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "return_context_logits"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "return_generation_logits"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "stop"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  },
  {
    name: "streaming"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  },
  {
    name: "prompt_embedding_table"
    data_type: TYPE_FP16
    dims: [ -1, -1 ]
    optional: true
    allow_ragged_batch: true
  },
  {
    name: "prompt_vocab_size"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  # the unique task ID for the given LoRA.
  # To perform inference with a specific LoRA for the first time `lora_task_id` `lora_weights` and `lora_config` must all be given.
  # The LoRA will be cached, so that subsequent requests for the same task only require `lora_task_id`.
  # If the cache is full the oldest LoRA will be evicted to make space for new ones.  An error is returned if `lora_task_id` is not cached.
  {
    name: "lora_task_id"
	data_type: TYPE_UINT64
	dims: [ 1 ]
    reshape: { shape: [ ] }
	optional: true
  },
  # weights for a lora adapter shape [ num_lora_modules_layers, D x Hi + Ho x D ]
  # where the last dimension holds the in / out adapter weights for the associated module (e.g. attn_qkv) and model layer
  # each of the in / out tensors are first flattened and then concatenated together in the format above.
  # D=adapter_size (R value), Hi=hidden_size_in, Ho=hidden_size_out.
  {
    name: "lora_weights"
	data_type: TYPE_FP16
	dims: [ -1, -1 ]
	optional: true
	allow_ragged_batch: true
  },
  # module identifier (same size a first dimension of lora_weights)
  # See LoraModule::ModuleType for model id mapping
  #
  # "attn_qkv": 0     # compbined qkv adapter
  # "attn_q": 1       # q adapter
  # "attn_k": 2       # k adapter
  # "attn_v": 3       # v adapter
  # "attn_dense": 4   # adapter for the dense layer in attention
  # "mlp_h_to_4h": 5  # for llama2 adapter for gated mlp layer after attention / RMSNorm: up projection
  # "mlp_4h_to_h": 6  # for llama2 adapter for gated mlp layer after attention / RMSNorm: down projection
  # "mlp_gate": 7     # for llama2 adapter for gated mlp later after attention / RMSNorm: gate
  #
  # last dim holds [ module_id, layer_idx, adapter_size (D aka R value) ]
  {
    name: "lora_config"
	data_type: TYPE_INT32
	dims: [ -1, 3 ]
	optional: true
	allow_ragged_batch: true
  }
]
output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1, -1 ]
  },
  {
    name: "sequence_length"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "cum_log_probs"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "output_log_probs"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  },
  {
    name: "context_logits"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  },
  {
    name: "generation_logits"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
  }
]
instance_group [
 {
   count: 1
   kind : KIND_CPU
 }
]
parameters: {
  key: "max_beam_width"
  value: {
    string_value: "1"
  }
}
parameters: {
  key: "FORCE_CPU_ONLY_INPUT_TENSORS"
  value: {
    string_value: "no"
  }
}
parameters: {
  key: "gpt_model_type"
  value: {
    string_value: "V1"
  }
}
parameters: {
  key: "gpt_model_path"
  value: {
    string_value: "/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1"
  }
}
```
</details>

#### 4.5. Inferencing via Triton client library
Here, we will introduce the process of performing inference using gRPC communication through Python code with the .
여기서는 [Triton client 라이브러리](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/README.html)를 통해 Python과 gRPC 통신으로 inference 하는 과정을 소개합니다.
만약 `tritonclient` 라이브러리가 설치 되지 않았다면 모델을 서빙하고 있는 도커 환경에 아래와 같은 명령어로 라이브러리를 설치하면 됩니다.
```bash
pip3 install tritonclient[all]
```
그리고 아래 Python 코드로 모델 추론을 할 수 있습니다.
```python
import time
import numpy as np
from transformers import AutoTokenizer
import tritonclient.grpc as grpcclient


prompt = """
<s><|user|>What is the color of the sky?<|end|>
<assistant>
"""

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
ids = tokenizer.encode(prompt, add_special_tokens=False)
ids = np.array(ids).reshape(1, len(ids), 1)
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

input_ids_arr = ids.astype(np.int32)
input_ids = grpcclient.InferInput("input_ids", input_ids_arr.shape, "INT32")
input_ids.set_data_from_numpy(input_ids_arr)

input_lengths_arr = np.array([[1024]]).astype(np.int32)
input_lengths = grpcclient.InferInput("input_lengths", input_lengths_arr.shape, "INT32")
input_lengths.set_data_from_numpy(input_lengths_arr)

request_output_len_arr = np.array([[1024]]).astype(np.int32)
request_output_len = grpcclient.InferInput("request_output_len", request_output_len_arr.shape, "INT32")
request_output_len.set_data_from_numpy(request_output_len_arr)

end_id_arr = np.array([[32007]]).astype(np.int32)
end_id = grpcclient.InferInput("end_id", end_id_arr.shape, "INT32")
end_id.set_data_from_numpy(end_id_arr)


response = triton_client.infer(
                model_name='tensorrt_llm', 
                inputs=[input_ids, input_lengths, request_output_len, end_id],
                request_id="llm_request"
            )

output = response.as_numpy("output_ids")
print('-'*100)
print(output[0][0][ids.shape[1]:])
print('-'*100)
print(tokenizer.decode(output[0][0][ids.shape[1]:]))
print('-'*100)
```
<br><br>

## Acknowledgement
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
* [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend)