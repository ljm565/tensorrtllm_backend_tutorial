# TensorRT-LLM Backend Tutorial
한국어 버전의 설명은 [여기](./docs/README_ko.md)를 참고하시기 바랍니다.

## Introduction
Here's a brief summary of the process of converting an LLM model to TensorRT and serving the converted model with Triton.
This tutorial uses the [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend) repositories, both based on version v0.10.0.
The entire process, from LLM conversion to Triton serving, can be carried out in a Linux environment.
The following tutorial uses Microsoft's [Phi3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) as an example and explains the process from installing the NVIDIA Container Toolkit to deploying Triton.
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
We will set up the TensorRT-LLM execution environment using [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags)'s docker image.
* `PATH_OF_THIS_REPO`: Path of `tensorrtllm_backend` repository.
* `PATH_OF_LOCAL_CACHE`: Path of cache folder. Generally, the cache folder is created in the home directory (e.g. `~/.cache/`).
```bash
docker run --name tensorrt-llm --runtime=nvidia --gpus all --entrypoint /bin/bash -it -d -v ${PATH_OF_THIS_REPO}:/tensorrtllm_backend -v ${PATH_OF_LOCAL_CACHE}:/root/.cache nvidia/cuda:12.4.0-devel-ubuntu22.04
```
After executing the above command, run the container named tensorrt-llm.
```bash
docker exec -it tensorrt-llm /bin/bash
```

#### 2.3. Setting up a Python environment
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
Here, we will use [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) model as an example.
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
There are several ways to create a Triton backend Docker image for model serving.
The first method involves using the [NGC Triton image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).
However, as of `06.16.2024`, the latest version of this image only supports up to TensorRT-LLM and TensorRT-LLM backend version v0.7.0.
Therefore, to use the latest models like Phi3, LLaMA3, etc., you need to use v0.10.0, and for this version, you need to build the image yourself using the following command.
I would take quite long time and you can get 72GB `triton_trt_llm` Docker image.
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
Copy all engine data generated in the process from 3.4 to `triton_model_repo`.
```bash
cd /tensorrtllm_backend
mkdir triton_model_repo

# Copy the example models to the model repository
cp -r all_models/inflight_batcher_llm/tensorrt_llm triton_model_repo/

# Copy the TRT engine to triton_model_repo/tensorrt_llm/1/
cp tensorrt_llm/examples/phi/phi-engine/* triton_model_repo/tensorrt_llm/1/
```
And then use one of the following commands to deploy a model on Triton.
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
# Your Python code here
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
  decoupled: true
}

dynamic_batching {
    preferred_batch_size: [ 1 ]
    max_queue_delay_microseconds: 1000
}

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
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
#instance_group [
#  {
#    count: 1
#    kind : KIND_CPU
#  }
#]
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
# parameters: {
#   key: "max_tokens_in_paged_kv_cache"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "max_attention_window_size"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "sink_token_length"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "batch_scheduler_policy"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "kv_cache_free_gpu_mem_fraction"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "kv_cache_host_memory_bytes"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "kv_cache_onboard_blocks"
#   value: {
#     string_value: ""
#   }
# }
# enable_trt_overlap is deprecated and doesn't have any effect on the runtime
# parameters: {
#   key: "enable_trt_overlap"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "exclude_input_in_output"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "cancellation_check_period_ms"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "stats_check_period_ms"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "iter_stats_max_iterations"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "request_stats_max_iterations"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "enable_kv_cache_reuse"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "normalize_log_probs"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "enable_chunked_context"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "gpu_device_ids"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "lora_cache_optimal_adapter_size"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "lora_cache_max_adapter_size"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "lora_cache_gpu_memory_fraction"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "lora_cache_host_memory_bytes"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "decoding_mode"
#   value: {
#     string_value: ""
#   }
# }
# parameters: {
#   key: "executor_worker_path"
#   value: {
#     string_value: "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"
#   }
# }
# parameters: {
#   key: "medusa_choices"
#     value: {
#       string_value: ""
#   }
# }
# parameters: {
#   key: "gpu_weights_percent"
#     value: {
#       string_value: ""
#   }
# }
```
</details>
<br><br>

## Acknowledgement
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
* [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend)