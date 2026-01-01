![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice2🤠&text2=Text-to-Speech%20💖%20Large%20Language%20Model&width=800&height=210)

## CosyVoice 2.0 - 0.5B

**CosyVoice 2.0**: [Demos](https://funaudiollm.github.io/cosyvoice2/); [Paper](https://arxiv.org/pdf/2412.10117); [Modelscope](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B); [HuggingFace](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)

## Highlight

**CosyVoice 2.0** is an advanced text-to-speech (TTS) system based on large language models (LLM), designed for streaming speech synthesis.

### Key Features
- **Streaming Support**: Supports both text-in streaming and audio-out streaming with low latency
- **Zero-Shot Voice Cloning**: Clone any voice with just a few seconds of audio prompt
- **Cross-Lingual Synthesis**: Synthesize speech in different languages while maintaining the voice characteristics
- **Natural Language Control**: Use natural language instructions to control speech style
- **vLLM Acceleration**: Supports vLLM for faster inference

## Install

### Clone and install

- Clone the repo
    ``` sh
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    # If you failed to clone the submodule due to network failures, please run the following command until success
    cd CosyVoice
    git submodule update --init --recursive
    ```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

    ``` sh
    conda create -n cosyvoice -y python=3.10
    conda activate cosyvoice
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

    # If you encounter sox compatibility issues
    # ubuntu
    sudo apt-get install sox libsox-dev
    # centos
    sudo yum install sox sox-devel
    ```

### Model download

Download the pretrained `CosyVoice2-0.5B` model and optionally `CosyVoice-ttsfrd` resource for better text normalization.

``` python
# modelscope SDK model download
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')

# for oversea users, huggingface SDK model download
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

Optionally, you can unzip `ttsfrd` resource and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use wetext by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### Basic Usage

Follow the code in `example.py` for detailed usage.
```sh
python example.py
```

#### vLLM Usage
CosyVoice2 supports **vLLM 0.11.x+ (V1 engine)** and **vLLM 0.9.0 (legacy)**.
Older vllm version(<0.9.0) do not support CosyVoice inference, and versions in between (e.g., 0.10.x) are not tested.

Notice that `vllm` has a lot of specific requirements. You can create a new env in case your hardware does not support vllm.

``` sh
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
# for vllm==0.9.0
pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# for vllm>=0.11.0
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
python vllm_example.py
```

#### Start web demo

You can use our web demo page to get familiar with CosyVoice2 quickly.

``` python
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

#### Advanced Usage

For advanced users, training and inference scripts are provided in `examples/libritts/cosyvoice2`.

#### Using Nvidia TensorRT-LLM for deployment

Using TensorRT-LLM to accelerate CosyVoice2 LLM gives 4x acceleration compared to the HuggingFace transformers implementation.
To quick start:

``` sh
cd runtime/triton_trtllm
docker compose up -d
```
For more details, check [runtime/triton_trtllm](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/FunAudioLLM/CosyVoice/issues).

## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Citations

``` bibtex
@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}
```

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
