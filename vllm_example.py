import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm


def cosyvoice2_example():
    """ CosyVoice2 vllm usage
    """
    cosyvoice = CosyVoice2(model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_cross_lingual(
            'The sun rose over the mountains, casting golden rays across the peaceful valley below.',
            './asset/cross_lingual_prompt.wav',
            stream=False
        )):
            continue


def main():
    cosyvoice2_example()


if __name__ == '__main__':
    main()
