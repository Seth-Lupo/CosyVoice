import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio


def cosyvoice2_example():
    """ CosyVoice2 Usage, check https://funaudiollm.github.io/cosyvoice2/ for more details
    """
    cosyvoice = CosyVoice2(model_dir='pretrained_models/CosyVoice2-0.5B')

    # Cross-lingual inference using English prompt audio
    prompt_wav = './asset/cross_lingual_prompt.wav'

    # Basic cross-lingual synthesis
    for i, j in enumerate(cosyvoice.inference_cross_lingual(
        'The sun rose over the mountains, casting golden rays across the peaceful valley below.',
        prompt_wav
    )):
        torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # Fine grained control with laughter
    for i, j in enumerate(cosyvoice.inference_cross_lingual(
        'While telling that absurd story, he suddenly [laughter] stopped because he made himself laugh [laughter].',
        prompt_wav
    )):
        torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # Instruct mode - speak with emotion
    for i, j in enumerate(cosyvoice.inference_instruct2(
        'I am so excited to share this wonderful news with everyone today!',
        'Speak with enthusiasm and joy.<|endofprompt|>',
        prompt_wav
    )):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # Streaming with text generator (useful for LLM output)
    def text_generator():
        yield 'The weather today is absolutely beautiful. '
        yield 'The sun is shining brightly in the clear blue sky. '
        yield 'A gentle breeze rustles through the leaves. '
        yield 'It is a perfect day for a walk in the park.'

    for i, j in enumerate(cosyvoice.inference_cross_lingual(text_generator(), prompt_wav, stream=False)):
        torchaudio.save('streaming_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def main():
    cosyvoice2_example()


if __name__ == '__main__':
    main()
