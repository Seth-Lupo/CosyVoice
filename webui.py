# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['Cross-Lingual', 'Instruct']
instruct_dict = {
    'Cross-Lingual': '1. Upload or record a prompt audio file (max 30s)\n2. Enter the text you want to synthesize\n3. Click Generate',
    'Instruct': '1. Upload or record a prompt audio file\n2. Enter the text and instruction\n3. Click Generate'
}
stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # Validation
    if prompt_wav is None:
        gr.Warning('Please provide a prompt audio file!')
        yield (cosyvoice.sample_rate, default_data)
        return

    if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
        gr.Warning('Prompt audio sample rate {} is below {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
        yield (cosyvoice.sample_rate, default_data)
        return

    if mode_checkbox_group == 'Instruct' and instruct_text == '':
        gr.Warning('Please enter instruction text for Instruct mode')
        yield (cosyvoice.sample_rate, default_data)
        return

    set_all_random_seed(seed)

    if mode_checkbox_group == 'Cross-Lingual':
        logging.info('Running cross-lingual inference')
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('Running instruct inference')
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) - Text to Speech")
        gr.Markdown("#### Enter text, select inference mode, and follow the instructions")

        tts_text = gr.Textbox(label="Text to Synthesize", lines=2,
                              value="Hello, I am a speech synthesis model. How can I help you today?")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Inference Mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Instructions", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            stream = gr.Radio(choices=stream_mode_list, label='Streaming', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed (non-streaming only)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Upload prompt audio (16kHz+ recommended)')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Or record prompt audio')
        instruct_text = gr.Textbox(label="Instruction Text (for Instruct mode)", lines=1,
                                   placeholder="e.g., Speak with enthusiasm<|endofprompt|>", value='')

        generate_button = gr.Button("Generate Audio")

        audio_output = gr.Audio(label="Generated Audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice2(model_dir=args.model_dir)

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
