import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import time

import torch
import kenlm
import subprocess
import streamlit as st
import soundfile as sf
# from bs4 import BeautifulSoup
# from pydub import AudioSegment
# from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
# from punctuator import Punctuator
from speech_recognition import Recognizer, Microphone
from streamlit_quill import st_quill
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


st.set_page_config(page_title="Nhập văn bản bằng giọng nói", page_icon="🎙️")

if 'content' not in st.session_state:
    st.session_state['content'] = ''

if 'recording' not in st.session_state:
    st.session_state['recording'] = False

@st.cache_resource()
def to_text(html_content):
    soup = BeautifulSoup(html_content, features="html.parser")
    return soup.get_text()

@st.cache_data() # Disable for Debugging
def load_wav2vec2(checkpoint):
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    return model, processor

@st.cache_resource() # Disable for Debugging
def get_decoder_ngram_model(_tokenizer, ngram_lm_path):
    tokenizer = _tokenizer
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    # convert ctc blank character representation
    vocab_list[tokenizer.pad_token_id] = ""
    # replace special characters
    vocab_list[tokenizer.unk_token_id] = ""
    # vocab_list[tokenizer.bos_token_id] = ""
    # vocab_list[tokenizer.eos_token_id] = ""
    # convert space character representation
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet,
                                language_model=LanguageModel(lm_model))
    
    return decoder

@st.cache_resource() # Disable for Debugging
def load_punctuator():
    punctuator = Punctuator()
    return punctuator

@st.cache_resource() # Disable for Debugging
def load_recognizer():
    recognizer = Recognizer()
    recognizer.energy_threshold = 400
    return recognizer

def infer(waveform, model, processor, language_model, punctuator) -> list:
    # tokenize
    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values

    # retrieve logits
    with torch.no_grad():
        logits = model(input_values).logits[0]

    # take argmax and decode
    pred_ids = torch.argmax(logits, dim=-1)
    greedy_search_output = processor.decode(pred_ids)
    beam_search_output = language_model.decode(logits.cpu().detach().numpy(), beam_width=500)
    print("Greedy search output: {}".format(greedy_search_output))
    print("Beam search output: {}".format(beam_search_output))

    result = punctuator.process(beam_search_output)[0]

    # breakpoint()

    return result

def pipeline(recognizer, model, processor, ngram_lm_model, punctuator):
    with Microphone(sample_rate=16000) as source:
        audio = recognizer.listen(source, phrase_time_limit=3) # Bytes

        data = io.BytesIO(audio.get_wav_data()) # Object(Bytes) Ex: 96300
        audio = AudioSegment.from_file(data) # Object(Object) Ex: 96300
        waveform = torch.FloatTensor(audio.get_array_of_samples()) # Tensor(Array(Int)) Ex: 48128 =>  1 Int = 2 Bytes

        start_time = time.time()
        transcription = infer(waveform, model, processor, ngram_lm_model, punctuator)
        end_time = time.time()
    
    return transcription, end_time - start_time

if __name__ == "__main__":
    st.sidebar.title("Nhập văn bản bằng giọng nói v1.0")
    st.sidebar.markdown("Lý Thành Lâm")
    
    st.title("Nhập văn bản bằng giọng nói")
    option = st.selectbox(
    'Ngôn ngữ sử dụng:',
    ('Tiếng Việt 🇻🇳', ''))

    language = {
        'Tiếng Việt 🇻🇳': 'vi',
        'English': 'en',
    }
    
    if language[option] == 'vi':
        # model, processor = load_wav2vec2('nguyenvulebinh/wav2vec2-base-vietnamese-250h')
        # ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, 'models/vi_lm_4grams.bin')
        # punctuator = load_punctuator()
        pass
    else:
        st.error('Ngôn ngữ chưa được hỗ trợ')
        st.stop()

    recognizer = load_recognizer()
    
    content = st_quill(
        value=st.session_state['content'],
        placeholder="Nhập văn bản hoặc ấn nút ghi âm và bắt đầu nói",
        html=True,
    )

    # st.session_state['content'] = content
    # st.write(st.session_state)
    st.write(content)

    notification = None

    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("Copy", use_container_width=True):
    #         subprocess.run("pbcopy", text=True, input=content)
    #         notification = "Sao chép thành công"
    # with col2:
    #     if st.button("Clear", use_container_width=True):
    #         st.session_state['content'] = ''
    #         st.rerun()

    if notification:
        st.toast(notification)

    # if not st.session_state['recording']:
    #     if st.button("Ghi âm", type="primary", use_container_width=True):
    #         st.session_state['recording'] = True
    #         st.rerun()
    # else:
    #     if st.button("Tạm dừng", use_container_width=True):
    #         st.session_state['recording'] = False
    #         st.rerun()

    if st.button("Ghi âm", type="primary", use_container_width=True) and not st.session_state['recording']:
        st.session_state['recording'] = True
        with Microphone(sample_rate=16000) as source:
            st.toast("Hãy bắt đầu nói...")
            audio_data = recognizer.listen(source, phrase_time_limit=3)
            wav_bytes = audio_data.get_wav_data()
            # wav_stream = io.BytesIO(wav_bytes)
            # audio_array, sampling_rate = sf.read(wav_stream)
            # waveform = torch.FloatTensor(audio_array)

            st.audio(wav_bytes)
        # transcription, runtime = pipeline(recognizer, model, processor, ngram_lm_model, punctuator)
        # st.toast(f"Kết quả trả về sau {runtime} giây")
        # if content[-4:] == '</p>':
        #     st.session_state['content'] = content[:-4] + transcription + ' ' + '</p>'
        # else:
        #     st.session_state['content'] = content + transcription + ' '
        st.session_state['recording'] = False
        # st.rerun()

