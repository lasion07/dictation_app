import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import time

import streamlit as st
from speech_recognition import Recognizer, Microphone
from streamlit_quill import st_quill


st.set_page_config(page_title="Nhập văn bản bằng giọng nói", page_icon="🎙️")

if 'content' not in st.session_state:
    st.session_state['content'] = ''

if 'recording' not in st.session_state:
    st.session_state['recording'] = False

@st.cache_resource() # Disable for Debugging
def load_recognizer():
    recognizer = Recognizer()
    recognizer.energy_threshold = 400
    return recognizer

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

