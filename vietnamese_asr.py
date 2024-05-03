import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import time

import torch
import kenlm
from pydub import AudioSegment
from pynput import keyboard
from pynput.keyboard import Controller
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from speech_recognition import Recognizer, Microphone
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from punctuator import Punctuator


def get_decoder_ngram_model(tokenizer, ngram_lm_path):
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

def on_press(key):
    global recording
    if recording == False and key == keyboard.Key.shift:
        recording = True
        
# Input

# import model, feature extractor, tokenizer
checkpoint = 'nguyenvulebinh/wav2vec2-base-vietnamese-250h'
model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
processor = Wav2Vec2Processor.from_pretrained(checkpoint)

language_model_path = 'models/vi_lm_4grams.bin'
ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, language_model_path)

punctuator = Punctuator()

recognizer = Recognizer()
recognizer.energy_threshold = 400

recording = False
history = []

keyboard_controller = Controller()

listener = keyboard.Listener(on_press=on_press)
listener.start()

start_time = time.time()
print('Only use English input source to avoid typo')

with Microphone(sample_rate=16000) as source:
  while True:
    if not recording:
      print('Press Shift to start record', end='\r')
      time.sleep(1)
      continue

    recording = False
    print("\nYou can start speaking now...")
    audio = recognizer.listen(source, phrase_time_limit=3) # Bytes
    
    data = io.BytesIO(audio.get_wav_data()) # Object(Bytes) Ex: 96300
    audio = AudioSegment.from_file(data) # Object(Object) Ex: 96300
    waveform = torch.FloatTensor(audio.get_array_of_samples()) # Tensor(Array(Int)) Ex: 48128 =>  1 Int = 2 Bytes

    start_time = time.time()
    transcription = infer(waveform, model, processor, ngram_lm_model, punctuator)
    end_time = time.time() - start_time
    
    print('After added punctuation:', transcription, end_time)

    history.append(transcription)

    transcription += ' '

    for letter in transcription:
      keyboard_controller.press(letter)
      keyboard_controller.release(letter)
      time.sleep(0.008)
    