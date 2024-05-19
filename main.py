import os
import io
import time
import json
import uvicorn
import logging

import torch
import kenlm
import librosa

from contextlib import asynccontextmanager

from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from punctuator import Punctuator

from fastapi import FastAPI, File, Request, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

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

async def infer(waveform) -> list:
    global ml_models
    model, processor, language_model, punctuator = ml_models["model"], ml_models["processor"], ml_models["language_model"], ml_models["punctuator"]
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

    transcription = punctuator.process(beam_search_output)[0]
    
    transcription += ' '

    return transcription

async def read_audio(audio_path):
    audio_array, sampling_rate = librosa.load(
        audio_path, sr=16000, res_type='kaiser_fast')
    return audio_array, sampling_rate

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # import model, feature extractor, tokenizer
    checkpoint = 'nguyenvulebinh/wav2vec2-base-vietnamese-250h'
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)

    language_model_path = 'models/vi_lm_4grams.bin'
    ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, language_model_path)

    punctuator = Punctuator()

    ml_models["model"] = model
    ml_models["processor"] = processor
    ml_models["language_model"] = ngram_lm_model
    ml_models["punctuator"] = punctuator
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="dictation_app.html")

@app.post("/recognize")
async def speech_recognize(file: UploadFile = File(...)):
    content_type = file.content_type

    # Check if the uploaded file is an audio file
    if "audio" not in str(content_type):
        raise HTTPException(
            status_code=415, detail="Unsupported Media Type. Please upload an audio file.")

    start_time = time.time()
    audio_bytes = await file.read()
    audio_chunk_path = "received_audio.wav"
    with open(audio_chunk_path, "wb") as f:
        f.write(audio_bytes)
    
    audio_array, sampling_rate = await read_audio(audio_chunk_path)
    waveform = torch.FloatTensor(audio_array)

    transcription = await infer(waveform)
    
    print(transcription, time.time() - start_time)

    return {"transcription": transcription}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)