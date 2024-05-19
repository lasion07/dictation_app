# DICTATION APP USING WAV2VEC 2.0

In a formal ASR system, two components are required: acoustic model and language model. Here ctc-wav2vec fine-tuned model works as an acoustic model. For the language model, [author](https://github.com/vietai/ASR) provide a [4-grams model](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h/blob/main/vi_lm_4grams.bin.zip) trained on 2GB of spoken text.

# Dependences
```
    pip install -r requirements.txt
```
**Please put 4-grams model into "models" directory**

# Usage
```
    fastapi run main.py
```

# REFENECES
```
@misc{Thai_Binh_Nguyen_wav2vec2_vi_2021,
  author = {Thai Binh Nguyen},
  doi = {10.5281/zenodo.5356039},
  month = {09},
  title = {{Vietnamese end-to-end speech recognition using wav2vec 2.0}},
  url = {https://github.com/vietai/ASR},
  year = {2021}
}
@inproceedings{omelianchuk-etal-2020-gector,
    title = "{GECT}o{R} {--} Grammatical Error Correction: Tag, Not Rewrite",
    author = "Omelianchuk, Kostiantyn  and
      Atrasevych, Vitaliy  and
      Chernodub, Artem  and
      Skurzhanskyi, Oleksandr",
    booktitle = "Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = jul,
    year = "2020",
    address = "Seattle, WA, USA â†’ Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bea-1.16",
    pages = "163--170",
    abstract = "In this paper, we present a simple and efficient GEC sequence tagger using a Transformer encoder. Our system is pre-trained on synthetic data and then fine-tuned in two stages: first on errorful corpora, and second on a combination of errorful and error-free parallel corpora. We design custom token-level transformations to map input tokens to target corrections. Our best single-model/ensemble GEC tagger achieves an F{\_}0.5 of 65.3/66.5 on CONLL-2014 (test) and F{\_}0.5 of 72.4/73.6 on BEA-2019 (test). Its inference speed is up to 10 times as fast as a Transformer-based seq2seq GEC system.",
}
```
