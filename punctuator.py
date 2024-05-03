import os
from gec_model import GecBERTModel


class Punctuator():
    def __init__(self):
        self.model = GecBERTModel(
            vocab_path="vocabulary",
            model_paths="dragonSwing/vibert-capu",
            split_chunk=True
        )

    def process(self, text: str) -> list[str]:
        return self.model(text)