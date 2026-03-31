from torch.utils.data import IterableDataset
from datasets import load_dataset
import random


class NLLBLanguageStream(IterableDataset):
    def __init__(self, pair_config: str, lang: str, split: str = "train"):
        self.pair_config = pair_config
        self.lang = lang
        self.split = split

    def __iter__(self):
        ds = load_dataset("allenai/nllb", self.pair_config, split=self.split, streaming=True,trust_remote_code=True)

        for row in ds:
            text = row["translation"][self.lang]
            if text is None:
                continue

            text = text.strip()
            if not text:
                continue

            yield {
                "text": text,
                "lang": self.lang,
                "pair": self.pair_config,
            }


class BalancedNLLBDataset(IterableDataset):
    def __init__(self, pair_configs: list[str], langs: list[str]):
        self.pair_configs = pair_configs
        self.langs = langs
        
        self.streams_by_lang = {}
        for pair_config, lang in zip(pair_configs, langs):
            self.streams_by_lang[lang] = NLLBLanguageStream(pair_config, lang)

    def __iter__(self):
        iterators = {lang: iter(ds) for lang, ds in self.streams_by_lang.items()}

        while True:
            random.shuffle(self.langs)
            for lang in self.langs:
                try:
                    yield next(iterators[lang])
                except StopIteration:
                    return
