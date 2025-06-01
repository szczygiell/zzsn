from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from tqdm import tqdm

#pip install numpy==1.26.4

class MBert():
    def __init__(self, device, target_lan):
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.device = device
        self.target = target_lan # "pl_PL", "de_DE"
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        self.tokenizer.src_lang = "en_XX"
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

    def translate(self, texts):
        translated_texts = []
        if isinstance(texts, str):
            encoded = self.tokenizer(texts, return_tensors="pt").to(self.device)
            generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target])
            translated_texts.append(self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])
        else:
            for t in tqdm(texts, desc="Tłumaczenie"):
                encoded = self.tokenizer(t, return_tensors="pt").to(self.device)
                generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target])
                translated_texts.append(self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])
        return translated_texts

