from tqdm import tqdm
import sacrebleu



def get_BLEU(translations, references):
    bleu = sacrebleu.corpus_bleu(translations, references)
    print(f"\nBLEU score: {bleu.score:.2f}")
    return bleu
