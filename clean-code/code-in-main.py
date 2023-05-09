
from customized_transformer import load_tokenizers, load_vocab, load_trained_model
if __name__ == '__main__':
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    training_loss = []
    val_loss = []
    LOSS = []

    model = load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en)