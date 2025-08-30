from __future__ import annotations

import spacy
import torch
from torchtext.data.metrics import bleu_score


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    """Translate a single sentence using the trained model."""
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load("es")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]


def calculate_bleu(model, test_data, src_field, trg_field, device, max_len=50): # noqa: PLR0913
    """Calculate BLEU score for the test dataset."""
    trgs = []
    pred_trgs = []
    for datum in test_data:
        source = vars(datum)["src"]
        target = vars(datum)["trg"]
        pred_trg = translate_sentence(source, src_field, trg_field, model, device, max_len)
        pred_trg = pred_trg[:-1]  # cut off <eos>
        pred_trgs.append(pred_trg)
        trgs.append([target])
    return bleu_score(pred_trgs, trgs)
