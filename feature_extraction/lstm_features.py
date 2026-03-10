import torch
import math
import pickle
from models.lstm.model import LSTMLanguageModel


# ------------------------------------------------------------
# Load vocabulary mapping
# ------------------------------------------------------------
def load_vocab(path):

    with open(path, "rb") as f:
        vocab = pickle.load(f)

    return vocab


# ------------------------------------------------------------
# Detect device automatically
# ------------------------------------------------------------
def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    return device


# ------------------------------------------------------------
# Load trained LSTM model
# ------------------------------------------------------------
def load_lstm_model(model_path, vocab_size, device=None):

    if device is None:
        device = get_device()

    model = LSTMLanguageModel(vocab_size)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device


# ------------------------------------------------------------
# Compute LSTM surprisal for a sentence
# ------------------------------------------------------------
def sentence_lstm_surprisal(sentence, model, vocab, device):

    words = sentence.split()

    word2idx = vocab["word2idx"]

    indices = [word2idx.get(w, word2idx["<UNK>"]) for w in words]

    input_tensor = torch.tensor(indices[:-1]).unsqueeze(0).to(device)
    target_tensor = torch.tensor(indices[1:]).to(device)

    with torch.no_grad():

        logits, _ = model(input_tensor)

        log_probs = torch.log_softmax(logits, dim=-1)

        total_surprisal = 0

        for i, target in enumerate(target_tensor):

            log_prob = log_probs[0, i, target].item()

            total_surprisal += -log_prob

    return total_surprisal


# ------------------------------------------------------------
# Compute LSTM features for dataset
# ------------------------------------------------------------
def compute_lstm_features(dataset, model, vocab, device):

    results = []

    for item in dataset:

        ref_sentence = item["reference"]
        var_sentence = item["variant"]

        s_ref = sentence_lstm_surprisal(ref_sentence, model, vocab, device)
        s_var = sentence_lstm_surprisal(var_sentence, model, vocab, device)

        delta = s_var - s_ref

        results.append({
            **item,
            "lstm_reference": s_ref,
            "lstm_variant": s_var,
            "delta_lstm": delta
        })

    return results