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
def sentence_lstm_surprisal(sentence, model, vocab, device="cpu"):

    words = sentence.split()

    # vocab is already word2idx
    word2idx = vocab

    indices = [word2idx.get(w, word2idx.get("<UNK>", 0)) for w in words]

    input_tensor = torch.tensor(indices[:-1]).unsqueeze(0).to(device)
    target_tensor = torch.tensor(indices[1:]).to(device)

    with torch.no_grad():

        logits, _ = model(input_tensor)

        log_probs = torch.log_softmax(logits, dim=-1)

        total_surprisal = 0

        for i, target in enumerate(target_tensor):

            prob = log_probs[0, i, target].item()

            total_surprisal += -prob

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


# ─────────────────────────────────────────────────────────────
# __main__ — standalone debug using real pair from features.csv
#
# Loads the trained LSTM and scores the first DOSV pair from
# features.csv, then compares against the stored delta.
#
# Expected (features.csv, sentence_id=0, DOSV):
#   reference      : इसे नवाब शाहजेहन ने बनवाया था ।
#   variant        : नवाब शाहजेहन ने इसे बनवाया था ।
#   stored delta_lstm ≈ -3.603
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, ".")

    LSTM_PATH  = "models/lstm/base_model.pt"
    VOCAB_PATH = "data/processed/vocab.pkl"

    df  = pd.read_csv("data/features/features.csv")
    row = df[df["construction_type"] == "DOSV"].iloc[0]
    ref = row["reference"]
    var = row["variant"]

    print(f"reference : {ref}")
    print(f"variant   : {var}")
    print(f"stored delta_lstm : {row['delta_lstm']:.4f}")

    print(f"\nLoading vocab from {VOCAB_PATH} ...")
    vocab = load_vocab(VOCAB_PATH)
    vocab_size = len(vocab["word2idx"])
    print(f"Vocab size : {vocab_size}")

    print(f"Loading LSTM model from {LSTM_PATH} ...")
    model, device = load_lstm_model(LSTM_PATH, vocab_size)
    print(f"Device     : {device}")

    s_ref = sentence_lstm_surprisal(ref, model, vocab["word2idx"], device)
    s_var = sentence_lstm_surprisal(var, model, vocab["word2idx"], device)
    delta = s_var - s_ref   # note: lstm uses var-ref convention in compute_lstm_features

    print(f"\nlstm_reference : {s_ref:.4f}")
    print(f"lstm_variant   : {s_var:.4f}")
    print(f"delta_lstm     : {delta:.4f}  (var - ref)")
    print(f"matches stored : {abs(delta - row['delta_lstm']) < 1e-3}")