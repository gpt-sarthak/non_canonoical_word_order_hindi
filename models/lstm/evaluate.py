import torch
import torch.nn.functional as F
import pickle
from model import LSTMLanguageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/lstm/base_model.pt"
VOCAB_PATH = "data/processed/vocab.pkl"


def load_model():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab["word2idx"])

    model = LSTMLanguageModel(vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, vocab


def compute_sentence_surprisal(sentence, model, vocab):
    word2idx = vocab["word2idx"]

    tokens = ["<SOS>"] + sentence.split() + ["<EOS>"]
    indices = [
        word2idx.get(t, word2idx["<UNK>"])
        for t in tokens
    ]

    input_tensor = torch.tensor(indices[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
    target_tensor = torch.tensor(indices[1:], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(input_tensor)
        log_probs = F.log_softmax(logits, dim=-1)

    surprisals = []
    for i in range(len(target_tensor)):
        word_log_prob = log_probs[0, i, target_tensor[i]]
        surprisals.append(-word_log_prob.item())

    return surprisals


if __name__ == "__main__":
    model, vocab = load_model()

    test_sentence = "भारत एक सुंदर देश है"
    surprisals = compute_sentence_surprisal(test_sentence, model, vocab)

    print("Word-level surprisals:")
    print(surprisals)
    print("Total surprisal:", sum(surprisals))