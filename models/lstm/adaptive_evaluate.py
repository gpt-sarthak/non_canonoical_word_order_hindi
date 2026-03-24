import torch
import torch.nn.functional as F
import pickle
from model import LSTMLanguageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/lstm/base_model.pt"
VOCAB_PATH = "data/processed/vocab.pkl"

ADAPT_LR = 0.005

# Turn this off later for HUTB experiments
PRINT_WORD_LEVEL = True


def load_model():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab["word2idx"])

    model = LSTMLanguageModel(vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    return model, vocab


def adaptive_surprisal(sentences):

    model, vocab = load_model()

    model.train()
    model.lstm.dropout = 0.0

    optimizer = torch.optim.SGD(model.parameters(), lr=ADAPT_LR)

    word2idx = vocab["word2idx"]

    all_sentence_surprisals = []

    for sentence in sentences:

        print("\nSentence:", sentence)
        print("-" * 40)

        tokens = ["<SOS>"] + sentence.split() + ["<EOS>"]

        indices = [
            word2idx.get(t, word2idx["<UNK>"])
            for t in tokens
        ]

        input_tensor = torch.tensor(indices[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
        target_tensor = torch.tensor(indices[1:], dtype=torch.long).to(DEVICE)

        logits, _ = model(input_tensor)

        log_probs = F.log_softmax(logits, dim=-1)

        surprisals = []

        for i in range(len(target_tensor)):

            word_log_prob = log_probs[0, i, target_tensor[i]]

            word_surprisal = -word_log_prob.item()

            predicted_word = tokens[i + 1]

            if PRINT_WORD_LEVEL:
                print(f"{predicted_word:<20} -> {word_surprisal:.4f}")

            surprisals.append(word_surprisal)

        sentence_surprisal = sum(surprisals)

        print("Sentence total surprisal:", round(sentence_surprisal, 4))

        all_sentence_surprisals.append(sentence_surprisal)

        # ----- Adaptation Step -----

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_tensor.view(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    return all_sentence_surprisals


if __name__ == "__main__":

    test_cases = {

        "test_rare_word_repetition": [
            "जैवप्रौद्योगिकी के क्षेत्र में अनुसंधान तेजी से विकसित हो रहा है",
            "जैवप्रौद्योगिकी के क्षेत्र में अनुसंधान तेजी से विकसित हो रहा है",
            "जैवप्रौद्योगिकी के क्षेत्र में अनुसंधान तेजी से विकसित हो रहा है"
        ],

        "test_exact_sentence_repetition": [
            "भारत एक विशाल और ऐतिहासिक देश है जो अपनी विविधता के लिए प्रसिद्ध है",
            "भारत एक विशाल और ऐतिहासिक देश है जो अपनी विविधता के लिए प्रसिद्ध है",
            "भारत एक विशाल और ऐतिहासिक देश है जो अपनी विविधता के लिए प्रसिद्ध है"
        ],

        "test_structural_pattern_repetition": [
            "सरकार ने नई नीति की घोषणा की है",
            "कंपनी ने नए उत्पाद की घोषणा की है",
            "विद्यालय ने नए कार्यक्रम की घोषणा की है"
        ],

        "test_topic_adaptation": [
            "भारत एक विशाल और ऐतिहासिक देश है",
            "भारत की संस्कृति और परंपराएँ विश्वभर में सराही जाती हैं",
            "भारत की आर्थिक प्रगति पिछले दशकों में उल्लेखनीय रही है"
        ],

        "test_reexposure_after_intervening_sentences": [
            "भारत एक विशाल और ऐतिहासिक देश है",
            "जैवप्रौद्योगिकी के क्षेत्र में अनुसंधान तेजी से विकसित हो रहा है",
            "सरकार ने नई नीति की घोषणा की है",
            "भारत एक विशाल और ऐतिहासिक देश है"
        ]
    }

    for test_name, sentences in test_cases.items():

        print("\n" + "=" * 60)
        print(f"Running: {test_name}")
        print("=" * 60)

        surprisals = adaptive_surprisal(sentences)

        print("\nSummary Surprisals:")

        for i, s in enumerate(surprisals):
            print(f"Sentence {i+1} Surprisal: {s:.4f}")

        print("-" * 60)