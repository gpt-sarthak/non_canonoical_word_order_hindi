import pickle
import math


# ------------------------------------------------------------
# Load trigram model
# ------------------------------------------------------------
def load_trigram_model(path):

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model


# ------------------------------------------------------------
# Compute trigram surprisal for a sentence
# ------------------------------------------------------------
def sentence_trigram_surprisal(sentence, model):

    words = sentence.split()

    total_surprisal = 0

    for i in range(2, len(words)):

        w1 = words[i-2]
        w2 = words[i-1]
        w3 = words[i]

        prob = model.score(w3, (w1, w2))

        if prob == 0:
            prob = model.score(w3)

        if prob == 0:
            prob = 1e-12

        total_surprisal += -math.log(prob)

    return total_surprisal


# ------------------------------------------------------------
# Compute trigram features for dataset
# ------------------------------------------------------------
def compute_trigram_features(dataset, model):

    results = []

    for item in dataset:

        ref_sentence = item["reference"]
        var_sentence = item["variant"]

        s_ref = sentence_trigram_surprisal(ref_sentence, model)
        s_var = sentence_trigram_surprisal(var_sentence, model)

        delta = s_var - s_ref

        results.append({
            **item,
            "trigram_reference": s_ref,
            "trigram_variant": s_var,
            "delta_trigram": delta
        })

    return results