import math


def sentence_pcfg_surprisal(sentence):
    """
    Placeholder PCFG surprisal computation.

    In the original paper, PCFG surprisal is computed from a
    probabilistic context-free grammar derived from a treebank.

    Since our dataset comes from dependency trees (UD), a direct
    constituency PCFG is not available without conversion.

    For now this function acts as a scaffold so the pipeline
    structure remains complete.

    TODO:
    Replace this with a real PCFG parser later.
    """

    words = sentence.split()

    # simple placeholder scoring
    # longer sentences receive larger surprisal
    return math.log(len(words) + 1)


def compute_pcfg_features(dataset):
    """
    Compute PCFG surprisal features for reference and variant sentences.
    """

    results = []

    for item in dataset:

        ref = item["reference"]
        var = item["variant"]

        s_ref = sentence_pcfg_surprisal(ref)
        s_var = sentence_pcfg_surprisal(var)

        delta = s_var - s_ref

        results.append({
            **item,
            "pcfg_reference": s_ref,
            "pcfg_variant": s_var,
            "delta_pcfg": delta
        })

    return results
