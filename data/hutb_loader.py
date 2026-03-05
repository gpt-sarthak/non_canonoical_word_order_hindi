import os
import itertools

TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"


def load_conllu(filepath):

    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")

            # skip multiword tokens (3-4)
            if "-" in parts[0]:
                continue

            # skip empty nodes (3.1)
            if "." in parts[0]:
                continue

            token = {
                "id": int(parts[0]),
                "word": parts[1],
                "lemma": parts[2],
                "upos": parts[3],
                "head": int(parts[6]),
                "deprel": parts[7]
            }

            current_sentence.append(token)

    return sentences


def is_valid_sov_sentence(sentence):

    root = None
    has_subject = False
    has_object = False

    for token in sentence:

        if token["head"] == 0:
            root = token

        if token["deprel"] == "nsubj":
            has_subject = True

        if token["deprel"] == "obj":
            has_object = True

    if root is None:
        return False

    if root["upos"] != "VERB":
        return False

    return has_subject and has_object


def get_subtree(sentence, token_id):

    subtree_ids = set()
    stack = [token_id]

    while stack:

        current = stack.pop()

        if current in subtree_ids:
            continue

        subtree_ids.add(current)

        for token in sentence:
            if token["head"] == current:
                stack.append(token["id"])

    subtree = []

    for token in sentence:
        if token["id"] in subtree_ids:
            subtree.append(token)

    return subtree


def subtree_tokens(sentence, token_id):

    subtree = get_subtree(sentence, token_id)

    subtree = sorted(subtree, key=lambda x: x["id"])

    return subtree


def generate_variants_subtrees(sentence, max_variants=99):

    root = None

    for token in sentence:
        if token["head"] == 0:
            root = token
            break

    if root is None:
        return []

    root_id = root["id"]

    preverbal_phrases = []
    postverbal_phrases = []

    for token in sentence:

        if token["head"] == root_id:

            if token["deprel"] == "punct":
                continue

            phrase_tokens = subtree_tokens(sentence, token["id"])

            if token["id"] < root_id:
                preverbal_phrases.append(phrase_tokens)
            else:
                postverbal_phrases.append((token["id"], phrase_tokens))

    # keep postverbal phrases in original order
    postverbal_phrases = sorted(postverbal_phrases, key=lambda x: x[0])

    postverbal_tokens = []
    for _, phrase in postverbal_phrases:
        postverbal_tokens.extend(phrase)

    variants = []

    perms = itertools.permutations(preverbal_phrases)

    for perm in perms:

        new_tokens = []

        for phrase in perm:
            new_tokens.extend(phrase)

        new_tokens.append(root)

        new_tokens.extend(postverbal_tokens)

        words = [t["word"] for t in new_tokens]

        variant = " ".join(words)

        variants.append(variant)

        if len(variants) >= max_variants:
            break

    return variants


def build_variant_dataset(sentences):

    dataset = []

    for i, sentence in enumerate(sentences):

        variants = generate_variants_subtrees(sentence)

        if len(variants) <= 1:
            continue

        reference = variants[0]

        for v in variants[1:]:

            dataset.append({
                "sentence_id": i,
                "reference": reference,
                "variant": v
            })

    return dataset


if __name__ == "__main__":

    sentences = load_conllu(TREEBANK_PATH)

    valid_sentences = [s for s in sentences if is_valid_sov_sentence(s)]

    print("Total sentences:", len(sentences))
    print("Valid SOV sentences:", len(valid_sentences))

    example = valid_sentences[0]

    print("\nExample sentence:\n")

    for token in example:
        print(
            token["id"],
            token["word"],
            token["upos"],
            "HEAD:", token["head"],
            "REL:", token["deprel"]
        )

    root_id = None

    for token in example:
        if token["head"] == 0:
            root_id = token["id"]
            break

    print("\nRoot:", root_id)

    print("\nRoot dependents:")

    for token in example:

        if token["head"] == root_id:

            phrase = subtree_tokens(example, token["id"])

            words = [t["word"] for t in phrase]

            print(token["deprel"], "->", " ".join(words))

    variants = generate_variants_subtrees(example)

    print("\nGenerated variants:")

    for v in variants:
        print(v)

    dataset = build_variant_dataset(valid_sentences)

    print("\nTotal variant pairs:", len(dataset))

    print("\nRaw variant repr:")
    print(repr(dataset[0]["variant"]))
    print("\nExample pair:")
    pair = dataset[0]

    print("\nSentence ID:", pair["sentence_id"])

    print("\nReference:")
    print(pair["reference"])

    print("\nVariant:")
    print(pair["variant"])
