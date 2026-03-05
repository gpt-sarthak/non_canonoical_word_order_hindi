import os
import itertools

TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"


def load_conllu(filepath):
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # sentence boundary
            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # skip comments
            if line.startswith("#"):
                continue

            parts = line.split("\t")

            # ignore multiword tokens like 3-4
            if "-" in parts[0] or "." in parts[0]:
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

    # root must be a verb
    if root["upos"] != "VERB":
        return False

    return has_subject and has_object

def extract_preverbal_tokens(sentence):

    root_id = None

    for token in sentence:
        if token["head"] == 0:
            root_id = token["id"]
            break

    if root_id is None:
        return []

    preverbal_tokens = []

    for token in sentence:
        if token["id"] < root_id and token["upos"] != "PUNCT":
            preverbal_tokens.append(token["word"])

    return preverbal_tokens

def generate_variants(sentence, max_variants=99):

    # find root verb
    root_id = None
    root_word = None

    for token in sentence:
        if token["head"] == 0:
            root_id = token["id"]
            root_word = token["word"]
            break

    if root_id is None:
        return []

    # split sentence
    preverbal = []
    postverbal = []

    for token in sentence:

        if token["id"] < root_id and token["upos"] != "PUNCT":
            preverbal.append(token["word"])

        if token["id"] > root_id and token["upos"] != "PUNCT":
            postverbal.append(token["word"])

    # generate permutations
    permutations = list(itertools.permutations(preverbal))

    variants = []

    for perm in permutations:

        variant = list(perm) + [root_word] + postverbal

        variants.append(" ".join(variant))

        if len(variants) >= max_variants:
            break

    return variants

def get_subtree(sentence, token_id):

    subtree = []
    stack = [token_id]

    while stack:
        current = stack.pop()

        for token in sentence:
            if token["id"] == current:
                subtree.append(token)

            if token["head"] == current:
                stack.append(token["id"])

    return subtree

def subtree_words(sentence, token_id):

    subtree = get_subtree(sentence, token_id)

    subtree = sorted(subtree, key=lambda x: x["id"])

    return [t["word"] for t in subtree]

def generate_variants_subtrees(sentence, max_variants=99):

    root_id = None
    root_word = None

    for token in sentence:
        if token["head"] == 0:
            root_id = token["id"]
            root_word = token["word"]
            break

    if root_id is None:
        return []

    phrases = []
    postverbal = []

    for token in sentence:

        if token["head"] == root_id:

            if token["deprel"] in ["punct"]:
                continue

            subtree = subtree_words(sentence, token["id"])

            # check if phrase is before verb
            if token["id"] < root_id:
                phrases.append(subtree)
            else:
                postverbal.append(subtree)

    # flatten postverbal phrases
    postverbal = [w for phrase in postverbal for w in phrase]

    variants = []

    perms = itertools.permutations(phrases)

    for perm in perms:

        new_sentence = []

        for phrase in perm:
            new_sentence.extend(phrase)

        new_sentence.append(root_word)

        new_sentence.extend(postverbal)

        variants.append(" ".join(new_sentence))

        if len(variants) >= max_variants:
            break

    return variants

if __name__ == "__main__":

    sentences = load_conllu(TREEBANK_PATH)

    valid_sentences = [s for s in sentences if is_valid_sov_sentence(s)]

    print("Total sentences:", len(sentences))

    print("Valid SOV sentences:", len(valid_sentences))

    print("\nExample sentence:\n")

    example = valid_sentences[0]

    for token in example:
        print(
            token["id"],
            token["word"],
            token["upos"],
            "HEAD:", token["head"],
            "REL:", token["deprel"]
        )
    preverbal = extract_preverbal_tokens(example)

    print("\nPreverbal tokens:")
    print(preverbal)

    variants = generate_variants(example)

    print("\nGenerated variants:")
    print("-------------------")

    for v in variants[:10]:
        print(v)
    
    root_id = None

    for token in example:
        if token["head"] == 0:
            root_id = token["id"]
            break

    print("\nRoot:", root_id)

    print("\nRoot dependents:")

    for token in example:

        if token["head"] == root_id:

            phrase = subtree_words(example, token["id"])

            print(token["deprel"], "->", " ".join(phrase))
    
    variants = generate_variants_subtrees(example)

    print("\nGenerated variants:")

    for v in variants:
        print(v)