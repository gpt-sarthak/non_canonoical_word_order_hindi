def compute_dependency_length(sentence):

    total_dl = 0

    for token in sentence:

        if token["head"] == 0:
            continue

        head_id = token["head"]
        dep_id = token["id"]

        total_dl += abs(dep_id - head_id)

    return total_dl