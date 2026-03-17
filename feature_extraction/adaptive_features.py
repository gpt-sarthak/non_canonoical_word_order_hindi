import torch
import torch.nn as nn

from feature_extraction.lstm_features import sentence_lstm_surprisal


def compute_adaptive_features(dataset, model, vocab, device):

    """
    Compute adaptive surprisal features.

    This follows the method described in the paper:
    ------------------------------------------------
    1. Compute surprisal for the reference sentence.
    2. Compute surprisal for the variant sentence.
    3. Record the difference between the two.
    4. Update (adapt) the language model using the reference sentence.

    The idea is to simulate human readers adapting to recently observed
    linguistic patterns.
    """

    results = []

    # ------------------------------------------------------------
    # Optimizer used to slightly update the model after each sentence
    # ------------------------------------------------------------
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Loss function for next-word prediction
    loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------
    # IMPORTANT: enable training mode for adaptation
    # Required for CUDA LSTM backward pass
    # ------------------------------------------------------------
    model.train()

    for item in dataset:

        # Extract sentence pair
        ref = item["reference"]
        var = item["variant"]

        # ------------------------------------------------------------
        # Step 1: Compute surprisal using the CURRENT model
        # ------------------------------------------------------------

        s_ref = sentence_lstm_surprisal(ref, model, vocab, device)
        s_var = sentence_lstm_surprisal(var, model, vocab, device)

        delta = s_var - s_ref

        results.append({
            **item,
            "adaptive_reference": s_ref,
            "adaptive_variant": s_var,
            "delta_adaptive": delta
        })

        # ------------------------------------------------------------
        # Step 2: Adapt the model using the reference sentence
        # ------------------------------------------------------------

        words = ref.split()

        word2idx = vocab

        indices = [word2idx.get(w, word2idx.get("<UNK>", 0)) for w in words]

        input_tensor = torch.tensor(indices[:-1]).unsqueeze(0).to(device)
        target_tensor = torch.tensor(indices[1:]).to(device)

        optimizer.zero_grad()

        logits, _ = model(input_tensor)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            target_tensor.view(-1)
        )

        loss.backward()

        optimizer.step()

    # ------------------------------------------------------------
    # Restore evaluation mode after adaptation
    # ------------------------------------------------------------

    model.eval()

    return results