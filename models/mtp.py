
import torch

def seq_to_mtp(
    long_input_ids: torch.Tensor,
    model_seq_len: int,
    n_future_tokens: int
) -> torch.Tensor:
    """
    Generates a tensor of future targets on the fly from a long input sequence.

    This version assumes `long_input_ids` contains both the tokens for the model's
    input AND the future tokens needed for the labels.
    It extracts the correct targets without adding artificial padding.

    Args:
        long_input_ids (torch.Tensor): The input sequences from the dataloader,
                                       shape (B, T + n_future_tokens).
        model_seq_len (int): The sequence length `T` that the model processes.
        n_future_tokens (int): The number of future tokens to predict for each time step.

    Returns:
        torch.Tensor: The target tensor of shape (B, T, n_future_tokens).
                      y[b, t, k] corresponds to the (k+1)-th token after input_ids[b, t].
    """
    B, total_len = long_input_ids.shape
    assert total_len >= model_seq_len + n_future_tokens, \
        "long_input_ids must be at least model_seq_len + n_future_tokens long."

    # 1. Create sliding windows (views) over the long tensor.
    # .unfold() is a highly efficient way to create sliding windows.
    # We create windows of size `n_future_tokens + 1`. For each time step `t`,
    # the window will contain the input token and its `n_future_tokens` targets.
    # Example (n=3, window_size=4):
    # For t=0, window is [t0, t1, t2, t3]
    # For t=1, window is [t1, t2, t3, t4]
    # Shape of windows: (B, total_len - n_future_tokens, n_future_tokens + 1)
    windows = long_input_ids.unfold(dimension=1, size=n_future_tokens + 1, step=1)

    # 2. Slice the windows to get only the targets.
    # We slice off the first element of each window (the input token itself)
    # to keep only the future tokens.
    # Example window [t0, t1, t2, t3] -> becomes targets [t1, t2, t3]
    all_targets = windows[:, :, 1:]

    # 3. Trim the result to match the model's output sequence length.
    # We only need the targets for the first `model_seq_len` positions.
    output_targets = all_targets[:, :model_seq_len, :]

    return output_targets.transpose(1, 2)
