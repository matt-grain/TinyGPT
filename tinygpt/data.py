"""
Dataset classes for TinyGPT pre-training and supervised fine-tuning (SFT).

- TextDataset      — exhaustive sliding window over a token array (pre-training)
- TextDatasetSmall — random sampling over a token array (pre-training, large corpora)
- encode_qa_pair   — encode a single Q&A pair into token IDs + loss mask
- SFTDataset       — structured Q&A dataset with per-token loss masking (SFT)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data
from typing import Callable


class TextDataset(torch.utils.data.Dataset):
    """
    Exhaustive sliding window dataset for language model pre-training.

    Every possible context window in the token array becomes one training sample.
    Window i yields input=tokens[i:i+L] and target=tokens[i+1:i+L+1] (next-token).

    Use this when the corpus is small enough that a full sweep is practical.
    For large corpora, prefer TextDatasetSmall to avoid a dataset of millions of
    near-duplicate windows.
    """

    def __init__(self, encoded: np.ndarray, context_length: int) -> None:
        self.encoded = encoded
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.encoded) - self.context_length - 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoded[i : i + self.context_length]
        y = self.encoded[i + 1 : i + self.context_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# --- KEY FIX: smaller dataset with random sampling ---
class TextDatasetSmall(torch.utils.data.Dataset):
    """
    Random-sampling dataset for language model pre-training.

    Instead of iterating over every possible window (which for 1 M tokens and
    context_length=64 gives ~1 M samples — mostly overlapping), we draw
    num_samples random start positions.  This:
      - Reduces one epoch to a manageable size (e.g. 20 000 samples).
      - Avoids training on near-duplicate windows in the same epoch.
      - Still covers the whole corpus over many epochs (random coverage).

    Trade-off: each epoch sees a random subset, not the full corpus.
    For very small corpora use TextDataset instead.
    """

    def __init__(
        self,
        encoded: np.ndarray,
        context_length: int,
        num_samples: int = 20000,
    ) -> None:
        self.encoded = encoded
        self.context_length = context_length
        # Random sample instead of exhaustive sliding window
        max_start = len(encoded) - context_length - 1
        self.indices = np.random.randint(0, max_start, size=num_samples)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[i]
        x = self.encoded[start : start + self.context_length]
        y = self.encoded[start + 1 : start + self.context_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def encode_qa_pair(
    question: str,
    answer: str,
    max_len: int,
    encode_fn: Callable[[str], list[int]],
    user_id: int,
    assistant_id: int,
    end_id: int,
) -> tuple[list[int], list[int]]:
    """
    Encode a Q&A pair into token IDs and a loss mask.

    Produces a sequence in the format:
        <|user|> q1 q2 ... <|assistant|> a1 a2 ... <|end|> <PAD> ...

    The loss mask is 0 for every prompt token and 1 for every answer token
    (including the final <|end|> token).  During SFT training only positions
    where mask=1 contribute to the gradient, so the model learns to produce
    answers — not to reproduce questions.

    Args:
        question:     Raw question string (will be tokenized by encode_fn).
        answer:       Raw answer string (will be tokenized by encode_fn).
        max_len:      Total sequence length after padding/truncation.
        encode_fn:    Callable that maps a string to a list of integer token IDs.
                      Must use the same vocabulary as the pre-trained model.
        user_id:      Token ID for the <|user|> special token.
        assistant_id: Token ID for the <|assistant|> special token.
        end_id:       Token ID for the <|end|> special token.

    Returns:
        input_ids:  List of length max_len — token IDs, right-padded with 0.
        loss_mask:  List of length max_len — 1 where the model should learn, else 0.
    """
    q_tokens = encode_fn(question)
    a_tokens = encode_fn(answer)

    # Build full sequence
    input_ids = [user_id] + q_tokens + [assistant_id] + a_tokens + [end_id]

    # Loss mask: 0 for prompt, 1 for answer portion.
    # The mask marks positions where we WANT the model to learn.
    # Prompt length = <|user|> + question tokens + <|assistant|>
    prompt_len = 1 + len(q_tokens) + 1
    loss_mask = [0] * prompt_len + [1] * len(a_tokens) + [1]  # include <|end|> in loss

    # Truncate if too long
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        loss_mask = loss_mask[:max_len]

    # Pad to max_len
    pad_len = max_len - len(input_ids)
    input_ids += [0] * pad_len  # 0 = <PAD>
    loss_mask += [0] * pad_len  # don't train on padding either

    return input_ids, loss_mask


class SFTDataset(torch.utils.data.Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT).

    Unlike pre-training (sliding window over raw text), SFT uses structured
    pairs where each sample is a complete Q&A interaction.  The dataset is
    small (tens to thousands of pairs) but high quality.

    Each item is a dict with:
        "input_ids"  — LongTensor of shape (max_len,)
        "loss_mask"  — FloatTensor of shape (max_len,), 1 on answer tokens only

    The caller is responsible for shifting input/target during the training loop:
        x    = input_ids[:, :-1]
        y    = input_ids[:, 1:]
        mask = loss_mask[:, 1:]   # aligned with targets
    """

    def __init__(
        self,
        qa_pairs: list[dict[str, str]],
        max_len: int,
        encode_fn: Callable[[str], list[int]],
        user_id: int,
        assistant_id: int,
        end_id: int,
    ) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        for pair in qa_pairs:
            input_ids, loss_mask = encode_qa_pair(
                question=pair["question"],
                answer=pair["answer"],
                max_len=max_len,
                encode_fn=encode_fn,
                user_id=user_id,
                assistant_id=assistant_id,
                end_id=end_id,
            )
            self.samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.samples[i]
