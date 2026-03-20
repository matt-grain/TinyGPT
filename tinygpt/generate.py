from __future__ import annotations

import torch
import torch.nn.functional as F

from tinygpt.model import TinyGPT
from tinygpt.tokenizer import Tokenizer


def generate(
    model: TinyGPT,
    tokenizer: Tokenizer,
    seed_text: str,
    context_length: int,
    num_words: int = 50,
    temperature: float = 0.8,
    stop_token_id: int | None = None,
    device: torch.device | None = None,
) -> str:
    """Autoregressively generate tokens starting from seed_text.

    WORD-LEVEL vs CHAR-LEVEL:
    Char-level: each mistake loses one letter — model recovers quickly.
    Word-level: each mistake loses an entire word — errors are more visible.
    The generation loop is identical; only the token vocabulary changes.

    TEMPERATURE:
    Scales logits before softmax: low T → sharp/confident/repetitive,
    high T → flat/creative/random. At T=1 the raw model distribution is used.

    MODEL OUTPUT CONTRACT:
    The model outputs logits over ALL vocab tokens at every position.
    Post-processing (softmax + multinomial here) decides what to do with them.
    Greedy decoding (argmax) and beam search are alternative strategies.

    AUTOREGRESSIVE GENERATION:
    At each step, feed the last `context_length` tokens, predict the next one,
    append it, and repeat. The model never sees future tokens (causal masking).
    """
    token_ids: list[int] = tokenizer.encode(seed_text)

    if device is not None:
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(num_words):
            x = torch.tensor(token_ids[-context_length:]).unsqueeze(0)
            if device is not None:
                x = x.to(device)

            logits = model(x)
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

            if stop_token_id is not None and next_id == stop_token_id:
                break

            token_ids.append(next_id)

    return tokenizer.decode(token_ids)


def generate_answer(
    model: TinyGPT,
    tokenizer: Tokenizer,
    question: str,
    context_length: int,
    user_id: int,
    assistant_id: int,
    end_id: int,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    device: torch.device | None = None,
) -> str:
    """Generate an answer to a question using the chat-format prompt.

    Builds the prompt as token IDs directly: [user_id] + encode(question) + [assistant_id].
    Special tokens (user_id, assistant_id) cannot round-trip through decode→encode
    because the tokenizer regex splits "<|user|>" into ['<','|','user','|','>'],
    so we operate on IDs throughout and do not delegate to generate().
    """
    # Build prompt: <|user|> <question tokens> <|assistant|>
    prompt_ids: list[int] = [user_id] + tokenizer.encode(question) + [assistant_id]
    prompt_length = len(prompt_ids)

    token_ids = list(prompt_ids)

    if device is not None:
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor(token_ids[-context_length:]).unsqueeze(0)
            if device is not None:
                x = x.to(device)

            logits = model(x)
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

            if next_id == end_id:
                break

            token_ids.append(next_id)

    # Return only the tokens generated after the assistant marker.
    answer_ids = token_ids[prompt_length:]
    return tokenizer.decode(answer_ids)
