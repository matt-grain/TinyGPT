from __future__ import annotations

import re
from collections import Counter
from typing import ClassVar


class Tokenizer:
    # WORD-LEVEL TOKENIZER: moves complexity from sequence length to vocabulary dimension.
    # Char-level: vocab=~126, model learns spelling + grammar + meaning. Slow to converge.
    # Word-level: vocab=~10K, spelling is "free" (baked into embeddings),
    #   model focuses on word RELATIONSHIPS (grammar, meaning). Faster convergence,
    #   but embedding table explodes: 126*64=8K params → 10003*128=1.28M params (160x).
    # Real GPT-2 uses BPE (byte-pair encoding) — a middle ground between char and word.
    #
    # Split on spaces but keep punctuation as SEPARATE tokens:
    # "Bonjour, dit-il." → ["Bonjour", ",", "dit", "-", "il", "."]
    # Punctuation is predicted by the model just like words, not "decoration".
    PATTERN: ClassVar[str] = r"\w+|[^\w\s]"

    def __init__(self, word_to_id: dict[str, int], id_to_word: dict[int, str]) -> None:
        self._word_to_id = word_to_id
        self._id_to_word = id_to_word

    def encode(self, text: str) -> list[int]:
        """Tokenize text to a list of token IDs. Unknown words map to unk_id."""
        tokens = re.findall(self.PATTERN, text, re.UNICODE)
        return [self._word_to_id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to text."""
        return " ".join(self._id_to_word[i] for i in ids)

    def add_special_tokens(self, tokens: list[str]) -> None:
        """Add special tokens (e.g. <|user|>, <|assistant|>) to the vocabulary."""
        for token in tokens:
            if token not in self._word_to_id:
                new_id = len(self._word_to_id)
                self._word_to_id[token] = new_id
                self._id_to_word[new_id] = token

    @classmethod
    def from_corpus(cls, text: str, max_vocab: int = 10000) -> Tokenizer:
        """Build a Tokenizer from raw text, keeping the top max_vocab words."""
        tokens = re.findall(cls.PATTERN, text, re.UNICODE)

        print(f"Total tokens in text: {len(tokens)}")

        freq: Counter[str] = Counter(tokens)
        print(f"Unique words/punctuation: {len(freq)}")
        print(f"Most common: {freq.most_common(20)}")

        # Keep only top max_vocab words + special tokens.
        # <PAD> for padding sequences to equal length.
        # <UNK> for unknown words at inference time.
        # <EOS> for end of sequence — the model learns to predict this token
        #       to signal "I'm done generating."
        special: list[str] = ["<PAD>", "<UNK>", "<EOS>"]
        vocab = special + [word for word, _ in freq.most_common(max_vocab)]

        word_to_id = {w: i for i, w in enumerate(vocab)}
        id_to_word = {i: w for i, w in enumerate(vocab)}

        print(f"\nVocabulary size: {len(vocab)}")
        print(f"Coverage: words in top {max_vocab} / total unique words")

        # How many tokens will become <UNK>?
        unk_count = sum(count for word, count in freq.items() if word not in word_to_id)
        print(
            f"Tokens that become <UNK>: {unk_count} ({100 * unk_count / len(tokens):.1f}% of text)"
        )

        return cls(word_to_id, id_to_word)

    @property
    def vocab_size(self) -> int:
        return len(self._word_to_id)

    @property
    def unk_id(self) -> int:
        return self._word_to_id["<UNK>"]

    @property
    def eos_id(self) -> int:
        return self._word_to_id["<EOS>"]

    @property
    def pad_id(self) -> int:
        return self._word_to_id["<PAD>"]
