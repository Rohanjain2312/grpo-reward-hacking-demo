"""The reward signal: P(positive) from a real, pretrained sentiment classifier.

This is a legitimate reward model -- a DistilBERT fine-tuned on IMDB sentiment -- not a
keyword or length heuristic. The whole point of the demo is that a *real* classifier can
still be gamed.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentReward:
    def __init__(self, model_name: str, device, dtype=torch.float32, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=dtype
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        pos = [i for i, lab in id2label.items() if "pos" in lab or lab in {"label_1", "1"}]
        if not pos:
            raise RuntimeError(f"cannot find a positive class in {id2label}")
        self.positive_index = pos[0]

    @torch.no_grad()
    def score(self, texts, batch_size: int = 64):
        """Return P(positive) in [0, 1] for each text."""
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
            out.append(probs[:, self.positive_index].cpu())
        return torch.cat(out)
