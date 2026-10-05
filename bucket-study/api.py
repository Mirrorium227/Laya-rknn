"""Reuse Laya encoding/decoding; replace only the neural forward with RKNN buckets."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rknn-probe"))
from runtime import Runtime

LENGTHS = (96, 256, 1024)


def bucket_for(tokens):
    for length in LENGTHS:
        if 0 < tokens <= length:
            return length
    raise ValueError(f"Encoded input requires {tokens} tokens; supported total is 1..1024")


class Buckets:
    # ponytail: single caller experiment; serialize calls or use per-worker contexts for a service.
    def __init__(self, agent, precision, flags=0):
        self.agent, self.precision, self.flags = agent, precision, flags
        self.runtimes, self.last = {}, []
        self.original_encode, self.original_forward = agent._encode_state, agent.model.forward
        agent._encode_state, agent.model.forward = self.encode, self.forward

    def encode(self, state, ids, internal, max_len=None, head_max_len=None):
        # Count before upstream truncation. Question/option budgets remain upstream defaults.
        items = self.original_encode(state, ids, internal, max_len=8192, head_max_len=head_max_len)
        for item in items:
            total = len(item["ids"]) + item["state_stats"]["state_tokens_dropped"]
            bucket_for(total)
            if max_len is not None and total > max_len:
                raise ValueError(f"Encoded input requires {total} tokens; requested max_len={max_len}")
            if len(item["markers"]) > 6:
                raise ValueError("RKNN bucket capacity is six candidates per question")
        return items

    def runtime(self, length, suffix=""):
        if length not in self.runtimes:
            if self.precision.startswith('split_'):
                from split_backend import SplitRuntime
                assert not suffix
                self.runtimes[length] = SplitRuntime(length, self.precision, flags=self.flags)
            else:
                model = Path("bucket-study/artifacts") / f"l{length}_{self.precision}{suffix}" / "model.rknn"
                self.runtimes[length] = Runtime(model, [(1, 6), (1, 768)], flags=self.flags)
        return self.runtimes[length]

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        if marker_pos.shape[1] > 6:
            raise ValueError("RKNN bucket capacity is six candidates per question")
        outputs, pooled, self.last = [], [], []
        for row in range(input_ids.shape[0]):
            tokens = int(attention_mask[row].sum())
            length = bucket_for(tokens)
            ids = torch.nn.functional.pad(input_ids[row:row + 1, :tokens],
                (0, length - tokens), value=self.agent.tok.pad_token_id)
            mask = torch.nn.functional.pad(attention_mask[row:row + 1, :tokens], (0, length - tokens))
            positions = torch.nn.functional.pad(marker_pos[row:row + 1].clamp(min=0),
                (0, 6 - marker_pos.shape[1]))
            embeddings = self.agent.model.encoder.get_input_embeddings()(ids).numpy()
            logits, cls = self.runtime(length).run(
                [embeddings, mask.numpy(), positions.numpy(), qtype[row:row + 1].numpy()])
            outputs.append(torch.from_numpy(logits[:, :marker_pos.shape[1]]))
            pooled.append(torch.from_numpy(cls))
            self.last.append({"tokens": tokens, "bucket": length})
        logits = torch.cat(outputs).masked_fill(~marker_mask, -1e4)
        cls = torch.cat(pooled)
        p = torch.softmax(logits, -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(p * p.clamp_min(1e-9).log()).sum(-1) / k.log()
        top2 = p.topk(min(2, p.shape[1]), -1).values
        if p.shape[1] == 1:
            top2 = torch.cat([top2, torch.zeros_like(top2)], -1)
        features = torch.stack((top2[:, 0], top2[:, 0] - top2[:, 1], entropy, k / 255), -1)
        action = self.agent.model.act_head(torch.cat((cls, features), -1))
        assert torch.isfinite(logits).all() and torch.isfinite(action).all()
        return logits, action

    def close(self):
        self.agent._encode_state, self.agent.model.forward = self.original_encode, self.original_forward
        for runtime in self.runtimes.values():
            runtime.close()
        self.runtimes.clear()


if __name__ == "__main__":
    assert [bucket_for(n) for n in (1, 96, 97, 256, 257, 1024)] == [96, 96, 256, 256, 1024, 1024]
    for n in (0, 1025):
        try:
            bucket_for(n)
        except ValueError:
            pass
        else:
            raise AssertionError(n)
    print("bucket boundary checks passed")
