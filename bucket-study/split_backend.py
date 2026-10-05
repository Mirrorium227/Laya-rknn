"""Three RKNN contexts with exact CPU key-mask and NaN-only guards."""
import json
import time
from pathlib import Path

import numpy as np


def clear_nan(value):
    return np.where(np.isnan(value), np.float32(0), value)


def key_bias(mask):
    return np.where(mask[:, None, None, :] != 0, np.float32(0), np.float32(-np.inf))


class SplitRuntime:
    def __init__(self, length, precision, flags=0):
        from runtime import Runtime
        self.manifest = json.loads(Path(f'bucket-study/split/l{length}/manifest.json').read_text())
        self.parts, self.last = [], []
        self.tail_cut = precision in ('split_tail_fp16', 'split_tail_int8')
        self.record_guards = bool(flags & 8)
        try:
            for index, part in enumerate(self.manifest['parts']):
                variant = precision if index == (2 if 'head2' in precision else 0) else 'split_fp16'
                path = Path(f'bucket-study/artifacts/l{length}_{variant}/part{index}/model.rknn')
                shapes = part['output_shapes']
                if self.tail_cut:
                    variant = 'split_fp16'
                    path = Path(f'bucket-study/artifacts/l{length}_{variant}/part{index}/model.rknn')
                    if index == 2:
                        suffix = '' if length == 96 else f'_l{length}'
                        path = Path(f'bucket-study/artifacts/head_attention_probe{suffix}/stock_fp16/prefix.rknn')
                        shapes = [(1, length, 768)]
                self.parts.append(Runtime(path, shapes, flags))
            if self.tail_cut:
                policy = 'stock_fp16' if precision == 'split_tail_fp16' else 'ffn_int8'
                path = Path(f'bucket-study/artifacts/head_ffn_probe{suffix}/{policy}/prefix.rknn')
                self.parts.append(Runtime(path, self.manifest['parts'][2]['output_shapes'], flags))
            self.version, self.driver = self.parts[0].version, self.parts[0].driver
            self.mapped = self.parts[0].mapped
            assert all((p.version, p.driver, p.mapped) == (self.version, self.driver, self.mapped) for p in self.parts)
            self.model_path = [p.model_path for p in self.parts]
            self.input_attrs = [p.input_attrs for p in self.parts]
            self.output_attrs = [p.output_attrs for p in self.parts]
            # Inf key masks and guarded probabilities must never be normalized into INT8.
            assert self.parts[0].input_attrs[3]['type'] == self.parts[1].input_attrs[3]['type'] == 1
            assert all(p.output_attrs[0]['type'] == 1 for p in self.parts[:2])
            assert self.parts[1].input_attrs[0]['type'] == self.parts[2].input_attrs[0]['type'] == 1
            if 'head2' in precision or self.tail_cut:
                assert all(a['type'] == 1 for a in self.parts[-1].output_attrs)
            if self.tail_cut:
                assert self.parts[2].output_attrs[0]['type'] == self.parts[3].input_attrs[0]['type'] == 1
        except Exception:
            self.close()
            raise

    def run(self, inputs):
        embeddings, mask, marker, qtype = inputs
        bias = key_bias(mask)
        values, self.last = [embeddings, mask, qtype, bias], []
        for index, part in enumerate(self.parts):
            output = part.run(values)
            if index < 2:
                started = time.perf_counter() if self.record_guards else 0
                clean = clear_nan(output[0])
                if self.record_guards:
                    guard_ms = (time.perf_counter() - started) * 1000
                    self.last.append({'part': index, 'nan_values_cleared': int(np.isnan(output[0]).sum()),
                        'infinite_values_retained': int(np.isinf(output[0]).sum()), 'cpu_guard_ms': guard_ms})
                values = [clean, output[1], output[2], bias if index == 0 else marker]
                if index == 1 and self.tail_cut:
                    values = values[:3]
            elif index == 2 and self.tail_cut:
                values = [output[0], marker]
        return output

    def memory(self):
        return {'parts': {str(index): part.memory() for index, part in enumerate(self.parts)},
                'aggregation': 'Per-context queries; do not sum these with process RSS.'}

    def profile(self):
        return {f'part{index}': part.profile() for index, part in enumerate(self.parts)}

    def close(self):
        for part in self.parts:
            part.close()
        self.parts.clear()


if __name__ == '__main__':
    assert np.array_equal(clear_nan(np.array([np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)),
                          [0, np.inf, -np.inf, 0.25])
    assert np.array_equal(key_bias(np.array([[1, 0]])), [[[[0, -np.inf]]]])
    print('CPU NaN-only guard and key mask checks passed', flush=True)
