import torch
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

class SWEAWithCache:
    def __init__(self, model, tokenizer, sink_size=10, window_size=512, local_size=100, ensemble_n=2.0):
        self.model = model
        self.tokenizer = tokenizer
        self.sink_size = sink_size
        self.window_size = window_size
        self.local_size = local_size
        self.ensemble_n = ensemble_n
        self.min_tokens = sink_size + window_size + local_size
        self.kv_cache = {}

    def predict_next_logits(self, input_ids, n=None):
        _n = n if n is not None else self.ensemble_n
        total_len = input_ids.shape[1]
        mid_end = total_len - self.local_size

        if total_len < self.min_tokens:
            with torch.no_grad():
                out = self.model(input_ids)
            return out.logits[:, -1, :], "通常生成"

        windows = self._build_windows(total_len)
        all_probs = []

        for w_start, w_end in windows:
            logits = self._predict_window(input_ids, w_start, w_end, mid_end)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs ** _n)

        fused = torch.cat(all_probs, dim=0).mean(dim=0, keepdim=True)
        fused = fused / fused.sum(dim=-1, keepdim=True)
        return fused, f"アンサンブル+KVキャッシュ（窓数{len(windows)}, N={_n}）"

    def clear_cache(self):
        self.kv_cache.clear()

    def _build_windows(self, total_len):
        stride = self.window_size // 2
        mid_end = total_len - self.local_size
        windows = []
        pos = self.sink_size
        while pos + self.window_size <= mid_end:
            windows.append((pos, pos + self.window_size))
            pos += stride
        last_start = mid_end - self.window_size
        if last_start >= self.sink_size:
            if len(windows) == 0 or windows[-1] != (last_start, mid_end):
                windows.append((last_start, mid_end))
        return windows

    def _predict_window(self, input_ids, w_start, w_end, mid_end):
        window_key = (w_start, w_end)
        if window_key not in self.kv_cache:
            sink_window_ids = torch.cat([
                input_ids[:, :self.sink_size],
                input_ids[:, w_start:w_end]
            ], dim=1)
            with torch.no_grad():
                out_sw = self.model(sink_window_ids, use_cache=True)
            self.kv_cache[window_key] = out_sw.past_key_values

        local_ids = input_ids[:, mid_end:]
        with torch.no_grad():
            out = self.model(
                local_ids,
                past_key_values=self.kv_cache[window_key],
                use_cache=False
            )
        return out.logits[:, -1, :]
