"""Unified LLaDA-V generation loop with a pluggable refresh policy.

A copy of upstream `FastDLLMGenerationHook._fast_generate_with_embeds`
(third_party/LLaDA-V/train/llava/hooks/fast_dllm_hook.py) with one change: the refresh decision
`i % prefix_refresh_interval == 0` becomes `policy(t, i)`. Attention patching, transfer selection, the
transfer-count schedule and cache slicing are upstream's own methods, called on a registered hook, so with
`FixedInterval(k)` the output must be token-identical to upstream at `prefix_refresh_interval=k` (gate G1(i)).
CFG and generation suffixes are not used by this project and are left out.
"""
import torch

from llava.hooks.fast_dllm_hook import register_fast_dllm_hook

FORBIDDEN_TOKENS = [126081, 126080, 126346, 126347]


def attach(model):
    """Patch the attention layers once per loaded model; returns the hook whose helpers the loop uses."""
    return register_fast_dllm_hook(model)


@torch.no_grad()
def generate(hook, inputs_embeds, policy, steps=128, gen_length=128, block_length=128, threshold=None,
             mask_id=126336, tokenizer=None, stopping_criteria=None, probe=None):
    """Returns (generated token ids [1, L], trace) where trace[k] = (t, block, i, refreshed, n_transferred).

    `probe(t, cache, x_embeds, mask_index, block_start, block_end, n_tok)`, if given, is called after every
    refresh forward with the fresh prompt cache; it must not modify its arguments (src/scripts/probe_replacement.py).
    """
    lm = hook.model
    embed = lm.model.embed_tokens
    policy.reset()
    trace = []
    with torch.cuda.amp.autocast(enabled=True):
        n_prompt = inputs_embeds.shape[1]
        total_length = n_prompt + gen_length
        masked_embed = embed(torch.tensor([mask_id]).to(inputs_embeds.device))
        x_embeds = masked_embed.repeat(1, total_length, 1).to(inputs_embeds.device)
        x_embeds[:, :n_prompt] = inputs_embeds.clone()
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=inputs_embeds.device)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps = steps // num_blocks

        stop_position = n_prompt + gen_length
        found_stop_seq = False
        stop_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in (stopping_criteria or [])]

        t = 0
        for num_block in range(num_blocks):
            block_start = n_prompt + num_block * block_length
            block_end = n_prompt + (num_block + 1) * block_length
            if found_stop_seq and stop_position <= block_start:
                break

            block_mask_index = torch.all(torch.abs(x_embeds[:, block_start:block_end] - masked_embed) < 1e-5, dim=2)
            num_transfer_tokens = hook._get_num_transfer_tokens(block_mask_index, steps)

            i = 0
            while True:
                if threshold is None and i >= steps:
                    break
                mask_index = torch.all(torch.abs(x_embeds - masked_embed) < 1e-5, dim=2)
                if found_stop_seq and not mask_index[0, n_prompt:stop_position].any():
                    break
                if not mask_index[0, block_start:block_end].any():
                    break

                refresh = i == 0 or policy(t, i)
                n_tok = None if threshold is not None else num_transfer_tokens[:, i]
                if refresh:
                    cache = []
                    outputs = lm.model(inputs_embeds=x_embeds, fast_dllm_cache=cache)
                    cache = hook._create_cache_slice(cache, block_start)
                    off = 0
                    if probe is not None:
                        probe(t, cache, x_embeds, mask_index, block_start, block_end, n_tok)
                else:
                    outputs = lm.model(inputs_embeds=x_embeds[:, block_start:], fast_dllm_cache=cache)
                    off = block_start
                logits = lm.lm_head(outputs[0]).float()
                for token_id in FORBIDDEN_TOKENS:
                    logits[:, :, token_id] = torch.where(mask_index[:, off:], -float("inf"), logits[:, :, token_id])

                x0, transfer_index = hook._get_transfer_index(
                    logits, 0.0, "low_confidence", mask_index[:, off:], x[:, off:], n_tok,
                    found_stop_seq, stop_position - off, block_end - off, 0, threshold)
                x0_embeds = embed(x0)
                x0_embeds = torch.where(mask_index[:, off:].unsqueeze(-1).expand_as(x_embeds[:, off:]),
                                        x0_embeds, x_embeds[:, off:])
                x_embeds[:, off:][transfer_index] = x0_embeds[transfer_index]
                x[:, off:][transfer_index] = x0[transfer_index]
                trace.append((t, num_block, i, refresh, transfer_index.sum()))

                if stop_tokens:
                    generated_part = x[0, n_prompt:n_prompt + gen_length]
                    for stop_seq in stop_tokens:
                        for start_idx in range(generated_part.size(0) - len(stop_seq) + 1):
                            if torch.all(generated_part[start_idx:start_idx + len(stop_seq)]
                                         == torch.tensor(stop_seq, device=x.device)):
                                current_position = n_prompt + start_idx
                                if not found_stop_seq or current_position < stop_position:
                                    stop_position = current_position
                                    found_stop_seq = True
                                break
                        if found_stop_seq:
                            break
                i += 1
                t += 1

        end = stop_position if found_stop_seq else n_prompt + gen_length
        return x[:, n_prompt:end], [(t, b, i, r, int(n)) for t, b, i, r, n in trace]
