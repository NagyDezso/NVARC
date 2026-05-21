"""Per-puzzle solver for HRM-Text-1B.

For each ARC puzzle the pipeline is:

  1. Test-time training (TTT): reset the LoRA adapter, then briefly fine-tune
     it on dihedral+colour augmentations of the puzzle's demonstration pairs.
  2. Decode: for each test input, build 16 augmented prompts, run the batched
     `turbo_dfs` decoder (DFS over the token tree, pruned by cumulative NLL),
     invert the augmentation on each decoded grid.
  3. Score: re-score every candidate grid as the *answer* under 8 fresh
     augmentations (`calc_scores`) to get an augmentation-consistency signal.
  4. Dump per-subkey `{beam_score, score_aug, solution}` candidates for the
     selection stage (arc_decoder.py).

PrefixLM decoding: the prompt block is fed with `token_type_ids == 1`. HRM only
consults `token_type_ids` on the first (prefix) forward, so incremental decode
steps need none. `turbo_dfs` restores the KV cache between sibling DFS branches
with `DynamicCache.crop()`; HRM's recurrent forward writes one cache slot per
H/L-cycle invocation, all of equal sequence length, so a single crop is
consistent across slots.
"""

from __future__ import annotations

import bz2
import gc
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch

from arc_loader import ArcDataset, HrmFormatter
from serialize import DEFAULT_CONDITION, build_sample


# ----------------------------------------------------------------------------
# ARC token set on the HRM tokenizer
# ----------------------------------------------------------------------------

class ArcTokens:
    """Resolves the small set of tokens an ARC grid serialization can emit."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.digit_ids = []
        for d in range(10):
            ids = tokenizer.encode(str(d), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"digit {d!r} tokenizes to {len(ids)} tokens ({ids}); the "
                    "turbo-DFS decoder needs one token per digit. Run "
                    "HRM/prepare_tokenizer.py to build a digit-clean tokenizer."
                )
            self.digit_ids.append(ids[0])
        nl = tokenizer.encode("\n", add_special_tokens=False)
        if len(nl) != 1:
            raise ValueError(f"newline tokenizes to {len(nl)} tokens ({nl}).")
        self.newline_id = nl[0]
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_id
        # Tokens the decoder is allowed to emit inside a grid.
        self.tokens = self.digit_ids + [self.newline_id, self.eos_id]


# ----------------------------------------------------------------------------
# turbo-DFS decoder
# ----------------------------------------------------------------------------

def _crop_cache(cache, length: int) -> None:
    """Truncate a (Dynamic)Cache back to `length` tokens, in place."""
    if cache is None:
        return
    if hasattr(cache, "crop"):
        cache.crop(length)
    else:  # very old transformers fallback
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :length, :]
            cache.value_cache[i] = cache.value_cache[i][..., :length, :]


class _DFSNode:
    """One node of the turbo-DFS tree: a decode position and its open children.

    `candidates[i]` is the score-sorted list of next tokens still to explore for
    beam `i`. `suffixes` accumulates completed `(cumulative_nll, [token, ...])`
    paths discovered at or below this node. `parent_tokens` is the token vector
    the parent stepped to reach this node, prepended to every suffix on the way
    back up.
    """

    __slots__ = ("pos", "max_new_tokens", "candidates", "suffixes",
                 "parent", "parent_tokens")

    def __init__(self, pos, max_new_tokens, candidates, suffixes,
                 parent, parent_tokens):
        self.pos = pos
        self.max_new_tokens = max_new_tokens
        self.candidates = candidates
        self.suffixes = suffixes
        self.parent = parent
        self.parent_tokens = parent_tokens


def _dfs_node(logits, scores, pos, max_new_tokens, max_score, arc,
              parent, parent_tokens) -> _DFSNode:
    """Build a DFS node from next-token `logits` and parent cumulative `scores`.

    A token is opened as a child only while the beam's cumulative NLL stays
    below `max_score`; a token equal to EOS completes a suffix immediately.
    """
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) \
        - logits.float().cpu().log_softmax(-1)
    suffixes = defaultdict(list)
    candidates = {}
    for i in range(n):
        cand = []
        for t in arc.tokens:
            score = nll[i, t].item()
            if score < max_score:
                if t == arc.eos_id:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    cand.append((score, t))
        cand.sort(key=lambda x: x[0])
        candidates[i] = cand
    return _DFSNode(pos, max_new_tokens, candidates, suffixes,
                    parent, parent_tokens)


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache,
              start_time, end_time, arc, time_budget=540) -> dict:
    """Depth-first search over the token tree, batched across `n` live beams.

    Iterative (explicit stack): the search depth equals the grid's token length
    (a 30x30 grid is ~930), which would overflow Python's recursion limit. One
    batched decode step advances every beam by one token; `cache` is the single
    shared KV cache, cropped back to a node's `pos` before each of its sibling
    steps so branches do not contaminate each other.

    `logits` are the next-token logits for each beam at position `pos-1`.
    A child token is explored only while the beam's cumulative NLL stays below
    `max_score`. Returns {beam_id: [(cumulative_nll, [token, ...]), ...]}.
    """
    n = logits.size(0)
    root = _dfs_node(logits, scores, pos, max_new_tokens, max_score, arc,
                     parent=None, parent_tokens=None)
    stack = [root]
    time_up = False

    while stack:
        node = stack[-1]
        if not time_up and (time.time() - start_time >= time_budget
                            or time.time() >= end_time):
            time_up = True

        # Pick one candidate per beam for this node's next sibling step.
        batch_tokens, batch_scores, num_alive = [], [], 0
        if not time_up:
            for i in range(n):
                cand = node.candidates[i]
                if cand:
                    score, t = cand.pop(0)
                    batch_tokens.append(t)
                    batch_scores.append(score)
                    num_alive += 1
                else:
                    batch_tokens.append(arc.pad_id)
                    batch_scores.append(1000.0)

        if time_up or num_alive == 0:
            # Node exhausted: pop it and fold its suffixes into the parent.
            stack.pop()
            if node.parent is not None:
                for beam_id, beams in node.suffixes.items():
                    for score, suffix_tokens in beams:
                        suffix_tokens.insert(0, node.parent_tokens[beam_id])
                        node.parent.suffixes[beam_id].append((score, suffix_tokens))
            continue

        # Descend: one batched decode step from this node's position.
        _crop_cache(cache, node.pos)
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device,
                                   dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), node.pos, device=model.device,
                                    dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        stack.append(_dfs_node(
            outputs.logits[:, -1], batch_scores, node.pos + 1,
            node.max_new_tokens - 1, max_score, arc,
            parent=node, parent_tokens=batch_tokens,
        ))

    return root.suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_token_batch, token_type_batch,
                        max_new_tokens, max_score, end_time, arc, time_budget=540):
    """Run the prefix forward, then DFS-decode every grid continuation.

    `prefix_token_batch` is a list of equal-length token id lists (augmentations
    of one test input share a grid shape, hence a token count).
    """
    lengths = {len(p) for p in prefix_token_batch}
    if len(lengths) != 1:
        raise ValueError(f"turbo-DFS batch needs equal-length prefixes, got {sorted(lengths)}")

    input_ids = torch.tensor(prefix_token_batch, device=model.device, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_batch, device=model.device, dtype=torch.long)
    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,   # PrefixLM mask — first forward only
        use_cache=True,
        return_dict=True,
    )
    suffixes = turbo_dfs(
        model, outputs.logits[:, -1], max_new_tokens, max_score,
        scores=[0.0] * input_ids.size(0), pos=input_ids.size(1),
        cache=outputs.past_key_values, start_time=time.time(),
        end_time=end_time, arc=arc, time_budget=time_budget,
    )
    result = []
    for beam_id, beams in suffixes.items():
        result.append((beam_id, sorted(beams, key=lambda x: x[0])))
    return result


# ----------------------------------------------------------------------------
# Augmentation-consistency scoring
# ----------------------------------------------------------------------------

@torch.no_grad()
def calc_scores(query_token_lists, answer_token_lists, query_prefix_lens, model,
                arc, chunk_size: int = 4):
    """Negative log-likelihood of each answer given its query.

    `query_prefix_lens[i]` is how many leading tokens of query i are the
    bidirectional PrefixLM block (the whole query, here).

    The batch is processed in chunks of `chunk_size`: a full-vocab logit
    tensor is `[chunk, seq, vocab]` and HRM's vocab is 65k, so a large chunk
    materialises tens of GB. Answer log-probs are gathered on-device per
    sample and only scalars are returned, so peak memory is one chunk's
    logits, never the whole batch.
    """
    n = len(query_token_lists)
    result: list[float] = []
    for start in range(0, n, chunk_size):
        q_chunk = query_token_lists[start:start + chunk_size]
        a_chunk = answer_token_lists[start:start + chunk_size]
        plen_chunk = query_prefix_lens[start:start + chunk_size]

        batch = [q + a for q, a in zip(q_chunk, a_chunk)]
        max_len = max(len(toks) for toks in batch)

        input_ids = torch.full((len(batch), max_len), arc.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        token_type_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (toks, plen) in enumerate(zip(batch, plen_chunk)):
            input_ids[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
            attention_mask[i, :len(toks)] = 1
            token_type_ids[i, :plen] = 1   # query = bidirectional prefix

        outputs = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            token_type_ids=token_type_ids.to(model.device),
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits  # [chunk, seq, vocab], device, model dtype

        for i, (q, a) in enumerate(zip(q_chunk, a_chunk)):
            ql = len(q)
            # Logits at positions ql-1 .. ql-1+len(a)-1 predict the answer.
            ans_logits = logits[i, ql - 1: ql - 1 + len(a)].float()
            ans_logp = ans_logits.log_softmax(-1)
            idx = torch.arange(len(a), device=ans_logp.device)
            ans_tokens = torch.tensor(a, dtype=torch.long, device=ans_logp.device)
            ans_score = ans_logp[idx, ans_tokens].sum()
            result.append(-ans_score.item())

        del outputs, logits
    return result


# ----------------------------------------------------------------------------
# Test-time training
# ----------------------------------------------------------------------------

# A vocab at or below this size is treated as "cut" (HRM/prepare_tokenizer.py
# reduces HRM's 65k vocab to the ~40 ARC tokens). HRM's 4096-token context is a
# safe separator: nothing legitimate sits between ~40 and 65536.
CUT_VOCAB_MAX = 4096

# LoRA targets that always apply: the attention and MLP projections.
LORA_BASE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")


def build_lora(model, r=256, alpha=32, dropout=0.0,
               target_modules=None, use_rslora=True):
    """LoRA-wrap `model` for test-time training.

    `target_modules=None` auto-selects based on vocab size: the attention/MLP
    projections always, plus `lm_head` only when the vocab has been cut. HRM
    ties `lm_head` to `embed_tokens`; on the full 65k vocab, adapting that tied
    layer desyncs the input/output weights and makes PEFT serialize the whole
    embedding into every adapter. With a cut (~40-token) vocab those costs
    vanish and adapting the output head is worthwhile.
    """
    if target_modules is None:
        target_modules = list(LORA_BASE_TARGETS)
        embed = model.get_input_embeddings()
        vocab_size = embed.weight.shape[0] if embed is not None \
            else getattr(model.config, "vocab_size", CUT_VOCAB_MAX + 1)
        if vocab_size <= CUT_VOCAB_MAX:
            target_modules.append("lm_head")

    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none", task_type="CAUSAL_LM", use_rslora=use_rslora,
    )
    return get_peft_model(model, lora_cfg)


def test_time_train(model, tokenizer, formatter, puzzle_ds, *,
                    n_aug=16, lr=5e-5, max_length=4096, seed=1, device="cuda"):
    """Fine-tune the (already LoRA-wrapped) model on augmented demo pairs.

    Each augmented puzzle variant contributes one PrefixLM sample: its train
    pairs minus the last form the prefix, the last is the causal target.
    `shuffle_ex` inside `augment` rotates which pair is held out.
    """
    train_ds = puzzle_ds.augment(n=n_aug, shfl_keys=True, seed=seed)
    train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_length)

    samples = []
    for key in train_ds.keys:
        pairs = train_ds.queries[key]["train"]
        if len(pairs) < 2:
            continue
        messages = []
        for p in pairs:
            messages.append({"role": "user", "content": _grid(p["input"])})
            messages.append({"role": "assistant", "content": _grid(p["output"])})
        s = build_sample(messages, tokenizer, formatter.condition, max_length)
        if s is not None:
            samples.append(s)
    if not samples:
        return model

    model.train()
    # HRM's forward re-applies 32 layers H_cycles*L_cycles times; without
    # gradient checkpointing every recurrent activation is retained for the
    # backward pass and a single seq-4096 sample needs tens of GB.
    # enable_input_require_grads is required for checkpointing to propagate
    # gradients into a LoRA adapter sitting on an otherwise-frozen base.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total = len(samples)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    warmup = max(1, total // 10)

    rng = random.Random(seed)
    order = list(range(total))
    rng.shuffle(order)
    for step, idx in enumerate(order):
        s = samples[idx]
        for g in opt.param_groups:                       # linear warmup
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        out = model(
            input_ids=torch.tensor([s.input_ids], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, len(s.input_ids)), dtype=torch.long, device=device),
            token_type_ids=torch.tensor([s.token_type_ids], dtype=torch.long, device=device),
            labels=torch.tensor([s.labels], dtype=torch.long, device=device),
            use_cache=False,   # incompatible with gradient checkpointing
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step + 1 >= warmup:
            sched.step()

    # Decoding needs the KV cache, which checkpointing disables.
    model.gradient_checkpointing_disable()
    model.eval()
    return model


def _grid(g):
    from serialize import grid_to_text
    return grid_to_text(g)


# ----------------------------------------------------------------------------
# Per-puzzle solve
# ----------------------------------------------------------------------------

def solve_puzzle(model, tokenizer, formatter, puzzle_ds, *,
                 max_seq_length=4096, max_new_tokens=None, max_score=None,
                 decode_batch=4, end_time=float("inf"), device="cuda"):
    """Decode + score one puzzle (TTT must already have been applied).

    Returns {subkey: [candidate_dict, ...]} where subkey is `{base}_{testidx}`.
    """
    if max_new_tokens is None:
        max_new_tokens = formatter.max_new_tokens()
    if max_score is None:
        max_score = -np.log(0.2)

    puzzle_ds_multi = puzzle_ds.split_multi_replies()
    eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
    eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input",
                                 max_len=max_seq_length - max_new_tokens)

    # Group augmentation subkeys by the test input they belong to.
    test_to_subkeys = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0]   # `{base}_{testidx}`
        test_to_subkeys[test_id].append(subkey)

    results = defaultdict(list)
    known_scores = {}

    for test_id, subkeys in test_to_subkeys.items():
        for batch_start in range(0, len(subkeys), decode_batch):
            if time.time() > end_time:
                return results
            batch = subkeys[batch_start: batch_start + decode_batch]

            prefix_tokens, prefix_ttids = [], []
            for subkey in batch:
                ids = tokenizer.encode(eval_ds.get(subkey, formatter)["input"],
                                       add_special_tokens=False)
                prefix_tokens.append(ids)
                prefix_ttids.append([1] * len(ids))   # whole prompt = prefix
            # Augmentations of one test input share a grid shape -> equal length.
            if len({len(p) for p in prefix_tokens}) != 1:
                # Fall back to per-subkey decoding when shapes differ.
                singles = [[i] for i in range(len(batch))]
            else:
                singles = [list(range(len(batch)))]

            for group in singles:
                gt = [prefix_tokens[i] for i in group]
                gtt = [prefix_ttids[i] for i in group]
                dfs = inference_turbo_dfs(model, gt, gtt, max_new_tokens,
                                          max_score, end_time, _arc(tokenizer))
                for local_id, scored_beams in dfs:
                    subkey = batch[group[local_id]]
                    base = subkey.split(".")[0]
                    candidates = []
                    for beam_score, toks in scored_beams:
                        array = formatter.convert_tokens_to_array(toks)
                        if array is None:
                            continue
                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                        grid_id = (base, tuple(map(tuple, solution)))
                        if grid_id in known_scores:
                            score_aug = known_scores[grid_id]
                        else:
                            score_aug = _augmentation_scores(
                                model, tokenizer, formatter, puzzle_ds_multi,
                                base, solution, max_seq_length, max_new_tokens)
                            known_scores[grid_id] = score_aug
                        candidates.append({
                            "beam_score": beam_score,
                            "score_aug": score_aug,
                            "solution": solution,
                        })
                    if candidates:
                        results[subkey].extend(candidates)
    return results


def _augmentation_scores(model, tokenizer, formatter, puzzle_ds_multi,
                         base, solution, max_seq_length, max_new_tokens):
    """Re-score a candidate `solution` as the answer under 8 augmentations."""
    aug = ArcDataset(
        keys=[base],
        queries={base: puzzle_ds_multi.queries.get(base)},
        replies={base: [solution.tolist()]},
    )
    aug = aug.augment(seed=hash(base) % (1024 ** 2))
    aug = aug.cut_to_len(formatter=formatter, name="input",
                         max_len=max_seq_length - max_new_tokens)
    q_tokens, q_prefix_lens, a_tokens = [], [], []
    for sample in aug.as_list(formatter):
        q = tokenizer.encode(sample["input"], add_special_tokens=False)
        a = tokenizer.encode(sample["reply"], add_special_tokens=False)
        q_tokens.append(q)
        q_prefix_lens.append(len(q))
        a_tokens.append(a)
    # calc_scores chunks the batch internally to bound peak memory.
    return calc_scores(q_tokens, a_tokens, q_prefix_lens, model, _arc(tokenizer))


_ARC_CACHE = {}


def _arc(tokenizer) -> ArcTokens:
    key = id(tokenizer)
    if key not in _ARC_CACHE:
        _ARC_CACHE[key] = ArcTokens(tokenizer)
    return _ARC_CACHE[key]


# ----------------------------------------------------------------------------
# Worker loop
# ----------------------------------------------------------------------------

def load_model_and_tokenizer(base, checkpoint=None, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    src = checkpoint or base
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=dtype, trust_remote_code=True, attn_implementation="sdpa",
    )
    # LoRA SFT checkpoint -> merge into the base before TTT re-wraps it.
    if checkpoint and os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=dtype, trust_remote_code=True, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    return model.to(device).eval(), tokenizer


def worker(rank, queue, end_time, *, base, checkpoint, tasks_path,
           out_dir, max_seq_length=4096, decode_batch=4,
           lora_r=256, ttt_lr=5e-5, ttt_aug=16, device="cuda"):
    """Pull puzzle keys off `queue`, solve each, dump candidates to `out_dir`."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    os.makedirs(out_dir, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(base, checkpoint, device=device)
    model = build_lora(model, r=lora_r)
    formatter = HrmFormatter(tokenizer, condition=DEFAULT_CONDITION)
    _arc(tokenizer)  # fail fast if the tokenizer can't represent ARC grids

    default_weights = {k: v.clone().detach()
                       for k, v in get_peft_model_state_dict(model).items()}

    arc_test = ArcDataset.from_dir(tasks_path) if os.path.isdir(tasks_path) \
        else ArcDataset.from_file(tasks_path)

    while not queue.empty():
        if time.time() > end_time:
            print(f"[rank {rank}] out of time")
            break
        try:
            key = queue.get_nowait()
        except Exception:
            break
        if key is None:
            break

        t0 = time.time()
        print(f"[rank {rank}] {key}: TTT...", flush=True)
        set_peft_model_state_dict(model, {k: v.clone() for k, v in default_weights.items()})
        puzzle_ds = arc_test.change_keys([key])

        test_time_train(model, tokenizer, formatter, puzzle_ds,
                        n_aug=ttt_aug, lr=ttt_lr, max_length=max_seq_length,
                        device=device)

        print(f"[rank {rank}] {key}: decoding "
              f"(TTT took {time.time() - t0:.1f}s)...", flush=True)
        with torch.inference_mode():
            results = solve_puzzle(model, tokenizer, formatter, puzzle_ds,
                                   max_seq_length=max_seq_length,
                                   decode_batch=decode_batch,
                                   end_time=end_time, device=device)
        for subkey, candidates in results.items():
            with bz2.BZ2File(os.path.join(out_dir, subkey), "w") as f:
                pickle.dump(candidates, f)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[rank {rank}] solved {key} in {time.time() - t0:.1f}s "
              f"({len(results)} subkeys)")
