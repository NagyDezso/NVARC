"""ARC dataset + augmentation machinery for the HRM-Text-1B solver.

`ArcDataset` and the augmentation helpers (`permute_mod`, `forward_mod`,
`invert_mod`, `augment`, `shuffle_ex`, `cut_to_len`, ...) are model-agnostic.
`HrmFormatter` emits the HRM-Text-1B PrefixLM wire format
(`<|im_start|>{condition}...<|im_end|>...<|EOS|>`) and derives the
`token_type_ids` that mark the bidirectional prefix.

See serialize.py for the format rationale.
"""

from __future__ import annotations

import json

import numpy as np
from serialize import DEFAULT_CONDITION, grid_to_text

# ----------------------------------------------------------------------------
# Augmentation primitives
# ----------------------------------------------------------------------------


def is_valid_solution(guess) -> bool:
    return (
        isinstance(guess, np.ndarray)
        and guess.ndim == 2
        and all(0 < x <= 30 for x in guess.shape)
    )


def shuffled(data_list):
    return np.random.permutation(data_list).tolist()


def permute_mod(a, descriptor, invert=False):
    """Apply (or invert) a colour permutation encoded in `descriptor`."""
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation) == list(range(10))
    a = np.asarray(a)
    if a.ndim == 3:
        if not invert:
            permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        assert a.ndim == 2
        if invert:
            permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a


def permute_rnd_all_(query):
    permutation = np.random.permutation(10).tolist()
    return "permute" + "".join(map(str, permutation))


class HrmFormatter:
    """Serializes ARC puzzles into the HRM-Text-1B PrefixLM wire format.

    A puzzle is a list of {"input", "output"} pairs. We emit one
    `<|im_start|>{cond}{input}<|im_end|>{output}<|EOS|>` block per pair. For the
    final (challenge) pair the assistant block is left open during inference.

    The whole demonstration block + the challenge input (up to and including the
    last `<|im_end|>`) is the *bidirectional prefix*; everything after is the
    *causal response*. `token_type_ids_for` recovers that split from token ids.
    """

    def __init__(self, tokenizer, condition: str = DEFAULT_CONDITION):
        self.tokenizer = tokenizer
        self.condition = condition
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        self.eos = tokenizer.eos_token  # `<|box_end|>` on the HRM tokenizer
        self.im_end_id = tokenizer.convert_tokens_to_ids(self.im_end)
        self.eos_id = tokenizer.eos_token_id

    def fmt_query(self, query) -> str:
        grid_input = grid_to_text(query[0]["input"])
        return f"{self.im_start}{self.condition}{grid_input}{self.im_end}"

    def fmt_reply(self, reply) -> str:
        return grid_to_text(reply[0]) + self.eos

    def fmt_train(self, train, last_is_challenge=False) -> str:
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            gi = grid_to_text(x["input"])
            go = grid_to_text(x["output"])
            text += f"{self.im_start}{self.condition}{gi}{self.im_end}{go}{self.eos}"
        if test is not None:
            text += self.fmt_query([test])
        return text

    def max_new_tokens(self) -> int:
        """Token budget for the largest (30x30) possible reply."""
        max_grid = np.zeros([30, 30], dtype=int)
        return (
            len(
                self.tokenizer.encode(
                    self.fmt_reply([max_grid]), add_special_tokens=False
                )
            )
            + 1
        )

    def token_type_ids_for(self, input_ids: list[int]) -> list[int]:
        """1 for prefix (bidirectional) tokens, 0 for response (causal) tokens.

        The prefix ends at the *last* `<|im_end|>` — that token closes the
        challenge input; everything after is the predicted grid.
        """
        last_imend = max(i for i, t in enumerate(input_ids) if t == self.im_end_id)
        n = len(input_ids)
        return [1] * (last_imend + 1) + [0] * (n - last_imend - 1)

    def convert_tokens_to_array(self, tokens, limit_rows: int = 30):
        """Decode a generated token sequence into an integer grid, or None."""
        if len(tokens) < 2:
            return None
        # Drop the trailing eos before decoding.
        text = self.tokenizer.decode(tokens[:-1], skip_special_tokens=True)
        try:
            lines = text.strip().split("\n")
            by_rows = [
                row
                for row in [[int(x) for x in line if x.isdigit()] for line in lines]
                if len(row)
            ]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if is_valid_solution(array):
                return array
        except Exception:
            pass
        return None


class ArcDataset:
    @staticmethod
    def forward_mod(a, key, use_perm=True):
        if a is None:
            return a
        for op in key.split(".")[1:]:
            if op == "rot90":
                a = np.rot90(a)
            elif op == "transpose":
                a = np.swapaxes(a, 0, 1)
            elif op.startswith("permute"):
                a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith(("copy", "out", "ex", "run")):
                a = np.copy(a) if op.startswith("copy") else a
            else:
                raise NotImplementedError(f"Operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True):
        if a is None:
            return a
        for op in key.split(".")[1:][::-1]:
            if op == "rot90":
                a = np.rot90(a, k=3)
            elif op == "transpose":
                a = np.swapaxes(a, 0, 1)
            elif op.startswith("permute"):
                a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith(("copy", "out", "ex", "run")):
                a = np.copy(a) if op.startswith("copy") else a
            else:
                raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies=None, keys=None, is_orig=False):
        replies = {} if replies is None else replies
        if keys is not None:
            keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = (
            replies if keys is None else {k: replies[k] for k in keys if k in replies}
        )
        self.is_orig = is_orig
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.transposed_dataset = None

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(
            queries=self.queries, replies=self.replies, keys=keys, **flags
        )

    @classmethod
    def from_file(cls, queries_file, keys=None):
        with open(queries_file) as f:
            queries = json.load(f)
        return cls(queries=queries, is_orig=True, keys=keys)

    @classmethod
    def from_dir(cls, tasks_dir, keys=None):
        """Load a directory of one-task-per-file ARC JSONs (ARC-AGI-2 layout)."""
        import glob
        import os

        queries = {}
        for path in sorted(glob.glob(os.path.join(tasks_dir, "*.json"))):
            task_id = os.path.splitext(os.path.basename(path))[0]
            with open(path) as f:
                queries[task_id] = json.load(f)
        return cls(queries=queries, is_orig=True, keys=keys)

    def load_replies(self, replies_file):
        with open(replies_file) as f:
            replies_parsed = json.load(f)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self):
        key_indices = [
            (k, i) for k in self.keys for i in range(len(self.queries[k]["test"]))
        ]
        return self.__class__(
            keys=[f"{k}_{i}" for k, i in key_indices],
            queries={
                f"{k}_{i}": {
                    "train": self.queries[k]["train"],
                    "test": [self.queries[k]["test"][i]],
                }
                for k, i in key_indices
            },
            replies={
                f"{k}_{i}": [self.replies[k][i]]
                for k, i in key_indices
                if k in self.replies
            },
        )

    def shuffled(self):
        return self.__class__(
            queries=self.queries, replies=self.replies, keys=shuffled(self.keys)
        )

    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys=[k for d in datasets for k in d.keys],
        )

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        queries, replies, keys = {}, {}, []
        for k0 in self.keys:
            desc = (
                ("copy{i}" if mod_func is np.copy else mod_func.__name__)
                if descriptor is None
                else descriptor
                if isinstance(descriptor, str)
                else descriptor(self.queries[k0])
            ).format(i=i)
            func = lambda a, d: np.asarray(
                mod_func(a) if descriptor is None else mod_func(a, d)
            ).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            queries[k1] = {
                m: [
                    {
                        t: (func(a, desc) if t == "input" or not inputs_only else a)
                        for t, a in x.items()
                    }
                    for x in e
                ]
                for m, e in self.queries[k0].items()
            }
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        return self.__class__(queries=queries, replies=replies, keys=keys)

    def mod(
        self,
        mod_func,
        descriptor=None,
        n=1,
        stack=None,
        keep=False,
        keep_key=False,
        shuffle=False,
        join=True,
        inputs_only=False,
    ):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None:
            stack = mod_func.__name__.startswith("rot")
        for i in range(n):
            cur = (cur if stack else self).mod_single(
                mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only
            )
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter: HrmFormatter):
        train = formatter.fmt_train(self.queries[key]["train"])
        query = formatter.fmt_query(self.queries[key]["test"])
        reply = formatter.fmt_reply(self.replies[key]) if key in self.replies else ""
        text = (
            train + query + reply
            if reply
            else formatter.fmt_train(self.queries[key]["train"], last_is_challenge=True)
        )
        return dict(
            key=key,
            train=train,
            query=query,
            reply=reply,
            input=train + query,
            text=text,
        )

    def as_list(self, formatter: HrmFormatter):
        return [self.get(key, formatter) for key in self.keys]

    def get_length(self, key, formatter: HrmFormatter, name, max_of_transposed=False):
        if formatter is None:
            if name == "input":
                return sum(
                    np.prod(np.shape(v))
                    for v3 in self.queries[key].values()
                    for v2 in v3
                    for v in v2.values()
                )
            if name == "reply":
                return sum(np.prod(np.shape(v)) for v in self.replies[key])
            raise AssertionError(name)
        datasets = [self]
        if max_of_transposed:
            if self.transposed_dataset is None:
                self.transposed_dataset = self.mod(
                    np.transpose, keep=False, keep_key=True
                )
            datasets.append(self.transposed_dataset)
        return max(
            len(
                formatter.tokenizer.encode(
                    ds.get(key, formatter=formatter)[name], add_special_tokens=False
                )
            )
            for ds in datasets
        )

    def cut_to_len(self, formatter, name, max_len, from_end=False):
        temp_ds = self.change_keys(self.keys)
        new_keys, new_queries, new_replies = [], {}, {}
        for key in self.keys:
            reply = temp_ds.replies.get(key)
            while max_len < temp_ds.get_length(key, formatter=formatter, name=name):
                query = temp_ds.queries[key]
                if not key.split(".")[-1].startswith("ex"):
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split(".")
                assert key_split[-1].startswith("ex")
                key = ".".join(
                    key_split[:-1]
                    + [f"ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}"]
                )
                temp_ds.queries[key] = {
                    k: ((v[:-1] if from_end else v[1:]) if k == "train" else v)
                    for k, v in query.items()
                }
                if reply is not None:
                    temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None:
                new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)

    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys, new_queries, new_replies = [], {}, {}
        for key in self.keys:
            n = len(self.queries[key]["train"])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None:
                p = p[:keep_max]
            new_key = f"{key}.ex" + ("-" if (p.max() > 9) else "").join(
                map(str, p.tolist())
            )
            new_keys.append(new_key)
            new_queries[new_key] = {
                k: (np.array(v, dtype=object)[p].tolist() if k == "train" else v)
                for k, v in self.queries[key].items()
            }
            if key in self.replies:
                new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def augment(self, n=1, shfl_keys=False, seed=42):
        """Dihedral + colour augmentation: transpose, 3x rot90, n colour perms."""
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex()
        return d

    def get_submission(self, results=None):
        assert self.is_orig, "Must be run on the original dataset."
        submission = {
            k: [
                {f"attempt_{i + 1}": [[0]] for i in range(2)}
                for _ in range(len(self.queries[k]["test"]))
            ]
            for k in self.keys
        }
        if results is not None:
            self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        for k, v in results.items():
            base_id, base_nr = k.split("_")
            target_dict = submission[base_id][int(base_nr)]
            for i, g in enumerate(v[: len(target_dict)]):
                target_dict[f"attempt_{i + 1}"] = (
                    g.tolist() if hasattr(g, "tolist") else g
                )

    def validate_submission(self, submission):
        assert self.is_orig, "Must be run on the original dataset."
        score = 0
        for k, v in self.replies.items():
            for i, r in enumerate(v):
                for attempt in ["attempt_1", "attempt_2"]:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score
