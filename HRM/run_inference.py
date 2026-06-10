"""Faithful ARChitects-style inference launcher for HRM-Text-1B.

Ports starter.py from the NVARC solution (ARC-AGI1/002_ivan_arc1.ipynb): fills
a queue with puzzle keys, spawns one worker per GPU, then runs the selection
stage over the dumped candidates to produce a Kaggle submission.json.

Single-GPU and CPU runs are supported (nprocs falls back to 1).

Usage:
    uv run --project HRM python HRM/run_inference.py \\
        --base sapientinc/HRM-Text-1B \\
        --checkpoint checkpoints/hrm-arc \\
        --tasks external/ARC-AGI-2/data/evaluation \\
        --out submission.json \\
        --time-budget-hours 11

If a solutions file is given via --solutions, the selection algorithms are
benchmarked against ground truth after the run.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def _run_worker(rank, queue, end_time, kwargs):
    if torch.cuda.device_count() > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    from arc_solver import worker
    worker(rank, queue, end_time, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="SFT checkpoint (LoRA adapter dir or full weights)")
    ap.add_argument("--tasks", required=True,
                    help="ARC tasks: a directory of per-task JSONs or a single "
                         "challenges JSON")
    ap.add_argument("--solutions", default=None,
                    help="optional ground-truth solutions JSON for benchmarking")
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--store", default="inference_outputs",
                    help="directory for per-subkey candidate dumps")
    ap.add_argument("--time-budget-hours", type=float, default=11.5)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="0 = one worker per visible GPU (min 1)")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--decode-batch", type=int, default=4,
                    help="augmentations decoded together; lower it if the "
                         "recurrent KV cache OOMs")
    ap.add_argument("--lora-r", type=int, default=256)
    ap.add_argument("--ttt-lr", type=float, default=5e-5)
    ap.add_argument("--ttt-aug", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="solve only the first N puzzles (debugging)")
    ap.add_argument("--keys", default=None,
                    help="comma-separated puzzle keys to solve (debugging); "
                         "overrides --limit")
    args = ap.parse_args()

    # Fail fast on bad paths: the run can take hours, so validate inputs
    # before queuing any work rather than crashing in the selection stage.
    if not os.path.exists(args.tasks):
        ap.error(f"--tasks path does not exist: {args.tasks}")
    if args.solutions is not None and not os.path.isfile(args.solutions):
        ap.error(f"--solutions file does not exist: {args.solutions}")

    end_time = time.time() + args.time_budget_hours * 3600 - 600

    # Collect puzzle keys.
    if os.path.isdir(args.tasks):
        keys = sorted(os.path.splitext(f)[0]
                      for f in os.listdir(args.tasks) if f.endswith(".json"))
    else:
        with open(args.tasks) as f:
            keys = sorted(json.load(f).keys())
    if args.keys:
        wanted = [k.strip() for k in args.keys.split(",") if k.strip()]
        missing = [k for k in wanted if k not in keys]
        if missing:
            ap.error(f"--keys not present in --tasks: {missing}")
        keys = wanted
    elif args.limit:
        keys = keys[:args.limit]
    print(f"queued {len(keys)} puzzles")

    n_gpus = torch.cuda.device_count()
    n_workers = args.num_workers or max(1, n_gpus)

    worker_kwargs = dict(
        base=args.base, checkpoint=args.checkpoint, tasks_path=args.tasks,
        out_dir=args.store, max_seq_length=args.max_seq_length,
        decode_batch=args.decode_batch, lora_r=args.lora_r,
        ttt_lr=args.ttt_lr, ttt_aug=args.ttt_aug,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if n_workers == 1:
        import queue as _queue
        q = _queue.Queue()
        for k in keys:
            q.put(k)
        q.put(None)
        _run_worker(0, q, end_time, worker_kwargs)
    else:
        import torch.multiprocessing as mp
        manager = mp.Manager()
        q = manager.Queue()
        for k in keys:
            q.put(k)
        for _ in range(n_workers):
            q.put(None)
        mp.spawn(_run_worker, args=(q, end_time, worker_kwargs), nprocs=n_workers)

    # Selection stage.
    from arc_loader import ArcDataset
    from arc_decoder import ArcDecoder, score_full_probmul_3

    data = (ArcDataset.from_dir(args.tasks) if os.path.isdir(args.tasks)
            else ArcDataset.from_file(args.tasks))
    if args.solutions:
        data = data.load_replies(args.solutions)

    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results(args.store)
    submission = data.get_submission(decoder.run_selection_algo(score_full_probmul_3))

    with open(args.out, "w") as f:
        json.dump(submission, f)
    print(f"wrote {args.out}")

    if args.solutions:
        decoder.benchmark_selection_algos()
        print("*** submission score:", data.validate_submission(submission))


if __name__ == "__main__":
    main()
