"""Candidate aggregation + selection.

This stage is model-agnostic: it consumes the per-candidate dicts produced by
arc_solver.py — each carries `beam_score` (cumulative NLL from the turbo-DFS
decoder) and `score_aug` (a list of NLLs from re-scoring the candidate as the
answer under fresh augmentations) — groups identical grids, and ranks them.

Two selection algorithms:
  * score_full_probmul_3 — combines inference score and augmentation score.
  * score_kgmon          — KGMoN variant: vote count minus mean augmentation NLL.
"""

from __future__ import annotations

import bz2
import os
import pickle

import numpy as np


def hashable(guess):
    return tuple(map(tuple, guess))


def score_sum(guesses, getter):
    """Group candidate dicts by identical solution grid, score each group."""
    scores = {}
    for g in guesses.values():
        h = hashable(g["solution"])
        x = scores.setdefault(h, [[], g["solution"]])
        x[0].append(g)
    ranked = [(getter(group), solution) for group, solution in scores.values()]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [solution for _, solution in ranked]


def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline - g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score


def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)


def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]


class ArcDecoder:
    """Loads per-subkey candidate dumps and runs the selection algorithms."""

    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        for key in os.listdir(store):
            with bz2.BZ2File(os.path.join(store, key)) as f:
                outputs = pickle.load(f)
            base_key = key.split(".")[0]
            self.decoded_results.setdefault(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def add_results(self, base_key, candidates):
        """Register candidates in memory (for single-process runs that skip
        the bz2 round-trip)."""
        self.decoded_results.setdefault(base_key, {})
        offset = len(self.decoded_results[base_key])
        for i, sample in enumerate(candidates):
            self.decoded_results[base_key][f"{base_key}.mem.out{offset + i}"] = sample

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm(dict(v))
                for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        """Print per-algorithm puzzle accuracy when ground-truth replies exist."""
        print("*** Benchmark selection algorithms...")
        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0
        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():
            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(
                num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)
            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():
                solution = sample["solution"]
                beam_score = sample["beam_score"]
                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"
                if corr_str == "ALL_CORRECT":
                    aug_mean = np.mean(sample["score_aug"])
                    out_len = f"{solution.shape[0]}x{solution.shape[1]}"
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} "
                          f"{out_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        if correct_beam_scores:
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)
        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {
                k for k, v in selected.items()
                if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])
            }
            score = sum(1 / num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")
