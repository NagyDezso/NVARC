"""
Load TRM + ACT head, optionally patch inner for ``z_H`` seeding, train then test (same
control flow as ``TRM/eval-arc-k-10.py`` ``launch()``).

Requires ``TRM`` on ``sys.path`` (see ``run_local`` / ``python -m hybrid_arc.run_local``).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

from .arc_tokenize import batch_y_prior_tokens
from .trm_z_h_seed import SeedMode, patch_act_model_inner


def _load_eval_module(trm_root: Path):
    path = trm_root / "eval-arc-k-10.py"
    spec = importlib.util.spec_from_file_location("eval_arc_k10", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_arc_k10"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_identifiers(data_dir: Path) -> List[str]:
    path = data_dir / "identifiers.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compose_pretrain_config(trm_root: Path, config_name: str, overrides: List[str]) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(trm_root / "config")):
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides)
    d = OmegaConf.to_container(cfg, resolve=True)
    eak = _load_eval_module(trm_root)
    pc = eak.PretrainConfig
    mv = getattr(pc, "model_validate", None)
    if callable(mv):
        return mv(d)  # type: ignore[no-any-return]
    return pc(**d)  # type: ignore[arg-type]


def evaluate_with_z_h_seed(
    *,
    trm_root: Path,
    config: Any,
    train_state: Any,
    eval_loader: Any,
    eval_metadata: Any,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[Any],
    identifier_list: List[str],
    prior_batch_fn: Optional[Callable[[Dict[str, torch.Tensor], List[str]], List[Optional[Any]]]],
    seed_first_step_only: bool,
) -> Optional[Dict[str, Any]]:
    """
    Same as ``evaluate`` in ``eval-arc-k-10.py``, but optional ``y_prior_tokens`` on the
    first ACT iteration per batch when ``prior_batch_fn`` returns grids.
    """
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        save_preds: Dict[str, List[torch.Tensor]] = {}
        metric_keys: List[str] = []
        metric_values: Optional[torch.Tensor] = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            batch = {k: v.cuda() for k, v in batch.items()}
            id_tensor = batch["puzzle_identifiers"]
            row_names = [identifier_list[int(i)] for i in id_tensor.cpu().tolist()]

            if prior_batch_fn is not None:
                y_list = prior_batch_fn(batch, row_names)
                yt = batch_y_prior_tokens(
                    batch["inputs"],
                    y_list,
                    do_translation=False,
                    device=batch["inputs"].device,
                )
                if yt is not None:
                    batch["y_prior_tokens"] = yt

            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore[operator]

            inference_steps = 0
            while True:
                if seed_first_step_only and inference_steps > 0:
                    batch.pop("y_prior_tokens", None)
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            del metrics

        save_preds_stacked = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}
        del save_preds

        if config.checkpoint_path is not None and len(save_preds_stacked):
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds_stacked,
                os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"),
            )
        del save_preds_stacked

        if metric_values is not None:
            if world_size > 1:
                import torch.distributed as dist  # noqa: WPS433

                if dist.is_initialized():
                    dist.reduce(metric_values, dst=0)
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics


def run_hybrid_eval(
    *,
    trm_root: Path,
    config_overrides: List[str],
    load_checkpoint: str,
    gamma: float,
    seed_mode: SeedMode,
    use_z_h_seed: bool,
    prior_batch_fn: Optional[Callable[[Dict[str, torch.Tensor], List[str]], List[Optional[Any]]]],
    seed_first_step_only: bool = True,
    data_dir_for_identifiers: Optional[Path] = None,
    smoke_train_batches: Optional[int] = None,
    smoke_skip_eval: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Same outer loop as ``eval-arc-k-10.py`` ``launch()`` (train → eval, EMA, checkpoints),
    with test-time ``evaluate_with_z_h_seed`` when ``use_z_h_seed`` else stock ``evaluate``.

    ``smoke_train_batches``: if set, stop the inner training loop after this many optimizer
    steps (one epoch can otherwise be thousands of batches).

    ``smoke_skip_eval``: if True, still run the EMA / ``train_state_eval`` setup when
    applicable, but skip ``evaluate`` / checkpoint I/O (fast sanity check).
    """
    trm_root = trm_root.resolve()
    if str(trm_root) not in sys.path:
        sys.path.insert(0, str(trm_root))

    eak = _load_eval_module(trm_root)
    config = compose_pretrain_config(trm_root, "cfg_pretrain", config_overrides)
    config.load_checkpoint = load_checkpoint

    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = eak.create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    try:
        eval_loader, eval_metadata = eak.create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
    except Exception:
        print("NO EVAL DATA FOUND")
        eval_loader = None
        eval_metadata = None

    try:
        evaluators = eak.create_evaluators(config, eval_metadata)
    except Exception:
        print("No evaluator found")
        evaluators = []

    train_state = eak.init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    if use_z_h_seed:
        patch_act_model_inner(train_state.model, gamma=float(gamma), seed_mode=seed_mode)

    ident_path = data_dir_for_identifiers
    if ident_path is None and config.data_paths_test:
        ident_path = Path(config.data_paths_test[0])
    elif ident_path is None:
        ident_path = Path(config.data_paths[0])
    identifier_list = load_identifiers(ident_path)

    prior_in_eval = prior_batch_fn if use_z_h_seed else None

    torch.random.manual_seed(config.seed + RANK)

    progress_bar = None
    ema_helper = None
    if RANK == 0:
        _cap = None if smoke_train_batches is None else max(1, int(smoke_train_batches))
        pb_total = _cap * total_iters if _cap is not None else train_state.total_steps
        progress_bar = tqdm.tqdm(total=pb_total)
        print({"num_params": sum(x.numel() for x in train_state.model.parameters())})
        eak.save_code_and_config(config)
    if config.ema:
        print("Setup EMA")
        ema_helper = eak.EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    last_metrics: Optional[Dict[str, Any]] = None

    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        train_cap = None if smoke_train_batches is None else max(1, int(smoke_train_batches))
        train_batches_this_iter = 0
        for _set_name, batch, global_batch_size in train_loader:
            metrics = eak.train_batch(
                config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE
            )
            train_batches_this_iter += 1
            if RANK == 0 and metrics is not None and progress_bar is not None:
                progress_bar.update(train_state.step - progress_bar.n)
            if config.ema and ema_helper is not None:
                ema_helper.update(train_state.model)
            if train_cap is not None and train_batches_this_iter >= train_cap:
                if RANK == 0:
                    print(f"SMOKE: stopping train loop after {train_cap} batch(es)")
                break

        if _iter_id >= config.min_eval_interval:
            if RANK == 0:
                print("EVALUATE")
            if eval_loader is None or eval_metadata is None:
                if RANK == 0:
                    print("EVALUATE skipped (no eval loader)")
                if smoke_skip_eval and config.ema and ema_helper is not None:
                    print("SMOKE: building EMA eval state only (no test split)")
                    train_state_eval = eak.TrainState(
                        model=ema_helper.ema_copy(train_state.model),
                        optimizers=train_state.optimizers,
                        optimizer_lrs=train_state.optimizer_lrs,
                        carry=None,
                        step=train_state.step,
                        total_steps=train_state.total_steps,
                    )
                    train_state_eval.model.eval()
                    del train_state_eval
                    print("SMOKE OK (EMA eval-state construction)")
                continue

            if config.ema and ema_helper is not None:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()

            if smoke_skip_eval:
                if RANK == 0:
                    print("SMOKE: skipping evaluate() and checkpoint save")
                if config.ema and ema_helper is not None:
                    del train_state_eval
                print("SMOKE OK")
                continue

            if use_z_h_seed:
                last_metrics = evaluate_with_z_h_seed(
                    trm_root=trm_root,
                    config=config,
                    train_state=train_state_eval,
                    eval_loader=eval_loader,
                    eval_metadata=eval_metadata,
                    evaluators=evaluators,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP,
                    identifier_list=identifier_list,
                    prior_batch_fn=prior_in_eval,
                    seed_first_step_only=seed_first_step_only,
                )
            else:
                last_metrics = eak.evaluate(
                    config,
                    train_state_eval,
                    eval_loader,
                    eval_metadata,
                    evaluators,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP,
                )

            if RANK == 0 and last_metrics is not None:
                print(last_metrics, train_state.step)

            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                eak.save_train_state(config, train_state_eval)

            if config.ema and ema_helper is not None:
                del train_state_eval

    return last_metrics
