#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from typing import List

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.configs.value_train import ValueTrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    Pistar06Policy,
    compute_normalized_value_targets,
)
from lerobot.values.pistar06.processor_pistar06 import make_pistar06_pre_post_processors


def build_cfg_from_cli_like_args(cli_args: List[str]) -> ValueTrainPipelineConfig:
    """
    用和 lerobot-value-train 一样的方式，通过 CLI 参数构造 ValueTrainPipelineConfig。
    这里直接用 draccus 解析传入的 CLI 参数。
    """
    cfg = draccus.parse(
        config_class=ValueTrainPipelineConfig,
        config_path=None,
        args=cli_args,
    )
    cfg.validate()
    return cfg


def load_dataset(cfg: ValueTrainPipelineConfig):
    dataset = make_dataset(cfg)
    return dataset


def load_policy_and_preprocessor(
    checkpoint_dir: str,
    cfg: ValueTrainPipelineConfig,
    device: str = "cuda",
):
    # 从 checkpoint 加载 Pistar06Policy
    policy = Pistar06Policy.from_pretrained(checkpoint_dir)
    policy.to(device)
    policy.eval()

    # 构建和训练时一致的 preprocessor（使用数据集统计信息）
    preprocessor, _ = make_pistar06_pre_post_processors(
        config=policy.config,
        dataset_stats=cfg.dataset.stats if hasattr(cfg.dataset, "stats") else None,
    )
    return policy, preprocessor


def collect_episode_indices(dataset, max_episodes: int):
    """
    从 dataset.meta.episodes 中取出 episode_index，返回前 max_episodes 个。
    """
    episodes_ds = dataset.meta.episodes.with_format("numpy")
    episode_indices = episodes_ds["episode_index"]
    unique_eps = np.unique(episode_indices)
    if max_episodes is not None:
        unique_eps = unique_eps[:max_episodes]
    return unique_eps.astype(int)


def get_episode_frames(dataset, episode_index: int):
    """
    返回单个 episode 的所有 frame（按 frame_index 排序后），
    并通过 dataset 的 __getitem__ + default_collate 构造出与训练时 DataLoader 一致的 batch dict。
    """
    hf_ds = dataset.hf_dataset.with_format("numpy")
    episode_indices = np.asarray(hf_ds["episode_index"])
    mask = episode_indices == episode_index
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return None

    # 按 frame_index 排序
    frame_indices = hf_ds["frame_index"][idx]
    order = np.argsort(frame_indices)
    sorted_idx = idx[order]

    samples = [dataset[int(i)] for i in sorted_idx]
    batch = default_collate(samples)
    # 额外返回 frame_index，方便调试 / 保存
    frame_index = np.asarray(hf_ds["frame_index"])[sorted_idx]
    batch["_frame_index_np"] = frame_index
    return batch


def eval_episodes_and_plot(
    dataset,
    policy: Pistar06Policy,
    preprocessor,
    episode_indices: np.ndarray,
    cfg: ValueTrainPipelineConfig,
    device: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # 先基于 dataset.meta.episodes 构建 episode_info 与 task_max_length，
    # 后面每条轨迹都直接用 compute_normalized_value_targets 现场重算 target。
    episodes_ds = dataset.meta.episodes.with_format("numpy")
    episodes = episodes_ds[:]
    n_episodes = len(episodes_ds)

    episode_info: dict[int, EpisodeTargetInfo] = {}
    task_max_length: dict[int, int] = {}
    has_success_field = cfg.targets.success_field in episodes_ds.column_names

    for i in range(n_episodes):
        ep_idx = int(episodes["episode_index"][i])
        ep_length = int(episodes["length"][i])

        # 任务名可能是标量字符串、list[str] 或 np.ndarray[str]
        tasks = episodes["tasks"][i]
        if isinstance(tasks, (list, tuple, np.ndarray)):
            task_name = tasks[0]
        else:
            task_name = tasks
        if isinstance(task_name, np.ndarray):
            task_name = task_name.item()
        task_name = str(task_name)

        if task_name not in dataset.meta.tasks.index:
            raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.")
        task_index = int(dataset.meta.tasks.loc[task_name].task_index)

        if has_success_field:
            explicit_success = episodes[cfg.targets.success_field][i]
            # 在训练里会通过 resolve_episode_success_label 归一化成 EPISODE_SUCCESS / FAILURE，
            # 这里直接按 bool(success) 来处理即可，可视化不依赖细粒度标签。
            ep_success = bool(explicit_success)
        else:
            # 如果数据集中没有 success 标记，就全部当作失败或使用默认配置；
            # 这里只是为了画图，可视化不会影响训练逻辑。
            ep_success = cfg.targets.default_success == "success"

        episode_info[ep_idx] = EpisodeTargetInfo(
            episode_index=ep_idx,
            task_index=task_index,
            length=ep_length,
            success=ep_success,
        )
        task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

    for i, ep_idx in enumerate(episode_indices):
        ep_batch = get_episode_frames(dataset, int(ep_idx))
        if ep_batch is None:
            continue

        plt.figure(figsize=(10, 6))

        # 从 batch 中拿出 frame_index，按论文公式现场重算 target
        frame_index_np = np.asarray(ep_batch.get("_frame_index_np"), dtype=np.int64)
        ep_indices_np = np.full_like(frame_index_np, int(ep_idx), dtype=np.int64)
        true_targets_np = compute_normalized_value_targets(
            episode_indices=ep_indices_np,
            frame_indices=frame_index_np,
            episode_info=episode_info,
            task_max_lengths=task_max_length,
            c_fail_coef=cfg.targets.c_fail_coef,
            clip_min=policy.config.bin_min,
            clip_max=policy.config.bin_max,
        )

        # preprocessor 期望的输入是 dict[str, Any]
        proc_batch = preprocessor(ep_batch)

        # 用 value 模型预测每一帧的 value
        with torch.no_grad():
            values = policy.predict_value(proc_batch)  # shape [T]
        values_np = values.detach().cpu().numpy()

        # 简单检查 target 是否单调非减，便于发现问题
        diffs = np.diff(true_targets_np)
        if np.any(diffs < -1e-6):
            first_bad = int(np.where(diffs < -1e-6)[0][0])
            print(
                f"[WARN] episode {int(ep_idx)} target 非单调: "
                f"idx {first_bad}->{first_bad+1}, "
                f"value {true_targets_np[first_bad]:.4f}->{true_targets_np[first_bad+1]:.4f}"
            )

        # 保存当前 episode 的数据，便于离线排查
        npz_path = os.path.join(output_dir, f"pistar06_value_eval_ep{int(ep_idx)}.npz")
        np.savez(
            npz_path,
            episode_index=int(ep_idx),
            frame_index=frame_index_np,
            target=true_targets_np,
            pred=values_np,
        )
        print(f"Saved data to {npz_path}")

        x = np.arange(len(values_np))
        plt.plot(x, values_np, label="pred")
        plt.plot(x, true_targets_np, "--", label="target")

        plt.xlabel("frame idx")
        plt.ylabel("value (expected / target)")
        plt.title(f"Pistar06 value over time (episode {int(ep_idx)})")
        plt.legend()

        save_path = os.path.join(output_dir, f"pistar06_value_eval_ep{int(ep_idx)}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Pistar06 pretrained checkpoint directory (e.g. .../checkpoints/last/pretrained_model)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="要评估的 episode 数量（从数据集中前几个 episode 开始取）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 或 cpu",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/value_eval",
        help="保存图片的目录",
    )
    # 为了简单，这里再允许你把当时训练的 CLI 直接传进来
    parser.add_argument(
        "--train_cli",
        type=str,
        default="",
        help=(
            "训练时的 CLI 参数字符串（不含命令本身），"
            "例如: \"--dataset.root=... --dataset.repo_id=... --value.type=pistar06 ...\""
        ),
    )

    args = parser.parse_args()

    # 用你提供的 train_cli 构造 config；如果你懒得传，也可以在代码里直接写死。
    if args.train_cli:
        train_cli_args = args.train_cli.strip().split()
    else:
        # 这里直接写一份默认的（按你现在的训练参数）；
        # 如果以后你改参数了，可以修改这一段。
        train_cli_args = [
            f"--dataset.root=/mnt/data/dataset/lerobot/arx_x5_single_demonstrations_slipper",
            f"--dataset.repo_id=arx_x5_single_demonstrations_slipper",
            "--value.type=pistar06",
            "--value.dtype=bfloat16",
            "--value.push_to_hub=false",
            "--batch_size=16",
            "--steps=10000",
            "--save_freq=1000",
            "--save_checkpoint=true",
            "--output_dir=outputs/value_train/arx_slipper_run1",
            "--job_name=arx_slipper_v1",
            "--wandb.enable=true",
            "--wandb.disable_artifact=true",
        ]

    cfg = build_cfg_from_cli_like_args(train_cli_args)
    dataset = load_dataset(cfg)

    policy, preprocessor = load_policy_and_preprocessor(
        checkpoint_dir=args.checkpoint_dir,
        cfg=cfg,
        device=args.device,
    )

    episode_indices = collect_episode_indices(dataset, max_episodes=args.num_episodes)

    eval_episodes_and_plot(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        episode_indices=episode_indices,
        cfg=cfg,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()