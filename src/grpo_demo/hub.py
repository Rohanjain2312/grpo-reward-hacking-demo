"""Hugging Face Hub checkpointing.

The Hub is the source of truth for resuming: a run started in a brand-new Colab runtime
(or an HF Job container) looks for `latest.json` in its model repo and picks up from the
step recorded there. Checkpoints live in step-numbered folders so they never overwrite
each other.
"""

import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

LATEST = "latest.json"
METRICS = "logs/metrics.jsonl"
SAMPLES = "logs/samples.jsonl"


class HubCheckpointer:
    def __init__(self, repo_id: str, local_dir: str, enabled: bool = True):
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled and bool(repo_id)
        self.api = HfApi() if self.enabled else None
        if self.enabled:
            self.api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

    # ---------------------------------------------------------------- resume
    def find_latest(self):
        """Return the parsed `latest.json` dict, or None if this run has not started."""
        if not self.enabled:
            return None
        try:
            path = hf_hub_download(self.repo_id, LATEST, repo_type="model")
        except (EntryNotFoundError, RepositoryNotFoundError, OSError):
            return None
        with open(path) as f:
            return json.load(f)

    def download_checkpoint(self, ckpt_path: str) -> Path:
        local = snapshot_download(
            self.repo_id, repo_type="model", allow_patterns=[f"{ckpt_path}/*"]
        )
        return Path(local) / ckpt_path

    def download_logs(self) -> tuple[list, list]:
        """Return (metrics, samples) already logged in previous sessions."""
        metrics, samples = [], []
        for remote, sink in ((METRICS, metrics), (SAMPLES, samples)):
            try:
                p = hf_hub_download(self.repo_id, remote, repo_type="model")
            except (EntryNotFoundError, RepositoryNotFoundError, OSError):
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sink.append(json.loads(line))
        return metrics, samples

    # ------------------------------------------------------------------ save
    def push_logs(self, metrics_path: Path, samples_path: Path, step: int):
        if not self.enabled:
            return
        ops = []
        for local, remote in ((metrics_path, METRICS), (samples_path, SAMPLES)):
            if local.exists():
                ops.append((str(local), remote))
        for local, remote in ops:
            self.api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"logs @ step {step}",
            )

    def push_checkpoint(self, step: int, model, tokenizer, optimizer, state: dict,
                        metrics_path: Path, samples_path: Path, with_optimizer: bool = True):
        """Write a full checkpoint locally and mirror it to the Hub, then update latest.json."""
        name = f"checkpoints/step_{step:06d}"
        staging = self.local_dir / name
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        if with_optimizer:
            torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        with open(staging / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        if not self.enabled:
            return name

        self.api.upload_folder(
            folder_path=str(staging),
            path_in_repo=name,
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=f"checkpoint @ step {step}",
        )
        self.push_logs(metrics_path, samples_path, step)
        latest = self.local_dir / LATEST
        with open(latest, "w") as f:
            json.dump({"step": step, "checkpoint": name, "has_optimizer": with_optimizer}, f)
        self.api.upload_file(
            path_or_fileobj=str(latest),
            path_in_repo=LATEST,
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=f"latest -> step {step}",
        )
        # Local staging copies are large; keep only the newest two on disk.
        self._prune_local()
        return name

    def _prune_local(self, keep: int = 2):
        root = self.local_dir / "checkpoints"
        if not root.exists():
            return
        dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        for d in dirs[:-keep]:
            shutil.rmtree(d, ignore_errors=True)

    def push_final(self, model, tokenizer, card: str, metrics_path: Path, samples_path: Path):
        """Publish the final weights at the repo root so `from_pretrained(repo)` works."""
        if not self.enabled:
            return
        final = self.local_dir / "final"
        if final.exists():
            shutil.rmtree(final)
        final.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        with open(final / "README.md", "w") as f:
            f.write(card)
        self.api.upload_folder(
            folder_path=str(final),
            path_in_repo=".",
            repo_id=self.repo_id,
            repo_type="model",
            commit_message="final model + model card",
        )
        self.push_logs(metrics_path, samples_path, -1)


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def rewrite_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
