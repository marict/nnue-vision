import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import train


class DummyWandbLogger:
    """Minimal stub for Lightning's WandbLogger used in tests."""

    def __init__(self, *args, **kwargs):
        # Provide just the attributes accessed in the training script
        self.experiment = SimpleNamespace(
            config={},
            url="http://wandb.local/run",
        )
        self.save_dir = "."
        self.version = "test"

    # Minimal API surface used by Lightning
    def log_metrics(self, *args, **kwargs):
        pass

    def log_hyperparams(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        pass

    def log_graph(self, *args, **kwargs):
        pass

    def save(self):
        pass

    # Gracefully handle any other method/attribute requests
    def __getattr__(self, item):
        def _dummy(*args, **kwargs):
            return None

        return _dummy


@pytest.mark.timeout(60)
def test_etinynet_cli_tiny(tmp_path, monkeypatch):
    """Run train_etinynet.main with a tiny synthetic dataset to ensure CLI works."""

    # ---------------------------------------------------------------------
    # 1. Create a minimal temporary config file
    # ---------------------------------------------------------------------
    cfg_path = tmp_path / "tiny_cfg.py"
    log_dir = tmp_path / "logs"
    cfg_path.write_text(
        """
name = 'tiny_cli_test'
batch_size = 2
max_epochs = 1
learning_rate = 0.1
etinynet_variant = '0.75'
use_asq = False
log_dir = r'{}'
""".format(
            log_dir
        )
    )

    # ---------------------------------------------------------------------
    # 2. Patch external dependencies (WANDB, data loaders)
    # ---------------------------------------------------------------------
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_MODE", "disabled")  # prevent network use

    # Replace WandbLogger with dummy implementation in training_framework
    import training_framework

    monkeypatch.setattr(training_framework, "WandbLogger", DummyWandbLogger)

    # Stub out wandb module used inside training_framework
    monkeypatch.setattr(
        training_framework,
        "wandb",
        SimpleNamespace(
            init=lambda *_, **__: SimpleNamespace(url="local", id="id"),
            log=lambda *_1, **_2: None,
            finish=lambda *_a, **_kw: None,
            Artifact=lambda *_, **__: SimpleNamespace(add_file=lambda x: None),
            log_artifact=lambda x: None,
        ),
    )

    # Patch replay_early_logs_to_wandb to no-op (signature mismatch in script)
    monkeypatch.setattr(
        training_framework, "replay_early_logs_to_wandb", lambda *_a, **_kw: None
    )

    # Provide tiny synthetic data loaders to avoid dataset download
    # Stub Trainer.test to avoid need for metrics
    from pytorch_lightning import Trainer as _PLTrainer

    monkeypatch.setattr(
        _PLTrainer, "test", lambda self, *a, **kw: [{"test_loss": 0.0, "test_acc": 0.0}]
    )

    with patch("etinynet_adapter.create_data_loaders") as mock_loaders:
        dummy_ds = torch.utils.data.TensorDataset(
            torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
        )
        dummy_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=2)
        mock_loaders.return_value = (dummy_loader, dummy_loader, dummy_loader)

        # -----------------------------------------------------------------
        # 3. Invoke unified train.py with etinynet model type
        # -----------------------------------------------------------------
        argv = [
            "train.py",
            "etinynet",
            "--config",
            str(cfg_path),
            "--variant",
            "0.75",
            "--max_epochs",
            "1",
            "--batch_size",
            "2",
        ]
        with patch.object(sys, "argv", argv):
            # The function prints but should not raise exceptions (sys.exit(0) is expected)
            with pytest.raises(SystemExit) as exc_info:
                train.main()
            assert (
                exc_info.value.code == 0
            ), f"Expected successful exit, got: {exc_info.value.code}"
