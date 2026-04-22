"""Tests for training loop functionality."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from train import (
    Trainer,
    build_model_from_contract,
    create_synthetic_dataloader,
    create_synthetic_regression_dataloader,
    create_synthetic_text_dataloader,
    create_synthetic_segmentation_dataloader,
    build_optimizer,
    build_scheduler,
    build_criterion,
)


def _make_model_contract(backbone="resnet18", num_classes=5):
    return {
        "task_type": "classification",
        "data_type": "image",
        "input_spec": {"shape": [3, 224, 224], "dtype": "float32"},
        "model_spec": {
            "family": "cnn",
            "architecture": "resnet",
            "backbone": backbone,
            "pretrained": False,
            "in_channels": 3,
            "feature_dim": 512,
        },
        "head_spec": {"type": "linear_cls", "num_classes": num_classes, "pooling": "avg", "dropout": 0.0},
        "forward_spec": {"output_shape": ["batch", num_classes]},
    }


class TestTrainer:
    def test_train_one_epoch(self):
        mc = _make_model_contract("resnet18", 3)
        model = build_model_from_contract(mc)
        loader = create_synthetic_dataloader(
            {"shape": [3, 224, 224], "dtype": "float32"}, 3, num_samples=50, batch_size=8
        )
        trainer = Trainer(model, "cpu", {"task_type": "classification", "optimizer": {"lr": 0.001}})
        history = trainer.train(loader, epochs=1)
        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_loss_decreases(self):
        mc = _make_model_contract("resnet18", 3)
        model = build_model_from_contract(mc)
        loader = create_synthetic_dataloader(
            {"shape": [3, 224, 224], "dtype": "float32"}, 3, num_samples=100, batch_size=8
        )
        config = {
            "task_type": "classification",
            "optimizer": {"name": "adam", "lr": 0.01},
            "scheduler": {"name": "none"},
        }
        trainer = Trainer(model, "cpu", config)
        history = trainer.train(loader, epochs=3)
        losses = history["train_loss"]
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_checkpoint_save_load(self):
        mc = _make_model_contract("resnet18", 3)
        model = build_model_from_contract(mc)
        loader = create_synthetic_dataloader(
            {"shape": [3, 224, 224], "dtype": "float32"}, 3, num_samples=50, batch_size=8
        )
        config = {"task_type": "classification", "optimizer": {"lr": 0.001}}
        trainer = Trainer(model, "cpu", config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train(loader, epochs=1, checkpoint_dir=tmpdir)
            assert os.path.exists(f"{tmpdir}/last_model.pt")
            assert os.path.exists(f"{tmpdir}/best_model.pt")

            model2 = build_model_from_contract(mc)
            trainer2 = Trainer(model2, "cpu", config)
            ckpt = trainer2.load_checkpoint(f"{tmpdir}/best_model.pt")
            assert "model_state_dict" in ckpt
            assert "optimizer_state_dict" in ckpt

    def test_resume_training(self):
        mc = _make_model_contract("resnet18", 3)
        loader = create_synthetic_dataloader(
            {"shape": [3, 224, 224], "dtype": "float32"}, 3, num_samples=50, batch_size=8
        )
        config = {"task_type": "classification", "optimizer": {"lr": 0.001}}

        model1 = build_model_from_contract(mc)
        trainer1 = Trainer(model1, "cpu", config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer1.train(loader, epochs=2, checkpoint_dir=tmpdir)

            model2 = build_model_from_contract(mc)
            trainer2 = Trainer(model2, "cpu", config)
            trainer2.load_checkpoint(f"{tmpdir}/last_model.pt")
            assert trainer2.current_epoch == 1  # 0-indexed

            history2 = trainer2.train(loader, epochs=1, checkpoint_dir=tmpdir)
            assert len(history2["train_loss"]) == 3  # 2 original + 1 new

    def test_efficientnet(self):
        mc = _make_model_contract("efficientnet_b0", 3)
        mc["model_spec"]["architecture"] = "efficientnet"
        mc["model_spec"]["feature_dim"] = 1280
        model = build_model_from_contract(mc)
        loader = create_synthetic_dataloader(
            {"shape": [3, 224, 224], "dtype": "float32"}, 3, num_samples=30, batch_size=8
        )
        trainer = Trainer(model, "cpu", {"task_type": "classification", "optimizer": {"lr": 0.001}})
        history = trainer.train(loader, epochs=1)
        assert len(history["train_loss"]) == 1

    def test_resnet50(self):
        mc = _make_model_contract("resnet50", 3)
        mc["model_spec"]["feature_dim"] = 2048
        model = build_model_from_contract(mc)
        loader = create_synthetic_dataloader(
            {"shape": [3, 224, 224], "dtype": "float32"}, 3, num_samples=30, batch_size=8
        )
        trainer = Trainer(model, "cpu", {"task_type": "classification", "optimizer": {"lr": 0.001}})
        history = trainer.train(loader, epochs=1)
        assert len(history["train_loss"]) == 1


class TestOptimizer:
    def test_adam(self):
        model = torch.nn.Linear(10, 2)
        opt = build_optimizer(model, {"name": "adam", "lr": 0.001})
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd(self):
        model = torch.nn.Linear(10, 2)
        opt = build_optimizer(model, {"name": "sgd", "lr": 0.01, "momentum": 0.9})
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self):
        model = torch.nn.Linear(10, 2)
        with pytest.raises(ValueError):
            build_optimizer(model, {"name": "rmsprop"})


class TestScheduler:
    def test_step(self):
        model = torch.nn.Linear(10, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = build_scheduler(opt, {"name": "step", "step_size": 5, "gamma": 0.1})
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_cosine(self):
        model = torch.nn.Linear(10, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = build_scheduler(opt, {"name": "cosine", "t_max": 10})
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_none(self):
        model = torch.nn.Linear(10, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = build_scheduler(opt, {"name": "none"})
        assert sched is None


class TestCriterion:
    def test_classification(self):
        loss = build_criterion("classification")
        assert isinstance(loss, torch.nn.CrossEntropyLoss)

    def test_regression(self):
        loss = build_criterion("regression")
        assert isinstance(loss, torch.nn.MSELoss)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_criterion("unknown")


class TestSyntheticDataLoader:
    def test_output_shapes(self):
        loader = create_synthetic_dataloader(
            {"shape": [3, 224, 224], "dtype": "float32"}, 10, num_samples=32, batch_size=8
        )
        inputs, labels = next(iter(loader))
        assert inputs.shape == (8, 3, 224, 224)
        assert labels.shape == (8,)
        assert inputs.dtype == torch.float32
        assert labels.dtype == torch.int64


class TestTabularTraining:
    def _make_tabular_cls_contract(self, num_features=20, num_classes=5):
        return {
            "task_type": "classification",
            "data_type": "tabular",
            "input_spec": {"shape": [num_features], "dtype": "float32"},
            "model_spec": {
                "family": "mlp", "architecture": "mlp", "backbone": "mlp",
                "pretrained": False, "in_features": num_features, "feature_dim": None,
            },
            "head_spec": {"type": "linear_cls", "num_classes": num_classes, "dropout": 0.0},
            "forward_spec": {"output_shape": ["batch", num_classes]},
        }

    def _make_tabular_reg_contract(self, num_features=15, output_dim=2):
        return {
            "task_type": "regression",
            "data_type": "tabular",
            "input_spec": {"shape": [num_features], "dtype": "float32"},
            "model_spec": {
                "family": "mlp", "architecture": "mlp", "backbone": "mlp",
                "pretrained": False, "in_features": num_features, "feature_dim": None,
            },
            "head_spec": {"type": "linear_regression", "output_dim": output_dim, "dropout": 0.0},
            "forward_spec": {"output_shape": ["batch", output_dim]},
        }

    def test_build_mlp_classification(self):
        mc = self._make_tabular_cls_contract()
        model = build_model_from_contract(mc)
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4, 5)

    def test_train_mlp_classification(self):
        mc = self._make_tabular_cls_contract()
        model = build_model_from_contract(mc)
        loader = create_synthetic_dataloader(
            {"shape": [20], "dtype": "float32"}, 5, num_samples=100, batch_size=8
        )
        trainer = Trainer(model, "cpu", {"task_type": "classification", "optimizer": {"lr": 0.01}})
        history = trainer.train(loader, epochs=3)
        assert len(history["train_loss"]) == 3
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_build_mlp_regression(self):
        mc = self._make_tabular_reg_contract()
        model = build_model_from_contract(mc)
        x = torch.randn(4, 15)
        out = model(x)
        assert out.shape == (4, 2)

    def test_train_mlp_regression(self):
        mc = self._make_tabular_reg_contract()
        model = build_model_from_contract(mc)
        loader = create_synthetic_regression_dataloader(
            {"shape": [15], "dtype": "float32"}, output_dim=2, num_samples=100, batch_size=8
        )
        trainer = Trainer(model, "cpu", {"task_type": "regression", "optimizer": {"lr": 0.01}})
        history = trainer.train(loader, epochs=3)
        assert len(history["train_loss"]) == 3
        assert history["train_loss"][-1] < history["train_loss"][0]
        # Regression should not have accuracy
        assert history["train_acc"][0] is None


class TestTextTraining:
    def _make_text_contract(self, num_classes=3, backbone="bert-tiny"):
        return {
            "task_type": "classification",
            "data_type": "text",
            "input_spec": {"shape": [64], "dtype": "int64", "max_seq_length": 64},
            "model_spec": {
                "family": "transformer_encoder", "architecture": "bert", "backbone": backbone,
                "pretrained": False, "hidden_size": 128,
            },
            "head_spec": {"type": "pooled_linear_cls", "num_classes": num_classes, "pooling": "cls_token", "dropout": 0.1},
            "forward_spec": {"output_shape": ["batch", num_classes]},
        }

    def test_build_transformer(self):
        mc = self._make_text_contract()
        model = build_model_from_contract(mc)
        batch = {"input_ids": torch.randint(1, 1000, (2, 64)), "attention_mask": torch.ones(2, 64, dtype=torch.long)}
        out = model(**batch)
        assert out.shape == (2, 3)

    def test_build_distilbert(self):
        mc = self._make_text_contract(backbone="distilbert-base-uncased")
        model = build_model_from_contract(mc)
        batch = {"input_ids": torch.randint(1, 1000, (2, 64)), "attention_mask": torch.ones(2, 64, dtype=torch.long)}
        out = model(**batch)
        assert out.shape == (2, 3)

    def test_train_text_classification(self):
        mc = self._make_text_contract()
        model = build_model_from_contract(mc)
        loader = create_synthetic_text_dataloader(
            {"max_seq_length": 64, "dtype": "int64"}, 3, num_samples=40, batch_size=4, seq_length=64
        )
        trainer = Trainer(model, "cpu", {"task_type": "classification", "optimizer": {"lr": 0.001}})
        history = trainer.train(loader, epochs=2)
        assert len(history["train_loss"]) == 2
        assert history["train_loss"][-1] < history["train_loss"][0]


class TestSegmentationTraining:
    def _make_deeplabv3_contract(self, num_classes=5):
        return {
            "task_type": "segmentation",
            "data_type": "image",
            "input_spec": {"shape": [3, 128, 128], "dtype": "float32"},
            "model_spec": {
                "family": "cnn_encoder_decoder", "architecture": "deeplabv3",
                "backbone": "deeplabv3_resnet50", "pretrained": False, "in_channels": 3,
            },
            "head_spec": {"type": "segmentation_head", "num_classes": num_classes},
            "forward_spec": {"output_shape": ["batch", num_classes, "H", "W"]},
        }

    def _make_unet_contract(self, num_classes=5):
        return {
            "task_type": "segmentation",
            "data_type": "image",
            "input_spec": {"shape": [3, 128, 128], "dtype": "float32"},
            "model_spec": {
                "family": "cnn_encoder_decoder", "architecture": "unet",
                "backbone": "unet", "pretrained": False, "in_channels": 3,
            },
            "head_spec": {"type": "segmentation_head", "num_classes": num_classes},
            "forward_spec": {"output_shape": ["batch", num_classes, "H", "W"]},
        }

    def test_build_deeplabv3(self):
        mc = self._make_deeplabv3_contract()
        model = build_model_from_contract(mc)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        # DeepLabV3 returns dict, unwrap
        if isinstance(out, dict):
            out = out["out"]
        assert out.shape[0] == 2
        assert out.shape[1] == 5

    def test_build_unet(self):
        mc = self._make_unet_contract()
        model = build_model_from_contract(mc)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 5

    def test_train_deeplabv3(self):
        mc = self._make_deeplabv3_contract()
        model = build_model_from_contract(mc)
        loader = create_synthetic_segmentation_dataloader(
            {"shape": [3, 128, 128], "dtype": "float32"}, 5, num_samples=20, batch_size=4
        )
        trainer = Trainer(model, "cpu", {"task_type": "segmentation", "optimizer": {"lr": 0.001}})
        history = trainer.train(loader, epochs=2)
        assert len(history["train_loss"]) == 2
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_train_unet(self):
        mc = self._make_unet_contract()
        model = build_model_from_contract(mc)
        loader = create_synthetic_segmentation_dataloader(
            {"shape": [3, 128, 128], "dtype": "float32"}, 5, num_samples=20, batch_size=4
        )
        trainer = Trainer(model, "cpu", {"task_type": "segmentation", "optimizer": {"lr": 0.001}})
        history = trainer.train(loader, epochs=2)
        assert len(history["train_loss"]) == 2
        assert history["train_loss"][-1] < history["train_loss"][0]
