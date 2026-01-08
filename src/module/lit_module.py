import os
from typing import Any, Dict, Optional, Callable

import torch
from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel


class LitMultimodalModule(LightningModule):
    """
    학습은 Lightning, 서빙 아티팩트는 Transformers save_pretrained로 생성
    """

    def __init__(
        self,
        model: PreTrainedModel,  # PreTrainedModel
        processor: Optional[Any] = None,  # ProcessorMixin
        optimizer: Optional[Callable[[], Optimizer]] = None,
        scheduler: Optional[Callable[[], LRScheduler]] = None,
        hf_export_dirname: str = "hf_export",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["hf_model", "processors"])

        self.hf_model = model
        self.processor = processor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hf_export_dirname = hf_export_dirname

    def forward(self, **inputs: torch.Tensor):
        return self.hf_model(**inputs, return_dict=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(**batch)
        if outputs.loss is None:
            raise ValueError("Model did not return loss. Ensure 'labels' is in batch.")
        self.log("train/loss", outputs.loss, prog_bar=True, on_step=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.forward(**batch)
        if outputs.loss is None:
            raise ValueError("Model did not return loss. Ensure 'labels' is in batch.")
        self.log("val/loss", outputs.loss, prog_bar=True, on_step=False, on_epoch=True)

        if outputs.logits is not None and "labels" in batch:
            preds = outputs.logits.argmax(dim=-1)
            acc = (preds == batch["labels"]).float().mean()
            self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        batch = dict(batch)
        batch.pop("labels", None)
        outputs = self.forward(**batch)
        return outputs.logits

    def on_train_end(self) -> None:
        if not self.trainer.is_global_zero:
            return

        root = getattr(self.trainer, "log_dir", None) or self.trainer.default_root_dir
        export_dir = os.path.join(root, self.hf_export_dirname)
        os.makedirs(export_dir, exist_ok=True)

        self.hf_model.save_pretrained(export_dir)
        if self.processor is not None:
            self.processor.save_pretrained(export_dir)

        print(f"[LitMultimodalHFModule] Exported HF artifacts to: {export_dir}")

    def configure_optimizers(self):
        if self.optimizer:
            optimizer = self.optimizer()
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(self.optimizer_cfg.get("lr", 1e-4)),
                weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.01)),
            )

        if not self.scheduler:
            return optimizer
        else:
            scheduler = self.scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
