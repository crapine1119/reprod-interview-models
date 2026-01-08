from typing import Optional, Any

from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class LitMultimodalDataModule(LightningDataModule):
    """
    Lightning DataModule:
    - train/val/test/predict Dataset을 받아서
    - 모델 forward(**batch)에 바로 넣을 수 있는 dict 배치를 생성합니다.
    """

    def __init__(
        self,
        *,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,  # Not supported for mps
        persistent_workers: bool = True,
        drop_last: bool = True,
        shuffle: bool = True,
        # collator 제어
        collator: Optional[Any] = None,
        processors: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.collator = collator
        self.processors = processors

    def setup(self, stage: Optional[str] = None) -> None:
        # Dataset 자체가 이미 외부에서 구성되므로 setup에서는 특별히 할 일이 없습니다.
        # (필요하면 stage에 따라 lazy load를 구현하세요.)
        return

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=self.drop_last,
            collate_fn=self.collator if self.collator is not None else self.processors.collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False,
            collate_fn=self.collator if self.collator is not None else self.processors.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False,
            collate_fn=self.collator if self.collator is not None else self.processors.collate_fn,
        )

    def predict_dataloader(self) -> Optional[DataLoader]:
        if self.predict_dataset is None:
            return None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False,
            collate_fn=self.collator if self.collator is not None else self.processors.collate_fn,
        )
