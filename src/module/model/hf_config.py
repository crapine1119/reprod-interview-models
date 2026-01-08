from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class MultimodalPreTrainedConfig(PretrainedConfig):
    """
    Transformers 저장/로딩 규약에 맞는 Config.
    - save_pretrained() 시 config.json에 직렬화됨
    - from_pretrained()/AutoConfig에서 복원됨
    """

    model_type = "multimodal_interview"

    def __init__(
        self,
        encoders: Optional[Dict[str, Dict[str, Any]]] = None,  # {"vision": {...}, "audio": {...}, ...}
        fusion: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        num_labels: Optional[int] = None,
        # 라벨 맵(선택)
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        # 운영 정책(선택): 허용 prefix 목록
        allowed_target_prefixes: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.encoders = encoders
        self.modalities = list(encoders)
        self.fusion = fusion
        self.head = head
        self.loss = loss
        if id2label is not None:
            self.id2label = {int(k): str(v) for k, v in id2label.items()}
        if label2id is not None:
            self.label2id = {str(k): int(v) for k, v in label2id.items()}

        # 예: ["src.module.multimodal.", "my_company.models."]
        self.allowed_target_prefixes = allowed_target_prefixes
