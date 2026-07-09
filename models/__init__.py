from transformers.models.auto.modeling_auto import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    _LazyAutoMapping
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

# NOTE: modeling_t5 uses deprecated transformers APIs incompatible with v5.x
# Fall back gracefully so other model modules can still be imported
try:
    from .modeling_t5 import T5ForTokenClassification, T5ForSequenceClassification
except ImportError:
    T5ForTokenClassification = None
    T5ForSequenceClassification = None

if T5ForTokenClassification is not None:
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES["t5"] = "T5ForTokenClassification"
if T5ForSequenceClassification is not None:
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES["t5"] = "T5ForSequenceClassification"

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)

from transformers.models import t5

if T5ForTokenClassification is not None:
    setattr(t5, "T5ForTokenClassification", T5ForTokenClassification)
if T5ForSequenceClassification is not None:
    setattr(t5, "T5ForSequenceClassification", T5ForSequenceClassification)

__version__ = "0.1"

__all__ = [
    "T5ForTokenClassification",
    "T5ForSequenceClassification"
]
