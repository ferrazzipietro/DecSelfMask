"""Programmatic entrypoints for the DecSelfMask package."""

from .classification_head import ClassificationHeadTrainer, ClassificationHeadTrainerConfig
from .relevance import RelevanceCalculator, RelevanceCalculatorConfig
from .sequences import DecSelfMaskSequencesMaker, SequenceMakerConfig
from .sft import SFTTrainer, SFTTrainerConfig
from .training import DecSelfMaskTrainer, DecSelfMaskTrainingArguments

SFTTrainerSFTTrainer = SFTTrainer

__all__ = [
    "RelevanceCalculator",
    "RelevanceCalculatorConfig",
    "DecSelfMaskSequencesMaker",
    "SequenceMakerConfig",
    "DecSelfMaskTrainingArguments",
    "DecSelfMaskTrainer",
    "SFTTrainer",
    "SFTTrainerSFTTrainer",
    "SFTTrainerConfig",
    "ClassificationHeadTrainer",
    "ClassificationHeadTrainerConfig",
]
