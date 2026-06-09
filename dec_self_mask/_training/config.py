from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The local path or huggingface hub name of the model and tokenizer to use."
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this"
                " dtype. If `auto` is passed, the dtype will be automatically derived"
                " from the model's weights. We will override this if we use quantization."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    use_lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use LoRA. If True, the model will be trained with LoRA: https://arxiv.org/abs/2106.09685"
            )
        },
    )

    quantization: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Whether to use '4' or '8' bit quantization. Requires bitsandbytes library:"
                " https://github.com/TimDettmers/bitsandbytes. This parameter is only used for training."
            )
        },
    )

    lora_weights_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If the model has been trained with LoRA, "
                "path or huggingface hub name or local path to the pretrained weights."
            )
        },
    )

    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "Lora attention dimension."},
    )

    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "The alpha parameter for Lora scaling."},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers."},
    )

    lora_target_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": (
                "The target modules to which LoRA will be applied. If not specified, We"
                " will use the default modules for the model in huggingface PEFT library."
            )
        },
    )

    force_auto_device_map: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to force the use of the auto device map. If set to True, the model will be split across "
                "GPUs and CPU to fit the model in memory. If set to False, a full copy of the model will be loaded "
                "into each GPU. Defaults to False."
            )
        },
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Trust the remote code from HuggingFace model hub. Defaults to False Defaults to False."
        },
    )

    use_flash_attention: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use FlashAttention. If True, the model will be trained with FlashAttention."
                "Flash attention must be installed, see: https://github.com/Dao-AILab/flash-attention "
                "for more details."
            )
        },
    )

    max_memory_MB: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Free memory per gpu in MB. Used to compute the device map when force_auto_device_map is set to True."
                "Defaults to None."
            )
        },
    )

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Override the tokenizer/model max sequence length used for truncation. "
                "If set, inputs will be truncated to this length (capped by the model's max_position_embeddings)."
            )
        },
    )

    merge_lora_after_training: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to merge LoRA layers after training. If True, the model will be trained with LoRA and then"
                " the LoRA layers will be merged into the model. We will save the merged model in"
                " {output_dir}/merged_model. Defaults to False."
            )
        },
    )

    rope_scaling_factor: Optional[float] = field(
        default=None,
        metadata={
            "help": "The scaling factor for the ROPE scaling. We will use dymanic scaling. "
            "If None we wont use ROPE scaling."
        },
    )

    tasks: Optional[List[str]] = field(
        default="all",
        metadata={
            "help": "The tasks to train the model on. Defaults to all tasks."
        },
    )

    max_examples: Optional[int] = field(
        default=10000,
        metadata={
            "help": "The maximum number of examples to train on per task. Defaults to 10000."
        },
    )
