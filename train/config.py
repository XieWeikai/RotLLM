import dataclasses

@dataclasses.dataclass
class QuantizeConfig:
    """
    Base onfiguration for quantization.
    """
    mode: str = 'dynamic'               # 'dynamic'(Only support per-channel and per-group) or 'static' (Only support per-tensor and per-channel)
    granularity: str = 'per_tensor'     # 'per_tensor' or 'per_channel' (When mode == static, granularity is effective.)

    num_bits: int = 8
    is_symmetric: bool = False
    groupsize: int = -1


@dataclasses.dataclass
class ActivationQuantizeConfig(QuantizeConfig):
    """
    Activation quantization config.
    """
    clip_ratio: float = 1.0
    

@dataclasses.dataclass
class WeightQuantizeConfig(QuantizeConfig):
    """
    Weight quantization config.
    """
    clip_ratio: float = 1.0

@dataclasses.dataclass
class BiasQuantizeConfig(QuantizeConfig):
    """
    Bias quantization config.
    """
    clip_ratio: float = 1.0


@dataclasses.dataclass
class KeyQuantizeConfig(QuantizeConfig):
    """
    Bias quantization config.
    """
    clip_ratio: float = 1.0


@dataclasses.dataclass
class ValueQuantizeConfig(QuantizeConfig):
    """
    Bias quantization config.
    """
    clip_ratio: float = 1.0


@dataclasses.dataclass
class AllQuantizeConfigs:
    activation: ActivationQuantizeConfig = dataclasses.field(default_factory=ActivationQuantizeConfig)
    weight: WeightQuantizeConfig = dataclasses.field(default_factory=WeightQuantizeConfig)
    bias: BiasQuantizeConfig = dataclasses.field(default_factory=BiasQuantizeConfig)
    key: KeyQuantizeConfig = dataclasses.field(default_factory=KeyQuantizeConfig)
    value: ValueQuantizeConfig = dataclasses.field(default_factory=ValueQuantizeConfig)

