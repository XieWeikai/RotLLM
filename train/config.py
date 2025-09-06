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
    int8_down_proj: bool = False
    

@dataclasses.dataclass
class WeightQuantizeConfig(QuantizeConfig):
    """
    Weight quantization config.
    """
    clip_ratio: float = 1.0
    mse: bool = False           # 表示是否启动搜索模式
    norm: float = 2.4           # 范数
    grid: int = 100             # 表示搜索的粒度（搜索多少步）  
    maxshrink: float = 0.8      # 表示搜索范围压缩比例（0~1）

    int8_down_proj: bool = False

    rtn: bool = False
    nsamples: int = 128
    percdamp: float = 0.01
    act_order: bool = False

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

