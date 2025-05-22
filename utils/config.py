from dataclasses import dataclass


@dataclass
class Configs:
    ROOT_DIR: str = '.'

    # GPU configs
    gpu_id: int = 3
    
    # Model configs
    input_channels: int = 3
    num_classes: int = 1
    num_filters: int = 17

    # Dataset configs
    batch_size: int = 2
    num_workers: int = 2

    # Optimizer configs
    lr: float = 0.0001
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0001

    # Scheduler configs
    step_size: int = 30
    gamma: float = 0.1

    # Training configs
    num_epochs: int = 300
    early_stopping: int = 30
    save_dir: str = 'checkpoints'