import numpy as np
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class SimulatorConfig:
    """
    Configuration parameters for the Economic Scenario Generator.
    Encapsulates all mathematical and structural constants.
    """
    duration_years: int = 60
    num_scenarios: int = 1000
    seed: int = 42

    # Chunking and thread worker optional overrides
    chunk_size: Optional[int] = None
    max_workers: Optional[int] = None

    # Structural yield/tenor constants
    tenors: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    )

    # Policy and inflation boundary thresholds
    mu_min: float = 0.010
    pi_min: float = -0.02

    # Model dynamics parameters
    alpha_smooth: float = 0.20
    beta_drag: float = 0.15
    eta_erp: float = 0.10
    lambda_irp: float = 0.05
    kappa_irp: float = 0.15

    # Merton Jump-Diffusion parameters
    lambda_J: float = 0.20
    mu_J: float = -0.15
    sigma_J: float = 0.10

    # Initial state values (Standard international economic terms)
    initial_rate: float = 0.10
    initial_inflation: float = 0.045

    # Calibrated parameter overrides (filled dynamically by the Calibrator)
    ou_mu: Optional[float] = None
    cir_theta_val: Optional[float] = None
    cir_sigma_val: Optional[float] = None
    cir_mu_val: Optional[float] = None
    ou_theta_val: Optional[float] = None
    ou_sigma_val: Optional[float] = None
    gbm_sigma_val: Optional[float] = None

    @property
    def steps(self) -> int:
        return self.duration_years * 12

    @property
    def dt(self) -> float:
        return 1.0 / 12.0

    @property
    def k_jump(self) -> float:
        return np.exp(self.mu_J + 0.5 * (self.sigma_J ** 2)) - 1.0

    def to_dict(self) -> dict:
        """
        Converts the configuration instance into a dictionary.
        Automatically converts NumPy arrays to standard lists for JSON serializability.
        """
        import dataclasses
        d = dataclasses.asdict(self)
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SimulatorConfig":
        """
        Constructs a SimulatorConfig instance from a dictionary.
        Handles both static constructor fields and dynamic attributes added by calibration.
        """
        valid_fields = {f.name for f in fields(cls)}

        init_kwargs = {}
        dynamic_attrs = {}
        for k, v in d.items():
            if k in valid_fields:
                init_kwargs[k] = v
            else:
                dynamic_attrs[k] = v

        if "tenors" in init_kwargs and isinstance(init_kwargs["tenors"], list):
            init_kwargs["tenors"] = np.array(init_kwargs["tenors"])

        config = cls(**init_kwargs)

        for k, v in dynamic_attrs.items():
            setattr(config, k, v)

        return config
