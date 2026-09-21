"""Small explicit settings with no platform-specific paths."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse
from baby_arcus.contracts import ContractError, fields, integer

@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    simulation_port: int = 8765
    artifact_port: int = 8766
    artifact_url: str = "http://127.0.0.1:8766"
    state_root: str = "runs/baby_arcus"
    request_timeout: float = 5.0
    signal_config: str = "configs/baby_arcus/signals.json"
    curriculum_config: str = "configs/baby_arcus/curriculum.json"
    inference_port: int = 8767
    training_port: int = 8768
    controller_port: int = 8769
    evaluator_port: int = 8770
    dashboard_port: int = 8771
    inference_url: str = "http://127.0.0.1:8767"
    training_url: str = "http://127.0.0.1:8768"
    controller_url: str = "http://127.0.0.1:8769"
    evaluator_url: str = "http://127.0.0.1:8770"
    simulation_url: str = "http://127.0.0.1:8765"
    device: str = "cuda"
    evaluation_config: str = "configs/baby_arcus/evaluation.json"

    @classmethod
    def load(cls, path=None):
        values = json.loads(Path(path).read_text()) if path else {}
        fields(values, (), asdict(cls()))
        result = cls(**values)
        for port in (result.simulation_port,result.artifact_port,result.inference_port,result.training_port,
                     result.controller_port,result.evaluator_port,result.dashboard_port):
            integer(port, 0, 65535)
        if not isinstance(result.host, str) or not result.host:
            raise ContractError("Invalid host")
        if not isinstance(result.state_root, str) or not result.state_root:
            raise ContractError("Invalid state root")
        if type(result.request_timeout) not in (int, float) or not 0 < result.request_timeout <= 30:
            raise ContractError("Invalid timeout")
        if result.device not in ("cpu","cuda"):
            raise ContractError("Invalid device")
        for url in (result.artifact_url,result.simulation_url,result.inference_url,result.training_url,result.controller_url,result.evaluator_url):
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
                raise ContractError("Invalid service URL")
        return result
