from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    data: bytes
    seed: int | None


class BaseGenerator(ABC):
    name: str
    label: int  # 0 = secure, 1 = weak

    @abstractmethod
    def generate_sample(self, n_bytes: int) -> Sample: ...

    @abstractmethod
    def params(self) -> dict: ...
