from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")  # 리턴 타입 일반화


class BaseAgent(ABC, Generic[T]):
    def __init__(self, generator):
        self.generator = generator

    @abstractmethod
    def generate(self, *args, **kwargs) -> T:
        pass
