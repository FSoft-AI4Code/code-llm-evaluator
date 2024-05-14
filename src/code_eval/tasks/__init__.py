from .humaneval import HumanEval
from .mbpp import MBPP

SUPPORTED_TASKS = ["humaneval", "multiple", "mbpp"]

__all__ = ["HumanEval", "MBPP"]