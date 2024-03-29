import os
import warnings
from abc import ABC
from typing import Any

from datasets import load_dataset

class TaskBase(ABC):
    TASK_NAME: str = None
    DATASET_NAME_OR_PATH: str = None
    def __init__(self) -> None:
        try:
            self.dataset = load_dataset(self.DATASET_NAME_OR_PATH)
        except Exception as e:
            warnings.warn("Encounter this error when try to load dataset "
                          "from HuggingFace Hub, `{}`. "
                          "Try loading dataset from local path".format(e))
            
            assert os.path.exists(self.DATASET_NAME_OR_PATH)
            assert os.path.isfile(self.DATASET_NAME_OR_PATH)
            
            # Load .json file
            self.dataset = load_dataset("json", data_files={"test": self.DATASET_NAME_OR_PATH})
                    
        
    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    def compute_metrics(self):
        raise NotImplementedError