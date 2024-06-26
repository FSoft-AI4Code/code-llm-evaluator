"""
Task interface.
"""
import os
import warnings
from abc import ABC
from typing import Any

from datasets import load_dataset

class TaskBase(ABC):
    """
    Initialize with TASK_NAME and DATASET_NAME_OR_PATH variable to load 
    evaluation dataset through ``datasets.load_dataset()`` function.
    Dataset can be store in Huggingface Hub or local path.
    
    :param TASK_NAME: name used for saving results
    :type TASK_NAME: str or None
    :param DATASET_NAME_OR_PATH: dataset name for loading using ``load_dataset()``
    :type DATASET_NAME_OR_PATH: str or None
    
    """
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
        """Pre-processing dataset."""
        raise NotImplementedError
    
    def postprocess_generation(self, generation, idx):
        pass
    
    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]
    
    def compute_metrics(self):
        """Task metric compute function."""
        raise NotImplementedError