"""Evaluator to load dataset and generate output. Customized config for 
generate output.

For example:

.. code-block:: python

    >>> from code_eval import Evaluator, HumanEval

    >>> task = HumanEval()
    >>> evaluator = Evaluator(task=task)

    >>> output = evaluator.generate(temperature=0.9, num_return_sequences=3)
    >>> result = evaluator.evaluate(output)

"""
import os
import sys
import json
import time
from tqdm import tqdm
from warnings import warn
from typing import Optional, Dict, List

import torch

from code_eval.tasks.base import TaskBase
from code_eval.tasks.mbpp import MBPP

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE=True
except ImportError:
    VLLM_AVAILABLE=False


class Evaluator:
    """Evaluator class.

        :param task: Evaluation task loader
        :type task: TaskBase
        :param model_name: Selected model for evaluating
        :type model_name: str
        :param peft_model: Adapter model, defaults to None
        :type peft_model: Optional[str], optional
        :param trust_remote_code: Huggingface argument, defaults to False
        :type trust_remote_code: Optional[bool], optional
        :param cache_dir: Downloaded cache directory, defaults to None
        :type cache_dir: Optional[str], optional
        :param batch_size: Generation batch size, defaults to 16
        :type batch_size: Optional[int], optional
        :param save_dir: Saving generation directory, defaults to "./output"
        :type save_dir: Optional[str], optional
    """
    
    def __init__(self, 
        task: TaskBase,
        model_name: str,
        peft_model: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        batch_size: Optional[int] = 16,
        save_dir: Optional[str] = "./output"
    ) -> None:
        self.TASK_NAME = task.TASK_NAME
        self.DATASET_NAME_OR_PATH = task.DATASET_NAME_OR_PATH
        self.compute_metrics = task.compute_metrics
        self.dataset = task.prepare_dataset()
        
        self.model_name = model_name
        self.peft_model = peft_model
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.save_dir = save_dir

    
    def generate(self,
        engine: Optional[str]="vllm",
        num_return_sequences: Optional[int]=1,
        max_tokens: Optional[int]=256,
        temperature: Optional[float]=0.9,
        repetition_penalty: Optional[float]=1.2
        ) -> List:
        """Start engine and generate output

        :param engine: Engine to inference model. 
            Choose between native ``transformers`` or ``vllms`` for fast infernce, defaults to "vllm"
        :type engine: Optional[str], optional
        :param num_return_sequences: Model generated n, defaults to 1
        :type num_return_sequences: Optional[int], optional
        :param max_tokens: Max new tokens, defaults to 256
        :type max_tokens: Optional[int], optional
        :param temperature: Model generate temperature, defaults to 0.9
        :type temperature: Optional[float], optional
        :param repetition_penalty: Repetition penalty, defaults to 1.2
        :type repetition_penalty: Optional[float], optional
        :raises NotImplementedError: _description_
        :return: List of generated result, stored in dictionary object 
            with ``task_id``, ``question`` and ``answer`` key.
        :rtype: List
        """

        
        gen_config = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
        
        if engine=="vllm":
            assert VLLM_AVAILABLE, "vllm not installed, try `pip install vllm`"
            self._vllm_initialize(**gen_config)
            self._generate_fn = self._vllm_generate
        
        elif engine == "hf":
            self._distributed_initialize(**gen_config)
            self._generate_fn = self._distributed_generate

        else:
            raise NotImplementedError
        
        start_time = time.time()
        ds_loader = [self.dataset[i:i+self.batch_size] 
                    for i in range(0, len(self.dataset), self.batch_size)]
        
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"{self.TASK_NAME}.generated.jsonl")
        writer = open(save_path, "w")
        
        results = []
        for batch_id, batch in tqdm(enumerate(ds_loader), total=len(ds_loader), desc="Generating"):
            outputs = self._generate_fn(batch['question'])

            for idx in range(len(outputs)):
                res = dict(
                    id=batch['task_id'][idx],
                    question=batch['question'][idx],
                    answer=outputs[idx]
                )
                results.append(res)
                json.dump(res, writer)
                writer.write("\n")

        writer.close()
        
        print("=======  Finished {}  =======".format(self.TASK_NAME))
        print("Completion time: %d s", (time.time() - start_time))
        
        return results
    
    def _distributed_initialize():
        pass
    
    def _distributed_generate(self, batch):
        raise NotImplementedError
    
    def _vllm_initialize(
        self,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float,
        num_return_sequences: int):
        """Initialize vllm engine

        :return: vLLM's model, sampling parameters and lora config
        :rtype: set
        """
        ngpus = torch.cuda.device_count()
        engine_kwargs = dict(
            disable_log_stats=True,
            tensor_parallel_size=ngpus,
            download_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        
        self.model = LLM(self.model_name, 
            enable_lora=True if self.peft_model else None,
            **engine_kwargs)
        
        self.lora_request = None
        if self.peft_model:
            self.lora_request=LoRARequest("lora", 1, self.peft_model)
            
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            n=num_return_sequences,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        
        return (self.model, self.sampling_params, self.lora_request)
    
    def _vllm_generate(self, batch: List) -> List: # type: ignore
        """Generated function

        :param batch: batched string inputs
        :type batch: List[str]
        :return: List of generated outputs
        :rtype: List[str]
        """
        outputs = self.model.generate(batch, 
                                   self.sampling_params, 
                                   lora_request=self.lora_request)
        
        for item in outputs:
            yield item.outputs[0].text
    
    def evaluate():
        pass
    