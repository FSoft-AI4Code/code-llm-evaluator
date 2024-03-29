import os
import sys
import json
import time
from tqdm import tqdm
from warnings import warn
from typing import Optional, Dict

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
    """Evaluator to load dataset and generate output. Customized config for 
    generate output.
    
    For example:
    >>> from code_eval import Evaluator, HumanEval
    
    >>> task = HumanEval()
    >>> evaluator = Evaluator(task=task)
    
    >>> output = evaluator.generate(temperature=0.9, num_return_sequences=3)
    >>> result = evaluator.evaluate(output)

    """
    def __init__(self, 
        task: TaskBase,
        model_name_or_path: str,
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
        
        self.model_name_or_path = model_name_or_path
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
        ) -> Dict:
        """Start engine and generate output
        
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
        
        for batch_id, batch in tqdm(enumerate(ds_loader), total=len(ds_loader), desc="Generating"):
            outputs = self._generate_fn(batch['question'])

            for idx in range(len(outputs)):
                res = dict(
                    id=batch['task_id'][idx],
                    question=batch['question'][idx],
                    answer=outputs[idx].outputs[0].text
                )
                json.dump(res, writer)
                writer.write("\n")

        writer.close()
        
        print("=======  Finished {}  =======".format(self.TASK_NAME))
        print("Completion time: %d s", (time.time() - start_time))
    
    def _distributed_initialize():
        pass
    
    def _distributed_generate(self, batch):
        raise NotImplementedError
    
    def _vllm_initialize(
        self,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float,
        num_return_sequences: int) -> Dict:
        
        ngpus = torch.cuda.device_count()
        engine_kwargs = dict(
            disable_log_stats=True,
            tensor_parallel_size=ngpus,
            download_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        
        self.model = LLM(self.model_name_or_path, 
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
    
    def _vllm_generate(self, batch):
        return self.model.generate(batch, 
                                   self.sampling_params, 
                                   lora_request=self.lora_request)
    
    def evaluate():
        pass
    