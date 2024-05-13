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
import torch.utils
import torch.utils.data
from tqdm import tqdm
from warnings import warn
from typing import Optional, Dict, List

import torch
from torch.utils.data import DataLoader

from code_eval.tasks.base import TaskBase
from code_eval.tasks.mbpp import MBPP

from accelerate import Accelerator, PartialState
from transformers import (
    HfArgumentParser,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria, 
    StoppingCriteriaList,
    DataCollatorWithPadding
)

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
        print(self.dataset)
        
        self.model_name = model_name
        self.peft_model = peft_model
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.save_dir = save_dir

    
    def generate(self,
        backend: Optional[str]="vllm",
        num_return_sequences: Optional[int]=1,
        max_tokens: Optional[int]=256,
        temperature: Optional[float]=0.9,
        repetition_penalty: Optional[float]=1.2
        ) -> List:
        """Start backend and generate output

        :param backend: backend to inference model. 
            Choose between native ``transformers`` or ``vllms`` for fast infernce, defaults to "vllm"
        :type backend: Optional[str], optional
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
            with ``task_id``, ``prompt`` and ``answer`` key.
        :rtype: List
        """
        print(f"Evaluating task: [{self.TASK_NAME}]")
        print(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        print(f"device compute capabilities={torch.cuda.get_device_capability()}")
        print(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")
        
        gen_config = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
        
        if backend == "vllm":
            assert VLLM_AVAILABLE, "vllm not installed, try `pip install vllm`"
            self._vllm_initialize(**gen_config)
            self._generate_fn = self._vllm_generate
        
        elif backend == "tf":
            self._distributed_initialize(**gen_config)
            self._generate_fn = self._distributed_generate

        else:
            raise NotImplementedError
        
        self.ds_loader = [self.dataset[i:i+self.batch_size] 
                    for i in range(0, len(self.dataset), self.batch_size)]
        
        start_time = time.time()
        self._generate_fn()
        
        # results = self._do_parallel_generate()
        
        print("=======  Finished {}  =======".format(self.TASK_NAME))
        print("Completion time: %d s", (time.time() - start_time))
        
        # return results
    
    def _do_parallel_generate(self):
        # os.makedirs(self.save_dir, exist_ok=True)
        # save_path = os.path.join(self.save_dir, f"{self.TASK_NAME}.generated.jsonl")
        # writer = open(save_path, "w")
        
        result = self._generate_fn()
        
        return result
        # for idx in range(len(outputs)):
        #         res = dict(
        #             task_id=batch['task_id'][idx],
        #             prompt=batch['question'][idx],
        #             response=outputs[idx]
        #         )
                
        #         # TODO: saving
        #         results.append(res)
        #         json.dump(res, writer)
        #         writer.write("\n")

        # writer.close()
        # return results
    
    def _distributed_initialize(
        self,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float,
        num_return_sequences: int):
        """Initilize native transformers backend"""
        self.accelerator = Accelerator()
        generate_args = dict(
            max_new_tokens=max_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )
        self.generation_config = GenerationConfig(**generate_args)
        
        model_kwargs = dict(
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            load_in_8bit=False
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs)
            
        except KeyError: # TODO: except load seq2seq model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, **model_kwargs)
        
        if self.peft_model:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.peft_model)
        
        self.model.to(self.accelerator.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=self.trust_remote_code,
            padding_side="left"
        )
        
        if not self.tokenizer.pad_token:
            print("Set EOS_TOKEN to PAD_TOKEN")
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def _distributed_generate(self):
        # ``Accelerate`` distribute data and model
        assert self.accelerator
        
        for i in range(len(self.ds_loader)):
            question = self.ds_loader[i]['question']
            self.ds_loader[i]['question'] = self.tokenizer(question, return_tensors="pt", padding=True)
        
        with self.accelerator.split_between_processes(self.ds_loader) as batched_prompts:
            for batch in tqdm(batched_prompts, desc="Generating"):
                batch = batch['question'].to(self.accelerator.device)
                outputs = self.model.generate(**batch, 
                                            generation_config=self.generation_config,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            eos_token_id=self.tokenizer.eos_token_id)
                
                outputs = [output[len(prompt) :] for prompt, output in zip(batch["input_ids"], outputs)]
                batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for res in batch_results:
                    print(res)
                    yield res
        

    def _vllm_initialize(
        self,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float,
        num_return_sequences: int):
        """Initialize vllm backend

        :return: vLLM's model, sampling parameters and lora config
        :rtype: set
        """
        ngpus = torch.cuda.device_count()
        backend_kwargs = dict(
            disable_log_stats=True,
            tensor_parallel_size=ngpus,
            download_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        
        self.model = LLM(self.model_name, 
            enable_lora=True if self.peft_model else None,
            **backend_kwargs)
        
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
    
    def _vllm_generate(self):
        for batch in tqdm(self.ds_loader, total=len(self.ds_loader), desc="Generating"):
            outputs = self.model.generate(batch['question'], 
                                          self.sampling_params, 
                                          lora_request=self.lora_request)
        
            for item in outputs:
                yield item.outputs[0].text
            
    def evaluate():
        pass
    