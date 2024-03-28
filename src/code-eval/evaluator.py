

class Evaluator:
    """Evaluator to load dataset and generate output. Customized config for 
    generate output.
    
    For example:
    >>> from code_eval import Evaluator
    >>> evaluator = Evaluator(task="humaneval")
    >>> output = evaluator.generate(temperature=0.9, n_return_sample=3)
    >>> result = evaluator.evaluate(output)
    
    """
    def __init__(self) -> None:
        pass
    
    def generate():
        pass
    
    def _distributed_generate():
        pass
    
    def _vllm_generate():
        pass
    
    def evaluate():
        pass