"""Explicit input view for an unchanged original faithfulness evaluator.

The evaluator's legacy Context/chat formatter is bypassed because attribution
used the frozen raw prompt. Only formatting is adapted. Tokenization, EOS choice,
all model scoring and the RISE/MAS function remain the original callables. This
is original-metric evaluation on the frozen input, not stock run_exp reproduction.
"""
class FixedInputMetricView:
    def __init__(self,original_evaluator,expected_prompt):
        self.original_evaluator=original_evaluator
        self.expected_prompt=expected_prompt

    def format_prompt(self,prompt):
        if prompt!=self.expected_prompt:
            raise ValueError('Metric received a different prompt than the frozen attribution input.')
        return prompt

    def __getattr__(self,name):
        return getattr(self.original_evaluator,name)
