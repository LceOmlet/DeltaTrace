"""CPU delegation checks; not a PPO numerical tolerance or training test."""
import functools
import json

import torch
from verl.trainer.ppo import core_algos

from capture_native_loss_backward_20261010 import observe_call, observe_gradient


def main():
    records = []
    for gap in [1.0,89.0]:
        expected_input = torch.tensor([[-1.0-gap]],requires_grad=True)
        actual_input = expected_input.detach().clone().requires_grad_()
        ref = torch.tensor([[-1.0]])
        expected = core_algos.kl_penalty(expected_input,ref,'low_var_kl')
        expected_gradient, = torch.autograd.grad(expected.sum(),expected_input)
        outputs, seen, errors, hooks = [], [], [], []

        @functools.wraps(core_algos.kl_penalty)
        def native(*args, **kwargs):
            value = core_algos.kl_penalty(*args,**kwargs)
            outputs.append(value)
            return value

        def after(value, arguments):
            assert value is outputs[-1]
            assert arguments['logprob'] is actual_input
            hooks.append(observe_gradient(actual_input,lambda g:seen.append(g.detach().clone()),errors.append))

        wrapped = observe_call(native,lambda:True,after,errors.append)
        actual = wrapped(actual_input,ref,'low_var_kl')
        actual_gradient, = torch.autograd.grad(actual.sum(),actual_input)
        # The assertions check transparent delegation, not a new loss tolerance.
        assert actual is outputs[0] and len(outputs)==1 and len(seen)==1 and not errors
        torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
        torch.testing.assert_close(actual_gradient,expected_gradient,rtol=0,atol=0,equal_nan=True)
        torch.testing.assert_close(seen[0],expected_gradient,rtol=0,atol=0,equal_nan=True)
        hooks[0].remove()
        records.append(dict(gap=gap,owner_called_once=True,same_return_object=True,
                            same_loss_and_gradient=True,nonfinite_gradient_preserved=gap==89.0))

    def fail(*args):
        raise RuntimeError('intentional diagnostic callback error')

    errors = []
    x = torch.tensor([2.0],requires_grad=True)
    handle = observe_gradient(x,fail,errors.append)
    (x*x).sum().backward()
    handle.remove()
    assert len(errors)==1 and float(x.grad)==4.0
    errors = []
    native_results = []
    def native(*args):
        result = core_algos.kl_penalty(*args)
        native_results.append(result)
        return result
    result = observe_call(native,lambda:True,fail,errors.append)(torch.tensor([-2.0]),torch.tensor([-1.0]),'low_var_kl')
    assert result is native_results[0] and len(errors)==1
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='passed',records=records,callback_errors_preserve_native_results=True,
        CUDA_initialized=False,model_calls=0,DT_calls=0,optimizer_calls=0,scope=__doc__)))


if __name__ == '__main__':
    main()
