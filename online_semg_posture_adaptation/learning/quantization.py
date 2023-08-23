"""
    Author(s):
    Marcello Zanghieri <marcello.zanghieri2@unibo.it>
    
    Copyright (C) 2023 University of Bologna and ETH Zurich
    
    Licensed under the GNU Lesser General Public License (LGPL), Version 2.1
    (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        https://www.gnu.org/licenses/lgpl-2.1.txt
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from __future__ import annotations
import os
import time

import numpy as np
import torch
from torch import nn

import quantlib.editing.graphs as qg
import quantlib.editing.editing as qe

from online_semg_posture_adaptation.learning import settings
from online_semg_posture_adaptation.learning import learning as learn
from online_semg_posture_adaptation.learning import goodness as good


DEVICE = settings.DEVICE


def helper_printer(
    model: torch.nn.Module,
    how: str = 'lw',
) -> None:

    assert how in ['lightweight', 'lw', 'tabular', 'call_modules']

    if how in ['lightweight' | 'lw']:
        # Traversal of the hierarchy:
        # overview of atoms, i.e. non-container Modules
        model_lw = qg.lw.quantlib_traverse(model)
        model_lw.show()

    elif how == 'tabular':
        model.graph.print_tabular()

    elif how == 'call_modules':
        # Helper to access the model graph and print modules for diagnostics.
        # The attribute node.op is a sort of "opcode" and has 6 subclasses:
        # - placeholder
        # - in/output
        # - getattr
        # - call_function
        # - call_method
        # - call_module (i.e., PyTorch doing a wrap)
        for node in model.graph.nodes:
            if node.op == 'call_module':
                target = getattr(model, node.target)
                print(node, '\t', target.__class__.__name__)

    else:
        raise NotImplementedError

    return


def intermediate_validation(
        xtrain: np.ndarray[np.float32],
        ytrain: np.ndarray[np.uint8],
        xvalid: np.ndarray[np.float32] | None,
        yvalid: np.ndarray[np.uint8] | None,
        model: torch.nn.Module,
        output_scale: float = 1.0,
        message_str: str | None = None,
):

    """
    Convenience wrapper for quickly calling an evaluation of the model at
    different stages of quantization.
    """

    # inference and evaluation,  on training and validation data

    yout_train = learn.do_inference(xtrain, model, output_scale)
    metrics_train = good.compute_classification_metrics(ytrain, yout_train)

    if xvalid is not None and yvalid is not None:
        yout_valid = learn.do_inference(xvalid, model, output_scale)
        metrics_valid = good.compute_classification_metrics(yvalid, yout_valid)
    else:
        yout_valid = None
        metrics_valid = None

    # display a report

    #if message_str is None:
    #    message_str = ''
    #print(
    #    f"\n"
    #    f"------------------------------------------------------------------\n"
    #    f"INTERMEDIATE VALIDATION: {message_str:s}\n"
    #)
    #print("TRAINING SET")
    #print(metrics_train)
    #print("VALIDATION SET")
    #print(metrics_valid)

    #print("------------------------------------------------------------------")
    #print("------------------------------------------------------------------")
    #print("------------------------------------------------------------------")

    return metrics_train, metrics_valid, yout_train, yout_valid


def do_onnx_export(
    model: torch.nn.Module,
    dummy_input: np.ndarray,
    onnx_filename: str,
) -> None:

    model.eval()
    model.cpu()

    # to trace the `nn.Module`
    # TODO: use the collate_fn instead of hardcoded conversion to torch.Tensor

    if len(dummy_input.shape) == 2:
        # add batch dimension 1
        dummy_input = np.expand_dims(dummy_input, axis=0)
    elif len(dummy_input) > 3:
        # no more dimensions than (batch, channels, samples)
        raise ValueError

    dummy_input = torch.tensor(
        dummy_input, dtype=torch.float32, requires_grad=False, device='cpu')

    # https://pytorch.org/docs/1.9.0/onnx.html#torch.onnx.export
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_filename,
        enable_onnx_checker=True,
    )

    return


def quantlib_flow(
    xtrain: np.ndarray[np.float32],
    ytrain: np.ndarray[np.uint8],
    model: torch.nn.Module,
    do_qat: bool,
    num_epochs_qat: int,
    input_scale: float,
    export: bool,
    onnx_filename: str,
    xvalid: np.ndarray[np.float32] | None = None,
    yvalid: np.ndarray[np.uint8] | None = None,
) -> tuple:

    assert (xvalid is None) == (yvalid is None)

    # quantlib-canonicalization
    model_fp = qg.fx.quantlib_symbolic_trace(root=model)
    del model

    # Float-to-Fake (F2F) transformation

    f2fconverter = qe.float2fake.F2F8bitPACTRoundingConverter()
    model_fq = f2fconverter(model_fp)

    # collect statistics about the floating-point `Tensor`s passing through the
    # quantisers, so that we can better fit the quantisers' hyper-parameters
    with qe.float2fake.calibration(model_fq):
        learn.do_inference(xtrain, model_fq)

    # add rounding to all PACT operators
    rounder = qe.float2fake.F2F8bitPACTRounder()
    model_fq_rounded = rounder(model_fq)
    del model_fq

    # Quantization-Aware Training/Tuning (QAT)

    if do_qat:
        model_fq_rounded, history, ysoft_train_qat, ysoft_valid_qat = \
            learn.do_training(
                xtrain=xtrain,
                ytrain=ytrain,
                xvalid=xvalid,
                yvalid=yvalid,
                model=model_fq_rounded,
                num_epochs=num_epochs_qat,
            )

    # retrieve the output scale
    # for some reason, product instead of ratio as in
    # https://github.com/pulp-platform/quantlib/blob/main/editing/editing/fake2true/epstunnels/finalremover/applier.py#L42
    named_modules_list = list(model_fq_rounded.named_modules())
    scale_secondtolast = named_modules_list[-2][1]._buffers['scale'].item()

    scale_last = named_modules_list[-1][1]._buffers['scale']
    scale_last = scale_last.detach().cpu().numpy().flatten()
    del named_modules_list
    output_scale = scale_last * scale_secondtolast

    # Fake-to-True (F2T) transformation

    x_shape = xtrain[:, 0].shape
    x_shape = (1,) + x_shape  # add batch dimension: minibatch size 1
    input_scale_tensor = torch.tensor(
        [input_scale], dtype=torch.float32, requires_grad=False, device='cpu')
    x_dict = {
        'x': {
            'shape': x_shape,
            'scale': input_scale_tensor,
        },
    }

    # Convert to TrueQuantized with converter
    log2_requantization_factor = 32  # working example by Alberto uses 20
    custom_editor = qe.f2t.FinalEpsTunnelRemover()
    f2tconverter = qe.f2t.F2TConverter(
        B=log2_requantization_factor,
        custom_editor=custom_editor,
    )
    model_fq_rounded.cpu()
    model_tq = f2tconverter(model_fq_rounded, x_dict)
    del model_fq_rounded

    # Final validation
    xtrain = xtrain  # / 255.0
    xvalid = xvalid  # / 255.0
    metrics_train, metrics_valid, ysoft_train, ysoft_valid = \
        intermediate_validation(
            xtrain, ytrain, xvalid, yvalid,
            model_tq, output_scale, message_str='INTEGER',
        )

    # Export in ONNX format
    if export:
        single_input = xtrain[:, 0]  # to trace the `nn.Module` (here just one)
        do_onnx_export(
            model=model_tq,
            dummy_input=single_input,
            onnx_filename=onnx_filename,
        )

    return (
        model_tq,
        output_scale,
        history,
        metrics_train,
        metrics_valid,
        ysoft_train,
        ysoft_valid
    )


def validate_from_onnx():
    '''
    import torch
    import onnx
    from onnx2torch import convert

    onnx_model = onnx.load(MODEL_FILE_FULLPATH)
    torch_model = convert(onnx_model)
    # short for torch_model = convert(MODEL_FILE_FULLPATH)

    import onnxruntime as ort

    # Create example data
    x = torch.ones((1, 9, 2048))

    out_torch = torch_model(x)

    ort_sess = ort.InferenceSession(MODEL_FILE_FULLPATH)

    input_name_str = onnx_model.graph.input[0].name
    input_feed = {input_name_str, x.numpy()}

    outputs_ort = ort_sess.run(
        output_names=None,
        input_feed=input_feed,
        run_options=None,
    )

    # Check the Onnx output against PyTorch
    print(torch.max(torch.abs(outputs_ort - out_torch.detach().numpy())))
    print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.e-7))
    '''
    pass


def main() -> None:
    pass


if __name__ == '__main__':
    main()
