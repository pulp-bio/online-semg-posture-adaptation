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

import torch  # for tensors
from torch import nn
import torchinfo

from online_semg_posture_adaptation import dataset as ds


"""
This module implements in PyTorch the 2-layer Multi-Layer Perceptron (MLP)
termed Neural Network (NN) and used in the paper:

B. Milosevic, E. Farella, S. Benatti,
Exploring Arm Posture and Temporal Variability in Myoelectric Hand Gesture
Recognition
https://doi.org/10.1109/BIOROB.2018.8487838

Differences:
- ReLU non-linear activation function is used instead of sigmoid;
- Batch-Normalization (BN) is added;
- the structure is adapted to be compliant with the DNN quantization tool
  quantlib (https://github.com/pulp-platform/quantlib): no biases, no final
  softmax inside the model, so that the model ends with a biasless linear.

"""


HIDDEN_UNITS = 8


class MLPUniboINAIL(nn.Module):

    def __init__(self):
        super(MLPUniboINAIL, self).__init__()

        self.fc0 = nn.Linear(ds.NUM_CHANNELS, HIDDEN_UNITS, bias=False)
        self.fc0_bn = nn.BatchNorm1d(8)
        self.fc0_relu = nn.ReLU()
        self.fc1 = nn.Linear(8, ds.NUM_CLASSES, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc0(x)
        x = self.fc0_bn(x)
        x = self.fc0_relu(x)
        y = self.fc1(x)
        return y


def summarize(
    model: nn.Module,
    verbose: 0 | 1 | 2 = 0,
) -> torchinfo.ModelStatistics:

    # set all parameters for torchsummary

    input_size = (ds.NUM_CHANNELS,)
    batch_dim = 0  # index of the batch dimension
    col_names = [
        'input_size',
        'output_size',
        'num_params',
        'params_percent',
        'kernel_size',
        'mult_adds',
        'trainable',
    ]
    device = 'cpu'
    mode = 'eval'
    row_settings = [
        'ascii_only',
        'depth',
        'var_names',
    ]

    # call the summary function

    model_stats = torchinfo.summary(
        model=model,
        input_size=input_size,
        batch_dim=batch_dim,
        col_names=col_names,
        device=device,
        mode=mode,
        row_settings=row_settings,
        verbose=verbose,
    )

    return model_stats


def main() -> None:

    # Display the summary of the MLP

    verbose = 1
    mlp_unibo_inail = MLPUniboINAIL()
    mlp_unibo_inail.eval()
    mlp_model_stats = summarize(mlp_unibo_inail, verbose=verbose)


if __name__ == '__main__':
    main()
