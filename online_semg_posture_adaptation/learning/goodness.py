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

# non-torch imports
import numpy as np
from sklearn import utils as sklutils
from sklearn import metrics as m
# torch imports
import torch  # just for tensors and datatypes
import torch.nn.functional as F

from online_semg_posture_adaptation import dataset as ds


def balanced_crossentropy_score(
    ytrue: np.ndarray[np.uint8],
    yout: np.ndarray[np.float32],
) -> float:

    """
    Shortcut by wrapping PyTorch to exploit the function.
    https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
    which redirects to
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """

    # compute class weights
    class_labels_array = np.arange(ds.NUM_CLASSES, dtype=np.uint8)
    class_weights = sklutils.class_weight.compute_class_weight(
        class_weight='balanced', classes=class_labels_array, y=ytrue)

    # convert to torch.Tensor
    # PyTorch's crossentropy wants int64 format for the target labels
    ytrue = torch.tensor(
        ytrue, dtype=torch.int64, requires_grad=False, device='cpu')
    yout = torch.tensor(
        yout, dtype=torch.float32, requires_grad=False, device='cpu')
    class_weights = torch.tensor(
        class_weights, dtype=torch.float32, requires_grad=False, device='cpu')

    # remember that PyTorch passes pred and true swapped wrt to Scikit-Learn
    balanced_crossentropy = F.cross_entropy(yout, ytrue, weight=class_weights)
    balanced_crossentropy = balanced_crossentropy.item()

    return balanced_crossentropy


def compute_classification_metrics(
    ytrue: np.ndarray[np.uint8],
    yout: np.ndarray[np.float32],
) -> dict:

    # compute metrics' values

    yhard = yout.argmax(1)  # yout has shape (num_examples, num_classes)
    yhard = yhard.astype(np.uint8)

    # balanced crossentropy
    balanced_crossentropy = balanced_crossentropy_score(ytrue, yout)

    # balanced accuracy
    balanced_accuracy = m.balanced_accuracy_score(ytrue, yhard)

    # accuracy
    accuracy = m.accuracy_score(ytrue, yhard)

    # store into a dictionary

    detection_metrics = {
        'balanced_crossentropy': balanced_crossentropy,
        'balanced_accuracy': balanced_accuracy,
        'accuracy': accuracy,
    }

    return detection_metrics


def main() -> None:
    pass


if __name__ == '__main__':
    main()
