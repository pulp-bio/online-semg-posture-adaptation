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
import enum
import os
import random
import time

# non-torch imports
import numpy as np
import sklearn.utils as sklutils
# torch imports
import torch
import torch.utils.data

from online_semg_posture_adaptation import dataset as ds
from online_semg_posture_adaptation.learning.settings import DEVICE
from online_semg_posture_adaptation.learning import goodness as good


NUM_EPOCHS = 16  # epochs of floating-point training
MINIBATCH_SIZE_TRAIN = 64  # minibatch size for training
MINIBATCH_SIZE_INFER = 8192  # minibatch size for inference


class UniboINAILSessionDataset():

    """
    For PyThorch's needs, a "dataset" is just an onbject with a __getitem__ and
    a __len__
    """

    def __init__(
        self,
        x: np.ndarray[np.float32],
        y: np.ndarray[np.uint8] | None = None,
    ):

        # "examples" is less ambiguous than "samples": not the single numbers
        num_channels, num_examples = x.shape
        assert num_channels == ds.NUM_CHANNELS
        if y is not None:
            assert len(y) == num_examples
            assert y.min() == 0 and y.max() == ds.NUM_CLASSES - 1

        self.x = x
        self.y = y
        self.num_examples = num_examples

    def __len__(self) -> int:
        return self.num_examples

    def __data_generation(
        self, idx_example: int,
    ) -> tuple[np.ndarray[np.float32], np.ndarray[np.uint8]]:

        # item index indicizes the second dimension because the format is
        # (num_channels, num_samples)

        if self.y is not None:
            return self.x[:, idx_example], self.y[idx_example]
        else:
            return self.x[:, idx_example]

    def __getitem__(
        self, idx: int,
    ) -> tuple[np.ndarray[np.float32], np.ndarray[np.uint8]]:
        return self.__data_generation(idx)


def collate_x_only(
    minibatch: list[np.ndarray[np.float32]]
) -> torch.Tensor[torch.float32]:

    # concatenating in NumPy first should be faster
    x = np.array(minibatch, dtype=np.float32)
    del minibatch
    x = torch.tensor(x, dtype=torch.float32, requires_grad=False, device='cpu')

    return x


def collate_xy_pairs(
    minibatch: list[tuple[np.ndarray[np.float32], np.ndarray[np.uint8]]]
) -> tuple[torch.Tensor[torch.float32], torch.Tensor[torch.uint8]]:

    # concatenating in NumPy first should be faster
    x = np.array([xy[0] for xy in minibatch], dtype=np.float32)
    y = np.array([xy[1] for xy in minibatch], dtype=np.uint8)
    del minibatch

    x = torch.tensor(x, dtype=torch.float32, requires_grad=False, device='cpu')
    # PyTorch's crossentropy wants int64 format for the target labels
    y = torch.tensor(y, dtype=torch.int64, requires_grad=False, device='cpu')

    return x, y


@enum.unique
class Mode(enum.Enum):
    TRAINING = 'TRAINING'
    INFERENCE = 'INFERENCE'


def dataset2dataloader(
    dataset: UniboINAILSessionDataset,
    mode: Mode,
) -> torch.utils.data.DataLoader:

    assert isinstance(mode, Mode)

    if mode == Mode.TRAINING:
        batch_size = MINIBATCH_SIZE_TRAIN
        drop_last = True
        shuffle = True
        sampler = None
    elif mode == Mode.INFERENCE:
        batch_size = MINIBATCH_SIZE_INFER
        drop_last = False
        shuffle = False
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        raise ValueError

    collate_fn = collate_x_only if dataset.y is None else collate_xy_pairs

    dataloader = torch.utils.data.DataLoader(
        dataset,  # just arg, not kwarg
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    return dataloader


def do_inference(
    x: np.ndarray[np.float32],
    model: torch.nn.Module,
    output_scale: float = 1.0,
) -> tuple:

    dataset = UniboINAILSessionDataset(x)
    dataloader = dataset2dataloader(dataset, mode=Mode.INFERENCE)

    model.eval()
    model.to(DEVICE)

    yout = np.zeros((0, ds.NUM_CLASSES), dtype=np.uint8)

    for x_b in dataloader:
        x_b = x_b.to(DEVICE)
        with torch.no_grad():
            yout_b = model(x_b)
        del x_b
        yout_b = yout_b.detach()
        yout_b = yout_b.cpu()
        yout_b = yout_b.numpy()
        yout = np.concatenate((yout, yout_b), axis=0)
        del yout_b

    yout *= output_scale

    return yout


def do_training(
    xtrain: np.ndarray[np.float32],
    ytrain: np.ndarray[np.uint8],
    xvalid: np.ndarray[np.float32] | None,
    yvalid: np.ndarray[np.uint8] | None,
    model: torch.nn.Module,
    criterion: torch.nn.Module | None = None,  # None as default (ugly)
    optimizer: torch.optim.Optimizer | None = None,  # None as default (ugly)
    num_epochs: int = NUM_EPOCHS,
) -> tuple:

    assert (xvalid is None) == (yvalid is None)

    dataset_train = UniboINAILSessionDataset(xtrain, ytrain)
    dataloader_train = dataset2dataloader(dataset_train, mode=Mode.TRAINING)

    model.to(DEVICE)
    model.train()

    if criterion is None:
        class_labels_array = np.arange(ds.NUM_CLASSES, dtype=np.uint8)
        class_weights = sklutils.class_weight.compute_class_weight(
            class_weight='balanced', classes=class_labels_array, y=ytrain)
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32,
            requires_grad=False, device='cpu',
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        criterion.to(DEVICE)

    if optimizer is None:
        params = model.parameters()
        lr = 0.001
        weight_decay = 0.001
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    history = {
        'epoch': {},
    }

    print(
        f"\n"
        f"\t\tTRAINING\t\tVALIDATION\n"
        f"\n"
        "EPOCH\t\tLoss\tBal.acc.\tLoss\tBal.acc.\tTime (s)\n"
    )
    for idx_epoch in range(num_epochs):

        t_start_epoch_s = time.time()

        for x_b, y_b in dataloader_train:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            yout_b = model(x_b)
            loss_b = criterion(yout_b, y_b)
            loss_b.backward()
            optimizer.step()

        yout_train = do_inference(xtrain, model)
        metrics_train_epoch = good.compute_classification_metrics(
            ytrain, yout_train)

        if xvalid is not None and yvalid is not None:
            yout_valid = do_inference(xvalid, model)
            metrics_valid_epoch = good.compute_classification_metrics(
                yvalid, yout_valid)
        else:
            yout_valid = None
            metrics_valid_epoch = None

        t_end_epoch_s = time.time()
        deltat_epoch_s = t_end_epoch_s - t_start_epoch_s

        if xvalid is not None and yvalid is not None:
            print("%d/%d\t\t%.4f\t%.4f\t\t%.4f\t%.4f\t\t%.1f" % (
                idx_epoch + 1,
                num_epochs,
                metrics_train_epoch['balanced_crossentropy'],
                metrics_train_epoch['balanced_accuracy'],
                metrics_valid_epoch['balanced_crossentropy'],
                metrics_valid_epoch['balanced_accuracy'],
                deltat_epoch_s,
            ))
        else:
            print("%d/%d\t\t%.4f\t%.4f\t\tnone\tnone\t\t%.1f" % (
                idx_epoch + 1,
                num_epochs,
                metrics_train_epoch['balanced_crossentropy'],
                metrics_train_epoch['balanced_accuracy'],
                deltat_epoch_s,
            ))

        history['epoch'][idx_epoch] = {
            'training': metrics_train_epoch,
            'validation': metrics_valid_epoch,
        }

    return model, history, yout_train, yout_valid


def main() -> None:
    pass


if __name__ == '__main__':
    main()
