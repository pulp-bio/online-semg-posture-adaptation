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

import numpy as np

from online_semg_posture_adaptation import dataset as ds
from online_semg_posture_adaptation.online import covariance as cov


def symm_orth_no_eig(
    w: np.ndarray[np.float32],
    num_iter: int,
) -> np.ndarray[np.float32]:

    norm_w = np.sum(np.abs(w))
    w /= norm_w

    for _ in range(num_iter):
        w = 1.5 * w - 0.5 * w @ w.T @ w

    return w


@enum.unique
class InitMode(enum.Enum):
    IDENTITY = 'IDENTITY'
    RANDOM = 'RANDOM'
    CUSTOM = 'CUSTOM'


def initialize_online_pca(
    num_channels: int,
    init_mode: InitMode,
    custom_init_matrix: np.ndarray[np.float32] | None = None,
) -> np.ndarray[np.float32]:

    assert isinstance(init_mode, InitMode)

    if init_mode == InitMode.IDENTITY:
        W_init = np.identity(num_channels, dtype=np.float32)

    elif init_mode == InitMode.RANDOM:
        W_init = np.random.randn(num_channels, num_channels).astype(np.float32)
        W_init = symm_orth_no_eig(W_init, num_iter=128)

    elif init_mode == InitMode.CUSTOM:
        assert custom_init_matrix is not None
        W_init = custom_init_matrix.astype(np.float32)

    else:
        raise NotImplementedError

    return W_init


def reorder_w_like_reference_pca(
    W: np.ndarray[np.float32],
    ref_pca_components: np.ndarray[np.float32],
) -> np.ndarray[np.float32]:

    dim, num_kept_pcs = W.shape
    coeffs = ref_pca_components @ W
    props = np.square(coeffs)

    taken_from_W = []

    rankings = np.argsort(props, axis=1)[:, ::-1]  # descending order

    for idx_pc in range(num_kept_pcs):

        for idx_look in range(num_kept_pcs):
            idx_in_w = rankings[idx_pc, idx_look]  # here just a candidate
            if idx_in_w in taken_from_W:
                continue  # look at the next one, right down in the ranking
            else:
                taken_from_W.append(idx_in_w)  # pick it
                break  # stop search

    W_ord = W[:, taken_from_W]  # orientation ("sign") not matched yet

    signs = np.sign(np.diag(ref_pca_components @ W_ord))
    signs = np.expand_dims(signs, axis=0)
    W_ord *= signs

    return W_ord


def oja_sga_step(
    x: np.ndarray[np.float32],
    W: np.ndarray[np.float32],
    gamma: float,
) -> np.ndarray[np.float32]:

    assert len(x.shape) == 1
    num_chnls = len(x)

    y = W.T @ x

    Delta_W = np.zeros((num_chnls, num_chnls), dtype=np.float32)

    for j in range(num_chnls):

        nsum = np.zeros(num_chnls, dtype=np.float32)
        for i in range(j):
            nsum += y[i] * W[:, i]

        Delta_W[:, j] = gamma * y[j] * (x - y[j] * W[:, j] - 2.0 * nsum)

    # update
    W = W + Delta_W

    # ----------------------------------------------------------------------- #
    # I add a normalization
    norms = np.sqrt(np.sum(np.square(W), axis=0))
    W = W / norms
    # ----------------------------------------------------------------------- #

    return W


def oja_sga_session(
    x: np.ndarray(np.float32),
    W_init: np.ndarray(np.float32),
    gamma_scheduled: np.ndarray(np.float32),
) -> np.ndarray(np.float32):

    num_chnls, num_samples = x.shape

    W_sequence = np.zeros(
        (num_samples + 1, num_chnls, num_chnls), dtype=np.float32)
    W_sequence[0] = W_init

    mean = np.zeros(ds.NUM_CHANNELS, dtype=np.float32)
    ncov = np.zeros((ds.NUM_CHANNELS, ds.NUM_CHANNELS), dtype=np.float32)

    for idx_sample in range(num_samples):

        mean, ncov, _ = cov.update_cov(
            x[:, idx_sample], mean, ncov, idx_sample + 1)
        scale = np.sqrt(np.diag(ncov) / (idx_sample + 1))

        W_sequence[idx_sample + 1] = oja_sga_step(
            (x[:, idx_sample] - mean) / scale,
            W_sequence[idx_sample],
            gamma_scheduled[idx_sample],
        )

    return W_sequence, mean, scale


def main() -> None:
    pass


if __name__ == '__main__':
    main()
