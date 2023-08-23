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

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch.nn  # just for nn.Module

from online_semg_posture_adaptation import dataset as ds
from online_semg_posture_adaptation.online import pca as opca
from online_semg_posture_adaptation.learning import learning as learn
from online_semg_posture_adaptation.learning import goodness as good


def calibration_experiment(
    xcalib: np.ndarray[np.float32],
    ycalib: np.ndarray[np.uint8],
    xvalid: np.ndarray[np.float32],
    yvalid: np.ndarray[np.uint8],
    adapt_flag: bool,
    stdscaler_train: StandardScaler | None,
    pca_train: PCA | None,
    model: torch.nn.Module,
    output_scale: float = 1.0,
) -> tuple[dict, dict]:

    xcalib_std = stdscaler_train.transform(xcalib.T).T
    xvalid_std = stdscaler_train.transform(xvalid.T).T
    del xcalib, xvalid

    if adapt_flag:

        # --------------------------------------------------------------------#
        # --------------------------------------------------------------------#
        # refit the PCA online

        W_init = opca.initialize_online_pca(
            ds.NUM_CHANNELS, opca.InitMode.CUSTOM, pca_train.components_.T)
        num_samples_calib = xcalib_std.shape[1]
        beta = 0.01
        ids_samples = np.arange(num_samples_calib)
        gamma_scheduled = 1.0 / (1.0 + ids_samples / beta)
        W_sequence, mean_calib, scale_calib = opca.oja_sga_session(
            xcalib_std, W_init, gamma_scheduled)
        W_calib = W_sequence[-1]
        W_calib = opca.reorder_w_like_reference_pca(
            W_calib, pca_train.components_)

        mean_calib = np.expand_dims(mean_calib, axis=1)
        scale_calib = np.expand_dims(scale_calib, axis=1)

        xcalib_std = (xcalib_std - mean_calib) / scale_calib
        xvalid_std = (xvalid_std - mean_calib) / scale_calib
        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #

        xcalib_pc = W_calib.T @ xcalib_std
        xvalid_pc = W_calib.T @ xvalid_std

    else:
        # no adptation
        xcalib_pc = pca_train.transform(xcalib_std.T).T
        xvalid_pc = pca_train.transform(xvalid_std.T).T

    del xcalib_std, xvalid_std

    # MLP inference
    yout_calib = learn.do_inference(xcalib_pc, model, output_scale)
    yout_valid = learn.do_inference(xvalid_pc, model, output_scale)
    del xcalib_pc, xvalid_pc

    metrics_calib = good.compute_classification_metrics(ycalib, yout_calib)
    metrics_valid = good.compute_classification_metrics(yvalid, yout_valid)

    return metrics_calib, metrics_valid


def main() -> None:
    pass


if __name__ == '__main__':
    main()
