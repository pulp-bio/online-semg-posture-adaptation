# %%
import itertools
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from alignments import dataset as ds
from alignments.learning import mlpuniboinail as mui
from alignments.learning import learning as learn
from alignments.learning import quantization as quant
from alignments import protocol

# %%
DOWNSAMPLING_FACTOR = 1

NUM_CALIB_REPETITIONS = 5

NUM_EPOCHS_FP = 4
QUANTIZE = True
NUM_EPOCHS_QAT = 8
INPUT_SCALE = 0.999

RESULTS_FILENAME = 'results_adaptation.pkl'
RESULTS_DIR_PATH = './results/'
RESULTS_FILE_FULLPATH = RESULTS_DIR_PATH + RESULTS_FILENAME

# %%
# structure for storing the results

results = {'subject': {}}

for idx_subject in range(ds.NUM_SUBJECTS):

    results['subject'][idx_subject] = {'day': {}}

    for idx_day in range(ds.NUM_DAYS):

        results['subject'][idx_subject]['day'][idx_day] = {'reference_posture': {}}

        for idx_ref_posture in range(ds.NUM_POSTURES):

            results['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture] = {'target_posture': {}}

            for idx_tgt_posture in range(ds.NUM_POSTURES):

                results['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture]['target_posture'][idx_tgt_posture] = {
                    'recalibration_mode': {
                        'none': {'calibration': {}, 'validation': {}},
                        'pca_online': {'calibration': {}, 'validation': {}},
                    },
                }

# %%
for idx_subject, idx_day, idx_ref_posture in itertools.product(
    range(ds.NUM_SUBJECTS), range(ds.NUM_DAYS), range(ds.NUM_POSTURES)
):
    
    # ----------------------------------------------------------------------- #

    # print a header
    print(
        f"\n"
        f"------------------------------------------------------------------\n"
        f"SUBJECT\t{idx_subject + 1 :d}/{ds.NUM_SUBJECTS:d}\n"
        f"DAY\t{idx_day + 1 :d}/{ds.NUM_DAYS:d}\n"
        f"POSTURE\t{idx_ref_posture + 1 :d}/{ds.NUM_POSTURES:d} AS REFERENCE\n"
        f"(all indices are one-based)\n"
        f"------------------------------------------------------------------\n"
        f"\n"
    )

    # ----------------------------------------------------------------------- #

    # load training data
    train_session_data_dict = ds.load_session(
        idx_subject, idx_day, idx_ref_posture)
    xtrain = train_session_data_dict['emg']
    ytrain = train_session_data_dict['relabel']
    del train_session_data_dict

    # ----------------------------------------------------------------------- #

    # downsampling
    xtrain = xtrain[:, ::DOWNSAMPLING_FACTOR]
    ytrain = ytrain[::DOWNSAMPLING_FACTOR]

    # standard scaling and de-correlation, as preprocessing before training
    stdscaler_train = StandardScaler()
    xtrain_stdscaled = stdscaler_train.fit_transform(xtrain.T).T
    del xtrain
    pca_train = PCA(n_components=ds.NUM_CHANNELS, whiten=False)
    xtrain_pc = pca_train.fit_transform(xtrain_stdscaled.T).T
    del xtrain_stdscaled

    # ----------------------------------------------------------------------- #

    # MLP training and validation

    mlp = mui.MLPUniboINAIL()
    mui.summarize(mlp)

    # full-precision training
    mlp, history, yout_train, yout_valid = learn.do_training(
        xtrain=xtrain_pc,
        ytrain=ytrain,
        model=mlp,
        xvalid=None,
        yvalid=None,
        num_epochs=NUM_EPOCHS_FP,
    )
    # Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
    if QUANTIZE:
        (
            mlp_q,
            output_scale,
            history_q,
            metrics_train_q,
            _,  # (in general, this is metrics_valid_q)
            yout_train_q,
            _,  # (in general, this is yout_valid_q)
        ) = quant.quantlib_flow(
            xtrain=xtrain_pc,
            ytrain=ytrain,
            model=mlp,
            xvalid=None,
            yvalid=None,
            do_qat=True,
            num_epochs_qat=NUM_EPOCHS_QAT,
            input_scale=INPUT_SCALE,
            export=False,
            onnx_filename=None,
        )
        mlp = mlp_q  # replace the model
        del mlp_q
    else:
        output_scale = 1.0

    # ----------------------------------------------------------------------- #

    # "tgt posture" stands for "target posture"
    for idx_tgt_posture in range(ds.NUM_POSTURES):

        # ------------------------------------------------------------------- #

        # print a header
        print(
            f"\n"
            f"--------------------------------------------------------------\n"
            f"TARGET POSTURE {idx_tgt_posture + 1 :d}\n"
            f"(trained on {idx_ref_posture + 1 :d})\n"
            f"--------------------------------------------------------------\n"
            f"\n"
        )

        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #

        # Do the two experiments:
        # - no adaptation
        # - refit the PCA online

        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #

        # load calibration and validation data

        calibvalid_session_data_dict = ds.load_session(
            idx_subject, idx_day, idx_tgt_posture)

        emg_calibvalid = calibvalid_session_data_dict['emg']
        relabel_calibvalid = calibvalid_session_data_dict['relabel']
        gesture_counter_calibvalid = \
            calibvalid_session_data_dict['gesture_counter']
        del calibvalid_session_data_dict

        xcalib, ycalib, xvalid, yvalid = ds.split_into_calib_and_valid(
            emg=emg_calibvalid,
            relabel=relabel_calibvalid,
            gesture_counter=gesture_counter_calibvalid,
            num_calib_repetitions=NUM_CALIB_REPETITIONS,
        )
        del emg_calibvalid, relabel_calibvalid, gesture_counter_calibvalid

        # ------------------------------------------------------------------- #

        # downsampling
        # NB: frozen standard scaling is included in the function
        # calibration_experiment
        xcalib = xcalib[:, ::DOWNSAMPLING_FACTOR]
        xvalid = xvalid[:, ::DOWNSAMPLING_FACTOR]
        ycalib = ycalib[::DOWNSAMPLING_FACTOR]
        yvalid = yvalid[::DOWNSAMPLING_FACTOR]
        
        # ------------------------------------------------------------------- #

        # no calibration

        adapt_flag = False
        
        metrics_calib, metrics_valid = protocol.calibration_experiment(
            xcalib=xcalib,
            ycalib=ycalib,
            xvalid=xvalid,
            yvalid=yvalid,
            adapt_flag=adapt_flag,
            stdscaler_train=stdscaler_train,
            pca_train=pca_train,
            model=mlp,
            output_scale=output_scale,
        )

        print('\nNO REFIT\n')
        print('CALIB METRICS')
        print(metrics_calib)
        print('VALID METRICS')
        print(metrics_valid)

        # store results
        results['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture][
            'target_posture'][idx_tgt_posture]['recalibration_mode']['none']['calibration'] = metrics_calib
        results['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture][
            'target_posture'][idx_tgt_posture]['recalibration_mode']['none']['validation'] = metrics_valid
        del adapt_flag, metrics_calib, metrics_valid

        # ------------------------------------------------------------------- #

        # online PCA

        adapt_flag = True

        metrics_calib, metrics_valid = protocol.calibration_experiment(
            xcalib=xcalib,
            ycalib=ycalib,
            xvalid=xvalid,
            yvalid=yvalid,
            adapt_flag=adapt_flag,
            stdscaler_train=stdscaler_train,
            pca_train=pca_train,  # used to initialize, reorder, and match sign
            model=mlp,
            output_scale=output_scale,
        )

        print('\nONLINE PCA\n')
        print('CALIB METRICS')
        print(metrics_calib)
        print('VALID METRICS')
        print(metrics_valid)

        # store results
        results['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture][
            'target_posture'][idx_tgt_posture]['recalibration_mode']['pca_online']['calibration'] = metrics_calib
        results['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture][
            'target_posture'][idx_tgt_posture]['recalibration_mode']['pca_online']['validation'] = metrics_valid
        del adapt_flag, metrics_calib, metrics_valid

        # ------------------------------------------------------------------- #
        
        # save to file
        # save the updated results dictionary after each validation
        results_outer_dict = {'results': results}
        Path(RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE_FULLPATH, 'wb') as f:
            pickle.dump(results_outer_dict, f)

# %%



