# %%
import itertools
from pathlib import Path
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from alignments import dataset as ds
from alignments.learning import mlpuniboinail as mui
from alignments.learning import learning as learn
from alignments.learning import quantization as quant
from alignments.learning import goodness as good

# %%
DOWNSAMPLING_FACTOR = 1

NUM_TRAIN_REPETITIONS = 5

NUM_EPOCHS_FP = 4
QUANTIZE = True
NUM_EPOCHS_QAT = 8
INPUT_SCALE = 0.999

RESULTS_FILENAME = 'results_multitrain.pkl'
RESULTS_DIR_PATH = './results/'
RESULTS_FILE_FULLPATH = RESULTS_DIR_PATH + RESULTS_FILENAME

# %%
# structure for storing the results

results = {'subject': {}}

for idx_subject in range(ds.NUM_SUBJECTS):

    results['subject'][idx_subject] = {'day': {}}

    for idx_day in range(ds.NUM_DAYS):

        results['subject'][idx_subject]['day'][idx_day] = {'posture': {}}

        for idx_valid_posture in range(ds.NUM_POSTURES):

            results['subject'][idx_subject]['day'][idx_day]['posture'][idx_valid_posture] = {
                # just classification metrics, no models or labels
                'training': {},  # classification metrics dictionary
                'validation': {},  # classification metrics dictionary
            }

# %%
for idx_subject, idx_day in itertools.product(
    range(ds.NUM_SUBJECTS), range(ds.NUM_DAYS)
):
    
    # ----------------------------------------------------------------------- #

    # print a header
    print(
        f"\n"
        f"------------------------------------------------------------------\n"
        f"SUBJECT\t{idx_subject + 1 :d}/{ds.NUM_SUBJECTS:d}\n"
        f"DAY\t{idx_day + 1 :d}/{ds.NUM_DAYS:d}\n"
        f"(all indices are one-based)\n"
        f"------------------------------------------------------------------\n"
        f"\n"
    )

    # ----------------------------------------------------------------------- #

    # load training data

    xtrain_list = []
    ytrain_list = []

    for idx_train_posture in range(ds.NUM_POSTURES):
        
        train_session_data_dict = ds.load_session(
            idx_subject, idx_day, idx_train_posture)

        emg_train = train_session_data_dict['emg']
        relabel_train = train_session_data_dict['relabel']
        gesture_counter_train = train_session_data_dict['gesture_counter']
        del train_session_data_dict

        # "_p" stands for single posture
        xtrain_p, ytrain_p, _, _ = ds.split_into_calib_and_valid(
            emg=emg_train,
            relabel=relabel_train,
            gesture_counter=gesture_counter_train,
            num_calib_repetitions=NUM_TRAIN_REPETITIONS,
        )
        del emg_train, relabel_train, gesture_counter_train

        # add to the lists
        xtrain_list.append(xtrain_p)
        ytrain_list.append(ytrain_p)
        del xtrain_p, ytrain_p

    # concatenate into single arrays
    xtrain = np.concatenate(xtrain_list, axis=1)
    ytrain = np.concatenate(ytrain_list, axis=0)
    del xtrain_list, ytrain_list

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
    
    del xtrain_pc, ytrain
    
    # ----------------------------------------------------------------------- #

    for idx_valid_posture in range(ds.NUM_POSTURES):

        # ------------------------------------------------------------------- #

        # print a header
        print(
            f"\n"
            f"--------------------------------------------------------------\n"
            f"VALIDATION ON POSTURE {idx_valid_posture + 1 :d}\n"
            f"--------------------------------------------------------------\n"
            f"\n"
        )

        # ------------------------------------------------------------------- #
        
        # load validation data

        valid_session_data_dict = ds.load_session(
            idx_subject, idx_day, idx_valid_posture)

        emg_valid = valid_session_data_dict['emg']
        relabel_valid = valid_session_data_dict['relabel']
        gesture_counter_valid = valid_session_data_dict['gesture_counter']
        del valid_session_data_dict

        # "_p" stands for single posture
        xtrain_p, ytrain_p, xvalid, yvalid = ds.split_into_calib_and_valid(
            emg=emg_valid,
            relabel=relabel_valid,
            gesture_counter=gesture_counter_valid,
            num_calib_repetitions=NUM_TRAIN_REPETITIONS,
        )
        del emg_valid, relabel_valid, gesture_counter_valid

        # ------------------------------------------------------------------- #

        # preprocessing

        xtrain_p = xtrain_p[:, ::DOWNSAMPLING_FACTOR]
        ytrain_p = ytrain_p[::DOWNSAMPLING_FACTOR]
        xvalid = xvalid[:, ::DOWNSAMPLING_FACTOR]
        yvalid = yvalid[::DOWNSAMPLING_FACTOR]

        xtrain_p_standardscaled = stdscaler_train.transform(xtrain_p.T).T
        xvalid_standardscaled = stdscaler_train.transform(xvalid.T).T
        del xtrain_p, xvalid
        xtrain_p_pc = pca_train.transform(xtrain_p_standardscaled.T).T
        xvalid_pc = pca_train.transform(xvalid_standardscaled.T).T
        del xtrain_p_standardscaled, xvalid_standardscaled

        # ------------------------------------------------------------------- #

        # MLP inference
        
        yout_train_p = learn.do_inference(xtrain_p_pc, mlp)
        yout_valid = learn.do_inference(xvalid_pc, mlp)
        del xtrain_p_pc, xvalid_pc

        metrics_train_p = good.compute_classification_metrics(ytrain_p, yout_train_p)
        metrics_valid = good.compute_classification_metrics(yvalid, yout_valid)

        print("\n\n")
        print("On training repetitions:")
        print(metrics_train_p)
        print("On validation repetitions:")
        print(metrics_valid)
        print("\n\n")
        
        # ------------------------------------------------------------------- #

        # store results
        results['subject'][idx_subject]['day'][idx_day]['posture'][idx_valid_posture]['training'] = metrics_train_p
        results['subject'][idx_subject]['day'][idx_day]['posture'][idx_valid_posture]['validation'] = metrics_valid
        
        # save to file
        # save the updated results dictionary after each validation
        results_outer_dict = {'results': results}
        Path(RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE_FULLPATH, 'wb') as f:
            pickle.dump(results_outer_dict, f)

# %%



