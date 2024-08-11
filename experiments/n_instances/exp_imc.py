import os
import time
import gzip
import pickle
import warnings

import matplotlib.pyplot as plt

import numpy as np

from tqdm import TqdmSynchronisationWarning
warnings.simplefilter("ignore", TqdmSynchronisationWarning)

PATH_TO_EXP = ''
PATH_DATA = os.path.join(PATH_TO_EXP, 'data/n_instances')

PATH_ARCHIVE = os.path.join(PATH_DATA, "arch_imc")
if not os.path.isdir(PATH_ARCHIVE):
    os.mkdir(PATH_ARCHIVE)

filenames = {
    "input": "data.gz",
    "output": "results_imc.gz"
}

filename_input = os.path.join(PATH_DATA, filenames["input"])

filename_output = os.path.join(PATH_DATA, filenames["output"])

if os.path.exists(filename_output):
    mdttm = time.strftime("%Y%m%d_%H%M%S")
    os.rename(filename_output, os.path.join(PATH_ARCHIVE, "%s%s" % (mdttm, filenames["output"])))

from sgimc.utils import mc_split
from sgimc.utils import get_submatrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from scipy.sparse import coo_matrix


def mc_get_scores(R_true, R_prob):
    diff = np.sum((R_prob.data - R_true.data) ** 2)
    norm = np.sum(R_true.data ** 2)

    return {"relative_error": diff / norm}

random_state = np.random.RandomState(0x0BADCAFE)

from sklearn.model_selection import ParameterGrid

grid_dataset = ParameterGrid({
    "train_size": np.arange(0.001, 0.02, 0.0015),
    "n_splits": [5],
})

grid_model = ParameterGrid({
    "C_lasso": [0.0],
    "C_group": [0.0],
    "C_ridge": [1e0],
    "lamb": [1e-4, 1e-3, 1e-2],
    "rank": [25]
})

from sgimc.utils import load, save

X, Y, R_full, R_noisy = load(filename_input)

dvlp_size, test_size = 0.9, 0.1

ind_dvlp, ind_test = next(mc_split(R_full, n_splits=1, random_state=random_state,
                                   train_size=dvlp_size, test_size=test_size))

R_test = get_submatrix(R_full, ind_test)

from sgimc import SparseGroupIMCRegressor

from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import train_test_split

results = []
for par_dtst in tqdm(grid_dataset):
    
    # prepare the train dataset: take the specified share from the beginnig of the index array
    ind_train_all, _ = train_test_split(ind_dvlp, shuffle=False, random_state=random_state,
                                        test_size=(1 - (par_dtst["train_size"] / dvlp_size)))

    # Run the experiment: the model 
    for par_mdl in grid_model:  # tqdm.tqdm(, desc="cv %02d" % (cv,))
        # set up the model
        C_lasso, C_group, C_ridge = par_mdl["C_lasso"], par_mdl["C_group"], par_mdl["C_ridge"]
        lamb = par_mdl["lamb"]
        imc = SparseGroupIMCRegressor(par_mdl["rank"], n_threads=8, random_state=42,
                                      C_lasso=C_lasso, C_group=C_group, C_ridge=C_ridge)

        # fit on the whole development dataset
        R_train = get_submatrix(R_noisy, ind_train_all)
        imc.fit(X, Y, R_train, sample_weight = np.ones(R_train.nnz) / lamb)

        # get the score
        prob_full = imc.predict(X, Y)
        prob_test = get_submatrix(prob_full, ind_test)
        scores_test = mc_get_scores(R_test, prob_test)

        # run the k-fold CV
        # splt = ShuffleSplit(**par_dtst, random_state=random_state)
        splt = KFold(par_dtst["n_splits"], shuffle=True, random_state=random_state)
        for cv, (ind_train, ind_valid) in enumerate(splt.split(ind_train_all)):

            # prepare the train and test indices
            ind_train, ind_valid = ind_train_all[ind_train], ind_train_all[ind_valid]
            R_train = get_submatrix(R_noisy, ind_train)
            R_valid = get_submatrix(R_noisy, ind_valid)

            # fit the model
            imc = SparseGroupIMCRegressor(par_mdl["rank"], n_threads=8, random_state=42,
                                           C_lasso=C_lasso, C_group=C_group, C_ridge=C_ridge)
            imc.fit(X, Y, R_train, sample_weight = np.ones(R_train.nnz) / lamb)

            # compute the class probabilities
            prob_full = imc.predict(X, Y)  # uses own copies of W, H
            prob_valid = get_submatrix(prob_full, ind_valid)

            scores_valid = mc_get_scores(R_valid, prob_valid)

            # record the results
            results.append({"train_size": par_dtst["train_size"],
                            "lamb": par_mdl["lamb"],
                            "cv": cv,
                            "val_score": scores_valid["relative_error"],
                            "test_score": scores_test["relative_error"]}
                          )
        # end for
    # end for
# end for

# Save the results in a pickle

with gzip.open(filename_output, "wb+", 4) as fout:
    pickle.dump(results, fout)