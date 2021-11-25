import numpy as np
import sklearn as sk
import pylab as plt
import pandas as pd
import os
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier

datadir = '/Users/ANONYMOUS/tmp/data/UCI/data'

def get_uci_info():
    info = []
    for idx, dataset in enumerate(sorted(os.listdir(datadir))):
        if not os.path.isdir(datadir + "/" + dataset):
            continue
        if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
            continue
        dic = dict()
        for k, v in map(lambda x: x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
            dic[k] = v
        c = int(dic["n_clases="])
        d = int(dic["n_entradas="])
        n_train = int(dic["n_patrons_entrena="])
        n_val = int(dic["n_patrons_valida="])
        n_train_val = int(dic["n_patrons1="])
        n_test = 0
        if "n_patrons2=" in dic:
            n_test = int(dic["n_patrons2="])
        n_tot = n_train_val + n_test

        p = dict(
            name=dataset,
            n=n_tot,
            num_classes=c,
            num_features=d
        )

        info.append(p)

    return pd.DataFrame(info)


def load_uci(dataset: str):
    dic = dict()
    for k, v in map(lambda x: x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test

    def load_dat(sub_name):
        f = open(datadir + '/' + dataset + "/" + sub_name, "r").readlines()[1:]
        X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
        y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))
        return X, y

    X, y = load_dat(dic["fich1="])
    if n_test > 0:  # if a test set is provided, add its samples too.
        X2, y2 = load_dat(dic["fich2="])
        X = np.concatenate((X, X2))
        y = np.concatenate((y, y2))

    fold = list(map(lambda x: list(map(int, x.split())),
                    open(datadir + "/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))
    # len(fold) == 8. Includes 4 cross-val splits.

    p = dict(n=n_tot,
             num_classes=c,
             num_features=d,
             X=X,
             y=y,
             train_fold=fold[0],
             test_fold=fold[1],
             folds=fold,
             )
    return p

