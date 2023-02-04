# Synthetic Data

This folder contains the code for the synthetic data tests. It models what was done in https://github.com/locuslab/qpth/blob/master/example-cls-layer.ipynb which can be applied to classification tasks, see https://github.com/locuslab/optnet/tree/master/cls for an application. For this synthetic data, we focus on the abstract task rather than the concrete application, primarily to showcase the effect of the attack and defense on different shapes of the constraint matrix.

## Training

To train the model, run the following (for 40x50):

```
for i in $(seq 1 10); do ./eq_train_all_cond.py -r $i -eps 200 -e 40 -cuda 0; done # for 40x50, B=200
for i in $(seq 1 10); do ./eq_train_all_cond.py -r $i -eps 0 -e 50 -cuda 0; done # for 50x50, B=0 (undefended model)
```

## Attack

To run all attacks methods, run the following:

```
for i in $(seq 1 10); do ./eq_attack_all.py -r $i -eps 2 -e 40 -cuda 0; done # attack all trained 40x50 models with the defense B=2
for i in $(seq 1 10); do ./eq_attack_all.py -r $i -eps 0 -e 50 -cuda 0; done # attack all trained 50x50 undefended models (B=0)
```