# Introduction

The folder details the Jigsaw Sudoku experiments. The dataset is created from an adapted version of the Sudoku creation script (https://github.com/locuslab/optnet/blob/master/sudoku/create.py) together with the standard MNIST dataset, converted to PNG (thanks to work at https://github.com/myleott/mnist_png).


The dataset used to train are in `data_one`.

# Generating the Dataset

To generate the dataset, run the following command:

```
./create.py --nSamples=70000
```

# Training

To train the data results in Table 2 and Figure 5, please run the following commands:

```
for i in $(seq 1 30); do ./train_nodef.py -r $i; done
for i in $(seq 1 30); do ./train_defense.py -r $i -e 2.0 -m "./model_def_eps_2"; done
for i in $(seq 1 30); do ./train_defense.py -r $i -e 10.0 -m "./model_def_eps_10"; done
for i in $(seq 1 30); do ./train_defense.py -r $i -e 100.0 -m "./model_def_eps_100"; done
for i in $(seq 1 30); do ./train_defense.py -r $i -e 200.0 -m "./model_def_eps_200"; done
```

# Attacking

To reproduce the results in Table 1, please run the following commands

AllZeroRowCol
```
./attack_num_cond_fn.py --train-sz=30 --loss-function=ZeroRowLoss --epsilon=0 -lr 4000000
```

ZeroSingularValue
```
./attack_num_cond_fn.py --train-sz=30 --loss-function=ZeroSingularValue -lr 4000000
```

ConditionGrad
```
./attack_num_cond_fn.py --train-sz=30 --loss-function=ConditionNumberLoss
```

To run against a model with the defense implemented, run with the following command:

```
./attack_num_cond_fn.py --train-sz=30 --loss-function=ZeroRowLoss --epsilon=2 -lr 4000000
```
