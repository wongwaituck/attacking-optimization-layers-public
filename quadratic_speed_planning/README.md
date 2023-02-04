# Speed Profile Planning

## Creating the Dataset

Simply run the following command to create the dataset. (Note that the dataset is already provided in `out`).

```
./create.py
```

## Training

To train the models as per the reported results, run the following commands

```
for i in $(seq 1 30); do ./models.py --cuda --batch-sz=1000 --epochs=30 --epsilon=10 --seed=$i; done # train the defended model with B=10 - repeat with B=2,100,200
for i in $(seq 1 250); do ./models.py --cuda --batch-sz=1000 --epochs=30 --epsilon=0 --seed=$i; done # train the undefended model i.e. B=10

```
Note that more seeds were used for the undefended model since the model would fail to train in some instances, and we need about 250 seeds to achieve 30 successful runs.

## Attacking

To run the attacks on the models, run the following command:

```
./attack.py --eps=200 --cuda # where attack a model with eps/B=200
```

We use seed=250 as the attacked model for the undefended model, and seed=1 for the attacked model for the defended models.

# Inequality Violation

To reproduce the inequality violations, run the following:

```
./negative.py --epsilon 0 --cuda # do this for each of the B/epsilon values you want to collate
```

The collated data is in `constraintfailure_all.csv`.