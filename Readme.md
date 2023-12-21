# Libraries

* https://www.tensorflow.org/

---

 * https://github.com/glouw/tinn
 * https://github.com/attractivechaos/kann
 * https://github.com/codeplea/genann

# Installation

* tensorflow

```bash
cd ./TensorFlow

python3 -m venv ./.venv
source ./.venv/bin/activate
pip3 install -r ./requirements.txt
```

* Submodules

```bash
git submodule init
git submodule update
```

# Notes

## Worflow

- load dataset(s) (train data)
- build  model
- compile model (optimizer, loss, metrics)
- train model (train data, epochs)
- [loss, acc] = evaluate model
- make prediction (raw data)

A model needs a loss function and an optimizer for training.

classification:  select a class from a list
regression: predict value

---

## Public Datasets

 * https://archive.ics.uci.edu/datasets
