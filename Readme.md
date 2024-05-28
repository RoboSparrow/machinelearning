This is tutorial stuff I collected while looking into machinelearning

----

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

---

>Loss is a measure of performance of a model. The lower, the better. When learning, the model aims to get the lowest loss possible.
>The target for multi-class classification is a one-hot vector, meaning it has 1 on a single position and 0’s everywhere else.

 * https://en.wikipedia.org/wiki/Artificial_neural_network
 * https://en.wikipedia.org/wiki/Feedforward_neural_network


 * https://www.guru99.com/machine-learning-tutorial.html

```
     1. Define a question
     2. Collect data
     3. Visualize data
 --> 4. Train algorithm
 |   5. Test the Algorithm
 |   6. Collect feedback
 |   7. Refine the algorithm
 --- 8. Loop 4-7 until the results are satisfying
     9. Use the model to make a prediction
```
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


---

>Loss is a measure of performance of a model. The lower, the better. When learning, the model aims to get the lowest loss possible.
>The target for multi-class classification is a one-hot vector, meaning it has 1 on a single position and 0’s everywhere else.

 * https://en.wikipedia.org/wiki/Artificial_neural_network
 * https://en.wikipedia.org/wiki/Feedforward_neural_network


 * https://www.guru99.com/machine-learning-tutorial.html

```
     1. Define a question
     2. Collect data
     3. Visualize data
 --> 4. Train algorithm
 |   5. Test the Algorithm
 |   6. Collect feedback
 |   7. Refine the algorithm
 --- 8. Loop 4-7 until the results are satisfying
     9. Use the model to make a prediction
```
