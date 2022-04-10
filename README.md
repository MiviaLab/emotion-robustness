# Benchmarking deep networks for facial emotion recognition in the wild

Emotion recognition from face images is a challenging task that gained interest in recent years for its applications to business intelligence and social robotics.

Existing emotion recognition systems evaluate their accuracy on common benchmark datasets that do not extensively cover possible variations that face images undergo in real environments. Following on investigations carried out in the field of object recognition, weevaluated the robustness of state-of-the-art models for emotion recognition when their input is subjected to alterations caused by factors present in real-world scenarios.

![samples from RAF-DB-P](perturbations.gif)

## This repository

This repository contains the code for the paper [*Benchmarking deep networks for facial emotion recognition in the wild
 - A. Greco, N. Strisciuglio, M. Vento, V. Vigilante - Deep-Patterns Emotion Recognition in the Wild 2022*](https://link.springer.com/article/10.1007/s11042-022-12790-7)

If you use this code in your research, please cite this paper.

This framework is provided by the [MIVIA Lab](https://mivia.unisa.it), including dataset generation, training and evaluation code.

The repository includes the code for generating the corrupted and perturbed versions of the RAF-DB emotion dataset (RAF-DB-C, RAF-DB-P), that we use to evaluate model robustness and stability.

## Setup
Obtain the [RAF-DB dataset from the original authors](https://www.whdeng.cn/raf/model1.html) and extract it in the `RAF-DB` directory.
Make sure to have the images under `RAF-DB/basic/Image/` and the **distribution type** annotation under `RAF-DB/distribute_basic.csv`.

We used Python 3.6 on Ubuntu 18.04.
```
# Prepare the environment
python3 -m venv myenv
source myenv/bin/activate
python3 -m pip install wheel cython wand
python3 -m pip install -r requirements.txt

# Generate the corrupted and pertubed datasets RAF-DB-C and RAF-DB-P
mkdir corrupted_raf_dataset
python3 rafdb_aug_dataset.py export
mkdir perturbed_raf_dataset
python3 rafdb_perturb_dataset.py export
```

## Use
To evaluate the robustness of a network use
```
python3 eval_corrupted_rafdb.py checkpoint.hdf5
```
(replace `checkpoint.hdf5` with your checkpoint path)

To evaluate the stability (flip rate) use
```
python3 eval_perturbed_rafdb.py checkpoint.hdf5 > /tmp/predictions.txt
python3 eval_perturbed_rafdb2.py /tmp/predictions.txt
```

To train use `train_fer.py`