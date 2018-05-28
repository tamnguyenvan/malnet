# MalNet - Detect malware using Convolutional Neural Networks
By Tam Nguyen Van
# Introduction
The repository contains all source code for training and evaluate  malware detection with **MalNet**.
# Requirements
1. Python 3.6
2. Keras (2.0.8)
3. Tensorflow (1.2.0)
# Installation
1. Clone the repository to your local.
`git clone https://github.com/tamnguyenvan/malnet`
2. Install all requirements (use **virtualenv** is recommneded). Note: It just works on Python 3.6 (maybe 3.5 but I haven't tested).
- `pip install tensorflow==1.2.0` (CPU only) or `pip install tensorflow-gpu==1.2.0` (GPU)
- `pip install -r requirements.txt`
3. Make data directory. For example, make a direcotory called **data** in the root project directory. The data can be found at [here](https://drive.google.com/drive/folders/1zUXAb7JnwOiBtfBheQI6LDFu4EG_XZ-_). After downloading, extract it and put all data files into the data directory.
# Training
If you have accomplished the installation step correctly, almost done. We just need run `python train.py` for training with default parameters.
Some options:
- `--model` Use specific model. For now, just `malnet`, `et` and `rt` are available.
- `--batch-size` Set batch size to fit our memory. Default is 32.
- `--epochs` Number of epochs will be trained. Default is 5.

Please see source code for more details.
# Evaluate
After training, the model will be saved in **result/checkpoint**. We can evaluate this or use my pretrained model that can be found at [here](https://drive.google.com/file/d/1zD99s0L9l1eVPmSo9o6c3WgkZrpa2e2o). The directory must contain 3 files:
- `model.h5` Model weights.
- `model.json` Model graph.
- `scaler.pkl` Pickle binary file contains an object for preprocessing scaler.

In order to evaluating, just run `python src/eval.py`.
# Deployment
In this section, we will try to use the model to predict samples from the real. We provided a script for this in **src/test.py**. So all we need to do is just run `python src/test.py --input [path/to/sample/file]`.
# Contact
Tam Nguyen Van (tamvannguyen200795@gmail.com)
Any questions can be left as issues in this repository. You're are welcome all.
