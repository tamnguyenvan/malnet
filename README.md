# MalNet - Detect malware using Convolutional Neural Networks
By Tam Nguyen Van
# Introduction
Malware detection using Convolutional Neural Networks.
# Requirements
1. Python >=3
2. Keras (>=2.0.8)
3. Tensorflow (>=1.15)
# Installation
1. Clone the repository to your local.
`git clone https://github.com/tamnguyenvan/malnet`
2. Install all requirements (**virtualenv** is recommneded).
- `pip install tensorflow==1.15` (CPU only) or `pip install tensorflow-gpu==1.15` (GPU)
- `pip install -r requirements.txt`
3. Download Ember dataset [here](https://pubdata.endgame.com/ember/ember_dataset_2018_2.tar.bz2). You can go to their [home page](https://github.com/endgameinc/ember) for more details. Extract to wherever you like.
4. Extract features by running: `python create_data.py --data_dir PATH_TO_DATA_DIR`. See `create_data.py` for the details. After that, some `.dat` file should be created in the same directory.
# Training model
Almost done, just run `python train.py --data_dir PATH_TO_DATA_DIR` for training. Show help to see additional options.

# Evaluate model
In case you want to regenerate validation result, run `python eval.py --data_dir PATH_TO_DATA_DIR--model_path MODEL_PATH --scaler_path SCALER_PATH`. Again, show help to see options.

# Deploy
Let's have some fun. We will try the pretrained model on real PE files. Download your PE file then run `python test.py --input_file INPUT_FILE --model_path MODEL_PATH`.

# Contact
Tam Nguyen Van (tamvannguyen200795@gmail.com)
Any questions can be left as issues in this repository. You're are welcome all.
