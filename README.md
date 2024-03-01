# Classifying Pathological Heartbeats from <br > ECG Signals

### [**Contents**](#)
1. [Project Description](#descr)
1. [Setup](#setup)
2. [Data Configuration](#dataset)
3. [Execution](#execution)
4. [Pre-training](#pre_training)
5. [Baseline Individual Classifiers](#baseline)
6. [Fine-tuning](#fine_tuning)
7. [Hyperparameter Tuning](#parameter_tuning)
8. [Team](#team)

---

### [**Project Description**](#) <a name="descr"></a>

In this project, we train deep Convolutional Neural Networks (CNNs) to perform binary classification of ECG beats to normal and abnormal. We use transfer learning in order to build models that are fine-tuned on specific patients’ data, after pre-training a generic network on a set of different ECGs selected from the MIT-BIH arrhythmia database. We then compare the
performance of the fine-tuned networks against that of individual networks, which are trained only on the ECG data of a single patient, in order to evaluate the overall efficacy of transfer learning on the given problem.

---

### [**Setup**](#) <a name="setup"></a>

**1.** We assume that Python3 is already installed on the system. This code has been tested on Python version 3.10, though it should also be compatible with earlier versions.

**2.** Clone this repository:

``` shell
$ git clone https://github.com/ChryssaNab/Machine_Learning_RUG.git
$ cd Machine_Learning_RUG
```

 **3.** Create a new Python environment and activate it:

``` shell
$ python3 -m venv env
$ source env/bin/activate
```

**4.** Modify the *requirements.txt* file: 

> If your machine **does NOT support** CUDA, add the following line at the top of the *requirements.txt* file:
>> --extra-index-url https://download.pytorch.org/whl/cpu
>
> If your machine **does support** CUDA, add the following line instead, replacing **115** with the CUDA version your machine supports:
>> --extra-index-url https://download.pytorch.org/whl/cu115

**5.** Install necessary requirements:

``` shell
$ pip install wheel
$ pip install -r requirements.txt
```

---

### [**Data Configuration**](#) <a name="dataset"></a>

Download the dataset from https://www.kaggle.com/datasets/mondejar/mitbih-database and copy the contents to the parent folder of this directory under a directory named *dataset/mitbih_database*.

---

### [**Execution**](#) <a name="execution"></a>
The primary execution script resides in the *main.py* file within the *src/* directory. To view the potential arguments for executing this script, use the following command:

``` shell
$ python3 src/main.py -h
```

These arguments are specified within the *opts.py* script. The subsequent arguments may require modification:

> --data_path: The directory under which the dataset lies. Needs to be specified if the dataset is not copied to the directory named above.
> 
> --output_path: The folder where the checkpoints and log files will be created (default="./output/").
> 
> --selected_patients_fine_tuning: This pertains to the list of patients earmarked for the experiments. Only applies to the baseline individual models and fine-tuning models that target certain patients.

---

### [**Pre-training**](#) <a name="pre_training"></a>

To perform supervised pre-training on all patients using the default settings, run the following command:

``` shell
$ python3 src/main.py --state pre-training [--args]
```

Executing this command initiates the pre-training phase, using the hyperparameters specified in the *opts.py* script. Resultantly, a folder named *output/* will be generated within the parent directory, housing model state checkpoints for each epoch and three log files encompassing metrics like loss, accuracy, and other evaluations for training, validation, and test sets. The output folder location can be specified using the `--output_path` flag, while other parameters in *opts.py* can be adjusted accordingly.

---

### [**Baseline Individual Classifiers**](#) <a name="baseline"></a>

In our baseline models, we individually train a CNN model from scratch for each patient in a fully-supervised mode, without incorporating pre-trained knowledge. To do this, run the following command:

``` shell
$ python3 src/main.py --state individuals [--args]
```

Executing this command initiates experiments where we individually train the CNN for each selected patient, employing the optimal parameter set determined through grid search. Consequently, a directory named *individuals/* will be generated within the existing *output/* folder, comprising one sub-folder per patient. Each patient's folder will contain identical files as those described in the pre-training section. The output folder location can be specified using the `--output_path` flag.

---

### [**Fine-tuning**](#) <a name="fine_tuning"></a>

To leverage transfer learning and conduct fine-tuning for each individual patient, execute the following command:

``` shell
$ python3 src/main.py --state fine_tuning --pretrain_path ./output/save_<x>.pth [--args]
```

In this command, the `--pretrain_path` argument indicates the model checkpoint intended for fine-tuning. We retain the model checkpoint from the epoch with the lowest validation loss during pre-training. Replace <x> with the corresponding epoch number.

Executing this command initiates experiments where we fine-tune the top-performing CNN produced during the pre-training phase for each patient within our curated subset. This process generates the *fine_tuning/* folder within the *output/* directory, comprising individual folders for each patient. Each patient's folder contains identical files as those described in the pre-training section. You can designate the output folder using the `--output_path` flag.

---

### [**Hyperparameter Tuning**](#) <a name="parameter_tuning"></a>

It is possible to tune the parameters of the experiments from the command line. However, not all parameters defined in opts.py can be changed without unpredictable results. The list of tunable parameters is:

- weight_decay: The value of the weight decay parameter of the optimizer.
- n_epochs: The maximum number of epochs for which the experiment will run.
- batch_size: The batch size used for training
- learning_rate: The initial learning rate
- weighted_sampling: Whether weighed sampling is enabled or not

Note that for the _individuals_ and _fine-tuning_ phases, setting the parameters _batch\_size_, _learning\_rate_ and _weighted\_sampling_ will have no effect, since they will be overriden by the optimal paramer set for each patient.

---

### [**Team**](#) <a name="team"></a>

- [Chryssa Nampouri](https://github.com/ChryssaNab)
- [Philip Andreadis](https://github.com/philip-andreadis)
- Christodoulos Hadjichristodoulou
- Marios Souroulla
