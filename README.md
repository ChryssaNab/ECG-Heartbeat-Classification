# Classifying Pathological Heartbeats from <br > ECG Signals

### [**Contents**](#)
1. [Project Description](#descr)
1. [Setup](#setup)
2. [Data Configuration](#dataset)
3. [Execution](#execution)
4. [Pre-training](#pre_training)
5. [Baseline Individual Classifiers](#baseline)
6. [Fine-tuning Classifiers](#fine_tuning)
7. [Team](#team)

---

### [**Project Description**](#) <a name="descr"></a>

<!---
In this project, we train deep Convolutional Neural Networks (CNNs) to perform binary classification of ECG beats to normal and abnormal. We use transfer learning in order to build models that are fine-tuned on specific patients’ data, after pre-training a generic network on a set of different ECGs selected from the MIT-BIH arrhythmia database [[1]](#1). We then compare the performance of the fine-tuned networks against that of individual networks, which are trained only on the ECG data of a single patient, in order to evaluate the overall efficacy of transfer learning on the given problem.
-->

In this project, we train deep Convolutional Neural Networks (CNNs) for binary classification of ECG beats to normal and abnormal. Initially, we pre-train a generic network on a collection of ECGs sourced from the MIT-BIH arrhythmia database [[1]](#1). Subsequently, we fine-tune models for each individual patient separately. Finally, we evaluate the performance of the fine-tuned networks against individual models trained from scratch solely on the ECG data of a single patient. This evaluation aims to assess the overall effectiveness of transfer learning and pre-trained knowledge for the given task.

The current project was implemented in the context of the course "Machine Learning" taught by [Prof. Herbert Jaeger](https://scholar.google.de/citations?hl=de&user=0uztVbMAAAAJ&view_op=list_works&sortby=pubdate) at [University of Groningen](https://www.rug.nl/?lang=en). For a comprehensive overview of the methodology and final results, please refer to the [Report](https://github.com/ChryssaNab/ECG-Heartbeat-Classification/blob/main/report/Heartbeat_Classification_ECG.pdf).

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

Download the dataset from https://www.kaggle.com/datasets/mondejar/mitbih-database and copy the contents to the parent folder under a directory named *dataset/mitbih_database*.

---

### [**Execution**](#) <a name="execution"></a>
The primary execution script for the entire project is the *main.py* file within the *src/* directory. The possible arguments for configuring and training the models are specified within the *opts.py* script. To view usage information run the following command:

``` shell
$ python3 src/main.py -h
```

The subsequent arguments may require modification:

> --data_path: The directory under which the dataset lies.
> 
> --output_path: The folder where the checkpoints and log files are created.
> 
> --selected_patients_fine_tuning: This pertains to the list of patients earmarked for the experiments. Only applies to the baseline individual models and fine-tuning models that target certain patients.
> 
> --weight_decay: The weight decay hyperparameter of the optimizer
> 
> --n_epochs: The maximum number of epochs for training
> 
> --batch_size: The batch size used for training.
> 
> --learning_rate: The initial learning rate
> 
> --weighted_sampling: Indicates whether weighted sampling is enabled.

Note that for the _individuals_ and _fine-tuning_ phases, the parameters `--batch_size`, `--learning_rate`, and `--weighted_sampling` are determined through a grid-search approach for each patient separately.

---

### [**Pre-training**](#) <a name="pre_training"></a>

To perform supervised pre-training on all patients using the default settings, run the following command:

``` shell
$ python3 src/main.py --state pre-training
```

Executing this command initiates the pre-training phase, using the hyperparameters specified in the *opts.py* script. Resultantly, a folder named *output/* will be generated within the parent directory, containing model state checkpoints for each epoch and three log files encompassing metrics like loss, accuracy, and other evaluations for training, validation, and test sets. The output folder location can be specified using the `--output_path` flag, while other hyperparameters in *opts.py* can be adjusted accordingly.

---

### [**Baseline Individual Classifiers**](#) <a name="baseline"></a>

In our baseline models, we train a CNN model from scratch for each patient individually, using fully-supervised mode, without integrating pre-trained knowledge. To initiate this process, execute the following command:

``` shell
$ python3 src/main.py --state individuals
```

Executing this command initiates experiments where we individually train a CNN model for each selected patient, employing the optimal hyperparameter set determined through grid-search. Consequently, a folder named *individuals/* will be generated within the existing *output/* directory, comprising one sub-folder per patient. Each patient's folder will contain identical files as those described in the pre-training section. The output folder location can be specified using the `--output_path` flag.

---

### [**Fine-tuning Classifiers**](#) <a name="fine_tuning"></a>

To leverage transfer learning and conduct fine-tuning for each individual patient, execute the following command:

``` shell
$ python3 src/main.py --state fine_tuning --pretrain_path ./output/save_<x>.pth
```

In this command, the `--pretrain_path` argument indicates the model checkpoint intended for fine-tuning. We retain the model checkpoint from the epoch with the lowest validation loss during pre-training. Replace \<x\> with the corresponding epoch number.

Executing this command initiates experiments where we optimize the pre-trained CNN model for each patient within our curated subset separately. This process generates the *fine_tuning/* folder within the existing *output/* directory, comprising individual folders for each patient. Each patient's folder contains identical files as those described in the pre-training section. You can designate the output folder using the `--output_path` flag.

---

## References
<a id="1">[1]</a> 
G. B. Moody and R. G. Mark (2001). The impact of the MIT-BIH Arrhythmia Database. *In IEEE Engineering in Medicine and Biology Magazine (pp. 45-50)*. DOI: 10.1109/51.932724.

---

### [**Team**](#) <a name="team"></a>

- [Chryssa Nampouri](https://github.com/ChryssaNab)
- [Philip Andreadis](https://github.com/philip-andreadis)
- Christodoulos Hadjichristodoulou
- Marios Souroulla
