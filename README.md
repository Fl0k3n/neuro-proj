# EEG-based Schizophrenia detection

## 1. Introduction

what is schizophrenia, what is eeg, explain this auditory task, what are N100 and P200,...

## 2. Methods

We implemented two approaches to solving this problem:
- a classical one that uses feature extraction based on studies on schizophrenia,
- a deep learning one that attempts to solve the problem without any domain knowledge.

### 2.1. Dataset

EEG data were recorded using a 64-channel system during the passive listening task, involving the presentation of 100 sounds: 1,000 Hz, 80 dB sound pressure level (SPL), 50 ms duration tones. Individual segments (trials - one per tone presented) were created from continuous recordings with a duration of 3000 ms, time-locked to tone onset with 1500 ms pre-stimulus and 1500 ms post-stimulus. Basic preprocessing was performed prior to uploading the dataset, the preprocessing included filtering, removing outliers, removing muscle and white noise artifacts. The dataset containts samples for 81 subjects - 49 schizopchrenic and 32 healthy. For each subject, about 100 trials were available.


### 2.2. Classical approach

For the classical approach, we used ERP (Event-related potentials) of the EEG, which are signals built from averaging signals from all trials for a given electrode. 

For our experiments, we used 5 middle-scalp electrodes (Fz, FCz, Cz, CPz, Pz) and constructed features from ERP data. For all electrodes, slopes of signal transition before N100, between N100 and P200, and after P200 responses were computed. We also computed average signal values for time ranges of those responses (the area between the yellow regions below). For FCz and Cz electrodes, we also used amplitudes and latencies of N100 and P200 peaks, as illustrated below. We ended up with 33 features and we trained random forest classifier on that.

<p align="center">
  <img src="./imgs/rf_features.png"/>
</p>

### 2.3. Deep learning approach

For a deep learning approach, we used individual trial signals, also for the same 5 electrodes. First, we normalized signals for each electrodes, then we selected a time window from -100 to 400 ms time-locked to sound onset, comprising the expected latencies of N100 and P200 peaks. Then we subsampled those 512 samples ending up with 256 samples per trial.

We implemented the model as proposed in [1]. First, we used a temporal convolution blocks built of the following layers:
- 1D convolution applied individually to each electrode, followed by batch normalization and dropout,
- same as above but without dropout,
- 1D max polling, which reduces time dimension twice

We used 3 such blocks, then we used a single block of spatial convolutions (applied on different electrodes, but for the same time offset), followed by batch normalization. Then we used a 2D convolution, followed by the dropout and flatten layers. Finally, we had 3 feed-forward networks with dropouts that perform the final classification. The whole architecture is depicted below.

<p align="center">
  <img src="./imgs/sznet.png"/>
</p>

The temporal blocks were meant to find correlations between points in the temporal course of the EEG data, spatial convolutions were meant for correlating features from different electrodes. 2D convolutions were used to ensure that patterns are learned from both dimensions.

## 3. Results 

We trained a random forest classifier with 100 estimators, Gini criterion, maximum number of features set to 15 and a maximum depth of 2. We tested other configurations, mostly changing maximum number of features and number of classifiers, but this one gave the best results.

We achieved the following results:

| metric   | score  |
| -------- | -----  |
| accuracy | 0.65   |
| precision| 0.79   |
| recall   | 0.66   |
| f1       | 0.7    |
| specificity | 0.73|


As for the neural network, we tested mutliple parameters, we also tried applying stronger regularization techniques (larger dropout, L2 penalty) and reducing layer sizes or even removing 1 feed-forward layer, but the network was overfitting nevertheless. We didn't obtain any valid results and accuracy was about 0.55. Plots below illustrate one of the trainig processes.

<p align="center">
  <img src="./imgs/sznet_training.png"/>
</p>

## 4. Discussion

We didn't manage to reproduce the results of [1], where the same network architecture managed to achieve almost 0.8 accuracy. The probable reason is due to the size of the dataset, authors of [1], in addition to the dataset used by us, had about twice as much data that they collected themselves. They also had the same proportion of healthy and schizophrenic subjects, whereas we had about 70% schizophrenic subjects. The authors didn't comment on the reason of collecting additional data, we can only guess that results were significantly worse without that.

The simpler, classical approach with random forest performed better and achieved about 0.07 worse accuracy (0.65) than the accuracy obtained in [1], but again, results are not directly comparable as datasets were different. We can conclude that the features used are indeed helpful in predicting whether the subject suffers from schizophrenia. The most important features are related to P200 response, especially signal transitions around P200 peak, as depicted below.

<p align="center">
  <img src="./imgs/feature_importances.png"/>
</p>


## References

[1] From Sound Perception to Automatic Detection of Schizophrenia: An EEG-Based Deep Learning Approach, Barros C. et al.

[2] https://www.kaggle.com/datasets/broach/button-tone-sz