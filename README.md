# Ambulance Detect V1

Coded for the competition Pragyan.

## What it is?

Ambulance Saves Life. When Its gets Struck in Traffic, Bad Things can Happen. This Project Gives a Proposal to automatically Prioritize Traffic Lights for the Ambulance Route. 

## Implementation

1. Record Audio Clip From Microphone
2. Convert The waveform to Spectrogram image
3. Classify the Spectrogram Image Using CNN (Convolutional Neural Network)

#### The Model is Currently Trained to Classify 

- Ambulance Sound
- Firetruck Sound
- Traffic Sound

The Model is Trained With this kaggle Dataset: https://www.kaggle.com/vishnu0399/emergency-vehicle-siren-sounds

## How To Run the Model?

Install Dependencies: 

```
pip install -r requirements.txt
```

There are Two Files Provided: 

- **mic.py** - To run inference on Microphone
- **train.ipynb** - To train the model using above mentioned Dataset

A pretrained model is already provided in the **model** folder, **mic.py** can be run on that model without hassle