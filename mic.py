from rich.console import Console
import argparse
import pyfiglet

import pyaudio
import os
import numpy as np
import wave
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
CHUNK = 1050
FORMAT = pyaudio.paInt16
CHANNELS = 1
OUTPUT_SR = RATE = 22050*2
MODEL_SR = 22050
DUR = RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "output.wav"
HOST_API = pyaudio.paMME #MME (default)
labels = ['ambulance', 'firetruck', 'traffic']
frames = []


p = pyaudio.PyAudio()
console = Console()
model = tf.keras.models.load_model('./model/v1')
#plt.ion()

def list_devices():
    devlist = []
    devcount = p.get_host_api_info_by_type(HOST_API)['deviceCount']
    api_idx = p.get_host_api_info_by_type(HOST_API)['index']
    # print(devcount)
    for i in range(devcount):
        dev = p.get_device_info_by_host_api_device_index(api_idx, i)
        if(dev['maxInputChannels'] == 0):
            continue
        devlist.append(dev)
    for i, item in enumerate(devlist):
        console.print(str(i+1) + ') ' + item['name'])
    ip = int(console.input('Enter option: '))
    return devlist[ip-1]['index']


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros(
        [MODEL_SR*DUR] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def decode_audio(audio_binary):
    audio, rate = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def write_wav(data):
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()


def main():
    try:
      img = plt.imshow(np.zeros((2, 2)))
      global frames
      stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=int(list_devices()))
      while True:
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            # clear_output(wait=True)
            write_wav(frames)
            audio_binary = tf.io.read_file(WAVE_OUTPUT_FILENAME)
            waveform = decode_audio(audio_binary)
            waveform = tfio.audio.resample(waveform, RATE, MODEL_SR)
            # print(tensor)
            print(waveform.shape)
            spectrogram = get_spectrogram(waveform)
            #plt.imshow(spectrogram)
            img.set_data(spectrogram)
            img.autoscale()
            plt.show(block=False)
            plt.pause(0.1)
            spectrogram = tf.expand_dims(spectrogram, -1)
            # spectrogram = np.reshape(spectrogram, (171, 129, 1))
            # print(spectrogram.shape)
            test = [spectrogram.numpy()]
            pred = model.predict(np.array(test))
            console.print(pred[0])
            idx = np.argmax(pred)
            console.print(labels[idx])
            frames = []

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    result = pyfiglet.figlet_format("Ambulance Detect V 1", font = "slant"  ) 
    console.print(result)
    main()
