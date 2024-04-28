#!/usr/bin/env python3
import signal
import sys
import librosa
import pyaudio
import numpy as np
import threading
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io.wavfile import write
import time
import panns_inference
from models import Cnn6, Cnn10
from panns_inference.inference import AudioTagging, SoundEventDetection, labels
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import wave


BUTTON_GPIO = 23 
RECORD_LED_GPIO = 27
TRIGGER_LED_GPIO = 22
record_led_high_sec = [0]
FS=32000
LEN=32000*2
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16, channels=1, rate=FS, input=True,
            input_device_index = 1, frames_per_buffer=2*LEN)
'''
checkpoint_path = '{}/panns_data/Cnn6_mAP=0.343.pth'.format(str(Path.home()))
model = Cnn6(sample_rate=FS, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=len(labels))
'''
checkpoint_path = '{}/panns_data/Cnn10_mAP=0.380.pth'.format(str(Path.home()))
model = Cnn10(sample_rate=FS, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=len(labels))

'''
buffer = stream.read(LEN, exception_on_overflow=False)
array = np.frombuffer(buffer, dtype=np.int16)
array = array.astype(np.float32)
array = array.reshape(1, -1)  # (batch_size, segment_samples)
'''
(audio, _) = librosa.core.load('golden.wav', sr=32000, mono=True)
array = audio[None, 0:64000]  # (batch_size, segment_samples)

device = 'cpu' # 'cuda' | 'cpu'
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device=device, data=array)

golden_embedding = np.ones([1, 512])
golden_logmel = np.ones([1, 201*64])
golden_spectrogram = np.ones([1, 513])
golden_lock = 0;
test_lock = 0;

def normalize(array):
    return array/32768.0

def data_saver(filename, data_frames):
    wf = wave.open(filename+'.wav','wb') # open .wav file for saving
    wf.setnchannels(1) # set channels in .wav file 
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16)) # set bit depth in .wav file
    wf.setframerate(FS) # set sample rate in .wav file
    wf.writeframes(b''.join(data_frames)) # write frames in .wav file
    wf.close() # close .wav file

def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)

def button_pressed_callback(channel):
    print("Button pressed!")
    GPIO.output(RECORD_LED_GPIO, True)
    t = threading.Timer(3.0, check_record_led)
    t.start()

def close_trigger_led():
    GPIO.output(TRIGGER_LED_GPIO, False)

def check_record_led():
    GPIO.output(RECORD_LED_GPIO, False)
    buffer = stream.read(LEN, exception_on_overflow=False)
    data_saver('golden', [buffer])
    (audio, _) = librosa.core.load('golden.wav', sr=32000, mono=True)
    array = audio[None, 0:64000]  # (batch_size, segment_samples)
    '''
    array = np.frombuffer(buffer, dtype=np.int16)
    array = array.astype(np.float32)
    array = normalize(array)
    np.save("golden.npy", array)
    array = array.reshape(1, -1)  # (batch_size, segment_samples)
    '''

    while test_lock:
        a = 0;
    print('------ Golden Audio tagging ------')
    print(time.time())
    golden_lock = 1;
    (logmel, embedding, spectrogram) = at.inference(array)
    logmel = logmel.reshape(1, -1)
    spectrogram = spectrogram.mean(axis=2).reshape(1, -1)
    golden_lock = 0;
    print(time.time())
    global golden_embedding
    golden_embedding = embedding
    global golden_logmel
    golden_logmel = logmel
    global golden_spectrogram
    golden_spectrogram = spectrogram


if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(RECORD_LED_GPIO, GPIO.OUT)
    GPIO.setup(TRIGGER_LED_GPIO, GPIO.OUT)
    GPIO.add_event_detect(BUTTON_GPIO, GPIO.FALLING, 
            callback=button_pressed_callback, bouncetime=100)
    
    signal.signal(signal.SIGINT, signal_handler)

    while 1:
        buffer = stream.read(LEN, exception_on_overflow=False)
        data_saver('test', [buffer])
        (audio, _) = librosa.core.load('test.wav', sr=32000, mono=True)
        array = audio[None, 0:64000]  # (batch_size, segment_samples)
        '''
        array = np.frombuffer(buffer, dtype=np.int16)
        array = array.astype(np.float32)
        array = normalize(array)
        np.save("test.npy", array)
        array = array.reshape(1, -1)  # (batch_size, segment_samples)
        print(array.shape)
        print(array.dtype)
        '''

        while golden_lock:
            a = 0;
        print('------ Audio tagging ------')
        print(time.time())
        test_lock = 1
        (logmel, embedding, spectrogram) = at.inference(array)
        logmel = logmel.reshape(1, -1)
        spectrogram = spectrogram.mean(axis=2).reshape(1, -1)
        test_lock = 0
        print(time.time())

        embedding_sim = cosine_similarity(embedding, golden_embedding) 
        print("embedding similarity", embedding_sim)
        spectrogram_sim = cosine_similarity(spectrogram, golden_spectrogram) 
        print("spectrogram similarity", spectrogram_sim)

        if embedding_sim > 0.94 and spectrogram_sim > 0.8:
            GPIO.output(TRIGGER_LED_GPIO, True)
            t = threading.Timer(3.0, close_trigger_led)
            t.start()

        
            


