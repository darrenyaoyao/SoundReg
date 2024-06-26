import os
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import torch.quantization
from pathlib import Path
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig

from .pytorch_utils import move_data_to_device
from .models import Cnn14, Cnn14_DecisionLevelMax
from .config import labels, classes_num


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


class AudioTagging(object):
    def __init__(self, model=None, checkpoint_path=None, device='cuda', data=None):
        """Audio tagging inference wrapper.
        """
        if not checkpoint_path:
            checkpoint_path='{}/panns_data/Cnn14_mAP=0.431.pth'.format(str(Path.home()))
        print('Checkpoint path: {}'.format(checkpoint_path))
        
        if not os.path.exists(checkpoint_path):
            create_folder(os.path.dirname(checkpoint_path))
            zenodo_path = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
            os.system('wget -O "{}" "{}"'.format(checkpoint_path, zenodo_path))

        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.labels = labels
        self.classes_num = classes_num

        # Model
        if model is None:
            self.model = Cnn14(sample_rate=32000, window_size=1024, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=self.classes_num)
        else:
            self.model = model

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])


        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

        self.model.eval()
        prep_config_dict = PrepareCustomConfig().set_non_traceable_module_names(["spectrogram_extractor", "logmel_extractor"])
        qconfig_mapping = get_default_qconfig_mapping('qnnpack')
        self.model = prepare_fx(self.model, qconfig_mapping, example_inputs=data, prepare_custom_config=prep_config_dict)

        data = move_data_to_device(data, self.device)
        self.model(data)
        self.model = convert_fx(self.model)



    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            output_dict = self.model(audio, None)

        logmel = output_dict['logmel'].data.cpu().numpy()
        embedding = output_dict['embedding'].data.cpu().numpy()
        spectrogram = output_dict['spectrogram'].data.cpu().numpy()

        return logmel, embedding, spectrogram


class SoundEventDetection(object):
    def __init__(self, model=None, checkpoint_path=None, device='cuda', interpolate_mode='nearest'):
        """Sound event detection inference wrapper.

        Args:
            model: None | nn.Module
            checkpoint_path: str
            device: str, 'cpu' | 'cuda'
            interpolate_mode, 'nearest' |'linear'
        """
        if not checkpoint_path:
            checkpoint_path='{}/panns_data/Cnn14_DecisionLevelMax.pth'.format(str(Path.home()))
        print('Checkpoint path: {}'.format(checkpoint_path))

        if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 3e8:
            create_folder(os.path.dirname(checkpoint_path))
            os.system('wget -O "{}" https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1'.format(checkpoint_path))

        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.labels = labels
        self.classes_num = classes_num

        # Model
        if model is None:
            self.model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=self.classes_num, interpolate_mode=interpolate_mode)
        else:
            self.model = model
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

    def inference(self, audio):
        print("Start")
        audio = move_data_to_device(audio, self.device)
        print("End Move data")

        with torch.no_grad():
            self.model.eval()
            print("Feed input")
            output_dict = self.model(
                input=audio, 
                mixup_lambda=None
            )
            print("Finish")

        framewise_output = output_dict['framewise_output'].data.cpu().numpy()

        return framewise_output
