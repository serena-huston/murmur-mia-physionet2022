from cnn_segmentation.cnn_data import CNNData
from cnn_segmentation.cnn_data_preprocessing import CNNDataPreprocessing
from cnn_segmentation.unet_segmentation import * 
import numpy as np
from tqdm import trange
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import ConcatDataset, DataLoader

from springer_segmentation.train_segmentation import create_segmentation_array


MODEL_PATH = "cnn_segmentation/models/cnn_segmentation_model_weights_2016.pt"

def train_cnn_segmentation(recordings, annotations, recording_freq=4000, feature_freq=50):
    cnn_dataset = get_cnn_data(recordings, annotations, recording_freq=recording_freq, feature_freq=feature_freq)
    cnn_dataset = ConcatDataset(cnn_dataset)
    trainloader = DataLoader(dataset=cnn_dataset, batch_size=1, shuffle=True)
    set_up_model()
    fitted_model = fit_model(trainloader)
    return fitted_model.state_dict()

def get_cnn_data(recordings, annotations, recording_freq=4000, feature_freq=50):
    cnn_data = [] 
    for rec_idx in trange(len(recordings)):
        full_recording = recordings[rec_idx]

        if annotations[rec_idx].shape[0] == 3 and annotations[rec_idx].shape[1] != 3: # hacky workaround to hackier data handling
            annotation = annotations[rec_idx].T
        else:
            annotation = annotations[rec_idx]

        if annotation.shape[0] <= 1:
            continue
        clipped_recording, segmentations = create_segmentation_array(full_recording,
                                                                    annotation,
                                                                    recording_frequency=recording_freq,
                                                                    feature_frequency=recording_freq)
        try:
            dp = CNNDataPreprocessing(clipped_recording[0], segmentations[0]-1, recording_freq)
            
            x_patches = dp.extract_env_patches()
            y_patches = dp.extract_segmentation_patches()
            cnn_data.append(CNNData(np.array(x_patches), np.array(y_patches)))
        except:
            continue 
    return cnn_data

def set_up_model():
    global model, optimiser, criterion 
    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()


def fit_model(trainloader, epochs=8, patience=5):
    model.train(True)

    for epoch in range(epochs):
        model.train()
        for x,y in trainloader:
            optimiser.zero_grad()
            yhat = model(x[0])
            loss = criterion(torch.t(yhat), y[0])
            loss.backward()
            optimiser.step()
    
    return model

 