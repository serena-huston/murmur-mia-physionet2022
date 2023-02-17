
from cnn_segmentation.cnn_data_preprocessing import CNNDataPreprocessing
from cnn_segmentation.unet_segmentation import * 
from springer_segmentation.upsample_states import * 
import numpy as np
import torch 
from librosa import resample

PATCH_SIZE = 64
STRIDE = 8 

def run_cnn_segmentation(audio_data,
                         models,
                         Fs = 4000,
                         ):

    model = UNet()
    model.load_state_dict(models)

    model.eval()
    predictions = [] 

    # for recording in audio_data:

    dp = CNNDataPreprocessing(audio_data, np.array([]), Fs)
    x_patches = dp.extract_env_patches()
    window_predictions = []
    for patch in x_patches: 
        window_probabilities = model(torch.tensor(patch, requires_grad=True).type(torch.float32))
        window_predictions.append(make_segmentation_predictions(window_probabilities))
    combined_windows = make_sample_prediction(window_predictions, int(len(audio_data)/(Fs/50)))
    predictions = upsample_states(combined_windows, 50, Fs, len(audio_data))
    return np.array(predictions)

def make_sample_prediction(patches, new_length):
    index_options = {key: [] for key in range(new_length)}

    for i in range(len(patches)):
        for j in range(len(patches[i])):
            max_index = (PATCH_SIZE-1)+(STRIDE*i)
            if max_index >= new_length:
                shift = max_index - new_length + 1 
                index_options[j+(STRIDE*i)-shift].append(patches[i][j])
            else: 
                index_options[j+(STRIDE*i)].append(patches[i][j])
    prediction = [statistics.mode(value) for (key, value) in index_options.items()] 
    return np.array(prediction)


def make_segmentation_predictions(window_probabilities):
    softmax = F.softmax(window_probabilities, dim=1)
    _, yhat = torch.max(softmax, 0)
    for i in range(1, yhat.shape[0]): 
        if yhat[i] != (yhat[i-1] + 1) % 4:
            yhat[i] = yhat[i-1]
    return yhat 