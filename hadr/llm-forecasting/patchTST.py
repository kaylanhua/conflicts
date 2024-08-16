# https://huggingface.co/docs/transformers/en/model_doc/patchtst
from transformers import PatchTSTConfig, PatchTSTModel
from huggingface_hub import hf_hub_download
import torch

# # Initializing an PatchTST configuration with 12 time steps for prediction
# configuration = PatchTSTConfig(prediction_length=12)

# # Randomly initializing a model (with random weights) from the configuration
# model = PatchTSTModel(configuration)

# # Accessing the model configuration
# configuration = model.config
# print(configuration)


file = hf_hub_download(
    repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = PatchTSTModel.from_pretrained("namctin/patchtst_etth1_pretrain")

# during training, one provides both past and future values
outputs = model(
    past_values=batch["past_values"],
    future_values=batch["future_values"],
)

last_hidden_state = outputs.last_hidden_state


## PREDICTION

from huggingface_hub import hf_hub_download
import torch
from transformers import PatchTSTConfig, PatchTSTForPrediction

file = hf_hub_download(
    repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

# Prediction task with 7 input channels and prediction length is 96
model = PatchTSTForPrediction.from_pretrained("namctin/patchtst_etth1_forecast")

# during training, one provides both past and future values
outputs = model(
    past_values=batch["past_values"],
    future_values=batch["future_values"],
)

loss = outputs.loss
loss.backward()

# during inference, one only provides past values, the model outputs future values
outputs = model(past_values=batch["past_values"])
prediction_outputs = outputs.prediction_outputs