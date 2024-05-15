import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

from .augmentations import GaussianSmoothing
from .model import GRUDecoder



class LightningModel(pl.LightningModule):
    def __init__(self, neural_dim, n_classes, hidden_dim, layer_dim, 
                 nDays=24, dropout=0, device="cuda", strideLen=4,
                 kernelLen=14, gaussianSmoothWidth=0, bidirectional=False):
        super(LightningModel, self).__init__()
        
        # Defining the number of layers and the nides in each layer
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.nDays = nDays
        self.dropout = dropout
        # self.device = device
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(self.nDays, self.neural_dim, self.neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(self.nDays, 1, self.neural_dim))

        # for i in range(self.nDays):
        #     self.dayWeights.data[i, :, :] = torch.eye(self.neural_dim)



        self.model = GRUDecoder(
            neural_dim=self.neural_dim,
            n_classes=self.n_classes,
            hidden_dim=self.hidden_dim,
            layer_dim=self.layer_dim,
            nDays=self.nDays,
            dropout=self.dropout,
            device=device,
            strideLen=self.strideLen,
            kernelLen=self.kernelLen,
            gaussianSmoothWidth=self.gaussianSmoothWidth,
            bidirectional=self.bidirectional
        ).to(device)
        print("Model device", self.model.device)

    def forward(self, neuralInput, dayIdx):
        return self.model.forward(neuralInput, dayIdx)



    # self.model = GRUDecoder()

# class LightningModel(pl.LightningModule):
#     def __init__(self, neural_dim, n_classes, hidden_dim, layer_dim, 
#                  nDays=24, dropout=0, device="cuda", strideLen=4, 
#                  kernelLen=14, gaussianSmoothWidth=0, bidirectional=False):

#         super(LightningModel, self).__init__()
#         # Defining the number of layers and the nodes in each layer
#         self.neural_dim = neural_dim
#         self.n_classes = n_classes
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#         self.nDays = nDays
#         self.dropout = dropout
#         self.device = device
#         self.strideLen = strideLen
#         self.kernelLen = kernelLen
#         self.gaussianSmoothWidth = gaussianSmoothWidth
#         self.bidirectional = bidirectional
#         self.inputLayerNonlinearity = torch.nn.Softsign()
#         self.unfolder = torch.nn.Unfold(
#             (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
#         )
#         self.gaussianSmoother = GaussianSmoothing(
#             neural_dim, 20, self.gaussianSmoothWidth, dim=1
#         )
#         self.dayWeights = torch.nn.Parameter(torch.randn(self.nDays, self.neural_dim, self.neural_dim))
#         self.dayBias = torch.nn.Parameter(torch.zeros(self.nDays, 1, self.neural_dim))

#         for i in range(self.nDays):
#             self.dayWeights.data[i, :, :] = torch.eye(self.neural_dim)
        
#         # GRU layers
#         self.gru_decoder = nn.GRU(
#             (self.neural_dim) * self.kernelLen,
#             self.hidden_dim,
#             self.layer_dim,
#             batch_first=True,
#             dropout=self.dropout,
#             bidirectional=self.bidirectional,
#         )

#         for name, param in self.gru_decoder.named_parameters():
#             if "weight_hh" in name:
#                 nn.init.orthogonal_(param)
#             elif "weight_ih" in name:
#                 nn.init.xavier_uniform_(param)
#             elif "bias" in name:
#                 nn.init.constant_(param.data, 0)

#         # Input layers
#         for i in range(nDays):
#             setattr(self, "inpLayer" + str(i), nn.Linear(self.neural_dim, self.neural_dim))

#         for i in range(nDays):
#             thisLayer = getattr(self, "inpLayer" + str(i))
#             thisLayer.weight = torch.nn.Parameter(
#                 thisLayer.weight + torch.eye(self.neural_dim)
#             ) 

#         # rnn outputs
#         if self.bidirectional:
#             self.fc_decoder_out = nn.Linear(
#                 self.hidden_dim * 2, self.n_classes + 1
#             )  # +1 for CTC blank
#         else:
#             self.fc_decoder_out = nn.Linear(
#                 self.hidden_dim, self.n_classes + 1
#             )  # +1 for CTC blank
             

#     def forward(self, neuralInput, dayIdx):
#         # Define the forward pass of your model here
#         neuralInput = torch.permute(neuralInput, (0, 2, 1))
#         neuralInput = self.gaussianSmoother(neuralInput)
#         neuralInput = torch.permute(neuralInput, (0, 2, 1))

#         # apply day layer
#         dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
#         transformedNeutral = torch.einsum(
#             'btd, bdk -> btk', neuralInput, dayWeights        
#         ) + torch.index_select(self.dayBias, 0, dayIdx)
#         transformedNeural = self.inputLayerNonlinearity(transformedNeural)

#         # stride/kernel
#         stridedInputs = torch.permute(
#             self.unfolder(
#                 torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
#             ),
#             (0, 2, 1),
#         )

#         # apply RNN layer
#         if self.bidirectional:
#             h0 = torch.zeros(
#                 self.layer_dim * 2,
#                 transformedNeural.size(0), 
#                 self.hidden_dim,
#                 device=self.device,
#             ).requires_grad_()
#         else:
#             h0 = torch.zeros(
#                 self.layer_dim,
#                 transformedNeural.size(0),
#                 self.hidden_dim,
#                 device=self.device,
#             ).requires_grad_()
        
#         hid, _ = self.gru_decoder(stridedInputs, h0.detach())

#         # get seq
#         seq_out = self.fc_decoder_out(hid)
#         return seq_out
            


#     def training_step(self, batch, batch_idx):
#         # Define the training step logic here
#         pass

#     # def validation_step(self, batch, batch_idx):
#     #     # Define the validation step logic here
#     #     pass

#     # def test_step(self, batch, batch_idx):
#     #     # Define the test step logic here
#     #     pass

#     def configure_optimizers(self):
#         # Define your optimizer and learning rate scheduler here
#         pass

#     # def train_dataloader(self):
#     #     # Define your training data loader here
#     #     pass

#     # def val_dataloader(self):
#     #     # Define your validation data loader here
#     #     pass

#     # def test_dataloader(self):
#     #     # Define your test data loader here
#     #     pass

if __name__ == '__main__':
    model = LightningModel()
    trainer = pl.Trainer()
    trainer.fit(model)