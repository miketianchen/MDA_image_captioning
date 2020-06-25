# Author: Fanli Zhou
# Date: 2020-06-09
#
# This script defines CNNModel, RNNModel and CaptionModel

import torch
import torch.nn as nn
from torchvision import models


class CNNModel(nn.Module):

    def __init__(self, pretrained=True, path='data/vgg16.hdf5'):
        """
        Initializes a CNNModel

        Parameters:
        -----------
        pretrained: bool (default: True)
            use pretrained model if True
        path: str (default: 'data/vgg16.hdf5')
            the path to load the pre-trained model
        """

        super(CNNModel, self).__init__()
        
        print(f'Loading pre-trained vgg16 CNN model...')
        
        try:
            self.model = torch.load(path)
        except:
            self.model = models.vgg16(pretrained=pretrained)
            
        # remove the last two layers in classifier
        self.model.classifier = nn.Sequential(
          *list(self.model.classifier.children())[:-2]
        )
        self.input_size = 224     

    def forward(self, img_input, train=False):
        """
        forward of the CNNModel

        Parameters:
        -----------
        img_input: torch.Tensor
            the image matrix
        train: bool (default: False)
            use the model only for feature extraction if False

        Return:
        --------
        torch.Tensor
            image feature matrix
        """
        if not train:
            # set the model to evaluation model
            self.model.eval()

        return self.model(img_input)


class RNNModel(nn.Module):

    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_size=256,
        embedding_matrix=None, 
        embedding_train=False
    ):
      
        """
        Initializes a RNNModel

        Parameters:
        -----------
        vocab_size: int
            the size of the vocabulary
        embedding_dim: int
            the number of features in the embedding matrix
        hidden_size: int (default: 256)
            the size of the hidden state in LSTM
        embedding_matrix: torch.Tensor (default: None)
            if not None, use this matrix as the embedding matrix
        embedding_train: bool (default: False)
            not train the embedding matrix if False
        """

        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if embedding_matrix is not None:

            self.embedding.load_state_dict({
              'weight': torch.FloatTensor(embedding_matrix)
            })
            self.embedding.weight.requires_grad = embedding_train

        self.dropout = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
 

    def forward(self, captions):
        """
        forward of the RNNModel

        Parameters:
        -----------
        captions: torch.Tensor
            the padded caption matrix

        Return:
        --------
        torch.Tensor
            word probabilities for each position
        """

        # embed the captions
        embedding = self.dropout(self.embedding(captions))

        outputs, (h, c) = self.lstm(embedding)

        return outputs

class CaptionModel(nn.Module):

    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_size=256,
        embedding_matrix=None, 
        embedding_train=False
    ):

        """
        Initializes a CaptionModel

        Parameters:
        -----------
        vocab_size: int
            the size of the vocabulary
        embedding_dim: int
            the number of features in the embedding matrix
        feature_size: int
            the number of features in the image matrix
        hidden_size: int (default: 256)
            the size of the hidden state in LSTM
        embedding_matrix: torch.Tensor (default: None)
            if not None, use this matrix as the embedding matrix
        embedding_train: bool (default: False)
            not train the embedding matrix if False
        """    
        super(CaptionModel, self).__init__() 

        # set feature_size based of vgg16
        self.feature_size = 4096

        self.decoder = RNNModel(
            vocab_size, 
            embedding_dim,
            hidden_size,
            embedding_matrix,
            embedding_train
        )
        
        self.dropout = nn.Dropout(p=0.5)
        self.dense1 = nn.Linear(self.feature_size, hidden_size) 
        self.relu1 = nn.ReLU()
          
        self.dense2 = nn.Linear(hidden_size, hidden_size) 
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(hidden_size, vocab_size) 

    def forward(self, img_features, captions):
        """
        forward of the CaptionModel

        Parameters:
        -----------
        img_features: torch.Tensor
            the image feature matrix
        captions: torch.Tensor
            the padded caption matrix

        Return:
        --------
        torch.Tensor
            word probabilities for each position
        """

        img_features =\
        self.relu1(
            self.dense1(
                self.dropout(
                    img_features
                )
            )
        )

        decoder_out = self.decoder(captions)

        # add up decoder outputs and image features
        outputs =\
        self.dense3(
            self.relu2(
                self.dense2(
                    decoder_out.add(
                        (img_features.view(img_features.size(0), 1, -1))\
                        .repeat(1, decoder_out.size(1), 1)
                    )
                )
            )
        )

        return outputs

