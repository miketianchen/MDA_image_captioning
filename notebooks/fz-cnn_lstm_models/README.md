### CNN-LSTM Models

|CNN-LSTM Models|CNN|Embeddings|DataLoader|Purpose|
|-|-|-|-|-|
|`9.1.1-fz-cnn_lstm_dataset_v1.ipynb`|`InceptionV3`|`glove.6B.200d`|Slow dataloader|Test on the valid dataset|
|`9.1.1.1-fz-cnn_lstm_dataset_v1_cv.ipynb`|`InceptionV3`|`glove.6B.200d`|Slow dataloader|Cross-validation on the train-valid dataset|
|`9.1.1.2-fz-cnn_lstm_dataset_v1.ipynb`|`InceptionV3`|`glove.6B.200d`|Slow dataloader|Final model|
|`9.1.2-fz-cnn_lstm_dataset_v2.ipynb`|`InceptionV3`|`glove.6B.200d`|Fast dataloader|Test on the valid dataset|
|`9.1.2.1-fz-cnn_lstm_dataset_v2_cv.ipynb`|`InceptionV3`|`glove.6B.200d`|Fast dataloader|Cross-validation on the train-valid dataset|
|`9.1.2.2-fz-cnn_lstm_dataset_v2.ipynb`|`InceptionV3`|`glove.6B.200d`|Fast dataloader|Final model|
|`9.1.3-fz-cnn_lstm_vgg16.ipynb`|`vgg16`|`glove.6B.200d`|Slow dataloader|Test on the valid dataset|
|`9.1.3.1-fz-cnn_lstm_vgg16_cv.ipynb`|`vgg16`|`glove.6B.200d`|Slow dataloader|Cross-validation on the train-valid dataset|
|`9.1.3.2-fz-cnn_lstm_vgg16.ipynb`|`vgg16`|`glove.6B.200d`|Slow dataloader|Final model|
|`9.3-fz-cnn_lstm_dataset_v2_cnn_v2_wiki_embed.ipynb`|`InceptionV3`, feature vectors extracted from convolutional layer|`enwiki_20180420_500d`|Fast dataloader|Test on the valid dataset|
|`9.3.1-fz-cnn_lstm_dataset_v2_cnn_v2_wiki_embed_cv.ipynb`|`InceptionV3`, feature vectors extracted from convolutional layer|`enwiki_20180420_500d`|Fast dataloader|Cross-validation on the train-valid dataset|
|`9.3.2-fz-cnn_lstm_dataset_v2_cnn_v2_wiki_embed.ipynb`|`InceptionV3`, feature vectors extracted from convolutional layer|`enwiki_20180420_500d`|Fast dataloader|Final model|
|`9.4-fz-cnn_lstm_dataset_v2_compare_embeddings.ipynb`|`InceptionV3`|`enwiki_20180420_500d`|Fast dataloader|Compare pre-trained embeddings and embeddings learned from scratch|