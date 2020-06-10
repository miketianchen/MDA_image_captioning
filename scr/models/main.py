# main.py
#
# Author: Fanli Zhou
#
# Date: 2020-06-09
#
# This script

import json
from tqdm import tqdm
import pickle
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare_data import get_img_info, get_vocab, get_word_dict,\
get_embeddings, SampleDataset, my_collate, encode_image,\
extract_img_features, hms_string
from model import CNNModel, RNNModel, CaptionModel
from train import train, init_weights
from generate_captions import generateCaption
EPOCHS = 10
START = "startseq"
STOP = "endseq"

if __name__ == "__main__":

    root_captioning = '../../../s3'
    model_path = f'{root_captioning}/final_model.hdf5'

    train_paths, train_descriptions, max_length_train =\
    get_img_info(root_captioning, 'train')
    valid_paths, valid_descriptions, max_length_valid =\
    get_img_info(root_captioning, 'valid')
    test_paths, test_descriptions, max_length_test =\
    get_img_info(root_captioning, 'test')
    sydney_paths, sydney_descriptions, max_length_sydney =\
    get_img_info(root_captioning, 'sydney')
    
    print(f'{len(train_paths)} images from RSICD and UCM datasets for training')
    print(f'{len(test_paths)} images from RSICD and UCM datasets for testing')
    print(f'{len(sydney_paths)} images from the Sydney dataset for testing')
    
    train_paths.extend(valid_paths.copy())
    train_descriptions.extend(valid_descriptions.copy())
    # add a start and stop token at the beginning/end
    for v in train_descriptions:
        for d in range(len(v)):
            v[d] = f'{START} {v[d]} {STOP}'

    max_length = max(max_length_train, max_length_valid) + 2

    vocab = get_vocab(train_descriptions, word_count_threshold=10)
    idxtoword, wordtoidx = get_word_dict(vocab)

    vocab_size = len(idxtoword) + 1
    batch_size = 200
    hidden_size = 256
    embedding_dim = 200
    cnn_type = 'vgg16'

    embedding_matrix = get_embeddings(
        root_captioning,
        vocab_size,
        embedding_dim,
        wordtoidx
    ) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = CNNModel(cnn_type, pretrained=True)
    encoder.to(device)
    
    train_img_features = extract_img_features(
        'training',
        train_paths,
        encoder, 
        device
    )

    test_img_features = extract_img_features(
        'test',
        test_paths,
        encoder,
        device
    )

    sydney_img_features = extract_img_features(
        'sydney',
        sydney_paths,
        encoder,
        device
    )

    train_dataset = SampleDataset(
        train_descriptions,
        train_img_features,
        wordtoidx,
        max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        collate_fn=my_collate
    )


    caption_model = CaptionModel(
        cnn_type, 
        vocab_size, 
        embedding_dim, 
        hidden_size=hidden_size,
        embedding_matrix=embedding_matrix, 
        embedding_train=True
    )

    init_weights(
        caption_model,
        embedding_pretrained=True
    )

    caption_model.to(device)

    # we will ignore the pad token in true target set
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        caption_model.parameters(), 
        lr=0.01
    )

    clip = 1
    start = time()
    print(f'Training...')
    for i in tqdm(range(EPOCHS * 7)):

        loss = train(
            caption_model,
            train_loader,
            optimizer,
            criterion,
            clip,
            vocab_size,
            device
        )
        print(f'loss = {loss}')

    # reduce the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4

    for i in tqdm(range(EPOCHS * 7)):

        loss = train(
            caption_model,
            train_loader,
            optimizer,
            criterion,
            clip,
            vocab_size,
            device
        )
        print(f'loss = {loss}')

    torch.save(caption_model, model_path)
    print(f"\Training took: {hms_string(time()-start)}")

    # generate results
    for name, paths, img_features in [('test', test_paths, test_img_features),
    ('sydney', sydney_paths, sydney_img_features)]:
        results = {}
        print(f'Generating captions for the {name} dataset...')

        for n in range(len(paths)):
            # note the filename splitting depends on path
            filename = paths[n].split('/')[4]
            results[filename] = generateCaption(
                caption_model, 
                img_features[n],
                max_length,
                vocab_size,
                wordtoidx,
                idxtoword,
                device
            )
        with open(f'{root_captioning}/{name}_generated_captions.json', 'w') as fp:
            json.dump(results, fp)    