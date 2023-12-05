# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""

import sys
sys.path.insert(0,'../') 
sys.path.insert(0,'../my_models') 
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
from raug.raug.loader import get_data_loader
from raug.raug.train import fit_model
from raug.raug.train_kd import fit_model_kd
from raug.raug.eval import test_model
from my_model import set_model
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch
from aug_isic import ImgTrainTransform, ImgEvalTransform
import time
from sacred import Experiment
from sacred.observers import FileStorageObserver
from raug.raug.utils.loader import get_labels_frequency

from src.utils import create_label_encoder
from src.data_operations.data_preprocessing import calculate_class_weights, import_minimias_dataset, dataset_stratified_split
from src.data_operations.data_transformations import generate_image_transforms
import numpy as np
import config
from collections import Counter


os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Starting sacred experiment
ex = Experiment()

@ex.config
def cnfg():

    # Dataset variables
    _folder = 2
    _base_path = "../../data/ISIC2019"
    _csv_path_train = os.path.join(_base_path, "train", "ISIC2019_parsed_train.csv")
    _imgs_folder_train = os.path.join(_base_path, "train", "imgs")

    _csv_path_test = os.path.join(_base_path, "test", "ISIC2019_parsed_test.csv")
    _imgs_folder_test = os.path.join(_base_path, "test", "imgs")

    _use_meta_data = False 
    _neurons_reducer_block = 0
    _comb_method = None 
    _comb_config = None 
    _batch_size = 2 
    _epochs = 100 

    # Training variables
    _best_metric = "loss"
    _pretrained = True
    _lr_init = 0.001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 5 #15
    _weights = "frequency"

    _model_name = 'vgg-19'
    
    _save_folder = "results/" + str(_comb_method) + "_" + _model_name + "_fold_" + str(_folder) #+ "_" + str(time.time()).replace('.', '')

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_folder, _csv_path_train, _imgs_folder_train, _csv_path_test, _imgs_folder_test, _lr_init, _sched_factor,
          _sched_min_lr, _sched_patience, _batch_size, _epochs, _early_stop, _weights, _model_name, _pretrained,
          _save_folder, _best_metric, _neurons_reducer_block, _comb_method, _comb_config, _use_meta_data):

    meta_data_columns = ['age_approx', 'female', 'male', 'anterior torso', 'head/neck', "lateral torso",
                         'lower extremity', 'oral/genital', 'palms/soles', 'posterior torso',  'upper extremity']

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

     # Create label encoder.
    l_e = create_label_encoder()

    images, labels = import_minimias_dataset(data_dir="/home/aikansh_yadav/Documents/MTP Breast Cancer Detection/images_processed".format(config.dataset),
                                                     label_encoder=l_e)

    labels_before_data_aug = labels
    # images, labels = generate_image_transforms(images, labels)
    labels_after_data_aug = labels
    # np.random.shuffle(labels)

    # print(y_train_before_data_aug, y_train_after_data_aug)

    def one_hot_to_label(one_hot_encoded):
      return np.argmax(one_hot_encoded, axis=1)
    
    labels_before_data_aug = one_hot_to_label(labels_before_data_aug)
    labels_after_data_aug = one_hot_to_label(labels_after_data_aug)

    # print("Before data augmentation:")
    # print(Counter(list(map(str, labels_before_data_aug))))
    # print("After data augmentation:")Teacher
    # print(Counter(list(map(str, labels_after_data_aug))))

    # Split dataset into training/test/validation sets (80/20% split).
    X_train, X_test, y_train, y_test = dataset_stratified_split(split=0.20, dataset=images, labels=labels_after_data_aug)

    # Create CNN model and split training/validation set (80/20% split). # older value is 25
    # model = CnnModel(config.model, l_e.classes_.size)
    X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.15,
                                                                dataset=X_train,
                                                                labels=y_train)
  
    val_data_loader = get_data_loader (X_val, y_val, None, transform=ImgEvalTransform(),
                                        batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    
    train_data_loader = get_data_loader (X_train, y_train, None, transform=ImgTrainTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    
    test_data_loader = get_data_loader(X_test, y_test, None, transform=ImgEvalTransform(),
                                        batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)   

    # if config.verbose_mode:
    #     print("Before data augmentation:")
    #     print(Counter(list(map(str, y_train_before_data_aug))))
    #     print("After data augmentation:")
    #     print(Counter(list(map(str, y_train_after_data_aug))))

    # # Fit model.
    # if config.verbose_mode:
    #     print("Training set size: {}".format(X_train.shape[0]))
    #     print("Validation set size: {}".format(X_val.shape[0]))
    #     print("Test set size: {}".format(X_test.shape[0]))
    # # Loading the csv file
    # csv_all_folders = pd.read_csv(_csv_path_train)

    # print("-" * 50)
    # print("- Loading validation data...")
    # val_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder) ]
    # train_csv_folder = csv_all_folders[ csv_all_folders['folder'] != _folder ]

    # # Loading validation data
    # val_imgs_id = val_csv_folder['image'].values
    # val_imgs_path = ["{}/{}.jpg".format(_imgs_folder_train, img_id) for img_id in val_imgs_id]
    # val_labels = val_csv_folder['diagnostic_number'].values
    # if _use_meta_data:
    #     val_meta_data = val_csv_folder[meta_data_columns].values
    #     print("-- Using {} meta-data features".format(len(meta_data_columns)))
    # else:
    #     print("-- No metadata")
    #     val_meta_data = None
    # val_data_loader = get_data_loader (val_imgs_path, val_labels, val_meta_data, transform=ImgEvalTransform(),
    #                                    batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    # print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))

    # print("- Loading training data...")
    # train_imgs_id = train_csv_folder['image'].values
    # train_imgs_path = ["{}/{}.jpg".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    # train_labels = train_csv_folder['diagnostic_number'].values
    # if _use_meta_data:
    #     train_meta_data = train_csv_folder[meta_data_columns].values
    #     print("-- Using {} meta-data features".format(len(meta_data_columns)))
    # else:
    #     print("-- No metadata")
    #     train_meta_data = None
    # train_data_loader = get_data_loader (train_imgs_path, train_labels, train_meta_data, transform=ImgTrainTransform(),
    #                                    batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    # print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))

    # print("-"*50)
    ####################################################################################################################

    # ser_lab_freq = get_labels_frequency(train_csv_folder, "diagnostic", "image")
    # _labels_name = ser_lab_freq.index.values
    # _freq = ser_lab_freq.values
    ####################################################################################################################
    print("- Loading", _model_name)
    print(l_e.classes_.size)

    model = set_model(_model_name, l_e.classes_.size, pretrained=_pretrained, neurons_reducer_block=_neurons_reducer_block)
    print(model)
    ####################################################################################################################
    # if _weights == 'frequency':
    #     _weights = (_freq.sum() / _freq).round(3)
    # weights_list = [v for v in _weights.values()]
    # weights_tensor = torch.Tensor(weights_list).cuda()
    # weight=torch.Tensor(weights_tensor).cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience)
    ####################################################################################################################

    total_params = sum(
	    param.numel() for param in model.parameters()
    )

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Total params : ", total_params)
    print("Trainable params : ", trainable_params)

    print("- Starting the training phase...")
    print("-" * 50)


    
    fit_model (model, train_data_loader, val_data_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs,
               epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None,
               device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=False,
               history_plot=True, val_metrics=["balanced_accuracy"], best_metric=_best_metric)
    ####################################################################################################################

    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model (model, val_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
                partition_name='eval', metrics_to_comp='all', class_names=l_e.classes_, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)
    ####################################################################################################################

    ####################################################################################################################

    # if _csv_path_test is not None:
    #     print("- Loading test data...")
    #     csv_test = pd.read_csv(_csv_path_test)
    #     test_imgs_id = csv_test['image'].values
    #     test_imgs_path = ["{}/{}.jpg".format(_imgs_folder_test, img_id) for img_id in test_imgs_id]
    #     test_labels = csv_test['diagnostic_number'].values
    #     csv_test['lateral torso'] = 0
    #     if _use_meta_data:
    #         test_meta_data = csv_test[meta_data_columns].values
    #         print("-- Using {} meta-data features".format(len(meta_data_columns)))
    #     else:
    #         test_meta_data = None
    #         print("-- No metadata")

    #     _metric_options = {
    #         'save_all_path': os.path.join(_save_folder, "test_pred"),
    #         'pred_name_scores': 'predictions.csv',
    #     }
    #     test_data_loader = get_data_loader(test_imgs_path, test_labels, test_meta_data, transform=ImgEvalTransform(),
    #                                        batch_size=_batch_size, shuf=False, num_workers=16, pin_memory=True)
    #     print("-" * 50)
                   
    print("- Evaluating the test partition...")
    test_model (model, test_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
                partition_name='Test', metrics_to_comp='all', class_names=l_e.classes_, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)                   
                   
    ####################################################################################################################
  
    
