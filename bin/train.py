# PREVENT: PRotein Engineering by Variational frEe eNergy approximaTion
# Copyright (C) 2024  Giovanni Stracquadanio, Evgenii Lobzaev

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training functions.

TODO:
1. Check if we need <eos>, <unk>, <pad> for regression task.
"""
import torch
import torch.nn as nn
import logging
import copy
import pickle
import typer
from torch.utils.data import DataLoader,WeightedRandomSampler
from models import (
                        TransformerVAE,
                        PredictorGRU,
                        PredictorTFEncoder,
                        simple_average_model
                    )
from utilities import (
                        default_w2i_i2w,
                        setup_logger,
                        ProteinSequencesDataset,
                        load_checkpoint,
                        save_data_to_csv,
                        train_step,            # TransformerVAE (sequence + property)
                        validation_step,       # TransformerVAE (sequence + property)
                        train_step_gru_tf,     # GRU prediction (sequence -> property)
                        validation_step_gru_tf # GRU prediction (sequence -> property)
                    )

# main app
app        = typer.Typer()

# for TransformerVAE
tf_app     = typer.Typer()
app.add_typer(tf_app, name="transformer")

# for predicting
predictor_app = typer.Typer()
app.add_typer(predictor_app, name="predictor")

######################################### VAE ############
@tf_app.command("gaussian")
def train_gaussian(
                    training_fasta_file         : str,            # FASTA file with training sequences
                    validation_fasta_file       : str,            # FASTA file with validation sequences
                    log_name_batch              : str,            # logger for batch updates during training
                    log_name_epoch              : str,            # logger for epoch updates during training
                    log_name_val_epoch          : str,            # logger for epoch updates during validation
                    log_name_seqs_batch         : str,            # logger for keeping track what sequences were in a batch update
                    checkpoint_pattern          : str,            # checkpoint generic name
                    csv_epoch_name              : str,            # filename for CSV with training losses results by epoch
                    csv_validation_epoch_name   : str,            # filename for CSV with validation losses results by epoch
                    pickle_name                 : str,            # filename for model input params (to instantiate correct object later)
                    pt_name_train               : str,            # filename for model trained layers (corresponding to smallest training error)
                    pt_name_val                 : str,            # filename for model trained layers (corresponding to smallest validation error)
                    epochs                      : int   = 10,     # number of epochs to train NN
                    learning_rate               : float = 0.0001, # learning rate for gradient descent
                    lambda_constant             : float = 0.001,  # L2 constant
                    validation_freq_epoch       : int   = 2,      # how frequently compute validation loss
                    checkpoint_freq_epoch       : int   = 2,      # how frequently save model snapshots
                    max_sequence_length         : int   = 100,    # maximum sequence length
                    batch_size                  : int   = 16,     # how many sequences in mini-batch
                    embedding_size              : int   = 512,    # embedding size (must be divisible by number of heads)
                    latent_size                 : int   = 64,     # dimensionality of the latent space
                    condition_on_energy         : bool  = False,  # whether to scale z by energy when passed to sequence decoder
                    weighted_sampling           : bool  = False,  # whether to use weighted sampling (right now using WeightedRandomSampler)
                    dropout_prob                : float = 0.1,    # dropout probability (same for all the dropouts)
                    masking_prob                : float = 0.1,    # probability of replacing AA with <unk> token for the input
                    heads                       : int   = 8,      # number of heads in (self)attention
                    num_layers_encoder          : int   = 6,      # number of transformer blocks in encoder
                    num_layers_decoder          : int   = 4,      # number of transformer blocks in decoder
                    seed                        : int   = 0,      # random seed
                    pt_checkpoint               : str   = None    # checkpoint file (default is None)
                  ) -> None:
    # set up loggers
    logger_batch     = setup_logger( 'logger_batch', log_name_batch, level = logging.INFO )
    logger_epoch     = setup_logger( 'logger_epoch', log_name_epoch, level = logging.INFO )
    logger_val_epoch = setup_logger( 'logger_val_epoch', log_name_val_epoch, level = logging.INFO )
    logger_sequences = setup_logger( 'logger_sequences', log_name_seqs_batch, level = logging.INFO )

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model info
    str_info  = "Training details:\n"
    str_info += f"\tNeural Network core architecture                                    : Transformer\n"
    str_info += f"\tPrior and variational distribution                                  : standard multivariate Gaussian\n"
    str_info += f"\tTraining set file                                                   : {training_fasta_file}\n"
    str_info += f"\tValidation set file                                                 : {validation_fasta_file}\n"
    str_info += f"\tNumber of epochs                                                    : {epochs}\n"
    str_info += f"\tValidation frequency (in epochs)                                    : {validation_freq_epoch}\n"
    str_info += f"\tCheckpoint frequency (in epochs)                                    : {checkpoint_freq_epoch}\n"
    str_info += f"\tEmbedding size                                                      : {embedding_size}\n"
    str_info += f"\tDropout probability                                                 : {dropout_prob}\n"
    str_info += f"\tMasking probability (for replacing with <unk>)                      : {masking_prob}\n"
    str_info += f"\tLatent size                                                         : {latent_size}\n"
    str_info += f"\tCondition latent z on predicted energy for sequence decoding        : {condition_on_energy}\n"
    str_info += f"\tUse weighted sampling for mini-batches( using WeightedRandomSampler): {weighted_sampling}\n"
    str_info += f"\tLearning rate                                                       : {learning_rate}\n"
    str_info += f"\tL2 penalty constant                                                 : {lambda_constant}\n"
    str_info += f"\tBatch size                                                          : {batch_size}\n"
    str_info += f"\tMax sequence length                                                 : {max_sequence_length}\n"
    str_info += f"\tNumber of heads in (self)attention                                  : {heads}\n"
    str_info += f"\tNumber of transformer blocks in encoder                             : {num_layers_encoder}\n"
    str_info += f"\tNumber of transformer blocks in decoder                             : {num_layers_decoder}\n"
    str_info += f"\tRandom seed                                                         : {seed}\n"
    str_info += f"\tCheckpoint to start a model                                         : {pt_checkpoint}\n"
    str_info += f"\tDevice                                                              : {device}\n"

    # log
    logger_epoch.info(str_info)
    
    # set seed
    torch.manual_seed(seed)

    # create default w2i and i2w maps
    w2i,i2w = default_w2i_i2w()

    # create Datasets
    # TODO: provide sequence weights
    dataset = ProteinSequencesDataset(
                                        training_fasta_file,
                                        w2i,
                                        i2w,
                                        device,
                                        max_sequence_length = max_sequence_length,
                                        extract_energy      = True
                                     )
    if weighted_sampling:
        train_sequence_weights = dataset.sequence_weights # obtain sequence weights
        train_weighted_sampler = WeightedRandomSampler(
                                                        weights     = train_sequence_weights,
                                                        num_samples = len(train_sequence_weights),
                                                        replacement = True
                                                      ) # create a weighted Random Sampler
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, sampler = train_weighted_sampler)
        sampler_string = "Using WeightedRandomSampler."

    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        sampler_string = "Using default sampling scheme."

    ## log info
    str_  = f"Train dataset contains {len(dataset)} elements. "
    str_ += f"Batch size is {batch_size}. "
    str_ += f"Train dataloader length is {len(dataloader)}.\n"
    str_ += f"{sampler_string}"
    
    logger_epoch.info(str_)

    ### create validation dataset and dataloader
    dataset_val = ProteinSequencesDataset(
                                            validation_fasta_file,
                                            w2i,
                                            i2w,
                                            device,
                                            max_sequence_length = max_sequence_length,
                                            extract_energy      = True
                                            
                                         )      
    dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = True)

    ### log info
    str_  = f"Validation dataset contains {len(dataset_val)} elements. "
    str_ += f"Batch size is {batch_size}. "
    str_ += f"Validation dataloader length is {len(dataloader_val)}.\n"
    logger_val_epoch.info(str_)

    #### compute basic model -> average as predictor
    mean_train, mse_train, mse_validation = simple_average_model(
                                                                    dataset.energies,
                                                                    dataset_val.energies
                                                                )
    # write to both train and validation logger
    str_ = f"Average energy prediction: {mean_train:0.4f}. Training MSE using this estimator: {mse_train:0.4f}\n"
    logger_epoch.info(str_)

    str_ = f"Average energy prediction: {mean_train:0.4f}. Validation MSE using this estimator: {mse_validation:0.4f}\n"
    logger_val_epoch.info(str_)

    # instantiate model
    vars = {
                "vocab_size"          : dataset.vocab_size,
                "max_seq_length"      : dataset.max_seq_len,
                "pad_idx"             : dataset.pad_idx,
                "sos_idx"             : dataset.sos_idx,
                "eos_idx"             : dataset.eos_idx,
                "unk_idx"             : dataset.unk_idx,          
                "embedding_size"      : embedding_size,               
                "latent_size"         : latent_size,
                "condition_on_energy" : condition_on_energy,
                "num_layers_encoder"  : num_layers_encoder,
                "num_layers_decoder"  : num_layers_decoder,
                "heads"               : heads,
                "dropout_prob"        : dropout_prob, 
                "device"              : device             
            }
    # model input params
    pickle.dump(vars,open(pickle_name,"wb"))

    # print GPU usage after datasets instatiation
    if torch.cuda.is_available():
        print("GPU resources after datasets creation...")
        print(torch.cuda.memory_summary())


    model = TransformerVAE(**vars)
    logger_epoch.info("Model specification:")
    logger_epoch.info(model)
    
    # move the model (and its sub-modules(model.children()) to GPU)
    model = model.to(device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = lambda_constant)

    # print GPU usage after model instatiation
    if torch.cuda.is_available():
        print("GPU resources after model creation...")
        print(torch.cuda.memory_summary())

    # try loading checkpoint
    # it may fail if there is no checkpoint or some mismatch in model params
    try:
        load_checkpoint(model,optimizer,pt_checkpoint)
    except Exception as e:
        print("Exception occured:",e)
    else:
        print("Checkpoint loaded successfully")
    finally:
        # for debugging
        max_len = 0
        for name, param in model.named_parameters(recurse=True):
            if len(name) > max_len:
                max_len = len(name)
            print(f"Param: {name}, device: {param.device}")

        # loss objects for reconstruction error
        NLL       = nn.NLLLoss(ignore_index=dataset.pad_idx)
        NLL_val   = nn.NLLLoss(ignore_index=dataset_val.pad_idx)

        # loss objects for mse error
        MSE       = nn.MSELoss()
        MSE_val   = nn.MSELoss()

        # losses by epoch
        nll_losses_by_epoch   = []
        kl_losses_by_epoch    = []
        mse_losses_by_epoch   = []
        total_losses_by_epoch = []

        # validation losses (by epoch only)
        epoch_val               = []
        nll_val_by_epoch        = []
        kl_val_by_epoch         = []
        mse_val_by_epoch        = []
        total_loss_val_by_epoch = []

        # to keep track of epoch with smallest loss: overall
        best_loss_train       = float('inf')  # for train loss
        best_loss_val         = float('inf')  # for validation loss

        # actual training part
        for epoch in range(epochs):
            # add a note to logger_batch
            logger_batch.info(f"Epoch: {epoch}")
            
            
            # do a full pass over training set
            train_elbo,train_reconstruction,train_kl,train_mse = train_step(
                                                                                model,
                                                                                dataloader,
                                                                                NLL,
                                                                                MSE,
                                                                                optimizer,
                                                                                logger_batch,
                                                                                logger_sequences,
                                                                                masking_prob = masking_prob
                                                                            )

            # append to epoch lists
            nll_losses_by_epoch.append(train_reconstruction)
            kl_losses_by_epoch.append(train_kl)
            mse_losses_by_epoch.append(train_mse)
            total_losses_by_epoch.append(train_elbo)

            # write to epoch logger
            str_epoch = f"Epoch: {epoch}\n\t\t"
            str_epoch += f"Reconstruction loss: {train_reconstruction:0.4f} \t"
            str_epoch += f"KL loss: {train_kl:0.4f} \t"
            str_epoch += f"MSE loss: {train_mse:0.4f}\t"
            str_epoch += f"ELBO: {train_elbo:0.4f} \t"
            logger_epoch.info(str_epoch)


            # do a snapshot
            if (epoch % checkpoint_freq_epoch == 0) and (epoch != 0):
                
                checkpoint_name = checkpoint_pattern + f"-after-epoch-{epoch}.pytorch"
                
                torch.save({
                            'epoch'                : epoch,
                            'reconstruction'       : train_reconstruction,
                            'kl loss'              : train_kl,
                            'mse'                  : train_mse,
                            'elbo'                 : train_elbo,
                            'model_state_dict'     : copy.deepcopy(model.state_dict()),
                            'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                            }, checkpoint_name)


            # check validation error
            if (epoch % validation_freq_epoch == 0) and (epoch != 0):

                # run validation
                val_elbo,val_reconstruction,val_kl,val_mse = validation_step(
                                                                                model,
                                                                                dataloader_val,
                                                                                NLL_val,
                                                                                MSE_val
                                                                            )

                # append to validation lists
                epoch_val.append(epoch)              
                nll_val_by_epoch.append(val_reconstruction)
                kl_val_by_epoch.append(val_kl)
                mse_val_by_epoch.append(val_mse)
                total_loss_val_by_epoch.append(val_elbo)

                # write to val_epoch logger
                val_string_ = f"Validation after epoch: {epoch}\n\t\t"
                val_string_ += f"Reconstruction loss : {val_reconstruction:0.4f}\t"
                val_string_ += f"KL loss : {val_kl:0.4f}\t"
                val_string_ += f"MSE loss : {val_mse:0.4f}\t"   
                val_string_ += f"ELBO : {val_elbo:0.4f}" 
                logger_val_epoch.info(val_string_)

                # update best model params based on validation error
                if val_elbo < best_loss_val:
                    # create/update dictionary
                    best_results_val    = {
                                            'epoch'                : epoch,
                                            'reconstruction'       : val_reconstruction,
                                            'kl loss'              : val_kl,
                                            'mse'                  : val_mse,
                                            'elbo'                 : val_elbo,
                                            'model_state_dict'     : copy.deepcopy(model.state_dict()),
                                            'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                                          }
                    best_loss_val = val_elbo

            
            # need to update best model params based on training error
            if train_elbo < best_loss_train:
                # create/update dictionary
                best_results_train = {
                                        'epoch'                : epoch,
                                        'reconstruction'       : train_reconstruction,
                                        'kl loss'              : train_kl,
                                        'mse'                  : train_mse,
                                        'elbo'                 : train_elbo,
                                        'model_state_dict'     : copy.deepcopy(model.state_dict()),
                                        'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                                     }
                best_loss_train = train_elbo

        
        # once training is over, save all results
        # training results
        save_data_to_csv(
                        csv_epoch_name,
                        [list(range(len(nll_losses_by_epoch))),nll_losses_by_epoch,kl_losses_by_epoch,mse_losses_by_epoch,total_losses_by_epoch],
                        ["Epoch","NLL","KL","MSE","ELBO"]
                        )
        
        # validation results
        save_data_to_csv(
                        csv_validation_epoch_name,
                        [epoch_val,nll_val_by_epoch,kl_val_by_epoch,mse_val_by_epoch,total_loss_val_by_epoch],
                        ["Epoch","NLL","KL","MSE","ELBO"]
                        )

        # save best results (TRAIN)
        str_  = "Best model parameters (based on train error)\n"
        str_ += f"\tAchieved at epoch    : {best_results_train['epoch']}\n"
        str_ += f"\tReconstruction error : {best_results_train['reconstruction']:0.4f}\n"
        str_ += f"\tKL loss              : {best_results_train['kl loss']:0.4f}\n"
        str_ += f"\tMSE loss             : {best_results_train['mse']:0.4f}\n"
        str_ += f"\tELBO                 : {best_results_train['elbo']:0.4f}\n"
        logger_epoch.info(str_)
        # save model
        torch.save(best_results_train, pt_name_train)

        # save best results (VALIDATION)
        str_  = "Best model parameters (based on validation error)\n"
        str_ += f"\tAchieved at epoch    : {best_results_val['epoch']}\n"
        str_ += f"\tReconstruction error : {best_results_val['reconstruction']:0.4f}\n"
        str_ += f"\tKL loss              : {best_results_val['kl loss']:0.4f}\n"
        str_ += f"\tMSE loss             : {best_results_val['mse']:0.4f}\n"
        str_ += f"\tELBO                 : {best_results_val['elbo']:0.4f}\n"
        logger_val_epoch.info(str_)
        # save model
        torch.save(best_results_val, pt_name_val)

######################################### GRU predictive ############
@predictor_app.command("gru")
def train_gru(
                    training_fasta_file         : str,            # FASTA file with training sequences
                    validation_fasta_file       : str,            # FASTA file with validation sequences
                    log_name_batch              : str,            # logger for batch updates during training
                    log_name_epoch              : str,            # logger for epoch updates during training
                    log_name_val_epoch          : str,            # logger for epoch updates during validation
                    checkpoint_pattern          : str,            # checkpoint generic name
                    csv_epoch_name              : str,            # filename for CSV with training losses results by epoch
                    csv_validation_epoch_name   : str,            # filename for CSV with validation losses results by epoch
                    pickle_name                 : str,            # filename for model input params (to instantiate correct object later)
                    pt_name_train               : str,            # filename for model trained layers (corresponding to smallest training error)
                    pt_name_val                 : str,            # filename for model trained layers (corresponding to smallest validation error)
                    epochs                      : int   = 10,     # number of epochs to train NN
                    learning_rate               : float = 0.0001, # learning rate for gradient descent
                    lambda_constant             : float = 0.001,  # L2 constant
                    validation_freq_epoch       : int   = 2,      # how frequently compute validation loss
                    checkpoint_freq_epoch       : int   = 2,      # how frequently save model snapshots
                    max_sequence_length         : int   = 100,    # maximum sequence length
                    batch_size                  : int   = 16,     # how many sequences in mini-batch
                    embedding_size              : int   = 32,     # embedding size (must be divisible by number of heads)
                    hidden_size                 : int   = 32,     # hidden size for h in GRU
                    num_layers                  : int   = 1,      # number of stacked GRUs
                    bidirectional               : bool  = False,  # whether to use bidirectional GRU (default: no)
                    dropout_prob                : float = 0.1,    # dropout probability (same for all the dropouts)
                    seed                        : int   = 0,      # random seed
                    pt_checkpoint               : str   = None    # checkpoint file (default is None)
                  ) -> None:
    # set up loggers
    logger_batch     = setup_logger( 'logger_batch', log_name_batch, level = logging.INFO )
    logger_epoch     = setup_logger( 'logger_epoch', log_name_epoch, level = logging.INFO )
    logger_val_epoch = setup_logger( 'logger_val_epoch', log_name_val_epoch, level = logging.INFO )

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model info
    str_info = "Training details:\n"
    str_info += f"\tPredictive NN core architecture                 : GRU\n"
    str_info += f"\tTraining set file                               : {training_fasta_file}\n"
    str_info += f"\tValidation set file                             : {validation_fasta_file}\n"
    str_info += f"\tNumber of epochs                                : {epochs}\n"
    str_info += f"\tValidation frequency (in epochs)                : {validation_freq_epoch}\n"
    str_info += f"\tCheckpoint frequency (in epochs)                : {checkpoint_freq_epoch}\n"
    str_info += f"\tEmbedding size                                  : {embedding_size}\n"
    str_info += f"\tDropout probability                             : {dropout_prob}\n"
    str_info += f"\tLearning rate                                   : {learning_rate}\n"
    str_info += f"\tL2 penalty constant                             : {lambda_constant}\n"
    str_info += f"\tBatch size                                      : {batch_size}\n"
    str_info += f"\tMax sequence length                             : {max_sequence_length}\n"
    str_info += f"\tRandom seed                                     : {seed}\n"
    str_info += f"\tCheckpoint to start a model                     : {pt_checkpoint}\n"
    str_info += f"\tDevice                                          : {device}\n"

    # log
    logger_epoch.info(str_info)
    
    # set seed
    torch.manual_seed(seed)

    # create default w2i and i2w maps
    w2i,i2w = default_w2i_i2w()

    # create train set -> need ["input"] and ["energy"]
    dataset = ProteinSequencesDataset(
                                        training_fasta_file,
                                        w2i,
                                        i2w,
                                        device,
                                        max_sequence_length = max_sequence_length,
                                        extract_energy      = True
                                     )
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    ## log info
    str_  = f"Train dataset contains {len(dataset)} elements. "
    str_ += f"Batch size is {batch_size}. "
    str_ += f"Train dataloader length is {len(dataloader)}.\n"
    logger_epoch.info(str_)

    ### create validation set
    dataset_val = ProteinSequencesDataset(
                                            validation_fasta_file,
                                            w2i,
                                            i2w,
                                            device,
                                            max_sequence_length = max_sequence_length,
                                            extract_energy      = True
                                            
                                         )
    dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = True)

    ### log info
    str_  = f"Validation dataset contains {len(dataset_val)} elements. "
    str_ += f"Batch size is {batch_size}. "
    str_ += f"Validation dataloader length is {len(dataloader_val)}.\n"
    logger_val_epoch.info(str_)

    #### compute basic model -> average as predictor
    mean_train, mse_train, mse_validation = simple_average_model(
                                                                    dataset.energies,
                                                                    dataset_val.energies
                                                                )
    # write to both train and validation logger
    str_ = f"Average energy prediction: {mean_train:0.4f}. Training MSE using this estimator: {mse_train:0.4f}\n"
    logger_epoch.info(str_)

    str_ = f"Average energy prediction: {mean_train:0.4f}. Validation MSE using this estimator: {mse_validation:0.4f}\n"
    logger_val_epoch.info(str_)

    # instantiate GRU model
    vars = {
                "vocab_size"          : dataset.vocab_size,
                "embedding_size"      : embedding_size,
                "hidden_size"         : hidden_size,
                "device"              : device,
                "pad_idx"             : dataset.pad_idx,                    
                "p_dropout"           : dropout_prob,
                "num_layers"          : num_layers,
                "bidirectional"       : bidirectional          
            }
    model = PredictorGRU(**vars)
    
    logger_epoch.info("Model specification:")
    logger_epoch.info(model)
    
    # move the model (and its sub-modules(model.children()) to GPU)
    model = model.to(device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = lambda_constant)

    # try loading checkpoint
    # it may fail if there is no checkpoint or some mismatch in model params
    try:
        load_checkpoint(model,optimizer,pt_checkpoint)
    except Exception as e:
        print("Exception occured:",e)
    else:
        print("Checkpoint loaded successfully")
    finally:
        # for debugging
        max_len = 0
        for name, param in model.named_parameters(recurse=True):
            if len(name) > max_len:
                max_len = len(name)
            print(f"Param: {name}, device: {param.device}")
        
        # loss objects for mse error
        MSE       = nn.MSELoss()
        MSE_val   = nn.MSELoss()

        # loss by epoch
        mse_losses_by_epoch   = [] # train
        mse_val_by_epoch      = [] # validation
        epoch_val             = [] # epochs when validation loss is computed
       

        # to keep track of epoch with smallest loss: overall
        best_loss_train       = float('inf')  # for train loss
        best_loss_val         = float('inf')  # for validation loss

        # actual training part
        for epoch in range(epochs):

            # add a note to logger_batch
            logger_batch.info(f"Epoch: {epoch}")

            # do a full pass over training set
            train_mse = train_step_gru_tf(
                                            model,
                                            dataloader,
                                            MSE,
                                            optimizer,
                                            logger_batch
                                        )
            # append to epoch lists
            mse_losses_by_epoch.append(train_mse)

            # write to epoch logger
            str_epoch = f"Epoch: {epoch}\n\t\t"
            str_epoch += f"MSE loss: {train_mse:0.4f}\t"
            logger_epoch.info(str_epoch)

            # do a snapshot
            if (epoch % checkpoint_freq_epoch == 0) and (epoch != 0):
                
                checkpoint_name = checkpoint_pattern + f"-after-epoch-{epoch}.pytorch"
                
                torch.save({
                            'epoch'                : epoch,
                            'mse'                  : train_mse,
                            'model_state_dict'     : copy.deepcopy(model.state_dict()),
                            'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                            }, checkpoint_name)

            # check validation error
            if (epoch % validation_freq_epoch == 0) and (epoch != 0):

                # run validation
                val_mse = validation_step_gru_tf(
                                                    model,
                                                    dataloader_val,
                                                    MSE_val
                                                )

                # append to validation lists
                epoch_val.append(epoch)              
                mse_val_by_epoch.append(val_mse)

                # write to val_epoch logger
                val_string_ = f"Validation after epoch: {epoch}\n\t\t"
                val_string_ += f"MSE loss : {val_mse:0.4f}\t" 
                logger_val_epoch.info(val_string_)

                # update best model params based on validation error
                if val_mse < best_loss_val:
                    # create/update dictionary
                    best_results_val    = {
                                            'epoch'                : epoch,
                                            'mse'                  : val_mse,
                                            'model_state_dict'     : copy.deepcopy(model.state_dict()),
                                            'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                                          }
                    best_loss_val = val_mse

            # need to update best model params based on training error
            if train_mse < best_loss_train:
                # create/update dictionary
                best_results_train = {
                                        'epoch'                : epoch,
                                        'mse'                  : train_mse,
                                        'model_state_dict'     : copy.deepcopy(model.state_dict()),
                                        'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                                     }
                best_loss_train = train_mse

        # once training is over save all the results
        # training results
        save_data_to_csv(
                            csv_epoch_name,
                            [list(range(len(mse_losses_by_epoch))),mse_losses_by_epoch],
                            ["Epoch","MSE"]
                        )
        
        # validation results
        save_data_to_csv(
                            csv_validation_epoch_name,
                            [epoch_val,mse_val_by_epoch],
                            ["Epoch","MSE"]
                        )
        
        # model input params
        pickle.dump(vars,open(pickle_name,"wb"))

        # save best results (TRAIN)
        str_  = "Best model parameters (based on train error)\n"
        str_ += f"\tAchieved at epoch    : {best_results_train['epoch']}\n"
        str_ += f"\tMSE loss             : {best_results_train['mse']:0.4f}\n"
        logger_epoch.info(str_)
        # save model
        torch.save(best_results_train, pt_name_train)

        # save best results (VALIDATION)
        str_  = "Best model parameters (based on validation error)\n"
        str_ += f"\tAchieved at epoch    : {best_results_val['epoch']}\n"
        str_ += f"\tMSE loss             : {best_results_val['mse']:0.4f}\n"
        logger_val_epoch.info(str_)
        # save model
        torch.save(best_results_val, pt_name_val)


######################################### TF encoder predictive ############
@predictor_app.command("tf")
def train_tf(
                training_fasta_file         : str,            # FASTA file with training sequences
                validation_fasta_file       : str,            # FASTA file with validation sequences
                log_name_batch              : str,            # logger for batch updates during training
                log_name_epoch              : str,            # logger for epoch updates during training
                log_name_val_epoch          : str,            # logger for epoch updates during validation
                checkpoint_pattern          : str,            # checkpoint generic name
                csv_epoch_name              : str,            # filename for CSV with training losses results by epoch
                csv_validation_epoch_name   : str,            # filename for CSV with validation losses results by epoch
                pickle_name                 : str,            # filename for model input params (to instantiate correct object later)
                pt_name_train               : str,            # filename for model trained layers (corresponding to smallest training error)
                pt_name_val                 : str,            # filename for model trained layers (corresponding to smallest validation error)
                epochs                      : int   = 10,     # number of epochs to train NN
                learning_rate               : float = 0.0001, # learning rate for gradient descent
                lambda_constant             : float = 0.001,  # L2 constant
                validation_freq_epoch       : int   = 2,      # how frequently compute validation loss
                checkpoint_freq_epoch       : int   = 2,      # how frequently save model snapshots
                max_sequence_length         : int   = 100,    # maximum sequence length
                batch_size                  : int   = 16,     # how many sequences in mini-batch
                embedding_size              : int   = 512,    # embedding size (must be divisible by number of heads)
                latent_size                 : int   = 64,     # dimensionality of the latent space
                weighted_sampling           : bool  = False,  # whether to use weighted sampling (right now using WeightedRandomSampler)
                dropout_prob                : float = 0.1,    # dropout probability (same for all the dropouts)
                heads                       : int   = 8,      # number of heads in (self)attention
                num_layers_encoder          : int   = 6,      # number of transformer blocks in encoder
                seed                        : int   = 0,      # random seed
                pt_checkpoint               : str   = None    # checkpoint file (default is None)
            ) -> None:
    
    # set up loggers
    logger_batch     = setup_logger( 'logger_batch', log_name_batch, level = logging.INFO )
    logger_epoch     = setup_logger( 'logger_epoch', log_name_epoch, level = logging.INFO )
    logger_val_epoch = setup_logger( 'logger_val_epoch', log_name_val_epoch, level = logging.INFO )

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model info
    str_info = "Training details:\n"
    str_info += f"\tPredictive NN core architecture                                     : Transformer Encoder\n"
    str_info += f"\tTraining set file                                                   : {training_fasta_file}\n"
    str_info += f"\tValidation set file                                                 : {validation_fasta_file}\n"
    str_info += f"\tNumber of epochs                                                    : {epochs}\n"
    str_info += f"\tValidation frequency (in epochs)                                    : {validation_freq_epoch}\n"
    str_info += f"\tCheckpoint frequency (in epochs)                                    : {checkpoint_freq_epoch}\n"
    str_info += f"\tEmbedding size                                                      : {embedding_size}\n"
    str_info += f"\tLatent size                                                         : {latent_size}\n"
    str_info += f"\tUse weighted sampling for mini-batches( using WeightedRandomSampler): {weighted_sampling}\n"
    str_info += f"\tNumber of heads in self-attention                                   : {heads}\n"
    str_info += f"\tNumber of layers in TF encoder block                                : {num_layers_encoder}\n"
    str_info += f"\tDropout probability                                                 : {dropout_prob}\n"
    str_info += f"\tLearning rate                                                       : {learning_rate}\n"
    str_info += f"\tL2 penalty constant                                                 : {lambda_constant}\n"
    str_info += f"\tBatch size                                                          : {batch_size}\n"
    str_info += f"\tMax sequence length                                                 : {max_sequence_length}\n"
    str_info += f"\tRandom seed                                                         : {seed}\n"
    str_info += f"\tCheckpoint to start a model                                         : {pt_checkpoint}\n"
    str_info += f"\tDevice                                                              : {device}\n"

    # log
    logger_epoch.info(str_info)
    
    # set seed
    torch.manual_seed(seed)

    # create default w2i and i2w maps
    w2i,i2w = default_w2i_i2w()

    # create train set -> need ["input"] and ["energy"]
    dataset = ProteinSequencesDataset(
                                        training_fasta_file,
                                        w2i,
                                        i2w,
                                        device,
                                        max_sequence_length = max_sequence_length,
                                        extract_energy      = True
                                     )
    if weighted_sampling:
        train_sequence_weights = dataset.sequence_weights # obtain sequence weights
        train_weighted_sampler = WeightedRandomSampler(
                                                        weights     = train_sequence_weights,
                                                        num_samples = len(train_sequence_weights),
                                                        replacement = True
                                                        ) # create a weighted Random Sampler
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, sampler = train_weighted_sampler)
        sampler_string = "Using WeightedRandomSampler."

    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        sampler_string = "Using default sampling scheme."

        ## log info
        str_  = f"Train dataset contains {len(dataset)} elements. "
        str_ += f"Batch size is {batch_size}. "
        str_ += f"Train dataloader length is {len(dataloader)}.\n"
        str_ += f"{sampler_string}"
        logger_epoch.info(str_)


    ### create validation set
    dataset_val = ProteinSequencesDataset(
                                            validation_fasta_file,
                                            w2i,
                                            i2w,
                                            device,
                                            max_sequence_length = max_sequence_length,
                                            extract_energy      = True
                                            
                                         )
    dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = True)

    ### log info
    str_  = f"Validation dataset contains {len(dataset_val)} elements. "
    str_ += f"Batch size is {batch_size}. "
    str_ += f"Validation dataloader length is {len(dataloader_val)}.\n"
    logger_val_epoch.info(str_)

    #### compute basic model -> average as predictor
    mean_train, mse_train, mse_validation = simple_average_model(
                                                                    dataset.energies,
                                                                    dataset_val.energies
                                                                )
    # write to both train and validation logger
    str_ = f"Average energy prediction: {mean_train:0.4f}. Training MSE using this estimator: {mse_train:0.4f}\n"
    logger_epoch.info(str_)

    str_ = f"Average energy prediction: {mean_train:0.4f}. Validation MSE using this estimator: {mse_validation:0.4f}\n"
    logger_val_epoch.info(str_)

    # instantiate GRU model
    vars = {
                "vocab_size"          : dataset.vocab_size,
                "max_seq_length"      : dataset.max_seq_len,
                "pad_idx"             : dataset.pad_idx,         
                "embedding_size"      : embedding_size,               
                "latent_size"         : latent_size,
                "num_layers_encoder"  : num_layers_encoder,
                "heads"               : heads,
                "dropout_prob"        : dropout_prob          
            }
    model = PredictorTFEncoder(**vars)
    
    logger_epoch.info("Model specification:")
    logger_epoch.info(model)
    
    # move the model (and its sub-modules(model.children()) to GPU)
    model = model.to(device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = lambda_constant)

    # try loading checkpoint
    # it may fail if there is no checkpoint or some mismatch in model params
    try:
        load_checkpoint(model,optimizer,pt_checkpoint)
    except Exception as e:
        print("Exception occured:",e)
    else:
        print("Checkpoint loaded successfully")
    finally:
        # for debugging
        max_len = 0
        for name, param in model.named_parameters(recurse=True):
            if len(name) > max_len:
                max_len = len(name)
            print(f"Param: {name}, device: {param.device}")
        
        # loss objects for mse error
        MSE       = nn.MSELoss()
        MSE_val   = nn.MSELoss()

        # loss by epoch
        mse_losses_by_epoch   = [] # train
        mse_val_by_epoch      = [] # validation
        epoch_val             = [] # epochs when validation loss is computed
       

        # to keep track of epoch with smallest loss: overall
        best_loss_train       = float('inf')  # for train loss
        best_loss_val         = float('inf')  # for validation loss

        # actual training part
        for epoch in range(epochs):

            # add a note to logger_batch
            logger_batch.info(f"Epoch: {epoch}")

            # do a full pass over training set
            train_mse = train_step_gru_tf(
                                            model,
                                            dataloader,
                                            MSE,
                                            optimizer,
                                            logger_batch,
                                            model_type = 2
                                        )
            # append to epoch lists
            mse_losses_by_epoch.append(train_mse)

            # write to epoch logger
            str_epoch = f"Epoch: {epoch}\n\t\t"
            str_epoch += f"MSE loss: {train_mse:0.4f}\t"
            logger_epoch.info(str_epoch)

            # do a snapshot
            if (epoch % checkpoint_freq_epoch == 0) and (epoch != 0):
                
                checkpoint_name = checkpoint_pattern + f"-after-epoch-{epoch}.pytorch"
                
                torch.save({
                            'epoch'                : epoch,
                            'mse'                  : train_mse,
                            'model_state_dict'     : copy.deepcopy(model.state_dict()),
                            'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                            }, checkpoint_name)

            # check validation error
            if (epoch % validation_freq_epoch == 0) and (epoch != 0):

                # run validation
                val_mse = validation_step_gru_tf(
                                                    model,
                                                    dataloader_val,
                                                    MSE_val,
                                                    model_type = 2
                                                )

                # append to validation lists
                epoch_val.append(epoch)              
                mse_val_by_epoch.append(val_mse)

                # write to val_epoch logger
                val_string_ = f"Validation after epoch: {epoch}\n\t\t"
                val_string_ += f"MSE loss : {val_mse:0.4f}\t" 
                logger_val_epoch.info(val_string_)

                # update best model params based on validation error
                if val_mse < best_loss_val:
                    # create/update dictionary
                    best_results_val    = {
                                            'epoch'                : epoch,
                                            'mse'                  : val_mse,
                                            'model_state_dict'     : copy.deepcopy(model.state_dict()),
                                            'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                                          }
                    best_loss_val = val_mse

            # need to update best model params based on training error
            if train_mse < best_loss_train:
                # create/update dictionary
                best_results_train = {
                                        'epoch'                : epoch,
                                        'mse'                  : train_mse,
                                        'model_state_dict'     : copy.deepcopy(model.state_dict()),
                                        'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                                     }
                best_loss_train = train_mse

        # once training is over save all the results
        # training results
        save_data_to_csv(
                            csv_epoch_name,
                            [list(range(len(mse_losses_by_epoch))),mse_losses_by_epoch],
                            ["Epoch","MSE"]
                        )
        
        # validation results
        save_data_to_csv(
                            csv_validation_epoch_name,
                            [epoch_val,mse_val_by_epoch],
                            ["Epoch","MSE"]
                        )
        
        # model input params
        pickle.dump(vars,open(pickle_name,"wb"))

        # save best results (TRAIN)
        str_  = "Best model parameters (based on train error)\n"
        str_ += f"\tAchieved at epoch    : {best_results_train['epoch']}\n"
        str_ += f"\tMSE loss             : {best_results_train['mse']:0.4f}\n"
        logger_epoch.info(str_)
        # save model
        torch.save(best_results_train, pt_name_train)

        # save best results (VALIDATION)
        str_  = "Best model parameters (based on validation error)\n"
        str_ += f"\tAchieved at epoch    : {best_results_val['epoch']}\n"
        str_ += f"\tMSE loss             : {best_results_val['mse']:0.4f}\n"
        logger_val_epoch.info(str_)
        # save model
        torch.save(best_results_val, pt_name_val)




if __name__ == "__main__":
    app()







