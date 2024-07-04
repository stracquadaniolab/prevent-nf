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
Utility functions and classes.
"""

import torch
import torch.nn as nn
from autograd_minimize import minimize
from models import (
                        TransformerVAE,
                        PredictorTFEncoder
                    )
from torch.utils.data import Dataset,DataLoader
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import defaultdict
from typing import List,Tuple
from numpy.typing import ArrayLike
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import tqdm
import re
from matplotlib import pyplot as plt
import pickle

# class to store sequence data
class SequenceData(object):
    """
    Stores information about sequence:
    sequence as string,
    list of energies,
    list of logprobabilities,
    list of z (as numpy arrays)
    id (as string)
    description (as string)
    """
    def __init__(self,seq,id,first_energy,first_logp,first_z):
        # instantiate object -> save id, create and add first element to energy/logp/z lists
        self.id          = id
        self.energies    = [first_energy]
        self.logps       = [first_logp]
        self.zs          = [first_z]
        self.sequence    = seq
        self.description = ""
    def update_info(self,next_energy,next_logp,next_z):
        # update respective lists
        self.energies.append(next_energy)
        self.logps.append(next_logp)
        self.zs.append(next_z)
    def generate_description(self,seedID=None):
        # generate description of the SequenceData
        # print("List of all energies/zs:")
        # for z,energy in zip(self.zs,self.energies):
        #     print("energy:",energy)
        #     print("z     :",z)
        #     print("-"*10)
        
        # stats for energies
        energies_stats = self.__compute_stats(self.energies)
        # stats for logp 
        logps_stats    = self.__compute_stats(self.logps)
        # write energies
        self.description = (
                            f"Seed molecule: {seedID}||"
                            f"Number of datapoints: {energies_stats[0]}||"
                            f"Length: {len(self.sequence)}||"
                            f"Energy [avg,std,5%,95%]: [{energies_stats[1]:0.4f},{energies_stats[2]:0.4f},{energies_stats[3]:0.4f},{energies_stats[4]:0.4f}]||"
                            f"logp(x|z) [avg,std,5%,95%]: [{logps_stats[1]:0.4f},{logps_stats[2]:0.4f},{logps_stats[3]:0.4f},{logps_stats[4]:0.4f}]"
                        )

    @property
    def toTuple(self):
        # return as tuple
        tpl = (
                    self.sequence,
                    self.energies,
                    self.logps,
                    self.zs,
                    self.id,
                    self.description
                )
        return tpl
    @property
    def toSeqRecord(self):
        # return as SeqRecord
        record = SeqRecord(
                                Seq(self.sequence),
                                id = self.id,
                                description = self.description
                          )
        return record
    def __compute_stats(self,somelist):
        # compute stats for a list of numerical values: energies and logps
        n_elements = len(somelist)      # number of elements
        as_np      = np.array(somelist) # cast to np.array
        avg        = np.mean(as_np)     # mean
        std        = np.std(as_np)
        pct5       = np.percentile(as_np,5)
        pct95      = np.percentile(as_np,95)
        return n_elements,avg,std,pct5,pct95

#################################################### DATA PREPROCESSING ######################################
# mappings: amino acids (+ tokens) <-> integers 
def default_w2i_i2w() -> Tuple[dict,dict] :
    '''
    Constructs default maps that can be passed to ProteinSequencesDataset.
    If you want custom maps, you need to code them separately. You can reuse the code from here.
    '''
    w2i            = dict() # maps word(amino-acids) into index
    i2w            = dict() # maps index into word(amino-acid)
        
    # may need to rearrange the order later
    amino_acids    = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

    # first get the amino acids    
    for w in amino_acids:
        i2w[len(w2i)] = w 
        w2i[w]       = len(w2i)

    # then add special tokens
    for st in special_tokens:
        i2w[len(w2i)] = st 
        w2i[st]       = len(w2i)
            
    return w2i,i2w

# find a historgram frequency for an input value
def find_bin_value(
                      element: float, # element to find correpsonding bin frequency for. higher the frequency, lower should be the probability of sampling this value
                      n: List,        # List of bin frequencies
                      bins: List      # List of bin edges
                  )->float:

    import bisect

    # find min/max : leftmost/rightmost bin edges
    min_, max_ = bins[0], bins[-1]

    # if element is outside min_, max_, throw an error
    if (element < min_) or (element > max_):
      raise Exception(f"Sorry, cannot find frequency for element outside of [{min_};{max_}] range")

    # if element is equal to max_, assign to last bin (other edge cases should be handled correclt in the main body of the function)
    if element == max_:
      ind = len(n)-1
    else:
      ind = bisect.bisect(bins, element) - 1

    return n[ind]

# dataset class for processing sequence data
class ProteinSequencesDataset(Dataset):
    '''
    Custom dataset class that works with protein sequences.
        fasta_file         : FASTA file from Uniprot with protein sequences (needs to be prepared separately), string
        w2i                : map word-to-index, dictionary
        i2w                : map index-to-word, dictionary
        max_sequence_length: maximum length of protein sequence to be considered for VAE, whatever is beyond is ignored, int
        sequence_weights   : dictionary with sequence weights
                             
    '''
    def __init__(
                    self,
                    fasta_file          : str,
                    w2i                 : dict,
                    i2w                 : dict,
                    device              : torch.device,
                    max_sequence_length : int  = 500, 
                    sequence_weights    : dict = None,
                    extract_energy      : bool = False
                ):
        super().__init__()
        
        # save few variables
        self.max_seq_length = max_sequence_length + 1 # to account for <eos>/<sos>
        
        # need to create w2i and i2w dictionaries
        self.w2i, self.i2w  = w2i, i2w
        
        # keep device
        self.device = device

        # need to construct data object -> modified to exclude sequences with weird amino acids
        self.data, self.list_of_energies = self.__construct_data(
                                                                    fasta_file,
                                                                    sequence_weights = sequence_weights,
                                                                    extract_energy   = extract_energy
                                                                )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> dict:
        return self.data[idx]

    def __sym2num_conversion(
                                self,
                                input_  : List,
                                target_ : List
                            ) -> Tuple[torch.tensor,torch.tensor]:
        '''
        Conversion of string array into numeric array. Needed if we use embedding layers.
        Conversion is the SAME for input_ and target_
        EX.: ['<sos>','M','Q','H'] -> [2,4,7,10]
        
        INPUT:
            input_ : Input array of strings, list of strings
            target_: Next element predictions for the input_, list of strings
        OUTPUT:
            input_num, target_num : numpy numeric arrays. See EX. above.
        
        '''
        #input_num  = np.asarray([self.w2i.get(element,self.w2i['<unk>']) for element in input_])
        #target_num = np.asarray([self.w2i.get(element,self.w2i['<unk>']) for element in target_])
        input_num   = torch.tensor(
                                    [self.w2i.get(element,self.w2i['<unk>']) for element in input_],
                                    dtype = torch.long,
                                    device = self.device
                                  )
        target_num  = torch.tensor(
                                    [self.w2i.get(element,self.w2i['<unk>']) for element in target_],
                                    dtype = torch.long,
                                    device = self.device
                                  )

        return input_num,target_num
       
    def __construct_data(
                            self,
                            fasta_file        : str,
                            sequence_weights  : dict = None, # sequence weights (as dictionary)
                            extract_energy    : bool = False
                        ) -> defaultdict:
        '''
        Explicit construction of data object that is used in __getitem__ method.
        INPUT:
            fasta_file : FASTA file from Uniprot with protein sequences (needs to be prepared separately), string
        OUTPUT:
            data       : defaultdict that has a following format:
                         data[i] = {"input"     : Input array for element i (EX: <sos>ABCD)
                                    "target"    : Target array for element i (it is input shifted by 1 position to the right) (EX: ABCD<eos>)
                                    "energy"    : Free Energy value
                                    "length"    : length of input or output (they'are the same length) that does not take padding into account)
                                    "weight"    : some weight assigned to a sequence, default 1.0
                                    "reference" : id of a sequence
                                   }
        '''
        # create a nested dictionary with default dictionary
        data = defaultdict(dict)

        # create empty list
        list_of_energies = []
        
        # get list of sequences: now only sequences with ALL known amino acids are added
        orginal_records = SeqIO.parse(fasta_file,"fasta")
        records         = [record for record in orginal_records]

        # add weights (if sequence_weights object is not None)
        addweights = True if sequence_weights != None else False
        
        # populate data
        # loop over seqeunces
        for i,record in enumerate(records):
            
            # get reference id
            reference_        = record.id

            if extract_energy:
                # parse ID to obtain free energy
                energy = float(
                                reference_.split(":")[1]
                            )

                # append to energies list
                list_of_energies.append(energy)
                
                # cast to tensor
                energy_ = torch.tensor(  
                                        [energy],
                                        dtype  = torch.float,
                                        device = self.device
                                    )

            # convert to a list
            sequence          = list(record.seq)
            sequence_plus_sos = ['<sos>'] + sequence
            
            # obtain input and target as character arrays
            input_  = sequence_plus_sos[:self.max_seq_length]
            target_ = sequence[:self.max_seq_length-1] + ['<eos>']
            assert len(input_) == len(target_), "Length mismatch"
            len_    = len(input_)

            # cast to tensor
            len_    = torch.tensor( 
                                    len_,
                                    dtype=torch.long,
                                    device=self.device
                                  )
            
            # need to append <pad> tokens if necessary
            input_.extend(['<pad>'] * (self.max_seq_length-len_))
            target_.extend(['<pad>'] * (self.max_seq_length-len_))
            
            # need to convert into numerical format
            input_,target_ = self.__sym2num_conversion(input_,target_)
             
            # save to data: everything but reference_ is torch tensor (pushed to cpu or gpu, if available)
            data[i]["input"]     = input_
            data[i]["target"]    = target_
            data[i]["length"]    = len_
            data[i]["reference"] = reference_

            # add weights
            if addweights:
                data[i]["weight"]    = torch.tensor(
                                                        sequence_weights[reference_],
                                                        dtype  = torch.float, 
                                                        device = self.device
                                                    ) # add weights wrapped into tensor
                print(data[i]["weight"])
            # add energy
            if extract_energy:
                data[i]["energy"] = energy_

        return data,list_of_energies
    
    @property
    def vocab_size(self) -> int:
        return len(self.w2i)

    @property
    def max_seq_len(self) -> int:
        return self.max_seq_length

    @property
    def pad_idx(self) -> int:
        return self.w2i['<pad>']

    @property
    def sos_idx(self) -> int:
        return self.w2i['<sos>']

    @property
    def eos_idx(self) -> int:
        return self.w2i['<eos>']

    @property
    def unk_idx(self) -> int:
        return self.w2i['<unk>']
        
    @property
    def energies(self) -> ArrayLike:
        return np.array(self.list_of_energies)

    # new: return weights for each sequence/energy pair
    @property
    def sequence_weights(self) -> ArrayLike:
        # run histrogram on self.list_of_energies
        n, bins     = np.histogram(
                                    self.list_of_energies,
                                    "auto"
                                  )

        # compute weights as inverse of frequencies
        # higher the frequency, lower the probability of sampling
        weights  = [ 1.0/find_bin_value(energy,n,bins) for energy in self.list_of_energies ] 

        return weights

        

##################################################### RELATED TO TRAINING ########################################
# computing loss for Gaussian-based VAE
# loss is composed of 3 terms:
# 1) Reconstruction loss
# 2) KL loss (in closed form)
# 3) MSE loss
# weights are not used here, if sequences have different weights, this function needs to be modified!
def compute_loss_gaussian(
                            reconstruction_loss_criterion, # function/class to compute Reconstruction loss -> here NLLLoss
                            true,                          # true sequnce tokens
                            predicted,                     # predicted tokens probability/logprobability/logits (depends on a function used)
                            mean,                          # mean of q(z|x)
                            logv,                          # log-variance of q(z|x)
                            mse_loss_criterion,            # MSE loss class/function
                            energy_true,                   # true energy values
                            energy_predicted               # predicted energy values
                          ):
    # true: [Batch Sise, Seq Length]
    # predicted: [Batch Size, Seq Length, Vocab Size]
    B,S,E     = predicted.size()
    true      = true.reshape(-1)
    predicted = predicted.reshape(B*S,E)

    # reconstruction loss
    reconstruction_loss = reconstruction_loss_criterion(predicted,true)

    # KL loss
    kl_loss = -0.5 * torch.sum(
                            1 + logv - mean.pow(2) - logv.exp()
                         )

    # MSE loss
    mse_energy = mse_loss_criterion(energy_predicted,energy_true)

    return reconstruction_loss,kl_loss,mse_energy

# whole pass over dataloader
# optimization is performed
def train_step(
                model,           # VAE
                dataloader,      # dataloader
                criterion,       # NLL for reconstruction error
                criterion_mse,   # MSE for MSE loss
                optimizer,
                logger,
                logger_sequences, # logger to check which sequences are in batch
                masking_prob=0.0
              ):
    # training mode
    model.train()

    batch_loss           = [] # all
    batch_reconstruction = [] # reconstruction
    batch_kl             = [] # kl
    batch_mse            = [] # mse 

    for idx,batch in enumerate(dataloader):
        
        src           = batch["input"]     # size [N,S]
        tgt           = batch["target"]    # size [N,T]
        energy_true   = batch["energy"]    # size [N,1] or maybe [N] as energy is a scalar
        reference_ids = batch["reference"]

        # forward pass
        output,mean,logv,predicted_energy = model(
                                                    src, # Src in Transformer architecture
                                                    src, # Tgt in Transformer architecture
                                                    masking_prob = masking_prob # masking probability to mask Tgt
                                                 )

        # loss calculation (returns reconstruction, kl and mse losses individually)
        reconstruction_loss, kl_loss, mse_loss = compute_loss_gaussian(
                                                                        criterion,
                                                                        tgt,
                                                                        output,
                                                                        mean,
                                                                        logv,
                                                                        criterion_mse,
                                                                        energy_true,
                                                                        predicted_energy
                                                                      )

        # overall loss
        loss = reconstruction_loss + kl_loss + mse_loss

        if idx == 0:
            # print resources
            if torch.cuda.is_available():
                print(f"GPU resources just before gradient update:")
                print(torch.cuda.memory_summary())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Overall loss
        batch_loss.append(loss.item())
        
        # reconstruction
        batch_reconstruction.append(reconstruction_loss.item())
        
        # kl
        batch_kl.append(kl_loss.item())

        # mse
        batch_mse.append(mse_loss.item())

        # add to the logger
        str_ = f"\t Batch {idx}\t"
        str_ += f"Reconstruction error: {batch_reconstruction[-1]:0.4f}\t"
        str_ += f"KL: {batch_kl[-1]:0.4f}\t"
        str_ += f"MSE: {batch_mse[-1]:0.4f}\t"
        str_ += f"ELBO: {batch_loss[-1]:0.4f}"
        logger.info(str_)

        # add to logger_sequences
        logger_sequences.info(reference_ids)


    # compute average (for epoch)
    avg_loss           = sum(batch_loss)/len(batch_loss)
    avg_reconstruction = sum(batch_reconstruction)/len(batch_reconstruction)
    avg_kl             = sum(batch_kl)/len(batch_kl)
    avg_mse            = sum(batch_mse)/len(batch_mse)

    return avg_loss,avg_reconstruction,avg_kl,avg_mse

# whole pass over dataloader
# no optimization is performed
def validation_step(model,dataloader,criterion,criterion_mse):
    # evaluation mode
    model.eval()

    batch_loss           = [] # all
    batch_reconstruction = [] # reconstruction
    batch_kl             = [] # kl
    batch_mse            = [] # mse
    
    # don't keep gradients
    with torch.no_grad():
        for batch in dataloader:
            src         = batch["input"]  # size [N,S]
            tgt         = batch["target"] # size [N,T]
            energy_true = batch["energy"] # size [N,1] or just [N]

            # forward pass
            output,mean,logv,predicted_energy = model(src,src)

            # loss calculation
            reconstruction_loss, kl_loss, mse_loss = compute_loss_gaussian(
                                                                            criterion,
                                                                            tgt,
                                                                            output,
                                                                            mean,
                                                                            logv,
                                                                            criterion_mse,
                                                                            energy_true,
                                                                            predicted_energy
                                                                         )

            # overall loss
            loss = reconstruction_loss + kl_loss + mse_loss

            # all loss
            batch_loss.append(loss.item())
            # reconstruction
            batch_reconstruction.append(reconstruction_loss.item())
            # kl
            batch_kl.append(kl_loss.item())
            # mse
            batch_mse.append(mse_loss.item())

    
    # compute average
    avg_loss           = sum(batch_loss)/len(batch_loss)
    avg_reconstruction = sum(batch_reconstruction)/len(batch_reconstruction)
    avg_kl             = sum(batch_kl)/len(batch_kl)
    avg_mse            = sum(batch_mse)/len(batch_mse)

    return avg_loss,avg_reconstruction,avg_kl,avg_mse

# GRU predict energy from sequence
# whole pass over dataloader
# optimization is performed
def train_step_gru_tf(
                        model,
                        dataloader,
                        criterion_mse,
                        optimizer,
                        logger,
                        model_type: int = 1 # 1: GRU, 2: TF
                    ):
    # training mode
    model.train()

    # batch mse 
    batch_mse  = [] 

    for idx,batch in enumerate(dataloader):
        
        input         = batch["input"]  # size [N,S]
        input_length  = batch["length"]
        energy_true   = batch["energy"] # size [N,1]

        if model_type == 1:
            # GRU needs sequences and lengths
            predicted_energy = model(
                                        input,
                                        input_length
                                    )
        else:
            # else if Transformer here
            predicted_energy = model(
                                        input
                                    )

        # loss calculation
        # overall loss
        mse_loss = criterion_mse(predicted_energy,energy_true)

        # if idx == 0:
        #     # print resources
        #     if torch.cuda.is_available():
        #         print(f"GPU resources just before gradient update:")
        #         print(torch.cuda.memory_summary())

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        # mse
        batch_mse.append(mse_loss.item())

        # add to the logger
        str_ = f"\t Batch {idx}\t"
        str_ += f"MSE: {batch_mse[-1]:0.4f}\t"
        logger.info(str_)


    # compute average (for epoch)
    avg_mse            = sum(batch_mse)/len(batch_mse)

    return avg_mse

# GRU validation step
def validation_step_gru_tf(
                            model,
                            dataloader,
                            criterion_mse,
                            model_type: int = 1 # 1: GRU, 2: TF
                       ):
    # training mode
    model.eval()

    # batch mse 
    batch_mse  = [] 

    with torch.no_grad():
        for idx,batch in enumerate(dataloader):
            
            input         = batch["input"]  # size [N,S]
            input_length  = batch["length"]
            energy_true   = batch["energy"] # size [N,1]

            if model_type == 1:
                # GRU needs sequences and lengths
                predicted_energy = model(
                                            input,
                                            input_length
                                        )
            else:
                # else if Transformer here
                predicted_energy = model(
                                            input
                                        )

            # loss calculation
            # overall loss
            mse_loss = criterion_mse(predicted_energy,energy_true)

            # mse
            batch_mse.append(mse_loss.item())

    # compute average (for epoch)
    avg_mse            = sum(batch_mse)/len(batch_mse)

    return avg_mse

# procedure to get means/stds of the latent space
def get_means_stds(
                        output_filename_means : str,         # filename for means of q(z|x)
                        output_filename_stds  : str,         # filename for stds of q(z|x) 
                        pt_name               : str,         # pytorch file with pretraine dlayers
                        pickle_name           : str,         # pickle file with model inputs
                        fasta_file            : str,         # FASTA file with sequences
                        batch_size            : int = 4,     # how many sequences to process in a batch
                        max_sequence_length   : int = 500,   # max sequence length for Dataset construction
                        seed                  : int = 0
                  ):
    # instantiate model object
    vars  = pickle.load(open(pickle_name, "rb" ))
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # force cpu, if only cpu is available
    vars["device"]  = device

    # instantiate model
    model = TransformerVAE(**vars)

    # load trained parameters
    checkpoint = torch.load(pt_name,map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # move to GPU
    model = model.to(device)
    print(f"Device: {device}")

    # get i2w map and eos_idx
    w2i,i2w  = default_w2i_i2w()
    
    # create a list of improper tokens
    improper_token_list = [
                            w2i['<sos>'], # <sos>
                            w2i['<eos>'], # <eos>
                            w2i['<unk>'], # <unk>
                            w2i['<pad>']  # <pad>
                          ]

    # set random seed
    torch.manual_seed(seed)

    # dataset and dataloader
    dataset = ProteinSequencesDataset(
                                        fasta_file,
                                        w2i,
                                        i2w,
                                        device,
                                        max_sequence_length = max_sequence_length
                                     )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # list to store all means/stds
    means, stds, sequenceIDs = [], [], []

    # iterate over dataloader
    with torch.no_grad():
        for batch in dataloader:
            src           = batch['input']
            batch_seq_ids = batch["reference"]
            sequenceIDs.extend(batch_seq_ids)

            # forward pass
            # no masking probability here
            # saving mean/std
            _,mean,logv,_ = model(
                                                        src, # Src in Transformer architecture
                                                        src, # Tgt in Transformer architecture
                                                     )
            std = torch.exp(0.5 * logv)

            means.append(mean)
            stds.append(std)

    # concatenate all means/stds
    means = torch.cat(means,dim=0)
    stds  = torch.cat(stds,dim=0)

    # convert to numpy : size [# seq in training set, latent size]
    if torch.cuda.is_available():
        # convert to numpy (if cuda is enabled, move to cpu first)
        means   = means.cpu().numpy()
        stds    = stds.cpu().numpy()
    else:
        # convert to numpy (already on cpu)
        means   = means.numpy()
        stds    = stds.numpy()

        
    # create column names
    latent_size = vars["latent_size"]
    column_names = [f"dim-{i+1}" for i in range(latent_size)]

    # create dataframe
    df_means = pd.DataFrame(
                                means,
                                columns = column_names,
                                index   = sequenceIDs
                             )  
    df_means.to_csv(output_filename_means)
    
    df_stds  = pd.DataFrame(
                                stds,
                                columns = column_names,
                                index   = sequenceIDs
                            )
    df_stds.to_csv(output_filename_stds)


####################################################### RELATED TO SAMPLING ######################################
# convert indices to tokens
def decode(
                array_indices       : ArrayLike,        # numpy array with indices
                array_logp          : ArrayLike,        # numpy array with log-probabilities (must be same shape as indices)
                array_Z             : ArrayLike,        # latent space value used in decoder to generate log-probabilities and indices
                array_energies      : ArrayLike,        # energies obtained from z
                i2w                 : dict,             # dictionary that maps index to a letter
                improper_token_list : List              # list of tokens that are not 20 valid AAs
            ) -> List:
    # check that arrays are of the same length
    assert  array_indices.shape == array_logp.shape, "Shape mismatch between indices and log-probabilities numpy arrays"
    # assert  array_indices.shape == array_logp.shape == array_certainty.shape, "Shape mismatch between indices and log-probabilities and certainty numpy arrays"

    # check that number of elements is the same in array_indices and array_Z
    assert array_indices.shape[0] == array_Z.shape[0], "Number of samples must be the same in indices and Z numpy arrays"

    # check that number of elements is the same between indices and energies
    assert array_indices.shape[0] == array_energies.shape[0], "Number of samples must be the same in indices and energies numpy arrays"
    
    # empty list; list element is a (sequence,energy,logp,z) tuple: z is a latent space vector used to create a sequence
    sequences               = []
    
    # get number of sequences and number of elements per sequence
    n_samples               = array_indices.shape[0]
    n_elements_per_sequence = array_indices.shape[1]

    print(f"Number of sequences to decode   : {n_samples}")
    print(f"Number of elements in a sequence: {n_elements_per_sequence}")

    
    # loop over sequences
    for n in range(n_samples):
        
        temp_decoded    = []  # for decoding of a sequence

        cumulative_logp = 0.0

        
        
        # extract associated latent vector
        associated_z    = array_Z[n,:][np.newaxis,:] # to make size (1, latent_size)

        # extract energy
        associated_energy = array_energies[n][0]

        # loop over elements within a sequence
        for k in range(n_elements_per_sequence):
            # get element
            element = array_indices[n,k]
            
            # if an element is not in improper token list, append it
            # improper token list: <sos>,<eos>,<unk>,<pad>
            if element not in improper_token_list:
                # append character
                temp_decoded.append(i2w[element])
                # increase logp
                cumulative_logp += array_logp[n,k]
                # # add ceratinty element
                # temp_certainty.append(array_certainty[n,k])
            else:
                break
        
        # append to a list
        # IF len(temp) > 0. Otherwise, first sampled token is improper token already
        if len(temp_decoded) > 0:
            sequences.append(
                                (   
                                    temp_decoded,
                                    associated_energy,
                                    cumulative_logp,
                                    associated_z
                                )
                            )

    return sequences

# sample from gaussian prior
def sample_tf_using_prior(
                            output_filename     : str,
                            pt_name             : str,
                            pickle_name         : str,
                            n_samples           : int = 10000,
                            seed                : int = 0,
                            max_number_of_steps : int = 100
                         ) -> List:
    # instantiate model object
    vars  = pickle.load(open(pickle_name, "rb" ))
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # force cpu, if only cpu is available
    vars["device"]  = device

    # instantiate model
    model = TransformerVAE(**vars)

    # load trained parameters
    checkpoint = torch.load(pt_name,map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # move to GPU
    model = model.to(device)

    print(f"Device: {device}")

    # get i2w map and eos_idx
    w2i,i2w  = default_w2i_i2w()
    # create a list of improper tokens
    improper_token_list = [
                            w2i['<sos>'], # <sos>
                            w2i['<eos>'], # <eos>
                            w2i['<unk>'], # <unk>
                            w2i['<pad>']  # <pad>
                          ]

    # set seed
    torch.manual_seed(seed)

    # check how many times we need to do in batches of 100
    n_times   = n_samples//100
    
    # record objects list
    records   = []

    # sequences list
    sequences = []

    with torch.no_grad():
        for _ in range(n_times):
            
            # sample 100 samples
            if torch.cuda.is_available():
                logp,indices,z,energies,_  = model.sample_from_latent_space(100,max_length = max_number_of_steps) 
                logp,indices,z,energies    = logp.cpu().numpy(), indices.cpu().numpy(), z.cpu().numpy(), energies.cpu().numpy()
            else:
                logp,indices,z,energies,_ = model.sample_from_latent_space(100,max_length = max_number_of_steps) 
                logp,indices,z,energies   = logp.numpy(), indices.numpy(),z.numpy(),energies.numpy()
            
            # extend list with a batch of decoded sequences
            # each element of a list is a tuple:
            # (
            #    decoded sequence,  as list: ["A","B","C"]
            #    energy,            as float number
            #    logp,              as float number
            #    z                  as numpy array
            #  )
            sequences.extend(
                                decode(
                                            indices,
                                            logp,
                                            z,
                                            energies,
                                            i2w,
                                            improper_token_list
                                        )
                             )
    
    # sort by energies (1st element of the tuple), although doesn't really matter
    sorted_sequences = sorted(sequences, key=lambda tup: tup[1],reverse = False)
    
    # for current seed maintain a list of observed sequences
    # samples may be identical but have different log-probabilties
    # since sequences are sorted by log-probability
    # we will keep every unique sequence with the highest log-probability
    observed_sequences = []

    # counter
    cnt       = 1

    # seq_tuple: (seq,list(energies),logp(seq),associated z,id,description) as (list[str],list(float),numpy,str,str)
    seq_tuple = []

    # dictionary seq2object
    seq2obj   = {}

    # combine data to have the following structure: {(seq,all energies, all logps, all zs, id, description)}
    for element in sorted_sequences:

        # obtain sequence and its logp
        seq               = "".join(element[0]) # here as string, not as list, list not hashable
        associated_energy = element[1]
        logp_seq          = element[2]
        associated_z      = element[3]

        # if we haven't seen this sequence yet
        if seq not in observed_sequences:
            id_ = f"seq-{cnt}-prior"
            # generate new instance of class SequenceData
            seq2obj[seq] = SequenceData(
                                            seq,
                                            id_,
                                            associated_energy,
                                            logp_seq,
                                            associated_z
                                       )
            # add to the observed sequences, increase the counter
            observed_sequences.append(seq)
            cnt += 1
        else:
            # update an object of type SequenceData (but we need to update a specific object, need a quick lookup!)
            # lookup: seq -> SequenceData instance
            seq2obj[seq].update_info(
                                        associated_energy,
                                        logp_seq,
                                        associated_z
                                    )
    # once data is combined we need to generate description, add to seq_tuple, add to records
    # loop over pairs (seq, SequenceData)
    for k,v in seq2obj.items():
        v.generate_description()      # to have a human readable description
        seq_tuple.append(v.toTuple)   # append to list as a tuple
        records.append(v.toSeqRecord) # append to list as SeqRecord

    
    # save records list as FASTA file
    SeqIO.write(records,output_filename,"fasta")

    # return seq_tuple
    return seq_tuple

# sample from gaussian variational posterior
def sample_tf_using_posterior(
                                output_filename      : str,
                                pt_name              : str,
                                pickle_name          : str,
                                fasta_file           : str,
                                max_number_of_steps  : int   = 500, # for how many steps to perform autoregressive sampling
                                n_samples            : int   = 10,
                                mini_batch_size      : int   = 10,
                                seed                 : int   = 0,
                                T                    : float = 0.0,
                                skip_first_element   : bool  = True
                             ) -> List:
    # instantiate model object
    vars  = pickle.load(open(pickle_name, "rb" ))

    # get max_seq_length from vars dictionary (subtract 1 as 1 is added in dataset constructor)
    max_sequence_length = vars["max_seq_length"] - 1
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # force cpu, if only cpu is available
    vars["device"]  = device

    # instantiate model
    model = TransformerVAE(**vars)

    # load trained parameters
    checkpoint = torch.load(pt_name,map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # move to GPU
    model = model.to(device)

    print(f"Device: {device}")

    # get i2w map and eos_idx
    w2i,i2w  = default_w2i_i2w()
    # create a list of improper tokens
    improper_token_list = [
                            w2i['<sos>'], # <sos>
                            w2i['<eos>'], # <eos>
                            w2i['<unk>'], # <unk>
                            w2i['<pad>']  # <pad>
                          ]

    # set seed
    torch.manual_seed(seed)

    # record list and counter
    records    = []
    cnt        = 1


    # dataset and dataloader
    dataset = ProteinSequencesDataset(
                                        fasta_file,
                                        w2i,
                                        i2w,
                                        device,
                                        max_sequence_length = max_sequence_length
                                     )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # check that dataloader contains only 1 element
    assert len(dataloader) == 1, "Dataloader should contain a single batch of size 1. Check FASTA file with which you construct dataset."

    # seq_tuple: (seq,logp(seq),associated z,id,description) as (list[str],float,numpy,str,str)
    seq_tuple = []

    # dictionary seq2object
    seq2obj   = {}

    n_times = n_samples // mini_batch_size
    
    # loop over elements of dataset: outer loop
    for batch in dataloader:
        
        # get data (here: batch = 1 sequence)
        src        = batch['input']
        reference_ = batch["reference"][0]

        print("Src size:",src.size())

        # empty sequence list
        sequences = []

        # sample indices
        with torch.no_grad():
            
            # sample
            for _ in range(n_times):
                logp,indices,z,energies   = model.sample_around_molecule(    
                                                                            src,                      
                                                                            mini_batch_size,               
                                                                            T                  = T, 
                                                                            max_length         = max_number_of_steps, 
                                                                            skip_first_element = skip_first_element
                                                                        )
                if torch.cuda.is_available():
                    # convert to numpy (if cuda is enabled, move to cpu first)
                    logp,indices,z,energies   = logp.cpu().numpy(), indices.cpu().numpy(), z.cpu().numpy(), energies.cpu().numpy()
                else:
                    # convert to numpy (already on cpu)
                    logp,indices,z,energies   = logp.numpy(), indices.numpy(), z.numpy(), energies.numpy()

        
                # decode
                sequences.extend(
                                    decode(
                                                    indices,
                                                    logp,
                                                    z,
                                                    energies,
                                                    i2w,
                                                    improper_token_list
                                                )
                                )
        
        # sort by energies
        sorted_sequences = sorted(sequences, key=lambda tup: tup[1],reverse = True)

        # for current seed maintain a list of observed sequences
        # samples may be identical but have different log-probabilties
        # since sequences are sorted by log-probability
        # we will keep every unique sequence with the highest log-probability
        observed_sequences = []

        # populate list of records
        for element in sorted_sequences:

            # obtain sequence and its logp
            seq               = "".join(element[0]) # as string
            associated_energy = element[1]
            logp_seq          = element[2]
            associated_z      = element[3]
            
            # if we haven't seen this sequence yet
            if seq not in observed_sequences:
                # create a description and id
                id_          = f"seq-{cnt}-temperature-{T}"

                # generate new instance of class SequenceData
                seq2obj[seq] = SequenceData(
                                                seq,
                                                id_,
                                                associated_energy,
                                                logp_seq,
                                                associated_z
                                            )
                # increase counter
                cnt += 1

                # append to a list of observed sequences
                observed_sequences.append(seq)
            else:
                # update an object of type SequenceData (but we need to update a specific object, need a quick lookup!)
                # lookup: seq -> SequenceData instance
                seq2obj[seq].update_info(
                                            associated_energy,
                                            logp_seq,
                                            associated_z
                                        )

        # once data is combined we need to generate description, add to seq_tuple, add to records
        # loop over pairs (seq, SequenceData)
        for k,v in seq2obj.items():
            v.generate_description(seedID=reference_)      # to have a human readable description
            seq_tuple.append(v.toTuple)                    # append to list as a tuple
            records.append(v.toSeqRecord)                  # append to list as SeqRecord
    

    # write a file
    SeqIO.write(records,output_filename,"fasta")

    ### output:
    ### seq_tuple: (seq,logp(seq),associated z,id,description) as (list[str],float,numpy,str,str)
    return seq_tuple
    
# sample seq2energy (same function will be used for prior and posterior samples)
# will estimate elbo (reconstruction+kl) and energy for sampled variants (prior and posterior)
def seq2_energy_tf( 
                        output_filename_intermediate : str,        # csv output template (reconstruction/kl/elbo/energy)
                        pt_name                      : str,        # pytorch file with pretrained layers
                        pickle_name                  : str,        # pickle file with model input parameters
                        fasta_file                   : str,        # fasta with sequences
                        n_samples                    : int = 100,  # how many times sample from latent
                        batch_size                   : int = 10,   # how many samples to process at once
                        seed                         : int = 0     # random seed for sampling
                  ):

    #### small fucntion that computes reconstruction loss and KL individually for each batch element
    def compute_loss_gaussian(
                                NLL    : torch.tensor, 
                                logp   : torch.tensor, # [N,S,Vocab Size]
                                target : torch.tensor, # [N,S]
                                mean   : torch.tensor, # [N,latent size]
                                logv   : torch.tensor  # [N,latent size]
                            ):
        # get number of batches [N] and sequence length [S]
        N,S,_ = logp.size()
    
        # merge dimensions
        target = target.view(-1)
        logp   = logp.view(-1, logp.size(2))

                
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)     # size of [N*S]
        NLL_loss = NLL_loss.reshape(N,S) # size of [N,S]

        
        # KL Divergence -> by batch
        KL_loss = -0.5 * torch.sum(
                                    1 + logv - mean.pow(2) - logv.exp(),
                                    dim = -1 # if this is not used, overall summation will happen
                                )
        # both will be tensors of size [N]
        return NLL_loss.sum(dim=-1), KL_loss
    
    #### small function to save intermedaite results into dataframe
    def save(
                np_array: ArrayLike, 
                indices : List,
                columns : List,
                filename: str
            )->None:
        df = pd.DataFrame(
                            np_array, 
                            columns = columns, 
                            index   = indices
                         )
        df.to_csv(filename)

    ################################################################# MAIN BODY #################################
    # create a map id (which is "reference" field in dataset) to (sequence,description)
    id2seq = {}
    for record in SeqIO.parse(fasta_file,"fasta"):
        id_         = record.id
        seq_        = record.seq
        desc_       = record.description
        id2seq[id_] = (seq_,desc_)

    # instantiate model object
    vars  = pickle.load(open(pickle_name, "rb" ))

    # get the max_seq_length (for dataset construction, adjust by subtracting 1)
    max_sequence_length = vars["max_seq_length"] - 1
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # force cpu, if only cpu is available
    vars["device"]  = device

    # instantiate model
    model = TransformerVAE(**vars)

    # load trained parameters
    checkpoint = torch.load(pt_name,map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # move to GPU
    model = model.to(device)

    print(f"Device: {device}")

    # get i2w map and eos_idx
    w2i,i2w  = default_w2i_i2w()

    # set seed
    torch.manual_seed(seed)

    # dataset and dataloader
    dataset = ProteinSequencesDataset(
                                        fasta_file,
                                        w2i,
                                        i2w,
                                        device,
                                        max_sequence_length = max_sequence_length
                                     )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # empty matrices: each for reconstruction/KL/ELBO/energies [ # sequences, # samples ]
    prediction_matrix_reconstruction = torch.zeros((len(dataset),n_samples))
    prediction_matrix_kl             = torch.zeros((len(dataset),n_samples))
    prediction_matrix_elbo           = torch.zeros((len(dataset),n_samples))
    prediction_matrix_energy         = torch.zeros((len(dataset),n_samples))

    # list for sequenceIDs
    sequenceIDs = []

    # NLL object
    NLL = nn.NLLLoss(ignore_index=dataset.pad_idx,reduction='none')

    with torch.no_grad():
        # run over batches
        for i, batch in enumerate(tqdm.tqdm(dataloader, 'Looping through mutation batches')):
            # extract necessary input
            src           = batch["input"]     # size [N,S]
            tgt           = batch["target"]    # size [N,T]
            reference_ids = batch["reference"] 
            sequenceIDs.extend(reference_ids)

            # run over number of samples
            for j in tqdm.tqdm(range(n_samples), 'Looping through number of samples for batch #: '+str(i+1)):

                # do forward pass
                # predicted energy: [Batch Size, 1] -> need to get rid of last dimension
                logp,mean,logv,predicted_energy = model(
                                                            src, # Src in Transformer architecture
                                                            src, # Tgt in Transformer architecture
                                                         )

                # compute reconstruction/KL
                # dimensionality [Batch Size]
                NLL_loss, KL_loss = compute_loss_gaussian(
                                                            NLL,
                                                            logp,
                                                            tgt,
                                                            mean,
                                                            logv
                                                         )
                elbo = NLL_loss + KL_loss
                
                # write to matrices: [# sequences; # samples]
                prediction_matrix_reconstruction[i*batch_size:i*batch_size+len(src),j] = NLL_loss
                prediction_matrix_kl[i*batch_size:i*batch_size+len(src),j]             = KL_loss
                prediction_matrix_elbo[i*batch_size:i*batch_size+len(src),j]           = elbo
                prediction_matrix_energy[i*batch_size:i*batch_size+len(src),j]         = predicted_energy.squeeze()

    
    # compute averages across dim=1 : [# of sequences]
    avg_pm_reconstruction = prediction_matrix_reconstruction.mean(dim=1,keepdim=False)
    avg_pm_kl             = prediction_matrix_kl.mean(dim=1,keepdim=False)
    avg_pm_elbo           = prediction_matrix_elbo.mean(dim=1,keepdim=False)
    avg_pm_energy         = prediction_matrix_energy.mean(dim=1,keepdim=False)
                

    ################################################################# intermediate results ########################
    # convert to numpy : size [# seq in training set, # samples]
    if torch.cuda.is_available():
        # convert to numpy (if cuda is enabled, move to cpu first)
        prediction_matrix_reconstruction,prediction_matrix_kl,prediction_matrix_elbo,prediction_matrix_energy = prediction_matrix_reconstruction.cpu().numpy(), prediction_matrix_kl.cpu().numpy(), prediction_matrix_elbo.cpu().numpy(), prediction_matrix_energy.cpu().numpy()
    else:
        # convert to numpy (already on cpu)
        prediction_matrix_reconstruction,prediction_matrix_kl,prediction_matrix_elbo,prediction_matrix_energy = prediction_matrix_reconstruction.numpy(), prediction_matrix_kl.numpy(), prediction_matrix_elbo.numpy(), prediction_matrix_energy.numpy()

    # column names: integers representing sample
    columns = [f"{i+1}" for i in range(n_samples)]

    # reconstruction errors
    save(
            prediction_matrix_reconstruction,
            sequenceIDs,
            columns,
            output_filename_intermediate + "-reconstruction.csv"
        )
    
    # KLs: will be identical
    save(
            prediction_matrix_kl,
            sequenceIDs,
            columns,
            output_filename_intermediate + "-kl.csv"
        )

    # ELBO
    save(
            prediction_matrix_elbo,
            sequenceIDs,
            columns,
            output_filename_intermediate + "-elbo.csv"
        )

    # energies
    save(
            prediction_matrix_energy,
            sequenceIDs,
            columns,
            output_filename_intermediate + "-energy.csv"
        )

    ################################################################# final results ########################
    final_data = {
                    "seq-id"            : sequenceIDs,
                    "reconstruction"    : avg_pm_reconstruction.tolist(),
                    "kl"                : avg_pm_kl.tolist(),
                    "elbo"              : avg_pm_elbo.tolist(),
                    "energy"            : avg_pm_energy.tolist()
                }
    
    return final_data

# single optimisation run -> running SGD on prior sample
def sgd_optimisation(
                                pt_name              : str,
                                pickle_name          : str,
                                logger               : logging.Logger,
                                learning_rate        : float,
                                delta_f_tol          : float,
                                max_opt_steps        : int
                            ):
    # instantiate model object
    vars  = pickle.load(open(pickle_name, "rb" ))
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # force cpu, if only cpu is available
    vars["device"]  = device
    logger.info(f"Device used: {device}")

    # instantiate model
    model = TransformerVAE(**vars)

    # load trained parameters
    checkpoint = torch.load(pt_name,map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # extract mlp2energy
    f_z = model.get_submodule("mlp2energy")

    # move to GPU
    f_z   = f_z.to(device)
    model = model.to(device)
    
    f_z.eval()

    # turn off the gradients everywhere in model f_z
    for param in f_z.parameters():
        param.requires_grad_(False)

    # we will be working with a single z, so set batch_size to 1
    batch_size          = 1
    latent_size         = vars["latent_size"]

    # this is dataset length (inclusing <eos>), so we will sample this number of times
    max_sequence_length = vars["max_seq_length"]

    # get i2w map and eos_idx
    w2i,i2w  = default_w2i_i2w()
    
    # create a list of improper tokens
    improper_token_list = [
                            w2i['<sos>'], # <sos>
                            w2i['<eos>'], # <eos>
                            w2i['<unk>'], # <unk>
                            w2i['<pad>']  # <pad>
                          ]

    # create an input and require gradients
    z = torch.nn.Parameter(
                                # our initial guess of vector z
                                torch.randn(
                                                [batch_size, latent_size],  # (1,latent size)
                                                dtype  = torch.float,
                                                device = device
                                            ),
                                requires_grad = True  
                           )
    str_  = "Initial value:\n"
    str_ += f"{z}"
    logger.info(str_)

    # optimizer for z
    optim = torch.optim.SGD([z], lr = learning_rate)
    
    # assume starting loss is inifinity
    prev_loss  = float('inf')

    ### loop over steps and keep track of progress
    records_progressive = []
    
    for step in range(max_opt_steps):

        # clone z to z_prev and detach from computational graph
        z_prev = z.clone().detach()
        
        # predict energy
        energy = f_z(z)

        # energy is [1,1] tensor, so we need a sum for backprop
        loss = energy.sum()

        # gradient step
        optim.zero_grad()
        loss.backward()
        optim.step()

        # print results: change in energy and L2 and L0 norms of vector z change

        # change in z; L2 and L0 norms of a change
        dz    = z - z_prev
        l2_dz = torch.linalg.vector_norm(dz,ord=2)
        #l0_dz = torch.linalg.vector_norm(dz,ord=0) # not sure it is good

        # change in function
        df = abs(loss.item() - prev_loss)

        # log results
        step_results_str  = f"Iteration:{step+1}:\n"
        step_results_str += f"Previous loss:{prev_loss:0.7f}, current loss:{loss.item():0.7f}, delta:{df:0.7f}\n"
        step_results_str += f"L2-norm of change in z: {l2_dz.item()}\n"
        step_results_str += f"Current z: \n {z.tolist()}\n"
        #step_results_str += f"L2-norm of change in z: {l2_dz.item():0.7f}, L0-norm of change in z: {l0_dz.item():0.7f}"
        logger.info(step_results_str)

        # reassign prev_loss variable
        prev_loss = loss.item()

        # obtain sequence from current z
        with torch.no_grad():
            # all of them should be batch size = 1

            # logp and indices correspond to collection scheme
            # right now: argmax
            # TODO: introduce random sampling
            logp ,indices, current_z, current_energy, entropies = model.sample_from_latent_space(
                                                                                                    batch_size,
                                                                                                    max_length = max_sequence_length,
                                                                                                    z          = z,
                                                                                                    argmax     = False # will do sampling
                                                                                                )
            if torch.cuda.is_available():
                logp, indices, current_z, current_energy = logp.cpu().numpy(), indices.cpu().numpy(), current_z.cpu().numpy(), current_energy.cpu().numpy()
            else:
                logp, indices, current_z, current_energy = logp.numpy(), indices.numpy(), current_z.numpy(), current_energy.numpy()
            
            # decode
            # [ temp_decoded, associated_energy, cumulative_logp, associated_z]
            out = decode(
                            indices,
                            logp,
                            current_z,
                            current_energy,
                            i2w,
                            improper_token_list
                        )
            assert len(out) == 1, "Something is wrong with decoding step."
            seq               = "".join(out[0][0])
            associated_energy = out[0][1]
            logp_seq          = out[0][2]

            # append to records
            records_progressive.append(
                                         SeqRecord(
                                                    Seq(seq),
                                                    id          = f"optimization-step-{step+1}",
                                                    description = f"Step:{step+1}/{max_opt_steps}||Logp:{logp_seq:0.7f}||Energy:{associated_energy:0.7f}|| Entropy (mean,max,std):[{entropies[0]:0.7f},{entropies[1]:0.7f},{entropies[2]:0.7f}]"
                                                  )
                                      )
            # break out of for loop
            if df < delta_f_tol:
                final_str  = f"Stopped after iteration {step+1} due to {df:0.10f} difference between energy values at two consecutive steps. Optimal function value: {loss.item():0.7f}"
                final_str += f"\tOptimal argument value (z): {z}"
                logger.info(final_str)

                break

    # here, we reached end of for loop without breaking it
    final_str  = f"Stopped because maximum number of optimisation steps ({max_opt_steps}) have been reached. Optimal function value: {loss.item():0.7f}"
    final_str += f"\tOptimal argument value (z): {z}"
    logger.info(final_str)


    # detach and convert to numpy
    if torch.cuda.is_available():
        # convert to numpy (if cuda is enabled, move to cpu first)
        opt_z,opt_loss   = z.detach().cpu().numpy(), loss.detach().cpu().numpy()
    else:
        # convert to numpy (already on cpu)
        opt_z,opt_loss   = z.detach().numpy(), loss.detach().numpy()

    # return tuple(opt_z,opt_energy,progressive list of decoded sequences throughout optimisation//last element if the optimal sequence)
    return opt_z,opt_loss,records_progressive 

# constraint optimisation of energy NN (z -> E(z))
# optimal z will lie in mean(seq)+-3*sigma(seq)
def trust_region_optimisation(
                                f_energy: nn.Module,        # NN to predict energy 
                                x0      : np.ndarray,       # starting point (x0,x1,x2...)
                                bounds  : List,             # List of tuples [(x0_min,x0_max),(x1_min,x1_max) ...] or Bounds class from scipy. Not sure which will work
                                logger  : logging.Logger,   # logger
                                device  : torch.device      # cpu/cuda
                             ):
    # run optimisation problem
    res = minimize(
                    f_energy,
                    x0, 
                    bounds       = bounds,
                    method       = 'trust-constr',
                    backend      = 'torch',
                    precision    = 'float64', 
                    tol          = 1e-8,
                    torch_device = device,
                    options      = {'disp': True}
                  )
    # check the output :
    # res.x       <- solution (x_opt)
    # res.fun     <- F(x_opt)
    # res.success <- whether optimisation is successful
    # res.message <- optimisation msg

    # construct log string
    log_string  = f"Optimisation successful: {res.success}.\n"
    log_string += f"Optimisation message   : {res.message}.\n"
    log_string += f"F(x_opt)               : {res.fun}.\n"
    log_string += f"x_opt                  : {res.x}.\n"
    logger.info(log_string)

    # return x_opt, f(x_opt)
    return res.x, res.fun
    
####################################################### RELATED TO PREDICTIVE MODELS:GRU/TF encoder ######################################
def predict_energies_using_tf_encoder(
                                        output_filename     : str,        # output (FASTA) filename
                                        pt_name             : str,        # pytorch file with pretrained layers
                                        pickle_name         : str,        # pickle file with model input parameters
                                        fasta_file          : str,        # fasta with sequences
                                        seed                : int = 0     # random seed for sampling
                                     ):

    # create a map id (which is "reference" field in dataset) to (sequence,description)
    id2seq = {}
    for record in SeqIO.parse(fasta_file,"fasta"):
        id_         = record.id
        seq_        = record.seq
        desc_       = record.description
        id2seq[id_] = (seq_,desc_)

    # instantiate model input params
    vars  = pickle.load(open(pickle_name, "rb" ))

    # get the max_seq_length (for dataset construction, adjust by subtracting 1)
    max_sequence_length = vars["max_seq_length"] - 1

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # force cpu, if only cpu is available
    #vars["device"]  = device

    # instantiate model -> TF predictive
    model = PredictorTFEncoder(**vars)

    # load trained parameters
    checkpoint = torch.load(pt_name,map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # move to GPU
    model = model.to(device)

    print(f"Device: {device}")

    # get i2w map and eos_idx
    w2i,i2w  = default_w2i_i2w()

    # set seed
    torch.manual_seed(seed)

    # record list
    records    = []

    # dataset and dataloader
    dataset = ProteinSequencesDataset(
                                        fasta_file,
                                        w2i,
                                        i2w,
                                        device,
                                        max_sequence_length = max_sequence_length
                                     )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # to sort based on energy
    to_sort = []

    # energies data, to save as csv
    energies_data = {}
    predicted_energies, references = [],[]

    i = 0 # currently processed
    # loop over elements of dataset: outer loop
    for batch in dataloader:
        
        # get data (here: batch = 1 sequence)
        src        = batch['input']
        reference_ = batch["reference"][0]

        print("Src size:",src.size())

        # sample indices
        with torch.no_grad():
            # forward pass to obtain a predicted energy
            predicted_energy = model(src)
            
            if torch.cuda.is_available():
                # convert to numpy (if cuda is enabled, move to cpu first)
                predicted_energy   = predicted_energy.cpu().numpy()[0][0]
            else:
                # convert to numpy (already on cpu)
                predicted_energy   = predicted_energy.numpy()[0][0]

        print("predicted energy: ", predicted_energy )
        
        # add to lists
        references.append(reference_)
        predicted_energies.append(predicted_energy)
    

        # write additional description
        additional_description = (
                                    f"||Model predicted energy:{predicted_energy}"
                                 )

        # add to_sort list  -> we need reference, to retrieve sequence post sorting
        to_sort.append(
                        (reference_,predicted_energy,additional_description)
                      )
    
    # do sorting now (on avg energy), construct records objects and save
    sorted_to_sort = sorted(to_sort,key=lambda tup: tup[1],reverse = False)

    # create records
    records = []
    #(id,avg_energy,additional description)
    for tpl in sorted_to_sort:
        id_                                   = tpl[0]
        reference_seq_, original_description_ = id2seq[id_][0], id2seq[id_][1]
        additional_description_               = tpl[2]

        # need to create SeqRecord object
        record_ = SeqRecord(
                                reference_seq_,
                                id = id_,
                                description = original_description_ + additional_description_
                           )

        records.append(record_)

    # save
    SeqIO.write(records,output_filename,"fasta")

    # construct energies_data
    energies_data['reference-id']     = references
    energies_data['estimated-energy'] = predicted_energies

    return energies_data







    


######################################################## RELATED TO PROTEIN ANALYSIS ################################

# creates default dictionary to be populated in protein_analysis function
# keys should be changed accordingly (!)
def get_accumulator(
                        amino_acid_list:list
                   ) -> defaultdict:
    # create default dictionary specifying keys explicitly
    standard_keys = [
                        "MW",
                        "aromaticity",
                        "instability",
                        "flexibility",
                        "gravy",
                        "isoelectric",
                        "helix",
                        "turn",
                        "sheet",
                        "reduced",
                        "oxidized"
                    ]
    # extend with AAs
    standard_keys.extend(amino_acid_list)
    
    accumulator = defaultdict(
                                list,
                                { k:[] for k in standard_keys }
                             )
    return accumulator
    
# compute sequence params using ProteinAnalysis
def protein_analysis(
                        sequence            : str,           # input is a sequence but as a string
                        sequence_id         : str,           # sequence id
                        amino_acid_list     : list,          # list of strings (change to list if it complains!)
                        results_accumulator : defaultdict,   # we will keep accumulating results
                        logger              : logging.Logger # logger to keep log info
                    ) -> None:
    
    # instantiate ProteinAnalysis object for a given sequence
    sequence_analysis = ProteinAnalysis(sequence)

    # analysis of AA %
    aa_pct_d = sequence_analysis.get_amino_acids_percent()

    # loop over AA % (if not found,set to 0.0)
    for key_aa in amino_acid_list:
        aa_value = aa_pct_d.get(key_aa,0.0)
        # add to accumulator
        results_accumulator[key_aa].append(aa_value)
    
    # molecular weight
    results_accumulator["MW"].append(
                                        sequence_analysis.molecular_weight()
                                    )

    # aromaticity
    results_accumulator["aromaticity"].append(
                                                sequence_analysis.aromaticity()
                                             ) 

    # instability index
    results_accumulator["instability"].append(
                                                sequence_analysis.instability_index()
                                             ) 

    # flexibility -> is a list
    # we take average (not sure it is correct)
    flexibility_list =  sequence_analysis.flexibility()
    if len(flexibility_list) > 0:
        avg_flexibility  = sum(flexibility_list)/len(flexibility_list)
    else:
        # log if flexibility is not computed as expected
        avg_flexibility  = 0.0
        str_  = f"Issue with sequence ID : {sequence_id}\n"
        str_ += f"Sequence               : {sequence}\n"
        str_ += f"Flexibility list       : {flexibility_list}\n"
        str_ += f"Avg flexibility set to : {avg_flexibility}\n"
        logger.info(str_)

    results_accumulator["flexibility"].append(
                                                avg_flexibility
                                             )
    
    # gravy
    results_accumulator["gravy"].append(
                                            sequence_analysis.gravy()
                                       )

    # isoelectric point
    results_accumulator["isoelectric"].append(
                                                sequence_analysis.isoelectric_point()
                                             )

    # secondary structure fraction
    secondary_structure_fraction = sequence_analysis.secondary_structure_fraction()
    results_accumulator["helix"].append(secondary_structure_fraction[0])
    results_accumulator["turn"].append(secondary_structure_fraction[1])
    results_accumulator["sheet"].append(secondary_structure_fraction[2])

    # molar extinction coefficient
    epsilon_prot = sequence_analysis.molar_extinction_coefficient()
    results_accumulator["reduced"].append(epsilon_prot[0])
    results_accumulator["oxidized"].append(epsilon_prot[1])

# call protein_analysis for all sequences
def complete_protein_analysis(
                                sequences       : List,
                                sequence_ids    : List,
                                amino_acid_list : List,
                                logger          : logging.Logger
                             ) -> pd.DataFrame:
    # get accumulator
    data_accumulator = get_accumulator(amino_acid_list)

    # call protein analysis on each sequence in a list
    for id_,sequence_ in zip(sequence_ids,sequences):
        protein_analysis(
                            sequence_,
                            id_,
                            amino_acid_list,
                            data_accumulator,
                            logger
                        )
    
    # generate dataframe out of data_accumulator
    data = pd.DataFrame.from_dict(data_accumulator)

    return data

# compute correlations of two dataframes with identical number of rows
def correlate(
                df1 : pd.DataFrame,
                df2 : pd.DataFrame
            ) -> pd.DataFrame:
    assert df1.shape[0] == df2.shape[0], "Dataframes must have identical number of rows"
    return pd.concat([df1, df2], axis=1, keys=['df1', 'df2']).corr(method="spearman").loc['df2', 'df1']

# computes all correlations and saves all intermediate results
def present_results_of_protein_analysis(
                                            seq_tuple         : Tuple,
                                            amino_acids       : List,
                                            logger            : logging.Logger,
                                            filename_template : str
                                       ) -> pd.DataFrame:
    # need to extract first and third element from a tuple
    string_seqs = []
    Z           = []
    seq_ids     = []
    for element in seq_tuple:
        # create a string
        string_seqs.append(
                            "".join(element[0])
                          )
        # append to Z
        Z.append(
                    element[2]
                )

        # append to seq ids
        seq_ids.append(element[3])
    
    # redefine Z to be numpy array
    Z = np.concatenate(
                        Z,
                        axis = 0
                      )
    
    # check that number of elements in string_seqs matches number of elements in Z
    assert len(string_seqs) == Z.shape[0], "Number of generated strings must match number of vectors from latent space"

    # need to create a dataframe from Z
    # create a list of column names: [dim-1,dim-2,...,dim-N]
    z_columns = [f"dim-{dim+1}" for dim in range(Z.shape[1])]
    
    ########## create dataframes

    # for Z
    df_z      = pd.DataFrame(
                                data    = Z,
                                columns = z_columns,
                                index   = seq_ids
                            )
    # for amino-acid properties
    df_seq_properties = complete_protein_analysis(
                                                    string_seqs,
                                                    seq_ids,
                                                    amino_acids,
                                                    logger
                                                 )
    # reset index
    df_seq_properties.index = seq_ids
    #df_seq_properties.set_axis(seq_ids,axis="index")
    
    # check they have the same number of rows
    assert df_z.shape[0] == df_seq_properties.shape[0], "Number of rows in dataframes must be identical"

    # compute correlation df
    df_corr_latent = correlate(
                                df_z,
                                df_seq_properties
                              )
    
    # visualize as heatmap
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=df_corr_latent.shape)
    sns.heatmap(
                df_corr_latent, 
                ax                  = ax,
                linewidths          = 0.5   # add small separating line between squares in heatmap
               )
    
    ################## save everything
    # 1. dataframe for z
    df_z.to_csv(f"{filename_template}-dataframe-z.csv")

    # 2. dataframe sequence properties
    df_seq_properties.to_csv(f"{filename_template}-dataframe-sequence-properties.csv")

    # 3. correlation dataframe
    df_corr_latent.to_csv(f"{filename_template}-dataframe-correlations.csv")

    # 4. heatmap
    fig.savefig(f"{filename_template}-correlations-as-heatmap.pdf")

    return df_seq_properties


#################################################### MISCELLANEOUS ####################################
# get a logger
def setup_logger(
                    logger_name : str, 
                    log_file    : str, 
                    level       : int =logging.INFO
                ) -> logging.Logger:
    l             = logging.getLogger(logger_name)
    formatter     = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    fileHandler   = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

    return l

# load checkpoint
def load_checkpoint(model,optimizer,pt_checkpoint):
    checkpoint = torch.load(pt_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# create and save dataframe as csv file
def save_data_to_csv(
                        filename  : str,
                        list_vals : List,
                        list_ids  : List
                    ) -> None:
    '''
    INPUT:
        filename  : filename of csv file, string
        list_vals : list of lists. Each element(list) should contain numeric information, list
        list_ids  : list of strings. Each element(string) contains a name for a list in the same position from list_vals variable, list
    OUTPUT:
        None. Saves a csv file
    '''
    assert len(list_vals) == len(list_ids), "Length mismatch between list_vals  and list_ids"
    # construct dictionary
    data = {}
    for key,val in zip(list_ids,list_vals):
        data[key] = val

    # create a dataframe from it
    df = pd.DataFrame.from_dict(data)

    # save df
    df.to_csv(filename,index=False)

# compute statistics for a set of FASTA records
def compute_stats(
                    records: list,
                    filename: str
                 ) -> None:
    
    # create a list of sequences
    seq_list = [ record.seq for record in records ]   

    # total number of sequences
    n_total  = len(seq_list)

    # length: numpy array
    lengths  = np.array( [ len(seq) for seq in seq_list ] )

    # different stats -> can be expanded later if needed
    ave_length = np.mean(lengths)
    std_length = np.std(lengths)
    max_length = np.max(lengths)
    min_length = np.min(lengths)
    pct10      = np.percentile(lengths,10)
    pct25      = np.percentile(lengths,25)
    pct50      = np.percentile(lengths,50)
    pct75      = np.percentile(lengths,75)
    pct90      = np.percentile(lengths,90)

    # create string
    str_ = "FASTA file statistics\n"
    str_ += f"\t Number of sequences : {n_total}\n"
    str_ += f"\t Average length      : {ave_length: 0.4f}\n"
    str_ += f"\t Std of length       : {std_length: 0.4f}\n"
    str_ += f"\t Max    length       : {max_length: 0.4f}\n"
    str_ += f"\t Min    length       : {min_length: 0.4f}\n"
    str_ += f"\t 10th percentile     : {pct10: 0.4f}\n"
    str_ += f"\t 25th percentile     : {pct25: 0.4f}\n"
    str_ += f"\t 50th percentile     : {pct50: 0.4f}\n"
    str_ += f"\t 75th percentile     : {pct75: 0.4f}\n"
    str_ += f"\t 90th percentile     : {pct90: 0.4f}\n"
    
    # save in file
    f = open(filename,"w")
    f.write(str_)
    f.close()

# pass basic filtering criteria: presence of non-canonical AAs
def pass_filter(
                    record: SeqRecord,
                    w2i   : dict
                  ) -> bool:
        """
        INPUT:
            record: record object of BioPython module
        OUTPUT:
            True if no weird amino acids are found. else, False.
        """
        # obtain amino acids as a set
        set_amino_acids                  = set(w2i.keys())

        # obtain set of amino acids in a given record
        unique_set_of_amino_acids_in_ID = set(list(str(record.seq)))

        # do set difference
        set_diff                        = unique_set_of_amino_acids_in_ID - set_amino_acids

        # if set is empty, filtering criteri passed, else not
        if len(set_diff)==0:
            return True
        else:
            return False 

# obtain best results from dataframe
def obtain_best_results(
                        file_    : str,
                        columns_ : list
                       ) -> Tuple[List,List,List]:
    
    # read csv, explicitly saying no header
    try:
        df = pd.read_csv(file_,header=None)
    except Exception as e:
        print("Exception occured:",e)
        # create empty dataframe
        df         = pd.DataFrame()

    # if dataframe is empty
    if df.empty:
        return None,None,None
    
    # add columns
    df.columns = columns_

    # indices of max values (seperate for identity and similarity and bitscore)
    indices_identity   = df.groupby('qseqid').idxmax()['pident'].values
    indices_similarity = df.groupby('qseqid').idxmax()['ppos'].values
    indices_bitscore   = df.groupby('qseqid').idxmax()['bitscore'].values

    # obtain max values
    best_identities         = df['pident'].values[indices_identity]
    best_similarities       = df['ppos'].values[indices_similarity]
    best_bitscores          = df['bitscore'].values[indices_bitscore]

    # resave csv file with columns
    df.to_csv(
                 file_,
                 index = False # do not write row names
              )

    # we will return lists because they are easy to extend
    return list(best_identities), list(best_similarities), list(best_bitscores)

# compute (percentile) stats of an array
def stats_compute(
                    items      : list,
                    identifier : str
                 ) -> str:
    # number of elements
    n_total  = len(items)

    # convert to numpy
    items = np.array(items)

    # different stats -> can be expanded later if needed
    ave_   = np.mean(items)
    std_   = np.std(items)
    max_   = np.max(items)
    min_   = np.min(items)
    pct10  = np.percentile(items,5)
    pct25  = np.percentile(items,25)
    pct50  = np.percentile(items,50)
    pct75  = np.percentile(items,75)
    pct90  = np.percentile(items,95)

    # create string
    str_ = f"{identifier} statistics \n"
    str_ += f"\t Number of elements       : {n_total}\n"
    str_ += f"\t Average                  : {ave_: 0.4f}\n"
    str_ += f"\t Std                      : {std_: 0.4f}\n"
    str_ += f"\t Max                      : {max_: 0.4f}\n"
    str_ += f"\t Min                      : {min_: 0.4f}\n"
    str_ += f"\t 5th percentile           : {pct10: 0.4f}\n"
    str_ += f"\t 25th percentile          : {pct25: 0.4f}\n"
    str_ += f"\t 50th percentile          : {pct50: 0.4f}\n"
    str_ += f"\t 75th percentile          : {pct75: 0.4f}\n"
    str_ += f"\t 95th percentile          : {pct90: 0.4f}\n"
    
    return str_

# do mutations for FoldX
def mutate(
                wt          : list,                # WT sequence as list
                pos_mutate  : list,        # list of positions to mutate
                chainA_only : bool = True # whether to add to chain B
          ) -> Tuple[str,List]:
    
    # empty FoldX representation
    representation = ""

    # loop over sampled positions
    for pos in pos_mutate:
        original_aa = wt[pos] # original AA in WT
        new_aa      = np.random.choice(
                                        [aa for aa in 'ARNDCEQGHILKMFPSTWYV' if aa != original_aa]
                                      ) # new AA
        # mutate sequence
        wt[pos] = new_aa
        
        # add to representation (keep in mind +1 here!)
        representation += f"{original_aa}A{pos+1}{new_aa},"
        
        # add chain B if asked
        if chainA_only == False:
            representation += f"{original_aa}B{pos+1}{new_aa},"

    
    # at the end we have all necessary representations
    # just need to replace final , with ;
    representation = re.sub(r".$", ";", representation)

    return representation,wt