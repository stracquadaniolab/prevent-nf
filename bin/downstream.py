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

import torch
import torch.nn
from torch.utils.data import DataLoader
from Bio import SeqIO
import numpy as np
import pandas as pd
import typer
import logging
from typing import List 
import pickle
from models import TransformerVAE
from utilities import (
                        SequenceData,
                        default_w2i_i2w,
                        ProteinSequencesDataset,
                        decode,
                        get_means_stds,            # get means and stds of q(z|x) for a given dataset
                        sample_tf_using_prior,
                        sample_tf_using_posterior,
                        seq2_energy_tf,
                        predict_energies_using_tf_encoder,
                        sgd_optimisation,          # completely unconstrained optimisation, performed using SGD pytorch
                        trust_region_optimisation, # optimisation with trust-region, some constraints are put on x0, performed using autograd-minimize
                        setup_logger,
                        obtain_best_results,
                        stats_compute,
                        present_results_of_protein_analysis
                      )

# typer setup
app           = typer.Typer()
tf_app        = typer.Typer() # for Transformer VAE models
predictor_app = typer.Typer() # for predictive models (GRU/TF encoder)
protein_app   = typer.Typer() # for protein analysis

app.add_typer(tf_app, name="transformer") 
app.add_typer(predictor_app, name="predictor")
app.add_typer(protein_app,name="protanalysis") 


#### GLOBAL CONST
# output columns for blastp
CSV_COLUMNS = ['qseqid','sseqid','score','bitscore','evalue','pident','ppos','qcovs']


######################################################### VAE ########################

# sample from prior distribution
# sample sequence and energy, combine collapse identical sequences (by averaging energy and logp)
@tf_app.command("sample-prior")
def transformer_sample_prior(
                                pt_name                   : str, # pytorch file (trained model params)
                                pickle_name               : str, # pickle file (input model params)
                                pickle_name_for_seq_tuple : str, # pickle filename to save seq_tuple
                                output_fasta_filename     : str, # fasta filename to save VAE samples
                                n_samples                 : int = 10000,
                                seed                      : int = 0,
                                max_number_of_steps       : int = 100
                            ) -> None:
    
    seq_tuple = sample_tf_using_prior(
                                        output_fasta_filename,
                                        pt_name,
                                        pickle_name,
                                        n_samples             = n_samples,
                                        seed                  = seed,
                                        max_number_of_steps   = max_number_of_steps
                                      )

    # seq_tuple: (seq,logp(seq),associated z,id,description,seq certainty) as (list[str],float,numpy,str,str,list)
    dict_to_pickle = {
                        "WT" : None, 
                        "VAE": seq_tuple
                     }

    # save as pickle
    pickle.dump(
                    dict_to_pickle,
                    open(pickle_name_for_seq_tuple,"wb")
                )

# seq2energy prediction: take sampled sequences from "transformer_sample_prior" function
# run through encoder, generate {n_samples} latent variables z, decode energy
@tf_app.command("seq2energy")
def transformer_seq2energy_prior(
                                    output_filename_main           : str,       # filename of main output
                                    output_filename_template_csv   : str,       # filename template to save individual energies/elbos/etc
                                    pt_name                        : str,       # pytorch file (trained model params)
                                    pickle_name                    : str,       # pickle file (input model params)
                                    clean_samples_fasta            : str,       # fasta file with clean samples
                                    n_samples                      : int = 300, # total number of energies computed for a sequence
                                    batch_size                     : int = 100, # number of energies computed in a single pass of NN (should divide n_samples evenly)
                                    seed                           : int = 0    # random seed for sampling
                                ):
    
    # call seq2_energy_tf; save dictionary "data" as csv [main output]
    data = seq2_energy_tf(
                            output_filename_template_csv,
                            pt_name,
                            pickle_name,
                            clean_samples_fasta,
                            n_samples      = n_samples,
                            batch_size     = batch_size,
                            seed           = seed
                         )

    # create dataframe and save
    df = pd.DataFrame(data)
    df.to_csv(output_filename_main)

# sample from posterior distribution
@tf_app.command("sample-seeded")
def transformer_sample_posterior(
                                        pt_name                   : str, # pytorch file with trained model params
                                        pickle_name               : str, # pickle file with input params
                                        pickle_name_for_seq_tuple : str, # filename for a pickle file to save VAE (+WT) results
                                        seed_fasta_file           : str, # fasta file with single seed sequence
                                        output_fasta_filename     : str, # fasta filename to save VAE samples
                                        max_number_of_steps       : int   = 500, 
                                        n_samples                 : int   = 10,
                                        mini_batch_size           : int   = 10,
                                        seed                      : int   = 0,
                                        T                         : float = 0.0,
                                        skip_first_element        : bool  = True  
                                ) -> None:
    
    seq_tuple = sample_tf_using_posterior(
                                            output_fasta_filename,
                                            pt_name,
                                            pickle_name,
                                            seed_fasta_file,
                                            max_number_of_steps = max_number_of_steps,
                                            n_samples           = n_samples,
                                            mini_batch_size     = mini_batch_size,
                                            seed                = seed,
                                            T                   = T,
                                            skip_first_element  = skip_first_element
                                         )

    # save tuple
    # take WT from seed fasta file
    WT = [
                str(record.seq) for  record in SeqIO.parse(seed_fasta_file,"fasta")
         ][0]
    # seq_tuple: (seq,logp(seq),associated z,id,description,seq certainty) as (list[str],float,numpy,str,str,list)
    dict_to_pickle = {
                        "WT" : WT, 
                        "VAE": seq_tuple
                     }

    # save as pickle
    pickle.dump(
                    dict_to_pickle,
                    open(pickle_name_for_seq_tuple,"wb")
                )

# running SGD on energy function
@tf_app.command("sgd-optimization")
def optimize_latent_space(
                            pt_name                   : str,            # pytorch file with trained model params
                            pickle_name               : str,            # pickle file with input params
                            logger_name               : str,            # logger name to keep progress (useful for debugging or adapting parameters)
                            interm_results_template   : str,            # intermediate results filename template
                            best_results_filename     : str,            # best results filename
                            seed                      : int   = 0,      # random seed (important for random starting point)
                            learning_rate             : float = 0.0001, # learning rate for SGD
                            n_restarts                : int   = 5,      # number of restarts withing a single optimization run
                            delta_f_tol               : float = 0.0001, # tolerance for function change: if function change is smaller, stop the optimisation
                            max_opt_steps             : int   = 100,    # maximum number of optimisation steps
                         )->None:
    # set random seed
    torch.manual_seed(seed)
    
    # get logger
    logger = setup_logger( 'logger', logger_name, level = logging.INFO )
    logger.info(f"Random seed: {seed}")

    # to keep track of best results
    best_energy  = float('inf')
    best_results = {
        "opt_energy": None,
        "opt_z"     : None,
        "sequences" : None
    }

    # loop over attempts
    for restart in range(n_restarts):
        logger.info(f"Restart : {restart+1}")
        logger.info("-"*30)
        
        # get optimal results (optimal z; optimal energy; list of sequences obtained throughout optimisation)
        opt_z, opt_energy, sequences = sgd_optimisation(
                                                        pt_name,
                                                        pickle_name,
                                                        logger,
                                                        learning_rate,
                                                        delta_f_tol,
                                                        max_opt_steps
                                                       )
        filename_ = f"{interm_results_template}-restart-{restart}.fasta"
        result_string = f"Restart {restart+1}, optimal energy: {opt_energy}; optimisation progress saved in file:{filename_}"
        logger.info(result_string)

        # save results
        SeqIO.write(
                        sequences,
                        filename_,
                        "fasta"
                   )
        logger.info("-"*30)

        # determine best results
        if opt_energy < best_energy:
            best_results["opt_energy"] = opt_energy
            best_results["opt_z"]      = opt_z
            best_results["sequences"]  = sequences
            #logger.info(f"Best energy:{best_energy:0.7f}, current energy:{opt_energy:0.7f}")
            
            # redefine best(smallest) energy
            best_energy = opt_energy
            

    # now we can write best results
    result_string = f"Best energy achieved: {best_energy}; optimisation progress saved in file:{best_results_filename}"
    logger.info(result_string)

    # save
    SeqIO.write(
                    best_results["sequences"],
                    best_results_filename,
                    "fasta"
               )

# running trust region optimisation on energy function
# here: within N(0,1) but later need to change for N(mu(seq),std(seq))
##################### KEEPING THIS FOR NOW, MAY DELETE LATER #####################
@tf_app.command("trust-region-optimization")
def optimize_latent_space_trust_region(
                                            pt_name                   : str,            # pytorch file with trained model params
                                            pickle_name               : str,            # pickle file with input params
                                            logger_name               : str,            # logger name to keep progress (useful for debugging or adapting parameters)
                                            output_filename_fasta     : str,            # FASTA filename for results
                                            pickle_name_for_seq_tuple : str,            # pickle filename for results
                                            seed_fasta_file           : str   = None,   # FASTA file with a seed sequence (if not provided, prior distribution is used)
                                            max_number_of_steps       : int   = 100,    # maximum number of recurrent steps in sampling procedure (i.e. equal to WT seq length)
                                            n_samples                 : int   = 500,    # how many samples to generate from a given Z (should be a multiple of 100!)
                                            seed                      : int   = 0,      # random seed (important for random starting point)
                                      )->None:
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # get logger
    logger = setup_logger( 'logger', logger_name, level = logging.INFO )
    logger.info(f"Random seed: {seed}")

    # get the model
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

    # double precision
    model = model.double()
    f_z   = f_z.double()


    # batch size is always 1 here
    batch_size          = 1
    latent_size         = vars["latent_size"]


    # get i2w map and eos_idx
    w2i,i2w  = default_w2i_i2w()
    # create a list of improper tokens
    improper_token_list = [
                            w2i['<sos>'], # <sos>
                            w2i['<eos>'], # <eos>
                            w2i['<unk>'], # <unk>
                            w2i['<pad>']  # <pad>
                          ]

    ##### need to obtain mu/sigma if seed_fasta_file is provided
    if seed_fasta_file != None:
        
        ############## USING SEEDED DISTRIBUTION #################

        # get max_seq_length from vars dictionary (subtract 1 as 1 is added in dataset constructor)
        max_sequence_length = vars["max_seq_length"] - 1    
        # dataset and dataloader
        dataset = ProteinSequencesDataset(
                                            seed_fasta_file,
                                            w2i,
                                            i2w,
                                            device,
                                            max_sequence_length = max_sequence_length
                                        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # check that dataloader contains only 1 element
        assert len(dataloader) == 1, "Dataloader should contain a single batch of size 1. Check FASTA file with which you construct dataset."

        # do a forward pass to obtain mu/sigma
        with torch.no_grad():
            for batch in dataloader:
                # get data (here: batch = 1 sequence)
                src        = batch['input']
                reference_ = batch["reference"][0]

                # obtain mu and sigma
                _,mean,logv,_ = model(
                                        src, # Src in Transformer architecture
                                        src, # Tgt in Transformer architecture
                                        masking_prob = 0.0 # masking probability to mask Tgt
                                      )
                std  = torch.exp(0.5 * logv)

                # sample random value (will be a starting point)
                z = torch.randn(
                                    [batch_size, latent_size],
                                    dtype  = torch.double,
                                    device = device
                                )
                # [Batch Size, Latent Size] # this is our latent representation
                x0 = z * std + mean

                # create bounds: [Batch Size, Latent Size]
                left_bound  = mean - 3.0 * std
                right_bound = mean + 3.0 * std

                logger.info(f"Optimisation run for the seeded distribution for sequence {reference_}")
                logger.info(f"mean (as tensor)            : {mean}")
                logger.info(f"std (as tensor)             : {std}")
                logger.info(f"x0 (as tensor)              : {x0}")
                logger.info(f"left_bound (as tensor)      : {left_bound}")
                logger.info(f"right_bound (as tensor)     : {right_bound}")

                # move to CPU and to numpy/list
                x0 = np.array(
                                x0.squeeze(0).tolist() # [batch size = 1, latent size] -> [latent_size] -> move to CPU
                             )
                left_bound  = left_bound.squeeze(0).tolist()
                right_bound = right_bound.squeeze(0).tolist()

                bounds = [(left_bound[i],right_bound[i]) for i in range(latent_size)]
                
                logger.info(f"Initial starting point : {x0}")
                logger.info(f"Bounds                 : {bounds}")

                print("x0    :", x0)
                print("bounds:", bounds)
    
    ############## USING PRIOR DISTRIBUTION #################
    else:
        logger.info("Optimisation run for the prior distribution")
        reference_ = "None"
        
        # create starting point
        # x0 is a random draw from N(0,I)
        x0                  = np.random.randn(latent_size) 
        # create bounds: every coordinate of x should be between -3.0 and 3.0
        bounds = [(-3.0,3.0)]*latent_size

        logger.info(f"Initial starting point: {x0}")
        logger.info(f"Bounds                : {bounds}")

        print("x0    :", x0)
        print("bounds:", bounds)

    # call the function -> will return double
    z_opt, energy_opt = trust_region_optimisation(
                                                    f_z,
                                                    x0,
                                                    bounds,
                                                    logger,
                                                    device
                                                 )
    
    # now we can do sampling using z_opt (which corresponds to energy_opt value)
    # record objects list
    records   = []

    # sequences list
    sequences = []

    # we will do argmax and categorical sampling
    # check how many times we need to do in batches of 100
    n_times   = n_samples//100

    # need to batchify z; right now it is just a list, we need to create a torch.tensor of this list and copy it 100 times
    z_tensor = torch.tensor(
                                [z_opt],
                                dtype  = torch.double,
                                device = device
                           ) # should create a tensor of size [1,latent_size]
    z_tensor_batch = z_tensor.repeat([100,1]) # should be of size [100,latent_size] where each row is a copy of 1st row
    
    
    with torch.no_grad():
        ############################################## argmax sampling
        # sample and convert to numpy
        if torch.cuda.is_available():
            logp,indices,z_out,energies,_  = model.sample_from_latent_space(
                                                                            1,
                                                                            max_length = max_number_of_steps,
                                                                            z          = z_tensor,
                                                                            argmax     = True
                                                                        ) 
            logp,indices,z_out,energies    = logp.cpu().numpy(),indices.cpu().numpy(),z_out.cpu().numpy(),energies.cpu().numpy()
        else:
            logp,indices,z_out,energies,_ = model.sample_from_latent_space(     
                                                                            1,
                                                                            max_length = max_number_of_steps,
                                                                            z          = z_tensor,
                                                                            argmax     = True
                                                                        ) 
            logp,indices,z_out,energies   = logp.numpy(),indices.numpy(),z_out.numpy(),energies.numpy()
        
        # decode sequences and append to sequence list
        sequences.extend(
                            decode(
                                        indices,
                                        logp,
                                        z_out,
                                        energies,
                                        i2w,
                                        improper_token_list
                                    )
                            )
        ############################################## categorical sampling (T=1.0)
        for _ in range(n_times):
            # sample and convert to numpy
            if torch.cuda.is_available():
                logp,indices,z_out,energies,_  = model.sample_from_latent_space(
                                                                                100,
                                                                                max_length = max_number_of_steps,
                                                                                z          = z_tensor_batch,
                                                                                argmax     = False
                                                                           ) 
                logp,indices,z_out,energies    = logp.cpu().numpy(),indices.cpu().numpy(),z_out.cpu().numpy(),energies.cpu().numpy()
            else:
                logp,indices,z_out,energies,_ = model.sample_from_latent_space(     
                                                                                100,
                                                                                max_length = max_number_of_steps,
                                                                                z          = z_tensor_batch,
                                                                                argmax     = False
                                                                          ) 
                logp,indices,z_out,energies   = logp.numpy(),indices.numpy(),z_out.numpy(),energies.numpy()
            
            # decode sequences and append to sequence list
            sequences.extend(
                                decode(
                                            indices,
                                            logp,
                                            z_out,
                                            energies,
                                            i2w,
                                            improper_token_list
                                        )
                             )

    # create SeqRecords and save as FASTA
    # copied from sample_tf_using_prior
    
    # sequences we already observed
    observed_sequences = []

    # counter
    cnt       = 1

    # seq_tuple: (seq,list(energies),logp(seq),associated z,id,description) as (list[str],list(float),numpy,str,str)
    seq_tuple = []

    # dictionary seq2object
    seq2obj   = {}

    # combine data to have the following structure: {(seq,all energies, all logps, all zs, id, description)}
    for element in sequences:

        # obtain sequence and its logp
        seq               = "".join(element[0]) # here as string, not as list, list not hashable
        associated_energy = element[1]
        logp_seq          = element[2]
        associated_z      = element[3]

        # if we haven't seen this sequence yet
        if seq not in observed_sequences:
            # since we haven't sorted sequences, first element of ${sequences} is the one with argmax // actually it will be first one anyways
            if cnt == 1:
                id_ = f"seq-{cnt}-argmax"
            else:
                id_ = f"seq-{cnt}"
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
        v.generate_description(seedID = reference_)      # to have a human readable description
        seq_tuple.append(v.toTuple)                      # append to list as a tuple
        records.append(v.toSeqRecord)                    # append to list as SeqRecord

    
    # save records list as FASTA file
    SeqIO.write(records,output_filename_fasta,"fasta")

    # seq_tuple: (seq,logp(seq),associated z,id,description,seq certainty) as (list[str],float,numpy,str,str,list)
    dict_to_pickle = {
                        "WT" : None, 
                        "VAE": seq_tuple
                     }

    # save as pickle
    pickle.dump(
                    dict_to_pickle,
                    open(pickle_name_for_seq_tuple,"wb")
                )


@tf_app.command("estimate-latent-space")
def latent_space_estimation(
                            output_filename_means : str,         # filename for means of q(z|x)
                            output_filename_stds  : str,         # filename for stds of q(z|x) 
                            pt_name               : str,         # pytorch file with pretraine dlayers
                            pickle_name           : str,         # pickle file with model inputs
                            fasta_file            : str,         # FASTA file with sequences
                            batch_size            : int = 4,     # how many sequences to process in a batch
                            max_sequence_length   : int = 500,   # max sequence length for Dataset construction
                            seed                  : int = 0
                           ):
    # call function get_means_stds
    get_means_stds(
                    output_filename_means,
                    output_filename_stds,
                    pt_name,
                    pickle_name,
                    fasta_file,
                    batch_size          = batch_size,
                    max_sequence_length = max_sequence_length,
                    seed                = seed
                )

# obtain mu/sigma from a seed sequence -> construct bounds -> call trust_region_optimisation
# @tf_app.command("trust-region-optimization-seeded")
# def optimize_latent_space_trust_region_seeded(
#                                                 pt_name                   : str,            # pytorch file with trained model params
#                                                 pickle_name               : str,            # pickle file with input params
#                                                 seed_fasta                : str,            # FASTA file with a seed sequence
#                                                 logger_name               : str,            # logger name to keep progress (useful for debugging or adapting parameters)
#                                                 best_results_filename     : str,            # best results filename
#                                                 seed                      : int   = 0,      # random seed (important for random starting point)
#                                              )->None:
#     # set random seed
#     torch.manual_seed(seed)
#     np.random.seed(seed)
    
#     # get logger
#     logger = setup_logger( 'logger', logger_name, level = logging.INFO )
#     logger.info(f"Random seed: {seed}")

#     # get the model
#     # instantiate model object
#     vars  = pickle.load(open(pickle_name, "rb" ))
    
#     # set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
#     # force cpu, if only cpu is available
#     vars["device"]  = device
#     logger.info(f"Device used: {device}")

#     # instantiate model
#     model = TransformerVAE(**vars)

#     # load trained parameters
#     checkpoint = torch.load(pt_name,map_location = device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # extract mlp2energy
#     f_z = model.get_submodule("mlp2energy")

#     # move to GPU
#     f_z   = f_z.to(device)
#     model = model.to(device)
    
#     f_z.eval()

#     # forward pass through encoder to get mu/sigma/z

#     # double precision

#     # create starting point
#     # we will be working with a single z, so set batch_size to 1
#     batch_size          = 1
#     latent_size         = vars["latent_size"]
#     x0                  = np.random.randn(latent_size) # TODO: check if we need to use (batch_size, latent_size)

#     # create bounds: every coordinate of x should be between -3.0 and 3.0
#     bounds = [(-3.0,3.0)]*latent_size

#     # call the function -> will return double
#     x_opt, f_opt = trust_region_optimisation(
#                                                 f_z.double(),
#                                                 x0,
#                                                 bounds,
#                                                 logger,
#                                                 device
#                                             )





########################################################## PREDICTIVE MODELS #####################
@predictor_app.command("tf-predict-energy")
def predict_energy_tf(
                        output_filename_fasta     : str,       # filename to save FASTA
                        output_filename_csv       : str,       # filename to save energies CSV
                        pt_name                   : str,       # pytorch file (trained model params)
                        pickle_name               : str,       # pickle file (input model params)
                        clean_samples_fasta       : str,       # fasta file with clean samples
                        seed                      : int = 0    # random seed for sampling
                     )->None:
    # call seq2_energy_tf; save dictionary "data" as csv
    data = predict_energies_using_tf_encoder(
                                                output_filename_fasta,
                                                pt_name,
                                                pickle_name,
                                                clean_samples_fasta,
                                                seed  = seed
                                            )
    print("data:")
    print(data)

    # create dataframe and save
    df = pd.DataFrame(data)
    df.to_csv(output_filename_csv,index=False)

########################################################## OTHER ###################################

### compute statistics from csv file  (which we obtained from blastp and subsequent filtering)
@tf_app.command("csv-stats")
def csv2stats(
                csv_file : str, # csv file (output of blastp)
                logname  : str, # filename for log file
                n_samples: int  # number of requested samples (a multiple of 100)
             ) -> None:

    # logger
    logger = setup_logger( 'logger', logname, level = logging.INFO )

    # obtain identities/similarities/bitscores as lists
    identities, similarities, bitscores = obtain_best_results(
                                                                csv_file,
                                                                CSV_COLUMNS
                                                             )
    # if anything is None, we are done
    if (identities is None) and (similarities is None) and (bitscores is None):
        str_ = "No good samples were generated."
        logger.info(str_)

    else:
        # assert equal lengths
        assert len(identities) == len(similarities) == len(bitscores), "Length mismatch"
        n_good     = len(identities)
        n_good_pct = n_good/n_samples * 100
        str_  = f"Number of requested samples: {n_samples}, number of filtered samples: {n_good} ({n_good_pct:0.2f}%)\n"
        str_ += stats_compute(identities,"identity score")
        str_ += stats_compute(similarities,"similarity score")
        str_ += stats_compute(bitscores,"bitscore")
        logger.info(str_)

# not used
@protein_app.command("prior")
def do_protein_analysis_prior(
                                pickle_name        : str,   # pickle filename with with saved sequences
                                log_name           : str,   # log name 
                                model_identifier   : str    # correposnding to best train/validation error, etc
                            ) -> None:
    # some preparations
    # copy from enzyme_datasets
    amino_acids    = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

    # create logger
    setup_logger( 'logger', log_name, level = logging.INFO )
    logger   = logging.getLogger('logger')

    # load pickle with seq_tuple_dict [seq, logp(seq), associated z, id, description]
    seq_tuple_dict = pickle.load(open(pickle_name, "rb" ))
    seq_tuple      = seq_tuple_dict["VAE"]


    str_ = f"Working with sequences (from the prior) from model: {model_identifier}."
    logger.info(str_)
        
    # start protein analysis
    _ = present_results_of_protein_analysis(
                                                seq_tuple,
                                                amino_acids,
                                                logger,
                                                f"prior-samples-{model_identifier}"    
                                            )
    logger.info(f"Protein analysis completed.\n")

# call it
if __name__ == "__main__":
    app()

