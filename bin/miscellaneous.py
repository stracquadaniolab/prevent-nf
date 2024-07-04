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

from Bio import SeqIO,AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import typer
from typing import List 
import random
import pickle
import copy
import pandas as pd
import numpy as np
from utilities import (
                        setup_logger,
                        compute_stats,
                        pass_filter,
                        default_w2i_i2w,
                        mutate
                      )
import logging
from matplotlib import pyplot as plt

#### GLOBAL CONST
CSV_COLUMNS = ['qseqid','sseqid','score','bitscore','evalue','pident','ppos','qcovs']

# main app
app         = typer.Typer()
various_app = typer.Typer()
foldx_app   = typer.Typer()
app.add_typer(various_app, name = "various")
app.add_typer(foldx_app, name = "foldx")


# rebalance training/validation sets to achieve desired proportions
@various_app.command("rebalance-sets")
def rebalance_sets(
                        combined_fasta_file       : str,         # contains all clean sequences
                        pre_validation_fasta_file : str,         # fasta file which was obtained by mmseqs2 clustering
                        train_set_filename        : str,         # filename for train set FASTA file
                        validation_set_filename   : str,         # filename for validatin set FASTA file
                        log_filename              : str,         # filename for log file
                        val_pct                   : float = 0.2, # pct of total sequences that should be in validation set
                        seed                      : int = 0      # random seed
                    ) -> None:

    # some inline funtion to obtain a subset of records from a big record list
    def obtain_records(all_records:list, subset_of_seq_ids:list):
        subset_records = []
        for record in all_records:
            if record.id in subset_of_seq_ids:
                subset_records.append(record)
        return subset_records
    
    # make sure val_pct is in [0.0;0.5]
    assert 0.0<=val_pct<=0.5, "Validation percent argument must be between 0.0 and 0.5"

    # set seed
    random.seed(seed)

    # we will create a logger for this operation
    setup_logger( 'logger', log_filename, level = logging.INFO )
    logger = logging.getLogger('logger')

    ################################## LOAD AS IS: BEFORE REBALANCING #######################

    # let's load all records and their IDs
    all_records       =  [ record for record in SeqIO.parse(combined_fasta_file, "fasta") ]
    all_records_ids   =  sorted([ record.id for record in all_records ])

    # load pre-validation-records IDs
    pre_val_seq_ids   = sorted([ record.id for record in SeqIO.parse(pre_validation_fasta_file, "fasta") ])

    # get pre-train-ids: all those that are in all_records_ids but not in pre_val_seq_ids
    pre_train_seq_ids  = sorted(
                                    list(
                                            set(all_records_ids) - set(pre_val_seq_ids) # using set difference
                                    )
                                )
    # log results before rebalancing
    n_all                    = len(all_records_ids)
    n_pre_validation         = len(pre_val_seq_ids)
    n_pre_train              = len(pre_train_seq_ids)

    val_pct_effective        = n_pre_validation/n_all * 100
    train_pct_effective      = n_pre_train/n_all * 100
    
    # number of samples required to be in train/validation sets
    required_n_val   = int(val_pct * n_all)
    required_n_train = n_all - required_n_val

    # log this information
    str_  = "Before rebalancing...\n"
    str_ += f"Total number of records                                 : {n_all}\n"
    str_ += f"Number of records in validation file before rebalancing : {n_pre_validation} ({val_pct_effective:0.4f})%\n"
    str_ += f"Number of records in train file before rebalancing      : {n_pre_train} ({train_pct_effective:0.4f})%\n"
    str_ += f"Required number of records in validation set            : {required_n_val}\n"
    str_ += f"Required number of records in train set                 : {required_n_train}\n"
    logger.info(str_)

    assert n_pre_train + n_pre_validation == n_all, "Total number of records must be equal to number of records in train and validation sets before rebalancing."

    # consider three possibilities
    if n_pre_validation > required_n_val:
        # 1. you have too many sequences in validation set
        #    move some sequences from validation set to train set
        n_seq_to_move   = n_pre_validation - required_n_val            # you want to move the difference to train set
        seq_ids_to_move = random.sample(pre_val_seq_ids,n_seq_to_move) # select this number of objects from pre_validation_records
        
        # now we will drop these ids from validation and add them to train
        train_ids      = pre_train_seq_ids + seq_ids_to_move
        validation_ids = list(
                                set(pre_val_seq_ids) - set(seq_ids_to_move)
                             )
        
        # obtain records
        train_records      = obtain_records(all_records,train_ids)
        validation_records = obtain_records(all_records,validation_ids)

        # log information about rebalancing
        n_train_after_rabalcning = len(train_records)
        n_val_after_rebalancing  = len(validation_records)
        
        val_pct_effective   = n_val_after_rebalancing/n_all * 100
        train_pct_effective = n_train_after_rabalcning/n_all * 100
        
        str_  = f"Rebalancing from validation set into train set. {n_seq_to_move} will be moved.\n"
        str_ += f"Number of records in validation file after rebalancing : {n_val_after_rebalancing} ({val_pct_effective:0.4f})%\n"
        str_ += f"Number of records in train file after rebalancing      : {n_train_after_rabalcning} ({train_pct_effective:0.4f})%\n"
        str_ += f"Moved sequences:\n"
        for seq in seq_ids_to_move:
            str_ += f"{seq}\n"
        logger.info(str_)

        assert n_train_after_rabalcning + n_val_after_rebalancing == n_all, "Total number of records must be equal to number of records in train and validation sets after rebalancing."

        # now we can save
        # save train set
        SeqIO.write(
                        train_records,
                        train_set_filename,
                        "fasta"
                   )
        # and its stats
        compute_stats(
                        train_records,
                        "train-set-stats.txt"
                     )
        
        # save validation set
        SeqIO.write(
                        validation_records,
                        validation_set_filename,
                        "fasta"
                   )
        # and its stats
        compute_stats(
                        validation_records,
                        "validation-set-stats.txt"
                     )

    elif n_pre_validation < required_n_val:
        # 2. you have too little sequences in validation set
        #    move some sequence from train set to validation set
        
        n_seq_to_move = required_n_val - n_pre_validation                  # you want to move the difference from train set to validation set 
        seq_ids_to_move  = random.sample(pre_train_seq_ids,n_seq_to_move)  # select this number of objects from ${pre_train_records}

        # now we will drop these ids from train set and add them to validation set
        validation_ids = pre_val_seq_ids + seq_ids_to_move
        train_ids      = list(
                                set(pre_train_seq_ids) - set(seq_ids_to_move)
                             )
        
        # obtain records
        train_records      = obtain_records(all_records,train_ids)
        validation_records = obtain_records(all_records,validation_ids)

        # log information about rebalancing
        n_train_after_rabalncing = len(train_records)
        n_val_after_rebalancing  = len(validation_records)
        
        val_pct_effective   = n_val_after_rebalancing/n_all * 100
        train_pct_effective = n_train_after_rabalncing/n_all * 100
        
        str_  = f"Rebalancing from train set into validation set. {n_seq_to_move} will be moved.\n"
        str_ += f"Number of records in validation file after rebalancing : {n_val_after_rebalancing} ({val_pct_effective:0.4f})%\n"
        str_ += f"Number of records in train file after rebalancing      : {n_train_after_rabalncing} ({train_pct_effective:0.4f})%\n"
        str_ += f"Moved sequences:\n"
        for seq in seq_ids_to_move:
            str_ += f"{seq}\n"
        logger.info(str_)
        
        assert n_train_after_rabalncing + n_val_after_rebalancing == n_all, "Total number of records must be equal to number of records in train and validation sets after rebalancing."

        # now we can save
        # save train set
        SeqIO.write(
                        train_records,
                        train_set_filename,
                        "fasta"
                   )
        # and its stats
        compute_stats(
                        train_records,
                        "train-set-stats.txt"
                     )
        
        # save validation set
        SeqIO.write(
                        validation_records,
                        validation_set_filename,
                        "fasta"
                   )
        # and its stats
        compute_stats(
                        validation_records,
                        "validation-set-stats.txt"
                     )
    else:
        # 3. you satisfy the required pct, so no need for rebalancing
        #    save as is

        # obtain records
        train_records      = obtain_records(all_records,pre_train_seq_ids)
        validation_records = obtain_records(all_records,pre_val_seq_ids)

        # log information about rebalancing
        n_train_after_rabalncing = len(train_records)
        n_val_after_rebalancing  = len(validation_records)
        
        val_pct_effective   = n_val_after_rebalancing/n_all * 100
        train_pct_effective = n_train_after_rabalncing/n_all * 100
        
        str_ = f"No need for rabalancing.\n"
        str_ += f"Number of records in validation file  : {n_val_after_rebalancing} ({val_pct_effective:0.4f})%\n"
        str_ += f"Number of records in train file       : {n_train_after_rabalncing} ({train_pct_effective:0.4f})%\n"

        assert n_train_after_rabalncing + n_val_after_rebalancing == n_all, "Total number of records must be equal to number of records in train and validation sets after rebalancing."

        # save train set
        SeqIO.write(
                        train_records,
                        train_set_filename,
                        "fasta"
                   )
        # and its stats
        compute_stats(
                        train_records,
                        "train-set-stats.txt"
                     )
        
        # save validation set
        SeqIO.write(
                        validation_records,
                        validation_set_filename,
                        "fasta"
                   )
        # and its stats
        compute_stats(
                        validation_records,
                        "validation-set-stats.txt"
                     )

# preprocess input: remove duplicates/ keep sequences of certain length/ remove sequences with non-canonical AA
@various_app.command("preprocess-input")
def join_sequences(
                    fasta_files          : List[str],  # list of FASTA files to preprocess
                    output_filename      : str,        # filename for the final output
                    logger_name          : str,        # logger filename
                    Lmin                 : int = 10,   # minimum length threshold for inclusion
                    Lmax                 : int = 300   # maximum length threshold for inclusion  
                  ) -> None:
    # this function will take a user-provided list of FASTA files and do the following operations upon it:
    # concatenate, remove duplicates, remove non-canonical AAs, keep sequences only between certain range of length

    # we will create a logger for this operation
    setup_logger( 'logger', logger_name, level = logging.INFO )
    logger = logging.getLogger('logger')

    # will iteratively open each file in fasta_files and append unique record
    unique_records = []
    seen_ids       = []
    
    # first pass is to merge and remove duplicates
    for fasta_file in fasta_files:
        # open file
        local_counter = 0
        for record in SeqIO.parse(fasta_file, "fasta"):
            # if we haven't seen record yet AND sequence length lies between Lmin and Lmax
            if (record.id not in seen_ids) and (len(record.seq) >= Lmin) and (len(record.seq) <= Lmax):
                # append to record list
                unique_records.append(record)
                # make sure we have seen this id already
                seen_ids.append(record.id)
            # increase local counter
            local_counter += 1

        # print info for a file
        logger.info(f"Number of entries in the input file {fasta_file}: {local_counter}")
    
    # next: remove non-canonical AAs
    # we will also check real Lmin and Lmax
    
    # create dictionary
    w2i,_   = default_w2i_i2w()

    # create a list of clean records
    clean_records = [record for record in unique_records if pass_filter(record,w2i) == True]
    bad_records   = [record for record in unique_records if pass_filter(record,w2i) == False]

    logger.info(f"Number of filtered records                : {len(clean_records)}")
    logger.info(f"Number of records with non-canonical AAs  : {len(bad_records)}")

    # find min and max lengths
    lengths   = [len(record.seq) for record in clean_records]
    Lmin_real = min(lengths)
    Lmax_real = max(lengths)
    
    # log this information
    str_  = f"Minimum sequence length in the filtered records FASTA file: {Lmin_real}\n"
    str_ += f"Maximum sequence length in filtered records FASTA file    : {Lmax_real}"
    logger.info(str_)
    
    # print IDs of sequences with non canonical AAs
    if len(bad_records) > 0:
        str_ = "(removed) records with non-canonical AAs:\n"
        for r in bad_records:
           str_ += f"\t{r.id}\n"
        # log
        logger.info(str_)
    
    # save as new file
    SeqIO.write(
                    clean_records,
                    output_filename,
                    "fasta"
               )

# remove sequences (by ID), listed in ${id_file} from ${input} file and save result in ${output}
@various_app.command("remove-sequences")
def remove_sequences(
                        id_file : str, # id txt file
                        input   : str, # input FASTA file
                        output  : str, # output FASTA file
                    )->None:
    with open(id_file, mode='r', encoding='utf-8') as file:
        ids = set(file.read().splitlines())

    records = [SeqRecord(record.seq, id=record.id, description=record.description, name='') for record in SeqIO.parse(input, 'fasta')
               if record.id not in ids]
    SeqIO.write(records, output, "fasta")


# split single fasta file (output of mmseqs) into multiple
# obtain elements of various clusters (as FASTA) from a joint FASTA file
@various_app.command("mmseqs-split")
def mmseq2clusters(
                    all_seqs_fasta    : str,   # fasta file with all sequences
                    template_name     : str    # template for saving clusters: {template}-xxx.fasta

                  ) -> None:

    # open all records
    records = [
                    record for record in SeqIO.parse(
                                                        all_seqs_fasta,
                                                        "fasta"
                                                    )
              ]

    # create empty subset
    subset_records = None

    # cluster counter
    cnt = 0

    # keep while there are records left
    while len(records)>0:
        # take first element of the array
        current_record = records.pop(0) 
        # if current record is empty string, generate a new empty list
        # else: append to list
        if current_record.seq == "":
            # if there are actually records in list, save ...
            if subset_records is not None:
                cnt += 1
                SeqIO.write(
                                subset_records,
                                f"{template_name}-{cnt}.fasta",
                                "fasta"
                            )
            # ..., then reset list
            subset_records = []
        else:
            # else: append to list
            subset_records.append(current_record)

    # at the end all but last cluster will be saved
    # save final cluster
    cnt += 1
    SeqIO.write(
                    subset_records,
                    f"{template_name}-{cnt}.fasta",
                    "fasta"
                )

#### compute and plot sequence coverage
@various_app.command("sequence-coverage")
def seqcoverage(
                    alignment_file : str, # MSA alingment file
                    csv_filename   : str, # filename for csv (contains coverage information)
                    fig_filename_1 : str, # filename for plot for full coverage
                    fig_filename_2 : str, # filename for plot with exact matches
               ) -> None:

    # inline function: save plots
    def plot(array,seqlbl,yaxislbl,filename):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,10))
        
        ax.plot(
                    range(1,len(array)+1),  # x: sequence position
                    np.array(array),        # y: coverage (either exact match or number of non-gapped columns in that position)
                    linestyle       = '-',
                    color           = 'black',
                    label           = seqlbl
                )
        ax.set_xlabel("query sequence position")
        ax.set_ylabel(yaxislbl)
        ax.set_xlim(0,len(array)+1)
        ax.set_ylim(0,100)
        ax.legend()

        fig.savefig(filename)

    # open MSA file
    aln    = AlignIO.read(
                            alignment_file,
                            "fasta"
                         )
    aln_ex_VAE  = aln[1:,:] # this is MSA excluding VAE sample, which is 1st row of MSA file
    vae_sample  = aln[0]    # this is VAE sample

    seq_coverage            = [] # how many sequences are there in MSA for this a given position
    seq_coverage_full_match = [] # how many exact matches are there for a given position

    msa_len = len(aln_ex_VAE)

    # loop over gapped sequence and consider only non-gapped elements
    for clmn,element in enumerate(str(vae_sample.seq)):
        if element != "-":
            # extract column from alignment (index is clmn)
            aln_column = aln_ex_VAE[:,clmn] # 'str' type
            
            # need to compute how many non-gapped elements there are
            non_gapped_cnt = len(aln_column) - aln_column.count("-")

            # compute how many sequences have the same AA in MSA
            element_cnt    = aln_column.count(element)

            # append (make numbers relative to length of MSA)
            seq_coverage.append(
                                    (non_gapped_cnt/msa_len) * 100
                                )
            
            seq_coverage_full_match.append(
                                                (element_cnt/msa_len) * 100
                                        )

    # save results as csv
    data = {
                "coverage"    : seq_coverage,
                "exact-match" : seq_coverage_full_match
            }
    df    = pd.DataFrame(data)
    df.to_csv(csv_filename)

    # plot results
    seqlbl = vae_sample.id

    # full coverage
    plot(
            seq_coverage,
            seqlbl,
            "sequences,%",
            fig_filename_1
        )

    # exact match
    plot(
            seq_coverage_full_match,
            seqlbl,
            "sequences with same AA,%",
            fig_filename_2
        )

#### compute and extract sequence certainty
@various_app.command("sequence-certainty")
def seqcertainty(
                    fastafile    : str,    # fasta file with single entry
                    picklefile   : str,    # pickle file that containes all information about sequences
                    temperature  : float,  # temperature for this sequence
                    fig_filename : str,    # filename to save pdf file
                    csv_filename : str     # filename to save csv file

                ) -> None:
    # extract sequence ID
    id_ = [record.id for record in SeqIO.parse(fastafile,"fasta")][0] # single entry, so we extract first element of list

    # obtain a list of tuples
    # each tuple is (seq,logp(seq),associated z,id,description,seq certainty) as (list[str],float,numpy,str,str,numpy)
    data           = pickle.load( open( picklefile, "rb" ) )
    list_of_tuples = data["VAE"]

    # here we have 2 alternatives: either loop until ids match or extract a list item by index
    # we will try to extract a list item by index
    # index = int(seqID number - 1)
    # seqID = seq-N-temperature-X -> we need N (to be precise N-1)
    index = int(
                    id_.split('-')[1]
                ) - 1
    
    tpl = list_of_tuples[index]

    # extract seqID annd certainty measure from tuple
    seqID        = tpl[3] # string
    seqcertainty = tpl[5] # list

    # for T = 0.0 it is p(best) - p(2nd best option): the higher the value in each position the better
    # for T > 0.0 it is p(best) - p(chosen)         : the lower the value in each position, the better
    # we will plot the quantity 1 - prob difference : the higher the value in each position, the better

    if (temperature == 0.0):
        seq_certainty_np = np.array(seqcertainty)
    else:
        seq_certainty_np = 1.0 - np.array(seqcertainty)

    ### printing just to make sure the extracted sequence is what we need
    str_  = f"Sequence ID from FASTA file  : {id_}.\n"
    str_ += f"Sequence ID from pickle file : {seqID}.\n"
    print(str_)

    # plot certainty
    

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,10))
    ax.plot(
                range(1,len(seqcertainty)+1),  # x: sequence position
                seq_certainty_np,              # y: sequence certainty
                linestyle       = '-',
                color           = 'black',
                label           = seqID
            )
    ax.set_xlabel("sequence position")
    # y label: sequence certainty
    ax.set_ylabel("sequence certainty") 
    ax.set_xlim(0,len(seqcertainty)+1)
    ax.set_ylim(0,1.1) # probability range is [0,1]
    ax.legend()
    fig.savefig(fig_filename)

    # save csv
    data = {
                "certainty"    : seq_certainty_np
           }
    df    = pd.DataFrame(data)
    df.to_csv(csv_filename)

#### split FASTA file into individual FASTA files
@various_app.command("sequence-split")
def seqsplit(
                master_fasta_file : str  # filename of aggreagte fastafile
            ) -> None:
    # open FASTA file
    records = [record for record in SeqIO.parse(master_fasta_file,"fasta")]

    # loop over records and save them individually
    for i,record in enumerate(records,1):
        SeqIO.write(
                        [record],
                        f"{record.id}.fasta",
                        "fasta"
                    )

### compute sequence weights from distance matrix
@various_app.command("distmat-weights")
def seq_weights_from_distmat(
                                distmat_file : str,  # distance matrix from clustalo
                                pickle_name  : str,  # filename for pickle file (will pass to Dataset construction later)
                                csv_name     : str   # filename for csv (will save for the record)
                            ) -> None:
    # now open and parse distmat_file
    df_distances = pd.read_csv(
                                    distmat_file,          # file with matrix that is output of clustalo
                                    skiprows = [0],        # skip first row, which only contains number of strings
                                    header   = None,       # no headers
                                    sep      = "\s+"       # split everything by 1 or more whitespaces
                            )
    
    # weights: dict
    weights = {}

    # extract column 0 as list
    ids = df_distances.iloc[:,0].to_list()

    # extract everything else as numpy
    identity_matrix = df_distances.iloc[0:,1:].to_numpy()
    N               = identity_matrix.shape[0] # number of elements in matrix

    # populate weights: seqID -> weight
    for i,id_ in enumerate(ids):
        avg_idenity  = (np.sum(identity_matrix[i,:]) - 100.00)/(N-1) # we just subtract identity value of a sequence to itself
        weights[id_] = 100.0/avg_idenity
    
    # need to save weights as csv
    df = pd.DataFrame(
                        list(weights.items()),
                        columns = ['ID','Weight']
                    )
    df.to_csv(csv_name)

    # and as pickle
    pickle.dump(
                    weights,
                    open(pickle_name,"wb")
                )

### compute default sequence weights (assume they are 1.0)
### this is just to have pickle/csv file in line with previous function
@various_app.command("default-weights")
def seq_weights_default(
                            fasta_file   : str, # fasta file with sequences
                            pickle_name  : str,  # filename for pickle file (will pass to Dataset construction later)
                            csv_name     : str   # filename for csv (will save for the record)
                        ) -> None:
    # set default weights to 1.0
    weights = {}
    for record in SeqIO.parse(fasta_file,"fasta"):
            weights[record.id] = 1.0

    # need to save weights as csv
    df = pd.DataFrame(
                        list(weights.items()),
                        columns = ['ID','Weight']
                     )
    df.to_csv(csv_name)

    # and as pickle
    pickle.dump(
                    weights,
                    open(pickle_name,"wb")
                )

### append energy term to WT fasta
@foldx_app.command("add-energy-wt")
def append_energy_wt(
                        wt_fasta_file     : str,            # input FASTA file
                        wt_fasta_filename : str,            # FASTA filename to save
                        free_energy_wt    : float = 0.0     # energy value of WT
                    ):

    # print("Free energy...")
    # print(f"type  : {type(free_energy_wt)}")
    # print(f"value : {free_energy_wt}")

    new_records = []
    # open file
    for old_record in SeqIO.parse(wt_fasta_file,"fasta"):
        # append Free Energy to ID: id -> id:(free)energy
        id_ = f"{old_record.id}:{free_energy_wt}"
        new_record = SeqRecord(
                                    old_record.seq,
                                    id          = id_,
                                    description = old_record.description
                                )

        new_records.append(new_record)

    # save
    SeqIO.write(new_records,wt_fasta_filename,"fasta")

## function to generate mutants in the appropriate form for FoldX
@foldx_app.command("generate-mutants")
def generate_mutants(
                        wt_fasta_file         : str,      # FASTA file with WT (doesn't really matter if Free Energy is appended or not
                        pdb_reference         : str,      # PDB reference name 
                        output_filename       : str,      # FASTA filename for mutants
                        n_mutation_sites      : int = 5,  # how many mutations to make
                        n_mutants             : int = 10, # how many different mutants to generate
                        seed                  : int = 0,  # random seed to control random sampling
                    ):

    # fix seed
    np.random.seed(seed)

    # open WT seq
    wt     = [record for record in SeqIO.parse(wt_fasta_file,"fasta")][0]

    # WT sequences as list: ["N","A","L",...]
    wt_seq = list(str(wt.seq)) 
    wt_id  = wt.id
    
    print("length of WT:",len(wt_seq))

    # our positions to mutate (don't mutate first one)
    positions_to_mutate = [i for i in range(1,len(wt_seq))]

    # open file
    f = open("individual_list.txt", "w")

    # mutant records
    mutant_records = []

    # loop over mutants
    for i in range(n_mutants):
        # sample which positions in a sequence to mutate
        chosen_positions = np.random.choice(
                                                positions_to_mutate,        # sample from this list
                                                size    = n_mutation_sites, # how many samples
                                                replace = False             # no replacement, as we can't have [1,1,1,...]
                                           )
        # sort in ascending order
        mutation_sites = sorted(chosen_positions)

        print("Mutation sites:",mutation_sites)

        # get FoldX representation and mutated sequence
        foldx_representation,mutant = mutate(
                                                copy.deepcopy(wt_seq),  # just a copy of our WT
                                                mutation_sites,         # poistions to mutate
                                                chainA_only = True      # only single chain
                                            )

        # add FoldX representation
        f.write(foldx_representation)
        f.write('\n')

        # add mutant to FASTA
        # id will follow FoldX convention for easy matching

        description_ = f"reference-uniprot:{wt_id}||refence-pdb:{pdb_reference}||mutations:{foldx_representation}"
        id_          = f"{pdb_reference}_{i+1}"
        print("ID:",id_)
        mutant_record = SeqRecord(
                                    Seq(
                                        "".join(mutant),
                                    ),
                                    id          = id_,
                                    description = description_
                                )
        print(mutant_record)
        print("-"*10)

        mutant_records.append(mutant_record)


    f.close()

    # write to FASTA
    SeqIO.write(mutant_records,output_filename,"fasta")

@foldx_app.command("add-energy-mutant")
def append_energy_mutants(
                            mutant_fasta_file : str, # FASTA file with mutant sequences but no energy
                            mutant_file_csv   : str  # CSV file with mutant energy estimates
                         ):
    # energy as lookup
    d_energy = {}

    # open and parse mutant_file_csv
    df_energy = pd.read_csv(
                                mutant_file_csv,
                                header = None,
                                sep = ","
                           )

    # extract ids 
    ids = df_energy.iloc[:,0].to_list()

    # extract energies
    energies = df_energy.iloc[:,1].to_list()

    # populate dictionary
    for id_,energy_ in zip(ids,energies):
        d_energy[id_] = energy_

    # now update fasta file
    
    new_records = []
    # open file
    for old_record in SeqIO.parse(mutant_fasta_file,"fasta"):
        # append Free Energy to ID: id -> id:(free)energy
        old_id      = old_record.id
        energy_term = d_energy[old_id]

        # somehow the old description contains ID, so we get rid of it by splitting on space
        description_ = old_record.description.split(" ")[1]
        
        id_ = f"{old_id}:{energy_term}"
        print("New ID:",id_)
        new_record = SeqRecord(
                                    old_record.seq,
                                    id          = id_,
                                    description = description_
                                )
        print(new_record)
        print("-"*10)

        new_records.append(new_record)

    # save under the same name
    SeqIO.write(new_records,mutant_fasta_file,"fasta")

# call it
if __name__ == "__main__":
    app()
    

   
    