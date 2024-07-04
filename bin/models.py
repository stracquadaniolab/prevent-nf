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
Various layers and Neural Networks.
# check:
https://github.com/oriondollar/TransVAE/tree/master/transvae
https://github.com/Fraser-Greenlee/transformer-vae

TODO:
# VAE THAT DECODES BOTH SEQUENCE AND PROPERTY
- Check if we need to use memory_key_padding_mask in line 189/256 (right now mask is used)
  Logic: we don't use encoder output, but rather stochastic representation of the encoder output.
  Does it make sense to mask out <pad> positions from the encoder output?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from numpy.typing import ArrayLike
import math
######################################################### VAE THAT DECODES BOTH SEQUENCE AND PROPERTY ######################
# Positional Encoding, taken from pytorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncodingPytorch(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position       = torch.arange(max_len).unsqueeze(1)
        div_term       = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe             = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# small MLP (used inside TransformerVAE)
class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP,self).__init__()

        # input dimensionality
        self.input_dim = input_dim

        # MLP with 5 layers
        # input_dim -> 64 -> 32 -> 16 -> 4 -> 1
        self.layers        = nn.Sequential(     
                                                nn.Linear(input_dim,64),
                                                nn.ReLU(),
                                                nn.Linear(64,32),
                                                nn.ReLU(),
                                                nn.Linear(32,16), # project to 16-dim vector space
                                                nn.ReLU(),
                                                nn.Linear(16,4), # project to 4-dim vector space
                                                nn.ReLU(),
                                                nn.Linear(4,1) # project to 1-dim vector space (no ReLU is used as target output is mostly negative)
                                          )
    def forward(self,X):
        out = self.layers(X)
        return out
    
# Transfromer-based VAE: both encoder and decoder follow Transformer architecture
# here: using Gaussian latent space
class TransformerVAE(nn.Module):
    def __init__(
                    self,
                    vocab_size,                 # number of AAs + 4 tokens
                    max_seq_length,             # max sequence length (from Dataset)
                    pad_idx,                    # pad_idx
                    sos_idx,                    # sos idx
                    eos_idx,                    # eos idx
                    unk_idx,                    # unk idx
                    device,                     # device (cpu/gpu)
                    embedding_size      = 512,  # embedding dimensionality
                    latent_size         = 64,   # latent space dimensionality
                    num_layers_encoder  = 6,    # how many blocks to use in encoder
                    num_layers_decoder  = 4,    # how many blocks to use in decoder
                    heads               = 8,    # embedding size should be divisible by heads
                    dropout_prob        = 0.1,  # dropout probability
                    condition_on_energy = False # whether to scale latent variable z by predicted energy: False means z->energy, z->sequence; True means z->energy, z*energy->sequence
                    
                ):
        super(TransformerVAE,self).__init__()

        # variables
        self.vocab_size          = vocab_size
        self.max_seq_length      = max_seq_length
        self.pad_idx             = pad_idx
        self.sos_idx             = sos_idx
        self.eos_idx             = eos_idx
        self.unk_idx             = unk_idx
        self.embedding_size      = embedding_size
        self.latent_size         = latent_size
        self.num_layers_encoder  = num_layers_encoder
        self.num_layers_decoder  = num_layers_decoder
        self.heads               = heads
        self.dropout_prob        = dropout_prob
        self.device              = device
        self.condition_on_energy = condition_on_energy

        # embedding layer
        self.embedding            = nn.Embedding(vocab_size,embedding_size,padding_idx = pad_idx)

        # positional embedding
        self.positional_embedding = PositionalEncodingPytorch(embedding_size,dropout=dropout_prob)

        # MLP: logits -> MLP -> energy
        # z size: [Batch, Latent size] -> input_dim:  Latent size
        self.mlp2energy = MLP(
                                latent_size
                             )

        # transformer
        #self.transformer = nn.Transformer(d_model = embedding_size,nhead=heads)
        self.transformer_encoder = nn.TransformerEncoder(
                                                            nn.TransformerEncoderLayer(embedding_size,heads,batch_first = True,dropout=dropout_prob),   # transformer encoder layer
                                                            num_layers_encoder                                                                                  # how many layers to have
                                                        )

        self.transformer_decoder = nn.TransformerDecoder(
                                                            nn.TransformerDecoderLayer(embedding_size,heads,batch_first = True,dropout=dropout_prob),   # transformer decoder layer
                                                            num_layers_decoder                                                                                  # how many layers to have
                                                        )
        
        # latent to mean/standard deviation
        self.hidden2mean   = nn.Linear(embedding_size * max_seq_length, latent_size)
        self.hidden2logv   = nn.Linear(embedding_size * max_seq_length, latent_size)

        # latent to input to decoder
        self.latent2embed  = nn.Linear(latent_size, embedding_size * max_seq_length)

        # final step: convert to vocab_size
        self.out = nn.Linear(embedding_size, vocab_size)
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def create_attn_mask(self, size) -> torch.tensor:
        # Generates a square matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))    # Convert ones to 0
        mask = mask.to(self.device)
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        return mask

    def forward(self,src,tgt,masking_prob = 0.0):
        # src          : (Batch Size, Src Seq Length)
        # tgt          : (Batch Size, Tgt Seq Length)
        # masking_prob : probability of masking any AA in decoder input (tgt)

        batch_size = src.size(0)

        # get pad masks: [Batch Size, Seq Length]
        src_pad_mask = self.create_pad_mask(src,self.pad_idx)
        tgt_pad_mask = self.create_pad_mask(tgt,self.pad_idx)

        # replace certain elements in tgt with <unk>
        if masking_prob > 0.0:
            # generate random numbers
            prob = torch.rand( 
                                tgt.size(), 
                                dtype  = torch.float, 
                                device = self.device
                             )
            # never replace <sos>,<pad>
            prob[(tgt.data - self.sos_idx) * (tgt.data - self.pad_idx) == 0] = 1

            # now replace the rest of tokens (AAs) with <unk> randomly
            tgt[prob < masking_prob] = self.unk_idx

        # get attention mask (used in decoder, so applied to tgt)
        tgt_attn_mask = self.create_attn_mask(tgt.size(1)) # size(1) is Tgt Seq Length

        # do embedding (dense + positional) -> [ Batch Size, Seq Length, Embedding size ]
        src_emb = self.positional_embedding(
                                                self.embedding(src).transpose(0,1) # embedding: [Batch Size, Seq Lengh] - > [Batch Size, Seq Length, Embedding Size] -> (transpose) -> [Seq Length, Batch Size, Embedding Size]
                                             ).transpose(0,1) # [Seq Length, Batch Size, Embedding Size]
        
        # same
        tgt_emb = self.positional_embedding(
                                                self.embedding(tgt).transpose(0,1)
                                             ).transpose(0,1)

        # pass through encoder
        # only padding mask is needed here
        encoder_out = self.transformer_encoder(src_emb,src_key_padding_mask = src_pad_mask) # size: [Batch Size, Seq Length, Embedding Size]

        # reshape: [Batch Size, Seq Length, Embedding Size] -> [Batch Size, Seq Length * Embedding Size]
        encoder_out = encoder_out.reshape(batch_size,self.max_seq_length*self.embedding_size)

        # to mean and standard deviation
        mean = self.hidden2mean(encoder_out)
        logv = self.hidden2logv(encoder_out)
        std  = torch.exp(0.5 * logv)

        # do reparametrisation trick
        z = torch.randn(
                        [batch_size, self.latent_size],
                        dtype  = torch.float,
                        device = self.device
                       )
        # [Batch Size, Latent Size] # this is our latent representation
        z = z * std + mean

        # predict energy from z : [Batch Size, 1]
        energy = self.mlp2energy(z)

        # scale energy (this wont change dimensionality of z, so we can pass it safely to self.latent2embed)
        # use scaled latent vector z as stochastic memory representation
        if self.condition_on_energy:
            z_tilda = z * energy
        else:
            z_tilda = z

        # stochastic representation of encoder output:
        # [Batch Size, Latent Size] -> [Batch Size, Seq Lenth * Embedding Size] -> [Batch Size, Seq Length, Embedding Size]
        enc_memory_stochastic = self.latent2embed(z_tilda).reshape(batch_size,self.max_seq_length,self.embedding_size)
        
        # pass through decoder
        out         = self.transformer_decoder(
                                                tgt_emb,
                                                enc_memory_stochastic,
                                                tgt_mask                = tgt_attn_mask,    # prohibit unwanted attention
                                                tgt_key_padding_mask    = tgt_pad_mask,     # pad indicies for target (will be ignored)
                                                memory_key_padding_mask = src_pad_mask      # pad indices for source (will be ignored) : TODO: DEBATABLE IF NEEDED TO BE USED!
                                              )

        # [Batch Size, Sequence Length, Embedding Size] -> [Batch, Sequence Length, Vocab Size]
        out = self.out(out)

        # convert to logp by applying log_softmax along last dimension
        logp = nn.functional.log_softmax(
                                            out,
                                            dim = -1
                                        )

        

        return logp, mean, logv, energy

    # do forward pass through encoder only
    def __precompute_stochastic_memory(
                                        self,
                                        src, # tensor with numerical representation of seed sequence: dataloader -> batch["input"]
                                      ):

        batch_size = src.size(0)

        # get pad masks
        src_pad_mask = self.create_pad_mask(src,self.pad_idx)
        # do embedding (dense + positional) -> [ Batch Size, Seq Length, Embedding size ]
        src_emb = self.positional_embedding(
                                                self.embedding(src).transpose(0,1)
                                             ).transpose(0,1)
        # pass through encoder
        encoder_out = self.transformer_encoder(src_emb,src_key_padding_mask = src_pad_mask) # size: [Batch Size, Seq Length, Embedding Size]

        # reshape: [Batch Size, Seq length * Embedding Size]
        encoder_out = encoder_out.reshape(batch_size,self.max_seq_length*self.embedding_size)

        # to mean and standard deviation
        mean = self.hidden2mean(encoder_out)
        logv = self.hidden2logv(encoder_out)
        std  = torch.exp(0.5 * logv)

        return mean,std,src_pad_mask
    
    # a pass through decoder
    def __get_logits_logp(self,encoder_output,tgt,memory_pad_mask = None):
        # ENCODER output is a constant thing [Batch Size, Seq Length, Embedding Size]
        # either
        # prior(z) -> latent2embed -> reshape
        # or
        # posterior(z) -> latent2embed -> reshape

        # TGT will be changed in a for loop: [<sos>,AA1,AA2,..]
        
        tgt_pad_mask = self.create_pad_mask(tgt,self.pad_idx)
        attn_mask    = self.create_attn_mask(tgt.size(1))

        # do embedding for tgt -> [Batch Size, Seq Length, Embedding Size]
        tgt_emb = self.positional_embedding(
                                                self.embedding(tgt).transpose(0,1)
                                            ).transpose(0,1)

        # pass through transformer decoder
        # memory_key_padding_mask is None for prior samples (because encoder was not used, so we don't know what in the stochastic representation actually corresponds to <pad>)
        # memory_key_padding_mask is provided for posterior samples (because encoder was used)
        out = self.transformer_decoder(
                                        tgt_emb,
                                        encoder_output,
                                        tgt_mask                = attn_mask,
                                        tgt_key_padding_mask    = tgt_pad_mask,
                                        memory_key_padding_mask = memory_pad_mask
                                      )
        # size: [Batch, ?, Vocab Size]
        # second dimension will be changing because sequence will grow
        # we are only interested in last element of that dimension
        out = self.out(out)

        # convert to logp by applying log_softmax along last dimension
        logp = nn.functional.log_softmax(
                                            out,
                                            dim = -1
                                        )
        return out,logp

    # sample from the prior
    def sample_from_latent_space(self,
                                    batch_size,
                                    max_length = 100,
                                    z          = None,
                                    argmax     = True
                                ):

        #NB: z needs to have first dimension equal to batch_size to make this work

        if z is None:
            # sample from prior -> [Batch Size, Latent Space Size]
            z = torch.randn(
                                [batch_size, self.latent_size],
                                dtype  = torch.float,
                                device = self.device
                        )

        # get energy: [Batch Size, 1]
        energy = self.mlp2energy(z)

        # scale z by energy and pass to decoder: [Batch Size, Latent Size]
        if self.condition_on_energy:
            z_tilda = z * energy
        else:
            z_tilda = z

        # convert into input for transformer decoder -> [Batch Size, Seq Length, Embedding Size]
        enc_memory_stochastic = self.latent2embed(z_tilda).reshape(batch_size,self.max_seq_length,self.embedding_size)

        # create start tokens -> [Batch Size, 1]
        tgt =  self.sos_idx * torch.ones(
                                            [batch_size,1],
                                            dtype  = torch.long,
                                            device = self.device
                                        )

        # keep indices and logprobabilities                    
        INDICES  = []
        LOGPROBS = []

        # ENTROPIES: will collect entropy for each distribution (i.e. each AA position will have an entropy term)
        ENTROPIES = []

        # sample tokens (argmax)
        for step in range(max_length):
            
            # call _sample() -> obtain logp for predicted tokens
            # size [Batch Size, ? , Vocab Size] (sequence grows so 2nd dimension changes)
            _,logp_predicted = self.__get_logits_logp(
                                                        enc_memory_stochastic,
                                                        tgt
                                                    )

            # select last element
            # size [Batch Size, Vocab Size]
            logp_predicted_last = logp_predicted[:,-1,:]

            # compute entropy for this AA -> [Batch Size=1]
            H = -1.0 * torch.sum(
                                    logp_predicted_last * torch.exp(logp_predicted_last),
                                    dim = 1
                                )[0].item() # since batch size is 1, just take the first element
            ENTROPIES.append(H)
            
            # need to select best tokens according to last dimension
            # size: [Batch Size, 1]
            if argmax:
                # argmax selection
                logp_all,next_all = torch.max(
                                                    logp_predicted_last,
                                                    dim     = -1,
                                                    keepdim = True
                                                )
            else:
                # categorical sampling
                
                # convert logp -> p
                probs_temp_annealed_last = logp_predicted_last.exp()

                # sample
                next_all = torch.multinomial(
                                                probs_temp_annealed_last,
                                                1,
                                                replacement = False
                                            )
                # gather sampled indices
                logp_all = logp_predicted_last.gather(
                                                        1,
                                                        next_all
                                                     )
        
            # record
            INDICES.append(next_all)
            LOGPROBS.append(logp_all)

            # need to concatenate tgt with next_token
            # [Batch Size, ?] as sequence grows
            tgt = torch.cat(
                                [tgt, next_all], 
                                dim = 1
                           )

        # once loop is over concatenate lists
        # [Batch Size, Seq Length]
        logp_opt = torch.cat(
                                LOGPROBS,
                                dim  = 1
                            )

        indices = torch.cat(
                                INDICES,
                                dim  = 1
                           )

        # extract mean,max,std of entropies
        entropies = np.array(ENTROPIES)

        # get a list of entropies stats
        entropies_out  = [
                            np.mean(entropies), # mean
                            np.max(entropies),  # max
                            np.std(entropies)   # std
                        ]

        return logp_opt, indices, z, energy, entropies_out

    
        
    # sample around seed molecule
    def sample_around_molecule( 
                                self,
                                src,                      # tensor with numerical representation of seed sequence: dataloader -> batch["input"]
                                batch_size,               # how many samples for a seed molecule to do
                                T                  = 0.0, # if T = 0.0 do argmax sampling
                                max_length         = 100, # maximum number of sampling steps
                                skip_first_element = True # first element (should be M) will be taken as argmax, the rest will be sampled if T > 0.0
                              ):
        
        # obtain mu/sigma (just part of forward())
        # also returns a memory pad mask of size [1,Seq Length]
        mean,std,memory_pad_mask = self.__precompute_stochastic_memory(src)
        
        # keep indices and logprobabilities                    
        INDICES  = []
        LOGPROBS = []
        Z        = []
            
        # expand mu and sigma, pad_mask -> [Batch Size, Seq Length]
        mean_exp            = mean.repeat([batch_size,1])
        std_exp             = std.repeat([batch_size,1])
        memory_pad_mask_exp = memory_pad_mask.repeat([batch_size,1])

        # sample from prior
        z = torch.randn(
                            [batch_size, self.latent_size],
                            dtype  = torch.float,
                            device = self.device
                        )
            
        # convert to posterior: [Batch Size, Latent Size]
        z = mean_exp + z * std_exp

        # predict energy: as long as sampling := argmax operation, it is fine
        # if there is real sampling involved, unclear
        energy = self.mlp2energy(z) # [Batch Size, 1]

        # whether to condition on energy or not
        if self.condition_on_energy:
            z_tilda = z * energy
        else:
            z_tilda = z

        # convert into input for transformer decoder -> [Mini Batch Size, Seq Length, Embedding Size]
        enc_memory_stochastic = self.latent2embed(z_tilda).reshape(batch_size,self.max_seq_length,self.embedding_size)

        # create start tokens -> [Batch Size, 1]
        tgt =  self.sos_idx * torch.ones(
                                            [batch_size,1],
                                            dtype  = torch.long,
                                            device = self.device
                                        )
            
        # for loop: max_length
        for step in range(max_length):
            
            # call _sample() -> obtain logp for predicted tokens
            # size [Batch Size, Num of Elements in a Sequence , Vocab Size] (sequence grows)
            logits_predicted,logp_predicted = self.__get_logits_logp(
                                                                        enc_memory_stochastic,
                                                                        tgt,
                                                                        memory_pad_mask = memory_pad_mask_exp
                                                                    )

            # select last element
            # size [Batch Size, Vocab Size]
            logp_predicted_last = logp_predicted[:,-1,:]
                
            # need to select next token (and it's logprobability)
            # if T == 0.0 or T>0.0 but first element is forced to be selected according to argmax
            if (T == 0.0) or (step == 0 and T > 0.0 and skip_first_element == True):
                # size [Batch Size, 1]
                logp_all,next_all = torch.max(
                                                    logp_predicted_last,
                                                    dim     = -1,
                                                    keepdim = True
                                                )
            else:
                # do proper categorical sampling
                # convert logits to logp
                logp_temp_annealed = nn.functional.log_softmax(
                                                                logits_predicted/T,
                                                                dim = -1
                                                                )
                # select last element
                # size [Batch Size,Vocab Size]
                logp_temp_annealed_last = logp_temp_annealed[:,-1,:]

                # logp -> p
                probs_temp_annealed_last = logp_temp_annealed_last.exp()

                # sample
                next_all = torch.multinomial(
                                                probs_temp_annealed_last,
                                                1,
                                                replacement = False
                                            )
                # gather sampled indices
                logp_all = logp_temp_annealed_last.gather(
                                                            1,
                                                            next_all
                                                            )

            # record
            INDICES.append(next_all)
            LOGPROBS.append(logp_all)
            Z.append(z)

            # need to concatenate tgt with next_token
            tgt = torch.cat(
                                [tgt, next_all], 
                                dim = 1
                            )

        # at this point need to concatenate INDICES and LOGPROBS and Z
        # size: [Batch Size, Max Length]
        logp_opt = torch.cat(
                                LOGPROBS,
                                dim  = 1
                            )

        indices  = torch.cat(
                                INDICES,
                                dim  = 1
                            )
        z = torch.cat(
                        Z,
                        dim = 1
                     )

        return logp_opt,indices,z,energy

######################################################### SEQUENCE-TO-PROPERTY MODELS ######################################
# simple average: compute avg of the train set, compute MSE of the train set\validation set
def simple_average_model(
                            energies_train      : ArrayLike, # numpy array of energies in the train set
                            energies_validation : ArrayLike, # numpy array of energies in the validation set
                        ):

    # avg of train set
    avg_train = np.mean(energies_train)

    # mse of the train set
    mse_train = np.mean(
                            np.square(
                                        energies_train - avg_train
                                     )
                       )
    
    # mse of the validation set
    mse_validation = np.mean(
                            np.square(
                                        energies_validation - avg_train
                                     )
                       )
    return avg_train, mse_train, mse_validation

# GRU-to-property (we will assume here that sequences can be of variable length)
class PredictorGRU(nn.Module):
    '''
    INPUT:
        vocab_size     : number of tokens, int
        embedding_size : number of features in the input, int
        hidden_size    : number of features in hidden dimension in RNN, int
        num_layers     : number of stacked RNNs, int, default: 1
        bidirectional  : whether to use bidirectional RNNs, boolean, default: False
    '''
    def __init__(       self,
                        vocab_size,
                        embedding_size,
                        hidden_size,
                        device,
                        pad_idx,
                        p_dropout     = 0.2,
                        num_layers    = 1,
                        bidirectional = False
                ):
        super().__init__()

        self.vocab_size     = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.device         = device
        self.pad_idx        = pad_idx
        self.p_droput       = p_dropout
        self.num_layers     = num_layers
        self.bidirectional  = bidirectional

        # account for bidirectionality and number of layers
        self.hidden_factor  = (2 if bidirectional else 1) * num_layers

        # embedding layer + dropout
        self.embedding         = nn.Embedding(vocab_size,embedding_size,padding_idx=pad_idx)
        self.dropout_embedding = nn.Dropout(p = p_dropout)

        # GRU
        self.gru = nn.GRU(
                            embedding_size, 
                            hidden_size, 
                            num_layers    = num_layers, 
                            bidirectional = bidirectional,
                            batch_first   = True
                         )
        
        # small MLP to move from h_last to scalar
        #self.h2energy = MLP(hidden_size * self.hidden_factor)

        # 3 layers here
        self.h2energy = nn.Sequential(     
                                        nn.Linear(hidden_size * self.hidden_factor,16),
                                        nn.ReLU(),
                                        nn.Linear(16,4), # project to 4-dim vector space
                                        nn.ReLU(),
                                        nn.Linear(4,1) # project to 1-dim vector space (no ReLU is used as target output is mostly negative)
                                    )

    def forward(self,batch_of_input_sequences,input_sequences_lengths, h0 = None):

        # batch size
        batch_size  = batch_of_input_sequences.size(0)
        
        # sorting by sequence length
        batch_size                 = batch_of_input_sequences.size(0)
        sorted_lengths, sorted_idx = torch.sort(input_sequences_lengths, descending=True)
        X                          = batch_of_input_sequences[sorted_idx]

        # embed sequences
        X_embedded = self.dropout_embedding(
                                                self.embedding(X)
                                           )
        # now we need to pack them for efficient passing through RNN
        X_packed  = rnn_utils.pack_padded_sequence(X_embedded, sorted_lengths.data.tolist(), batch_first=True)

        # run through GRU -> we are only concerned about final h
        if h0 is None:
            _, hidden = self.gru(X_packed)
        else:
            _, hidden = self.gru(X_packed,h0)

        # need to take into account different sizes of h depending on bidirectionality and number of layers
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            encoder_hidden_at_T = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            encoder_hidden_at_T = hidden.squeeze()

        print("Size of encoder_hidden_at_T:",encoder_hidden_at_T.size())

        # now pass through MLP
        output = self.h2energy(encoder_hidden_at_T)

        return output

# TransformerEncoder-to-property (we will assume here that sequences can be of variable length)    
class PredictorTFEncoder(nn.Module):
    def __init__(
                    self,
                    vocab_size,               # number of AAs + 4 tokens
                    max_seq_length,           # max sequence length (from Dataset)
                    pad_idx,                  # pad_idx
                    embedding_size     = 512, # embedding dimensionality
                    latent_size        = 64,  # latent space dimensionality
                    num_layers_encoder = 6,   # how many blocks to use in encoder
                    heads              = 8,   # embedding size should be divisible by heads
                    dropout_prob       = 0.1  # dropout probability
                ):
        super(PredictorTFEncoder,self).__init__()

        # variables
        self.vocab_size          = vocab_size
        self.max_seq_length      = max_seq_length
        self.pad_idx             = pad_idx
        self.embedding_size      = embedding_size
        self.latent_size         = latent_size
        self.num_layers_encoder  = num_layers_encoder
        self.heads               = heads
        self.dropout_prob        = dropout_prob

        # embedding layer
        self.embedding            = nn.Embedding(vocab_size,embedding_size,padding_idx = pad_idx)

        # positional embedding
        self.positional_embedding = PositionalEncodingPytorch(embedding_size,dropout=dropout_prob)

        # MLP: logits -> MLP -> energy
        # z size: [Batch, Latent size] -> input_dim:  Latent size
        self.mlp2energy = MLP(
                                latent_size
                             )

        # TF encoder
        self.transformer_encoder = nn.TransformerEncoder(
                                                            nn.TransformerEncoderLayer(
                                                                                        embedding_size,
                                                                                        heads,
                                                                                        batch_first = True,
                                                                                        dropout=dropout_prob
                                                                                      ),   # transformer encoder layer
                                                            num_layers_encoder                                                                                  # how many layers to have
                                                        )

        # sequence (embedding) to hidden
        self.seq2latent   = nn.Linear(embedding_size * max_seq_length, latent_size)

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def forward(self,src):
            
        batch_size = src.size(0)

        # size : [N,S]
        src_pad_mask = self.create_pad_mask(src,self.pad_idx)

        # do embedding (dense + positional) -> [ Batch Size, Seq Length, Embedding size ]
        src_emb = self.positional_embedding(
                                                self.embedding(src).transpose(0,1) # embedding: [Batch Size, Seq Lengh] - > [Batch Size, Seq Length, Embedding Size] -> (transpose) -> [Seq Length, Batch Size, Embedding Size]
                                            ).transpose(0,1) # [Seq Length, Batch Size, Embedding Size]

        # pass through TF encoder
        encoder_out = self.transformer_encoder(src_emb,src_key_padding_mask = src_pad_mask) # size: [Batch Size, Seq Length, Embedding Size]
        # reshape to be 2D
        encoder_out = encoder_out.reshape(batch_size,self.max_seq_length*self.embedding_size)

        # to latent (aka z)
        latent = self.seq2latent(encoder_out)

        # finally latent to energy
        energy = self.mlp2energy(latent)

        return energy


        







            
            



        

       

