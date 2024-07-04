# prevent-nf

![](https://img.shields.io/badge/current_version-1.0.0-blue)

## Overview

PREVENT: PRotein Engineering by Variational frEe eNergy approximaTion

## Configuration

A user needs to specify a number of parameters in the configuration file under **params** scope to run `prevent` pipeline.
In addition to that, profile(s) describing the system where `prevent` is run need(s) to be provided in the same configuration file.

- **resultsDir**: directory name where all the results of the current run will be saved
- **seed**: random seed used in set rebalancing and model training (default: 0)

- **preprocessing.lmin**: minimum sequence length to consider sequence suitable for training (default: 0)
- **preprocessing.lmax**: maximum sequence length to consider sequence suitable for training (default: 1,000)
- **preprocessing.val_pct**: relative size of validation set, should be between 0.0 and 0.5 (default: 0.2)
- **preprocessing.mmseq_clustering**: `MMseqs2` clustering options to obtain cluster representative sequences (default: ''--min-seq-id 0.8'')
- **preprocessing.weight_sequences**: weight sequences by inverse of the average sequence identity within the train set (default: true) [NOT USED]
- **preprocessing.training_list**: list of FASTA files that will be used to generate train and validation sets (compulsory parameter)
- **preprocessing.seed_list**: list of FASTA files with seed sequences to generate novel variants (compulsory parameter)
- **training.epochs**: number of training epochs (default: 100)
- **training.val_freq**: epoch frequency of validation error calculation (default: 1)
- **training.checkpoint_freq**: frequency of model checkpoints, there must be at least one checkpoint, set the value accordingly (default: 50)
- **training.batch_size**: batch size for the gradient update step (default: 32)
- **training.learning_rate**: learning rate for the gradient update step (default: 0.0001)
- **training.L2**: **L2** normalisation constant (default: 0.0)
- **training.clipping_type**: gradient clipping technique, either ''norm'' or ''value'', see pytorch documentation for more details (default: ''value'')
- **training.hidden_size**: hidden units dimensionality for intermeidate layers for temporal convolution layers (default: 128)
- **training.latent_size**: dimensionality of latent distribution (default: 32)
- **training.condition_on_energy**: whether scale latent variable z by predicted energy value, when otaining p(sequence|z) (compulsory parameter). Start with setting it to "false".
- **training.entry_checkpoint**: pytorch model checkpoint to start training from (default: ''None'')
- **training.embedding_size**: embedding dimensionality for amino acids and special tokens (default: 32). Must be divisible by **training.heads**
- **training.dropout_prob**: embedding dropout probability (default: 0.2)
- **training.masking_prob**: input masking probability for transformer decoder (default: 0.0). Non-zero values are recommended to force latent space to learn more
- **training.heads**: number of heads in (self)attention layers of transformer. Default pytorch transformer class is used, see pytorch docs for more information
- **training.num_layers_encoder**: number of TransformerEncoderLayer in Transformer (default: 6)
- **training.num_layers_decoder**: number of TransformerDecoderLayer in Transformer (default: 4)

- **sampling.seed**: random seed used for sampling (default: 0)
- **sampling.n_samples**: number of requested samples, must be a multiple of 100 (default: 100)
- **sampling.mini_batch_size**: number of samples per mini-batch in seeded sampling procedure only. Must be a divisor of **sampling.n_samples**, not sure why this was implemented.
- **sampling.max_length**: maximum number of sampling steps (default: 200)
- **sampling.e_value**: e-value threshold for `BLASTP` (default: 0.0001)
- **sampling.query_coverage**: minimum allowed query coverage (default: 70.0)
- **sampling.temperature**: list of temperatures for temperature annealing, temperature 0.0 corresponds to argmax operation; the higher the temperature the more uniform amino acid sampling becomes (default: [1.0])
- **sampling.argmax_first_element**: force argmax selection of the first amino acid; relevant for posterior sampling for temperatures greater than 0.0 (default: true). This is done to avoid having other than methionine AA.
- **sampling.mmseq_clustering**: `MMseqs2` clustering options to obtain cluster representative sequences (best candidates) for subsequent downstream analysis; relevant for seeded sampling only (default: ''--min-seq-id 0.8'') [NOT USED]

There are other functionalities included in the `prevent` workflow. The most important is generating a set of mutants for further NN training. There are certain parameters that need to be specified in order to run this particular branch (entrypoint) of the workflow:

- **energy.fasta_file**: FASTA file with a single (wild-type;WT) sequence, for which mutants will be generated and free energy estimated. For the WT free energy is also estimated.
- **energy.pdb_file**: PDB file for WT sequence provided in **energy.fasta_file**. Needed for energy estimation using FoldX.

- **mutagenesis.os**: OS where code is run. Supported options are "macOS" and "linux" as they will determine which FoldX binary to use.
- **mutagenesis.n_mutation_sites**: list with number of mutation sites in a sequence, for example [1,2,3] means 1,2,3 positions in WT sequence will be mutated independetly. First position in WT is not mutated (compulsory parameter)
- **mutagenesis.n_mutants**: how many mutants to produce (compulsory parameter)
- **mutagenesis.seeds**: list of seeds for random mutagenesis.
- **mutagenesis.foldx_runs**: number of random restarts for FoldX (code won't work with 1 restart yet). Recommend to have between 2 and 5.

In the end you will have |**mutagenesis.n_mutation_sites**| _ **mutagenesis.n_mutants** _ |**mutagenesis.seeds**| mutants. Each mutant will be run **mutagenesis.foldx_runs** and average value for free energy will be taken.

There are few more entrypoints in `prevent` that can be useful. Each of them has a specific set of parameters that need to be specified in **params** scope.
First entrypoint is called **GRU_SUPERVISED** and it allows to train a supervised model to predict free energy from a sequence. The special parameters are:

- **training_gru.epochs**: number of training epochs for GRU supervised model
- **training_gru.val_freq**: frequency of valiation error calculation
- **training_gru.checkpoint_freq**: frequency of model checkpoints
- **training_gru.batch_size**: batch size
- **training_gru.learning_rate**: learning rate for the gradient update step
- **training_gru.L2**: L2 regularisation constant
- **training_gru.hidden_size**: hidden units dimensionality in GRU
- **training_gru.num_layers**: number of stacked GRUs
- **training_gru.bidirectional**: whether to use bidirectional GRU (true/false)
- **training_gru.entry_checkpoint**: model checkpoint to start training from (if no checkpoint, provide "None")
- **training_gru.embedding_size**: embedding size for AAs and special tokens
- **training_gru.dropout_prob**: dropout probability for embeddings

Another entrypoint is called **TF_SUPERVISED** and it allows to train a different supervised model to predict free energy from a sequence. The related parameters are:

- **training_tf.epochs**: number of training epochs for Transformer (encoder) supervised model
- **training_tf.val_freq**: frequency of valiation error calculation
- **training_tf.checkpoint_freq**: frequency of model checkpoints
- **training_tf.batch_size**: batch size
- **training_tf.learning_rate**: learning rate for the gradient update step
- **training_tf.L2**: L2 regularisation constant
- **training_tf.latent_size**: dimensionality of a latent space (a size of a space we project to following transformer encoder forward pass). From this latent space an MLP predicts energy. Unlike VAE, here latent space is a deterministic object
- **training_tf.embedding_size**: embedding size for AAs and special tokens; will be split across attention heads, so this number must be divisible by the number of heads
- **training_tf.heads**: number of attention heads
- **training_tf.num_layers_encoder**: number of of Transformer encoderl layers to be used
- **training_tf.entry_checkpoint**: model checkpoint to start training from (if no checkpoint, provide "None")
- **training_tf.dropout_prob**: embedding dropout probability

Next entrypoint is called **ENERGY_ESTIMATE** and it allows to run energy estimation (using 500 random samples) for a list of sequences from a FASTA file. May be useful for evaluating model performance on test set. The related parameters are:

- **energy_estimate.pickle_file**: pickle file with model params (often an output of a main entrypoint)
- **energy_estimate.pytorch_file**: pytorch file with pretrained model layers (often an output of a main entrypoint)
- **energy_estimate.fasta_file**: FASTA file with sequence for which energy needs to be estimated (for example, test set)

Finally, entrypoint called **OPTIMISATION**, it allows to run optimisation procedure in latent space to find proteins with smallest free energy (not complete!). The related parameters are:

- **optimisation.pickle_file**: pickle file with model params (often an output of a main entrypoint)
- **optimisation.pytorch_file**: pytorch file with pretrained model layers (often an output of a main entrypoint)
- **optimisation.learning_rate**: learning rate for SGD
- **optimisation.n_restarts**: number of restarts for a run (to achieve a better result)
- **optimisation.delta_f_tol**: absolute tolerance for change in predicted energy (if absolute change is smaller than this number, procedure stops)
- **optimisation.max_opt_steps**: maximum allowed number of optimisation steps for a single restart
- **optimisation.seeds**: list of random seeds (for example, [1,2,3]); each seed correponds to an individual process with **optimisation.n_restarts** restarts

## Running the workflow (example)

There are multiple entry points to the pipeline, depending on the desired outcome: `ENERGY_ESTIMATE`, `OPTIMISATION`, `LATENTVARIABILITY` and others. This README will focus on the main entry point that is called by default and trains a model and does sampling.

### Install the workflow

1. Install [Nextflow](https://www.nextflow.io/docs/latest/getstarted.html).

2. Download `prevent` as follows: 

   ```bash
      git clone https://github.com/stracquadaniolab/prevent-nf.git
   ```

3. Install [Docker](https://docs.docker.com/get-docker/) or
   [Singularity](https://sylabs.io/guides/3.7/user-guide/installation.html),
   depending on what container engine you want to use. You can find the software
   used in either `containers/environment.yml` file or `model.environment.txt`
   file, which is the output of the pipeline.

4. Since we cannot redistribute FoldX, you have to build your docker/singularity image using the file in `containers/Dockerfile` with :

   ```bash
   docker buildx build . -f containers/Dockerfile -t ghcr.io/stracquadaniolab/prevent-nf:1.0.0
   ```
and put the `FoldX` executables and its required files in the `${HOME}/.nextflow/assets/stracquadaniolab/prevent-nf/bin` folder.

5. A example of configuration files is attached to the release:

   - `reviewer-1-q3-latent-size-16-smaller-NN.conf` - a configuration file with
     all the necessary parameters to run the pipeline with a small transformer
     and latent size of 16. The results of this model were used to address the
     reviewer's question about hyperparameter tuning.

6. Inspect and modify the configuration file to be able to run the pipeline on
   your machine. Sensitive information, such as personal tokens or account names
   are deliberately masked.

```bash
nextflow run main.nf -profile singularity,singularitynv,cell -c conf/reviewer-1-q3-latent-size-16-smaller-NN.conf
```

```bash
nextflow run main.nf -profile singularity,singularitynv,cell  -entry ENERGY_ESTIMATE -c conf/reviewer-1-q3-latent-size-16-smaller-NN.conf
```

## Authors

- Evgenii Lobzaev (maintainer and main developer)
- Giovanni Stracquadanio (Principal Investigator)
