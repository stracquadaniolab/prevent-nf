// # PREVENT: PRotein Engineering by Variational frEe eNergy approximaTion
// # Copyright (C) 2024  Giovanni Stracquadanio, Evgenii Lobzaev

// # This program is free software: you can redistribute it and/or modify
// # it under the terms of the GNU Affero General Public License as published
// # by the Free Software Foundation, either version 3 of the License, or
// # (at your option) any later version.

// # This program is distributed in the hope that it will be useful,
// # but WITHOUT ANY WARRANTY; without even the implied warranty of
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// # GNU Affero General Public License for more details.

// # You should have received a copy of the GNU Affero General Public License
// # along with this program.  If not, see <https://www.gnu.org/licenses/>.
// enabling nextflow DSL v2
nextflow.enable.dsl=2

// (re)define variables: use user-inputs, otherwise use default values
// global : used in various subworkflows
resultsDir                           = params.resultsDir ? params.resultsDir : "${workflow.launchDir}/results" // default: "results"
seed                                 = params.seed ? params.seed : 0                                           // default: 0

// preprocessing
preprocessing_lmin                   = params.preprocessing.lmin ? params.preprocessing.lmin : 0                                          // default: 0
preprocessing_lmax                   = params.preprocessing.lmax ? params.preprocessing.lmax : 1000                                       // default: 1000
max_sequence_length                  = preprocessing_lmax                                                                                 // set to lmax
preprocessing_val_pct                = params.preprocessing.val_pct ? params.preprocessing.val_pct : 0.2                                  // default: 0.2 (20%)
preprocessing_mmseq_mmseq_clustering = params.preprocessing.mmseq_clustering ? params.preprocessing.mmseq_clustering : "--min-seq-id 0.8" // default: --min-seq-id 0.8
preprocessing_weight_sequences       = params.preprocessing.weight_sequences ? params.preprocessing.weight_sequences : true               // default: true


// training VAE
training_epochs                      = params.training.epochs ? params.training.epochs : 100                                            // default : 100
training_val_freq                    = params.training.val_freq ? params.training.val_freq : 1                                          // default : 1
training_checkpoint_freq             = params.training.checkpoint_freq ? params.training.checkpoint_freq : 50                           // default : 50 (just to have 1 checkpoint)
training_batch_size                  = params.training.batch_size ? params.training.batch_size : 32                                     // default : 32
training_learning_rate               = params.training.learning_rate ? params.training.learning_rate : 0.0001                           // default : 0.0001
training_L2                          = params.training.L2 ? params.training.L2 : 0.0                                                    // default : 0.0 (no L2 regularisation)
training_latent_size                 = params.training.latent_size ? params.training.latent_size : 32                                   // default : 32
training_entry_checkpoint            = params.training.entry_checkpoint ? params.training.entry_checkpoint : "None"                     // default : "None"
training_embedding_size              = params.training.embedding_size ? params.training.embedding_size : 512                             // default : 32
training_dropout_prob                = params.training.dropout_prob ? params.training.dropout_prob : 0.2                                // default : 0.2
training_masking_prob                = params.training.masking_prob ? params.training.masking_prob : 0.0                                // default : 0.0
training_heads                       = params.training.heads ? params.training.heads : 8
training_num_layers_encoder          = params.training.num_layers_encoder ? params.training.num_layers_encoder : 6
training_num_layers_decoder          = params.training.num_layers_decoder ? params.training.num_layers_decoder : 4

// taining GRU (supervised learning)
gru_training_epochs                  = params.training_gru.epochs ? params.training_gru.epochs : 100
gru_training_val_freq                = params.training_gru.val_freq ? params.training_gru.val_freq : 1
gru_training_checkpoint_freq         = params.training_gru.checkpoint_freq ? params.training_gru.checkpoint_freq : 50
gru_training_batch_size              = params.training_gru.batch_size ? params.training_gru.batch_size : 32
gru_training_learning_rate           = params.training_gru.learning_rate ? params.training_gru.learning_rate : 0.0001
gru_training_L2                      = params.training_gru.L2 ? params.training_gru.L2 : 0.0
gru_training_hidden_size             = params.training_gru.hidden_size ? params.training_gru.hidden_size : 32
gru_training_num_layers              = params.training_gru.num_layers ? params.training_gru.num_layers : 1
gru_training_embedding_size          = params.training_gru.embedding_size ? params.training_gru.embedding_size : 32
gru_training_dropout_prob            = params.training_gru.dropout_prob ? params.training_gru.dropout_prob : 0.2

// you sure this is correct ?
if (params.training_gru.bidirectional == true)
    gru_training_directionality = "--no-bidirectional"
else
    gru_training_directionality = "--bidirectional"

if (params.training.condition_on_energy == true)
    training_condition_on_energy = "--condition-on-energy"
else
    training_condition_on_energy = "--no-condition-on-energy"

if (params.training.weighted_sampling == true)
    training_weighted_sampling = "--weighted-sampling"
else
    training_weighted_sampling = "--no-weighted-sampling"

//sampling
sampling_seed                        = params.sampling.seed ? params.sampling.seed : 0                                          // default : 0
sampling_n_samples                   = params.sampling.n_samples ? params.sampling.n_samples : 100                              // default : 100
samling_mini_batch_size              = params.sampling.mini_batch_size ? params.sampling.mini_batch_size : 100                    // default : 100
sampling_max_length                  = params.sampling.max_length ? params.sampling.max_length : 200                            // default : 200
sampling_e_value                     = params.sampling.e_value ? params.sampling.e_value : 0.0001                               // default : 0.0001
samplng_query_coverage               = params.sampling.query_coverage ? params.sampling.query_coverage : 70.0                   // default : 70.0
sampling_argmax_first_element        = params.sampling.argmax_first_element ? params.sampling.argmax_first_element : true       // default : true
sampling_temperature                 = params.sampling.temperature ? params.sampling.temperature : [1.0]                        // default : [1.0]
sampling_mmseq_clustering            = params.sampling.mmseq_clustering ? params.sampling.mmseq_clustering : "--min-seq-id 0.8" // default: "--min-seq-id 0.8"


// printing message of the day
motd = """
--------------------------------------------------------------------------
prevent-nf (${workflow.manifest.version})
--------------------------------------------------------------------------
Name               : ${params.name}
Session ID         : ${workflow.sessionId}
--------------------------------------------------------------------------
Environment information
--------------------------------------------------------------------------
Container                                    : ${workflow.container}
Config files                                 : ${workflow.configFiles}
Project directory                            : ${workflow.projectDir}
Work directory                               : ${workflow.workDir}
Launch directory                             : ${workflow.launchDir}
Results directory                            : ${resultsDir}
Command line                                 : ${workflow.commandLine}
Repository                                   : ${workflow.repository}
CommitID                                     : ${workflow.commitId}
Revision                                     : ${workflow.revision}
--------------------------------------------------------------------------
Preprocessing
--------------------------------------------------------------------------
Minimum allowed sequence length               : ${preprocessing_lmin}
Maximum allowed sequence length               : ${preprocessing_lmax}
MMSEQS clustering options                     : ${preprocessing_mmseq_mmseq_clustering}
Validation set proportion                     : ${preprocessing_val_pct}
Add MSA-related weights to sequences          : ${preprocessing_weight_sequences}
Random seed                                   : ${seed}
--------------------------------------------------------------------------
Training (Transformver VAE)
-------------------------------------------------------------------------- 
Number of training epochs                     : ${training_epochs}
Validation error frequency estimation         : ${training_val_freq}
Model checkpoint frequency                    : ${training_checkpoint_freq}
Batch size                                    : ${training_batch_size}
Learning rate                                 : ${training_learning_rate}
L2 normalisation constant                     : ${training_L2}
Latent distribution dimensionality            : ${training_latent_size}
Conditioning on energy for decoding sequence  : ${training_condition_on_energy}
Weighted sampling                             : ${training_weighted_sampling}
Embeddings dimensionality                     : ${training_embedding_size}
Dropout probability                           : ${training_dropout_prob}
Input masking probability                     : ${training_masking_prob}
Number of heads in (self)-attention           : ${training_heads}
Number of layers in encoder                   : ${training_num_layers_encoder}
Number of layers in decoder                   : ${training_num_layers_decoder}
--------------------------------------------------------------------------
Sampling (Prior & Posterior Transformver VAE) 
--------------------------------------------------------------------------
Random seed                                   : ${sampling_seed}
Number of samples                             : ${sampling_n_samples}
Number of samples in mini batch               : ${samling_mini_batch_size}
E-Value                                       : ${sampling_e_value}
Query coverage                                : ${samplng_query_coverage}
Argmax selection of first AA (posterior only) : ${sampling_argmax_first_element}
Sampling temperatures (posterior only)        : ${sampling_temperature}
Number of autoregressive steps in sampling    : ${sampling_max_length}
MMSEQS clustering options                     : ${sampling_mmseq_clustering}
--------------------------------------------------------------------------
Training (GRU supervised)
-------------------------------------------------------------------------- 
Number of training epochs                     : ${gru_training_epochs}
Validation error frequency estimation         : ${gru_training_val_freq}
Model checkpoint frequency                    : ${gru_training_checkpoint_freq}
Batch size                                    : ${gru_training_batch_size}
Learning rate                                 : ${gru_training_learning_rate}
L2 normalisation constant                     : ${gru_training_L2}
Hidden size                                   : ${gru_training_hidden_size}
Embeddings dimensionality                     : ${gru_training_embedding_size}
Dropout probability                           : ${gru_training_dropout_prob}
Number of layers                              : ${gru_training_num_layers}
GRU directionalify flag                       : ${gru_training_directionality}
-------------------------------------------------------------------------- 
"""

log.info motd
//////////////////////////////////////////////// FOLDX ///////////////////////////////////////////

// repair WT PDB file
process RepairWT{
    tag "WT-repair"

    publishDir "${resultsDir}/preprocessing/energy-estimates", pattern: "${pdb_file.baseName}_Repair.pdb", mode: 'copy'
    publishDir "${resultsDir}/preprocessing/energy-estimates", pattern: "${pdb_file.baseName}_Repair_0_ST.fxout", mode: 'copy'

    input:
        path pdb_file   // WT PDB file

    output:
        path("${pdb_file.baseName}_Repair.pdb"), emit: repaired_pdb
        path("${pdb_file.baseName}_Repair_0_ST.fxout"), emit: energy_estimate_file
    /*
        Set of commands:
        1. FoldX: RepairPDB -> adjust some stuff in PDB, necessary for downstream analysis.
            Need repaired PDB file for downstream analysis.
        2. FoldX: Stability -> compute free energy estimate for WT structure
    */
    script:

    if (params.mutagenesis.os == "macOS")
        """
        FoldX_macOS --command=RepairPDB --pdb=${pdb_file}

        FoldX_macOS --command=Stability --pdb=${pdb_file.baseName}_Repair.pdb
        """
    else
        """
        FoldX_linux --command=RepairPDB --pdb=${pdb_file}

        FoldX_linux --command=Stability --pdb=${pdb_file.baseName}_Repair.pdb
        """
}
// extract Free Energy from Repaired WT PDB file
process ExtractEnergyWT{
    tag "extract-WT-Free-Energy"

    input:
        path wt_energy_file // fxout file that is output of FoldX Stability subcommand

    output:
        stdout emit: energy_wt
    
    script:
    """
    energy=`cat ${wt_energy_file} | awk -F"\t" '{print \$2}'`
    echo \$energy
    """
}
// append Free Energy to WT Fasta file
process AddFreeEnergyWT{

    tag "add-free-energy-to-WT"
    
    publishDir "${resultsDir}/preprocessing/energy-estimates", pattern: "${fasta_wt.baseName}-with-energy.fasta", mode:'copy'

    input:
        path fasta_wt  // WT fasta file
        val energy     // Free energy estimated for WT structure

    output:
        path("${fasta_wt.baseName}-with-energy.fasta"), emit: wt_with_energy

    script:
    """
    miscellaneous.py foldx add-energy-wt ${fasta_wt} ${fasta_wt.baseName}-with-energy.fasta --free-energy-wt ${energy}
    """
}

// main process to generate mutants
process Mutagenesis{
    tag "mutagenesis:seed-${seed};num-of-mutations-${n_mutation_sites}"

    // raw output
    publishDir "${resultsDir}/preprocessing/energy-estimates/seed-${seed}/num-of-mutations-${n_mutation_sites}", pattern: "Raw_${pdb_repaired.baseName}.fxout", saveAs: { filename -> "Raw-Energy-Estimates-${pdb_repaired.baseName}.txt" }, mode:'copy'
    // avg output
    publishDir "${resultsDir}/preprocessing/energy-estimates/seed-${seed}/num-of-mutations-${n_mutation_sites}", pattern: "mutants-energy*.csv", saveAs: { filename -> "mutants-energy.csv" }, mode:'copy'
    // mutants fasta file
    publishDir "${resultsDir}/preprocessing/energy-estimates/seed-${seed}/num-of-mutations-${n_mutation_sites}", pattern: "mutants*.fasta",saveAs: { filename -> "mutants.fasta" }, mode:'copy'

    input:
        path fasta_wt              // WT FASTA file
        path pdb_repaired          // PDB file (repaired)
        each seed                  // list of user provided seeds: for each seed there will be a run
        each n_mutation_sites      // list of user provided number of mutation sites: for each number there will be a run

    output:
        path("Raw_${pdb_repaired.baseName}.fxout"), emit: raw_energy_estimates
        path("mutants-energy*.csv"), emit: avg_energy_estimate
        path("mutants*.fasta"), emit: mutants_fasta

    /*
        Procedure:
            1. Generate mutants (mutants.fasta and individual_list.txt)
            2. Run FoldX to generate mutants
            3. Compute averge FreeEnergy for mutants (line by line):
               - keep only mutant names: 1gs5_Repair_1_0.pdb -> 1gs5_Repair_1 by removing trailing 6 characters
               - count unique values in first column and sum up values in the second column: (1gs5_Repair_1,avg energy)
               - sort by 1st column and save into csv file
            4. Update mutants.fasta with free energy estimates

    */
    script:
    if (params.mutagenesis.os == "macOS")
        """
        miscellaneous.py foldx generate-mutants ${fasta_wt}\
                                                ${pdb_repaired.baseName}\
                                                mutants-${seed}-${n_mutation_sites}.fasta\
                                                --n-mutation-sites ${n_mutation_sites}\
                                                --n-mutants ${params.mutagenesis.n_mutants}\
                                                --seed ${seed}

        FoldX_macOS --command=BuildModel --pdb=${pdb_repaired} --mutant-file=individual_list.txt --numberOfRuns=${params.mutagenesis.foldx_runs}

        cat Raw_${pdb_repaired.baseName}.fxout |\
        awk -F "\t" '/^${pdb_repaired.baseName}/ {print substr(\$1,1,length(\$1)-6) "\t" \$2}' |\
        awk '{v[\$1]+=\$2;n[\$1]++} END{for (i in n) print i","v[i]/n[i]}' |\
        sort -V -k1,1 > mutants-energy-${seed}-${n_mutation_sites}.csv

        miscellaneous.py foldx add-energy-mutant mutants-${seed}-${n_mutation_sites}.fasta\
                                                mutants-energy-${seed}-${n_mutation_sites}.csv
        """
    else
        """
        miscellaneous.py foldx generate-mutants ${fasta_wt}\
                                                    ${pdb_repaired.baseName}\
                                                    mutants-${seed}-${n_mutation_sites}.fasta\
                                                    --n-mutation-sites ${n_mutation_sites}\
                                                    --n-mutants ${params.mutagenesis.n_mutants}\
                                                    --seed ${seed}

        FoldX_linux --command=BuildModel --pdb=${pdb_repaired} --mutant-file=individual_list.txt --numberOfRuns=${params.mutagenesis.foldx_runs}

        cat Raw_${pdb_repaired.baseName}.fxout |\
        awk -F "\t" '/^${pdb_repaired.baseName}/ {print substr(\$1,1,length(\$1)-6) "\t" \$2}' |\
        awk '{v[\$1]+=\$2;n[\$1]++} END{for (i in n) print i","v[i]/n[i]}' |\
        sort -V -k1,1 > mutants-energy-${seed}-${n_mutation_sites}.csv

        miscellaneous.py foldx add-energy-mutant mutants-${seed}-${n_mutation_sites}.fasta\
                                                mutants-energy-${seed}-${n_mutation_sites}.csv
        """
}

// process to combine all mutants and WT FASTA files with energies
process CombineFASTA{
    tag "combine-FASTA-files"

    publishDir "${resultsDir}/preprocessing/energy-estimates", pattern: "dataset-with-energies.fasta", mode:'copy'

    input:
        path wt_fasta      // WT FASTA file with energy
        path mutants_fasta // Mutants FASTA files with energies

    output:
    path("dataset-with-energies.fasta"), emit: final_fasta
    
    script:
    """
    cat ${wt_fasta} ${mutants_fasta} > dataset-with-energies.fasta
    """
}

// run: nextflow main.nf -profile test -entry FreeEnergy
workflow FreeEnergy{
    // construct channels for FASTA file and PDB file
    ch_fasta = Channel.fromPath( params.energy.fasta_file )
    ch_pdb   = Channel.fromPath( params.energy.pdb_file )

    // call RepairWT
    // output: .repaired_pdb -> to be used for mutagenesis
    //         . energy_estimate_file -> to be used to extract Free Energy from file
    out_repair = RepairWT(
                            ch_pdb
                        )

    // extract energy through ExtractEnergyWT
    // output: .energy_wt -> WT energy value
    out_energy = ExtractEnergyWT(

                                    out_repair.energy_estimate_file
                                )

    // add energy to FASTA file
    // output: .wt_with_energy -> FASTA file with WT sequence and its Free Energy
    out_add_free_energy = AddFreeEnergyWT(
                                            ch_fasta,
                                            out_energy.energy_wt
                                         )

    ch_seeds            = Channel.fromList( params.mutagenesis.seeds )
    ch_n_mutation_sites = Channel.fromList( params.mutagenesis.n_mutation_sites )
    
    // mutagenesis
    out_mutagenesis = Mutagenesis(
                                    ch_fasta,
                                    out_repair.repaired_pdb,
                                    ch_seeds,
                                    ch_n_mutation_sites
                                )

    // combine output
    out_combine_fasta = CombineFASTA(
                                        out_add_free_energy.wt_with_energy,
                                        out_mutagenesis.mutants_fasta.collect()
                                    )

}

///////////////////////////////////////////////// PREPROCESSING //////////////////////////////////
// info about pipeline
process Info {
    tag "experiment-info"

    publishDir "${params.resultsDir}", mode: 'copy', overwrite: 'yes'

	output:
        tuple path('model.info.txt'), path('model.environment.txt')

    """
    echo '${motd}' > model.info.txt
    conda list --export > model.environment.txt
    """
}
// merge and clean user-provided FASTA files
process DataFiltering {
    tag "data-filtering"

    // publish log file
    publishDir "${resultsDir}/preprocessing/data-filtering", pattern: "logger-filtering.log", mode: 'copy'
    // filtered FASTA files (for training/validation)
    publishDir "${resultsDir}/preprocessing/data-filtering", pattern: "filtered-sequences.fasta", mode: 'copy'
    // seed FASTA file
    publishDir "${resultsDir}/preprocessing", pattern: "seeds.fasta", mode: 'copy'


    input:
        path fasta_files               // 1.list of FASTA files to concatenate (the result will be used for generating train and validation sets)
        path seed_fasta_files          // 3.list of FASTA files that will be used as seed molecules

    output:
        path("filtered-sequences.fasta"),        emit: clean_file           // file that will be split into train and validation set (full file)
        path("seeds.fasta"),                     emit: seed_file            // filtered seed sequences
        path("logger-filtering.log"),            emit: log_file             // log

    script:
    // 1st command: obtain unique (and clean) fasta file from ${fasta_files}
    // 2nd command: take first ${number_of_test_samples} lines and save it as SMALL file (useful for small testing)
    // 3rd command: same as 1st but for seed sequences
    // 4th command: join logs
    """
    miscellaneous.py various preprocess-input ${fasta_files}\
                                              filtered-sequences.fasta\
                                              logger-inputs.log\
                                              --lmin ${preprocessing_lmin}\
                                              --lmax ${preprocessing_lmax}
    
    miscellaneous.py various preprocess-input ${seed_fasta_files}\
                                              seeds.fasta\
                                              logger-seeds.log\
                                              --lmin ${preprocessing_lmin}\
                                              --lmax ${preprocessing_lmax}
    
    cat logger-inputs.log logger-seeds.log > logger-filtering.log                      
    """
}
// generate train and validation sets
process GenerateTrainValidationSet  {

    tag "train-and-validation-sets-generation"
    
    // log file
    publishDir "${resultsDir}/preprocessing/data-split", pattern: "logger-data-split.log", mode: 'copy'
    // fasta files
    publishDir "${resultsDir}/preprocessing", pattern: "train-set.fasta", mode: 'copy'
    publishDir "${resultsDir}/preprocessing", pattern: "validation-set.fasta", mode: 'copy'
    // sets stats
    publishDir "${resultsDir}/preprocessing", pattern: "train-set-stats.txt", mode: 'copy'
    publishDir "${resultsDir}/preprocessing", pattern: "validation-set-stats.txt", mode: 'copy'
    // representative sequences (for DEBUGGING)
    publishDir "${resultsDir}/preprocessing/data-split", pattern: "pre-validation-set.fasta", mode: 'copy'

    
    input:
        path(clean_fasta_file) // FASTA file
    
    output:
        path("train-set.fasta"),          emit: train_set_fasta_file
        path("validation-set.fasta"),     emit: validation_set_fasta_file
        path("train-set-stats.txt"),      emit: train_set_stats
        path("validation-set-stats.txt"), emit: validation_set_stats
        path("logger-data-split.log"),    emit: log_file
        path("pre-validation-set.fasta"), emit: representative_sequences_fasta_file

    // train and validation sets are created by applying mmseqs2 clustering to clean fasta file
    // representative sequences are considered to be (pre)validation set
    // mmseqs2 output may be tweaked (rebalanced) in order to maintain 80-20 (as an example; defined by user) ratio for train-validation set sizes
    script:
    """
    mmseqs easy-cluster ${preprocessing_mmseq_mmseq_clustering} ${clean_fasta_file} clustered-train tmp
    
    mv clustered-train_rep_seq.fasta pre-validation-set.fasta
    
    miscellaneous.py various rebalance-sets     ${clean_fasta_file}\
                                                pre-validation-set.fasta\
                                                train-set.fasta\
                                                validation-set.fasta\
                                                logger-data-split.log\
                                                --val-pct ${preprocessing_val_pct}\
                                                --seed ${seed}

    """
}
// generate weights for training and validation
process CalculateSequenceWeights{
    tag "sequence-weights-calculation"

    // publish csv,pickle
    publishDir "${resultsDir}/preprocessing/sequence-weights", pattern: "*.{pickle,csv}", mode: 'copy'

    input:
        path(train_set)      // FASTA file with train set
        path(validation_set) // FASTA file with validation set

    output:
        path("${train_set.baseName}-weights.pickle"),      emit: train_weights_pickle
        path("${train_set.baseName}-weights.csv"),         emit: train_weights_csv
        path("${validation_set.baseName}-weights.pickle"), emit: validation_weights_pickle
        path("${validation_set.baseName}-weights.csv"),    emit: validation_weights_csv

    
    // we will compute sequence weights for train set optionally, but leave weights for validation set as 1.0
    script:
    if (preprocessing_weight_sequences)
        """
        clustalo -i ${train_set}\
                -o ${train_set.baseName}-msa.aln\
                --distmat-out ${train_set.baseName}-distmat.csv\
                --full\
                --percent-id
        miscellaneous.py various distmat-weights ${train_set.baseName}-distmat.csv\
                                                 ${train_set.baseName}-weights.pickle\
                                                 ${train_set.baseName}-weights.csv

        miscellaneous.py various default-weights ${validation_set}\
                                                 ${validation_set.baseName}-weights.pickle\
                                                 ${validation_set.baseName}-weights.csv
        """
    else
        """
        miscellaneous.py various default-weights ${train_set}\
                                                 ${train_set.baseName}-weights.pickle\
                                                 ${train_set.baseName}-weights.csv

        miscellaneous.py various default-weights ${validation_set}\
                                                 ${validation_set.baseName}-weights.pickle\
                                                 ${validation_set.baseName}-weights.csv
        """

}
// combine sequences into single FASTA file
process MergeFasta{
    tag "merging-FASTA-files"

    input:
        path(fasta_files_to_merge) // [list of FASTA files]

    output:
        path("merged-files.fasta"), emit: merged_file
    
    script:
    """
    cat ${fasta_files_to_merge} > merged-files.fasta
    """
}

// MAIN PREPROCESSING WORKFLOW
workflow Preprocessing{
    take:
        fasta_files_to_join       // raw FASTA files
        seed_fasta_files_to_join  // raw FASTA files with seeds
    main:
        // clean data -> remove duplicates/non-canonical AAs
        out_data_filtering = DataFiltering(
                                                fasta_files_to_join.collect(),
                                                seed_fasta_files_to_join.collect()
                                               )
        
        // generate train and validation sets
        out_generate_train_validation_set = GenerateTrainValidationSet(
                                                                        out_data_filtering.clean_file
                                                                      )

        
        // compute weights for sequences: weights are computed for training set, for validation set they are set to 1.0
        // out_sequence_weights = CalculateSequenceWeights(
        //                                                     out_generate_train_validation_set.train_set_fasta_file,
        //                                                     out_generate_train_validation_set.validation_set_fasta_file
        //                                                )
    emit:
        train_file    = out_generate_train_validation_set.train_set_fasta_file      // clean train set file
        val_file      = out_generate_train_validation_set.validation_set_fasta_file // clean validation set file
        seed_file     = out_data_filtering.seed_file                                // clean seed file
        //train_weights = out_sequence_weights.train_weights_pickle                   // training weights
        //val_weights   = out_sequence_weights.validation_weights_pickle              // validation weights
}

//////////////////////////////////////////////// TRAINING ////////////////////////////////////////
// TransformerVAE model training process
process TrainModel{
    tag "model-training"

    // no checkpoint or sequence weights
    input:
        path(trainFile)      // train set
        path(validationFile) // validation set
        path(checkpoint)     // model checkpoint

    // csv files with losses: train by batch and by epoch; validation by epoch
    publishDir "${resultsDir}/training/losses/csv", pattern: "loss-*.csv", mode: 'copy'
    // log files with losses and gradients
    publishDir "${resultsDir}/training/losses/log", pattern: "*.log", mode: 'copy'

    // publish pytorch files (best model): will have 2 pytorch files and 1 pickle file
    publishDir "${resultsDir}/training/best-models", pattern: "best-*.pytorch", mode: 'copy'
    publishDir "${resultsDir}/training/best-models", pattern: "model-input.pickle", mode: 'copy'

    // publish intermediate model results + pickle file
    publishDir "${resultsDir}/training/model-checkpoints", pattern: "model-checkpoint-*.pytorch", mode: 'copy'
    publishDir "${resultsDir}/training/model-checkpoints", pattern: "model-input.pickle", mode: 'copy'

    // embedding weights
    //publishDir "${resultsDir}/training/embeddings", pattern: "embedding-weights-best-*.csv", mode: 'copy'

    output:
        path trainFile,  emit: train_set_fasta_file
        path validationFile, emit: validation_set_fasta_file
        // pickle (input parameters to the model)
        path("model-input.pickle"), emit: pickle_file
        // best models
        path("best-train-error.pytorch"), emit:best_train_pytorch
        path("best-validation-error.pytorch"), emit: best_validation_pytorch
        // logs
        path("loss-batch-train.log"), emit: train_batch_log
        path("loss-epoch-train.log"), emit: train_loss_log
        path("loss-epoch-validation.log"), emit: validation_loss_log
        path("sequence-batch-update.log"), emit: sequence_batch_log
        // checkpoints
        path("model-checkpoint-*.pytorch"), emit: model_chekpoints
        // embedding weights (for best models)
        //path("embedding-weights-best-*.csv"), emit: embedding_weights
        // csv files with losses -> needs to be separate
        //path("loss-batch-train.csv"), emit: csv_file_batch
        path("loss-epoch-train.csv"), emit: csv_file_epoch
        path("loss-epoch-validation.csv"), emit: csv_file_val_epoch

    // no checkpoint
    script:
    """
    train.py transformer gaussian ${trainFile}\
                                  ${validationFile}\
                                  loss-batch-train.log\
                                  loss-epoch-train.log\
                                  loss-epoch-validation.log\
                                  sequence-batch-update.log\
                                  model-checkpoint\
                                  loss-epoch-train.csv\
                                  loss-epoch-validation.csv\
                                  model-input.pickle\
                                  best-train-error.pytorch\
                                  best-validation-error.pytorch\
                                  --epochs ${training_epochs}\
                                  --learning-rate ${training_learning_rate}\
                                  --lambda-constant ${training_L2}\
                                  --validation-freq-epoch ${training_val_freq}\
                                  --checkpoint-freq-epoch ${training_checkpoint_freq}\
                                  --max-sequence-length ${max_sequence_length}\
                                  --batch-size ${training_batch_size}\
                                  --embedding-size ${training_embedding_size}\
                                  --latent-size ${training_latent_size}\
                                  ${training_condition_on_energy}\
                                  ${training_weighted_sampling}\
                                  --dropout-prob ${training_dropout_prob}\
                                  --masking-prob ${training_masking_prob}\
                                  --heads ${training_heads}\
                                  --num-layers-encoder ${training_num_layers_encoder}\
                                  --num-layers-decoder ${training_num_layers_decoder}\
                                  --seed ${seed}\
                                  --pt-checkpoint ${checkpoint}
    """
}

//  TransformerVAE model: get params of latent space
process EstimateLatentSpace{
    tag "latent-space-estimation:${model_identifier};${fasta_file.baseName}"


    // store csv file in the training folder
    publishDir "${resultsDir}/training/latent-space-estimation/${model_identifier}/${fasta_file.baseName}", pattern: "*.csv", mode: 'copy'

    input:
        tuple path(pickle_file), path(pytorch_file), val(model_identifier), path(fasta_file) // [pickle, pytorch, model identifier, FASTA file with sequences]

    output:
        path("${fasta_file.baseName}-means.csv"), emit: means
        path("${fasta_file.baseName}-stds.csv"), emit: stds

    script:
    """
    downstream.py transformer estimate-latent-space ${fasta_file.baseName}-means.csv\
                                                    ${fasta_file.baseName}-stds.csv\
                                                    ${pytorch_file}\
                                                    ${pickle_file}\
                                                    ${fasta_file}\
                                                    --batch-size ${training_batch_size}\
                                                    --max-sequence-length ${max_sequence_length}\
                                                    --seed 12345
    """
}

// supervised GRU model training process: sequence -> energy
process TrainModelPredictiveGRU{
    tag "model-training-supervised-GRU"

    input:
        path(trainFile)      // train set
        path(validationFile) // validation set
        path(checkpoint)     // model checkpoint

    // csv files with losses: train by batch and by epoch; validation by epoch
    publishDir "${resultsDir}/supervised-model-gru/training/losses/csv", pattern: "loss-*.csv", mode: 'copy'
    // log files with losses and gradients
    publishDir "${resultsDir}/supervised-model-gru/training/losses/log", pattern: "*.log", mode: 'copy'

    // publish pytorch files (best model): will have 2 pytorch files and 1 pickle file
    publishDir "${resultsDir}/supervised-model-gru/training/best-models", pattern: "best-*.pytorch", mode: 'copy'
    publishDir "${resultsDir}/supervised-model-gru/training/best-models", pattern: "model-input.pickle", mode: 'copy'

    // publish intermediate model results + pickle file
    publishDir "${resultsDir}/supervised-model-gru/training/model-checkpoints", pattern: "model-checkpoint-*.pytorch", mode: 'copy'
    publishDir "${resultsDir}/supervised-model-gru/training/model-checkpoints", pattern: "model-input.pickle", mode: 'copy'

    output:
        path trainFile,  emit: train_set_fasta_file
        path validationFile, emit: validation_set_fasta_file
        // pickle (input parameters to the model)
        path("model-input.pickle"), emit: pickle_file
        // best models
        path("best-train-error.pytorch"), emit:best_train_pytorch
        path("best-validation-error.pytorch"), emit: best_validation_pytorch
        // logs
        path("loss-batch-train.log"), emit: train_batch_log
        path("loss-epoch-train.log"), emit: train_loss_log
        path("loss-epoch-validation.log"), emit: validation_loss_log
        // checkpoints
        path("model-checkpoint-*.pytorch"), emit: model_chekpoints
        path("loss-epoch-train.csv"), emit: csv_file_epoch
        path("loss-epoch-validation.csv"), emit: csv_file_val_epoch

    // max_sequence_length is the same as for VAE
    // seed is the same as for VAE
    script:
    """
    train.py predictor gru  ${trainFile}\
                            ${validationFile}\
                            loss-batch-train.log\
                            loss-epoch-train.log\
                            loss-epoch-validation.log\
                            model-checkpoint\
                            loss-epoch-train.csv\
                            loss-epoch-validation.csv\
                            model-input.pickle\
                            best-train-error.pytorch\
                            best-validation-error.pytorch\
                            --epochs ${gru_training_epochs}\
                            --learning-rate ${gru_training_learning_rate}\
                            --lambda-constant ${gru_training_L2}\
                            --validation-freq-epoch ${gru_training_val_freq}\
                            --checkpoint-freq-epoch ${gru_training_checkpoint_freq}\
                            --max-sequence-length ${max_sequence_length}\
                            --batch-size ${gru_training_batch_size}\
                            --embedding-size ${gru_training_embedding_size}\
                            --hidden-size ${gru_training_hidden_size}\
                            --num-layers ${gru_training_num_layers}\
                            ${gru_training_directionality}\
                            --dropout-prob ${gru_training_dropout_prob}\
                            --seed ${seed}\
                            --pt-checkpoint ${checkpoint}
    """

    

}

// supervised TF encoder model training process : sequence -> energy
process TrainModelPredictiveTFEncoder{
    tag "model-training-supervised-TF"

    input:
        path(trainFile)          // train set
        path(validationFile)     // validation set
        path(checkpoint)         // model checkpoint
        val(useWeightedSampling) // whether to use WeitedRandomSampling or not

    // csv files with losses: train by batch and by epoch; validation by epoch
    publishDir "${resultsDir}/supervised-model-tf/training/losses/csv", pattern: "loss-*.csv", mode: 'copy'
    // log files with losses and gradients
    publishDir "${resultsDir}/supervised-model-tf/training/losses/log", pattern: "*.log", mode: 'copy'

    // publish pytorch files (best model): will have 2 pytorch files and 1 pickle file
    publishDir "${resultsDir}/supervised-model-tf/training/best-models", pattern: "best-*.pytorch", mode: 'copy'
    publishDir "${resultsDir}/supervised-model-tf/training/best-models", pattern: "model-input.pickle", mode: 'copy'

    // publish intermediate model results + pickle file
    publishDir "${resultsDir}/supervised-model-tf/training/model-checkpoints", pattern: "model-checkpoint-*.pytorch", mode: 'copy'
    publishDir "${resultsDir}/supervised-model-tf/training/model-checkpoints", pattern: "model-input.pickle", mode: 'copy'

    output:
        path trainFile,  emit: train_set_fasta_file
        path validationFile, emit: validation_set_fasta_file
        // pickle (input parameters to the model)
        path("model-input.pickle"), emit: pickle_file
        // best models
        path("best-train-error.pytorch"), emit:best_train_pytorch
        path("best-validation-error.pytorch"), emit: best_validation_pytorch
        // logs
        path("loss-batch-train.log"), emit: train_batch_log
        path("loss-epoch-train.log"), emit: train_loss_log
        path("loss-epoch-validation.log"), emit: validation_loss_log
        // checkpoints
        path("model-checkpoint-*.pytorch"), emit: model_chekpoints
        path("loss-epoch-train.csv"), emit: csv_file_epoch
        path("loss-epoch-validation.csv"), emit: csv_file_val_epoch

    // max_sequence_length is the same as for VAE
    // seed is the same as for VAE
    script:
    """
    train.py predictor tf   ${trainFile}\
                            ${validationFile}\
                            loss-batch-train.log\
                            loss-epoch-train.log\
                            loss-epoch-validation.log\
                            model-checkpoint\
                            loss-epoch-train.csv\
                            loss-epoch-validation.csv\
                            model-input.pickle\
                            best-train-error.pytorch\
                            best-validation-error.pytorch\
                            --epochs ${params.training_tf.epochs}\
                            --learning-rate ${params.training_tf.learning_rate}\
                            --lambda-constant ${params.training_tf.L2}\
                            --validation-freq-epoch ${params.training_tf.val_freq}\
                            --checkpoint-freq-epoch ${params.training_tf.checkpoint_freq}\
                            --max-sequence-length ${max_sequence_length}\
                            --batch-size ${params.training_tf.batch_size}\
                            --embedding-size ${params.training_tf.embedding_size}\
                            --latent-size ${params.training_tf.latent_size}\
                            ${useWeightedSampling}\
                            --dropout-prob ${params.training_tf.dropout_prob}\
                            --heads ${params.training_tf.heads }\
                            --num-layers-encoder ${params.training_tf.num_layers_encoder}\
                            --seed ${seed}\
                            --pt-checkpoint ${checkpoint}
    """
}


//////////////////////////////////////////////// PRIOR SAMPLING /////////////////////////////////////////
// sample from prior: at each sampling step we obtain (seq, energy), which is later merged into (seq, {energies})
process PriorSampling{

    tag "prior-sampling,model-params:${model_identifier}"

    // publish duplicated/unduplicated VAE samples FASTA file
    publishDir "${resultsDir}/prior/sampling-results/${model_identifier}", pattern: "all-prior-samples-all-${model_identifier}.fasta", mode: 'copy'
    publishDir "${resultsDir}/prior/sampling-results/${model_identifier}", pattern: "prior-samples-all-${model_identifier}.fasta", mode: 'copy'
    // publish pickle file
    publishDir "${resultsDir}/prior/sampling-results/${model_identifier}", pattern: "all-prior-results-${model_identifier}.pickle", mode: 'copy'
    // publish txt file with duplicates
    publishDir "${resultsDir}/prior/sampling-results/${model_identifier}", pattern: "duplicated.detail.txt", mode: 'copy'

    
    input:
        tuple path(pickle_file), path(pytorch_file), val(model_identifier), path(fasta_files_clean_against) //[pickle, pytorch, modelID, list of FASTA files that contain sequences that should be removed from samples file]

    output:
        tuple path("all-prior-results-${model_identifier}.pickle"), val(model_identifier), emit: output_for_protein_analysis_prior // tuple [pickle, modelID]
        tuple path(pickle_file), path(pytorch_file), path("prior-samples-all-${model_identifier}.fasta"), val(model_identifier), emit: output_for_filtering           // tuple [raw FASTA file, modelID]
        path("duplicated.detail.txt"), emit: duplicates_details
        path("all-prior-samples-all-${model_identifier}.fasta"), emit: initial_samples

    // command description:
    // 1st      : sample from the prior distribution. if identical sequences are sampled, they will be collapsed and the statistics will be aggregated
    // 2nd - 6th: from the output of 1st command remove sequences that can be found in ${fasta_files_clean_against}
    // 2nd      : join all FASTA file together
    // 3rd      : remove duplicated sequences from this joint file (called joint.fasta)
    // 4th      : if there are no duplicates, need to create duplicated.detail.txt, otherwise the code will break
    // 5th      : take duplicated.detail.txt file, remove first column then flatten it -> output.duplicated.detail.txt
    // 6th      : take output.duplicated.detail.txt and remove all sequences listed there from all-prior-samples* file
    script:
    """
    downstream.py transformer sample-prior ${pytorch_file}\
                                           ${pickle_file}\
                                           all-prior-results-${model_identifier}.pickle\
                                           all-prior-samples-all-${model_identifier}.fasta\
                                           --n-samples ${sampling_n_samples}\
                                           --seed ${sampling_seed}\
                                           --max-number-of-steps ${sampling_max_length}

    cat ${fasta_files_clean_against} all-prior-samples-all-${model_identifier}.fasta > joint.fasta

    cat joint.fasta | seqkit rmdup -s -o unduplicated-joint.fasta -d duplicated.fasta -D duplicated.detail.txt

    [ -f duplicated.detail.txt ] && echo "duplicated.detail.txt file exist." || touch duplicated.detail.txt

    cut -f2- duplicated.detail.txt | tr ',' '\n' | tr -d "[:blank:]" > output.duplicated.detail.txt

    miscellaneous.py various remove-sequences output.duplicated.detail.txt all-prior-samples-all-${model_identifier}.fasta prior-samples-all-${model_identifier}.fasta
    """
}
// filter prior samples obtained from PriorSampling process
process PriorFiltering{
    tag "prior-filtering,model-params:${modelID},database:${database.baseName}"

    // publish filtered VAE samples FASTA file
    publishDir "${resultsDir}/prior/sampling-results/${modelID}/${database.baseName}", pattern: "filtered-${vae_samples_fasta_file.baseName}.fasta", mode: 'copy'

    // publish csv file with filtered results (we will need it for statisitcs calculation)
    publishDir "${resultsDir}/prior/sampling-results/${modelID}/${database.baseName}", pattern: "${vae_samples_fasta_file.baseName}.csv", mode: 'copy'

    //publish log file
    publishDir "${resultsDir}/prior/sampling-results/${modelID}/${database.baseName}", pattern: "${vae_samples_fasta_file.baseName}-logger.log", mode: 'copy'

    input:
        tuple path(pickle_file),path(pytorch_file), path(vae_samples_fasta_file), val(modelID) // tuple: [pickle file, pytorch file, raw FASTA file, modelID] : pickle and pytorch are not used here but will be carried over to the next process
        each path(database) // train or validation  set
    output:
        path("${vae_samples_fasta_file.baseName}.csv"), emit: csv_file
        path("filtered-${vae_samples_fasta_file.baseName}.fasta"), emit: filtered_fasta_file
        path("${vae_samples_fasta_file.baseName}-logger.log"), emit: log_file
        tuple path(pickle_file), path(pytorch_file), val(modelID), path(database), path("filtered-${vae_samples_fasta_file.baseName}.fasta"), emit: output_for_energy_reestimation // [pickle, pytorch, modelID, database, clean FASTA]

    /* 
        1st command: create protein DB out of FASTA file with single seed
        2nd command: run blastp of ${vae_samples_fasta_file} against newly created DB -> receive csv file where filtering is done only by e-value: temp.csv!
        3rd command: filter by qcovs (last column)
        4th command: obtain list of seqIDs
        5th command: obtain filtered FASTA file
        6th command: compute statistics and save them to log file (csv headers added)
    */
    script:
    """
    makeblastdb -dbtype prot -in ${database}

    blastp -out temp.csv -outfmt '10 qseqid sseqid score bitscore evalue pident ppos qcovs' -query ${vae_samples_fasta_file} -db ${database} -evalue ${sampling_e_value}

    cat temp.csv | awk -F "," '/^seq/ { if (\$NF > ${samplng_query_coverage}) print \$0 }' > ${vae_samples_fasta_file.baseName}.csv

    cat ${vae_samples_fasta_file.baseName}.csv | cut -d ',' -f1 | uniq > seq-list.txt

    seqtk subseq ${vae_samples_fasta_file} seq-list.txt > filtered-${vae_samples_fasta_file.baseName}.fasta

    downstream.py transformer csv-stats ${vae_samples_fasta_file.baseName}.csv\
                                        ${vae_samples_fasta_file.baseName}-logger.log\
                                        ${sampling_n_samples}    
    """

}
// re-estimate (by running encoder) energies for (clean) sequences from PriorFiltering process
process ReestimateEnergiesPrior{
    errorStrategy 'ignore' // in case no sequences

    tag "energy-reestimation-prior,model-params:${model_identifier},database:${database.baseName}"

    // publish reestimated energies
    publishDir "${resultsDir}/prior/sampling-results/${model_identifier}/${database.baseName}/reestimated-energies", pattern: "*.csv", mode: 'copy'

    input:
        tuple path(pickle_file), path(pytorch_file), val(model_identifier), path(database), path(fasta_file) // [pickle file, pt file, model ID, database_name, FASTA with clean sequences]: database is relevant b/c FASTA was cleaned against that DB, so we want to output files in the correct folder
    
    output:
        path("*.csv"), emit: all_csv // all csv files
    
    script:
    """
    downstream.py transformer seq2energy ${fasta_file.baseName}-estimated-average-values.csv\
                                         ${fasta_file.baseName}-individual\
                                         ${pytorch_file}\
                                         ${pickle_file}\
                                         ${fasta_file}\
                                         --n-samples 300\
                                         --batch-size ${training_batch_size}\
                                         --seed ${sampling_seed}
    """
}

// protein analysis of prior samples
process ProteinAnalysisPrior {
    tag "prior-protein-analysis-${model_identifier}"
    errorStrategy 'ignore'

    // for now save all files
    publishDir "${resultsDir}/prior/protein-analysis-results/${model_identifier}", pattern: "*.{pdf,csv,log}", mode: "copy"

    input:
        tuple path(pickle_file), val(model_identifier)   // pickle file with sequence information and model identifier

    output:
        path("*.csv"), emit: csv_files
        path("*.pdf"), emit: pdf_files
        path("*.log"), emit: log_files
    
    script:
    """
    downstream.py protanalysis prior ${pickle_file}\
                                     protein-analysis-prior.log\
                                     ${model_identifier}

    """ 

    
}
// subworkflow to do prior sampling + analysis
workflow Prior{
    take:
        input_prior_sampling // input for prior sampling: tuple[pickle_file,pytorch_file,modelID]
        databases            // list of FASTA files to query against (either train or validation set)
        fasta_seq_to_exclude // FASTA file with sequences that should be excluded
    main:

        // modify ${input_prior_sampling}, extend each element with ${databases}
        input_prior_sampling_extended = input_prior_sampling.combine(fasta_seq_to_exclude) // [pickle, pytorch, modelID, FASTA file with sequences to exclude]

        // run prior sampling
        out_downstream_prior_sampling = PriorSampling(input_prior_sampling_extended)

        // filter prior samples (against DB)
        out_prior_filtering = PriorFiltering( 
                                                out_downstream_prior_sampling.output_for_filtering, // [[pickle file, pytorch file, raw FASTA file, modelID]
                                                databases
                                            )
        // reestimate energies by running filtered sequences through encoder
        ReestimateEnergiesPrior(
                                    out_prior_filtering.output_for_energy_reestimation
                               )

        // run protein analysis on prior results NOT NEEDED NOW
        // ProteinAnalysisPrior(
        //                         out_downstream_prior_sampling.output_for_protein_analysis_prior
        //                     )
}

////////////////////////////////////////////////// POSTERIOR SAMPLING ////////////////////////////////////////
// sample from variational (seeded) posterior: at each sampling step we obtain (seq, energy), which is later merged into (seq, {energies})
process SeededSampling{
    tag "seeded-sampling:${seed_fasta_file.baseName},model-params:${model_identifier},temperature:${temperature}"

    // publish seed FASTA file
    publishDir "${resultsDir}/seeded/sampling-results/${model_identifier}/temperature-${temperature}/${seed_fasta_file.baseName}", pattern: "${seed_fasta_file}", mode: 'copy'

    // publish duplicated/unduplicated VAE samples FASTA file
    publishDir "${resultsDir}/seeded/sampling-results/${model_identifier}/temperature-${temperature}/${seed_fasta_file.baseName}", pattern: "all-seeded-samples-all-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.fasta", mode: 'copy'
    publishDir "${resultsDir}/seeded/sampling-results/${model_identifier}/temperature-${temperature}/${seed_fasta_file.baseName}", pattern: "seeded-samples-all-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.fasta", mode: 'copy'
    // publish txt file with duplicates
    publishDir "${resultsDir}/prior/sampling-results/${model_identifier}", pattern: "duplicated.detail.txt", mode: 'copy'

    // publish pickle file
    publishDir "${resultsDir}/seeded/sampling-results/${model_identifier}/temperature-${temperature}/${seed_fasta_file.baseName}", pattern: "all-seeded-results-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.pickle", mode: 'copy'

    input:
        // ${model_identifier} is "train" or "validation"
        // ${first_element} tells whether to avoid sampling first element or not
        tuple path(pickle_file), path(pytorch_file), val(model_identifier), path(seed_fasta_file), val(first_element), path(fasta_files_clean_against) //[pickle, pytorch, modelID, seed FASTA, whether to do argmax for the first sample, list of FASTA files that contain sequences that should be removed from samples file]
        each temperature

    output:
        tuple val("${seed_fasta_file.baseName}"), val(model_identifier), val(temperature), path("seeded-samples-all-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.fasta"), path(seed_fasta_file), path(pickle_file), path(pytorch_file), emit: raw_to_filter // tuple [seedID,modelID,temperature,raw FASTA file with VAE samples, seed FASTA file, pickle, pytorch]
        tuple path("all-seeded-results-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.pickle"), val("${seed_fasta_file.baseName}"), val(model_identifier), val(temperature), emit: output_for_protein_analysis_posterior   //tuple [pickle file, seedID, modelID, temperature]
        path("duplicated.detail.txt"), emit: duplicates_details
        path("all-seeded-samples-all-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.fasta"), emit: initial_samples
    
    // command description:
    // 1st      : sample from the prior distribution. if identical sequences are sampled, they will be collapsed and the statistics will be aggregated
    // 2nd - 6th: from the output of 1st command remove sequences that can be found in ${fasta_files_clean_against}
    // 2nd      : join all FASTA file together
    // 3rd      : remove duplicated sequences from this joint file (called joint.fasta)
    // 4th      : if there are no duplicates, need to create duplicated.detail.txt, otherwise the code will break
    // 5th      : take duplicated.detail.txt file, remove first column then flatten it -> output.duplicated.detail.txt
    // 6th      : take output.duplicated.detail.txt and remove all sequences listed there from all-seeded-samples* file
    script:
    """
    downstream.py transformer sample-seeded ${pytorch_file}\
                                            ${pickle_file}\
                                            all-seeded-results-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.pickle\
                                            ${seed_fasta_file}\
                                            all-seeded-samples-all-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.fasta\
                                            --max-number-of-steps ${sampling_max_length}\
                                            --n-samples ${sampling_n_samples}\
                                            --mini-batch-size ${samling_mini_batch_size}\
                                            --seed ${sampling_seed}\
                                            --t ${temperature}\
                                            ${first_element}

    cat ${fasta_files_clean_against} all-seeded-samples-all-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.fasta > joint.fasta

    cat joint.fasta | seqkit rmdup -s -o unduplicated-joint.fasta -d duplicated.fasta -D duplicated.detail.txt

    [ -f duplicated.detail.txt ] && echo "duplicated.detail.txt file exist." || touch duplicated.detail.txt

    cut -f2- duplicated.detail.txt | tr ',' '\n' | tr -d "[:blank:]" > output.duplicated.detail.txt

    miscellaneous.py various remove-sequences output.duplicated.detail.txt all-seeded-samples-all-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.fasta seeded-samples-all-${seed_fasta_file.baseName}-${model_identifier}-temperature-${temperature}.fasta
    """
}
// filter posterior samples
process SeededFiltering{
    tag "seeded-filtering:${seedID},model-params:${modelID},temperature:${temperature}"

    // publish filtered VAE samples FASTA file
    publishDir "${resultsDir}/seeded/sampling-results/${modelID}/temperature-${temperature}/${seedID}", pattern: "filtered-${vae_samples_fasta_file.baseName}.fasta", mode: 'copy'

    // publish csv file with filtered results (we will need it for statisitcs calculation)
    publishDir "${resultsDir}/seeded/sampling-results/${modelID}/temperature-${temperature}/${seedID}", pattern: "${vae_samples_fasta_file.baseName}.csv", mode: 'copy'

    //publish log file
    publishDir "${resultsDir}/seeded/sampling-results/${modelID}/temperature-${temperature}/${seedID}", pattern: "${vae_samples_fasta_file.baseName}-logger.log", mode: 'copy'

    
    input:
        tuple val(seedID), val(modelID), val(temperature), path(vae_samples_fasta_file), path(seed_fasta_file), path(pickle_file), path(pytorch_file) // tuple [seedID,modelID,temperature,raw FASTA file with VAE samples, seed FASTA file, pickle, pytorch ]: pickle and pytorch are not needed, just carry over!
    output:
        tuple val(seedID), val(modelID), path("filtered-${vae_samples_fasta_file.baseName}.fasta"), emit: seed_id_clean_fasta_tuple // tuple [seed ID, model identifier, clean fasta file]
        path("${vae_samples_fasta_file.baseName}.csv"), emit: csv_file
        path("filtered-${vae_samples_fasta_file.baseName}.fasta"), emit: filtered_fasta
        path("${vae_samples_fasta_file.baseName}-logger.log"), emit: log_file
        tuple val(seedID), val(modelID), val(temperature), path(pickle_file), path(pytorch_file), path("filtered-${vae_samples_fasta_file.baseName}.fasta"), emit: output_for_energy_reestimation // [seedID, modelID, temperature, pickle, pytorch, clean FASTA]

    /* 
        1st command: create protein DB out of FASTA file with single seed
        2nd command: run blastp of ${vae_samples_fasta_file} against newly created DB -> receive csv file where filtering is done only by e-value: temp.csv!
        3rd command: filter by qcovs (last column)
        4th command: obtain list of seqIDs
        5th command: obtain filtered FASTA file
        6th command: compute statistics and save them to log file (csv headers added)
    */
    script:
    """
    makeblastdb -dbtype prot -in ${seed_fasta_file}
    
    blastp -out temp.csv -outfmt '10 qseqid sseqid score bitscore evalue pident ppos qcovs' -query ${vae_samples_fasta_file} -db ${seed_fasta_file} -evalue ${sampling_e_value}
    
    cat temp.csv | awk -F "," '/^seq/ { if (\$NF > ${samplng_query_coverage}) print \$0 }' > ${vae_samples_fasta_file.baseName}.csv

    cat ${vae_samples_fasta_file.baseName}.csv | cut -d ',' -f1 | uniq > seq-list.txt

    seqtk subseq ${vae_samples_fasta_file} seq-list.txt > filtered-${vae_samples_fasta_file.baseName}.fasta

    downstream.py transformer csv-stats ${vae_samples_fasta_file.baseName}.csv\
                                        ${vae_samples_fasta_file.baseName}-logger.log\
                                        ${sampling_n_samples}    
    """
}
// re-estimate (by running encoder) energies for (clean) sequences from SeededFiltering process
process ReestimateEnergiesPosterior{
    errorStrategy 'ignore' // in case no sequences

    tag "energy-reestimation,seed:${seedID},model-params:${modelID},temperature:${temperature}"

    // publish reestimated energies
    publishDir "${resultsDir}/seeded/sampling-results/${modelID}/temperature-${temperature}/${seedID}/reestimated-energies", pattern: "*.csv", mode: 'copy'

    input:
        tuple val(seedID), val(modelID), val(temperature), path(pickle_file), path(pytorch_file), path(fasta_file) // [seedID, modelID, temperature, pickle file, pytorch file, FASTA with clean sequences]
    
    output:
        path("*.csv"), emit: all_csv // all csv files
    
    script:
    """
    downstream.py transformer seq2energy ${fasta_file.baseName}-estimated-average-values.csv\
                                         ${fasta_file.baseName}-individual\
                                         ${pytorch_file}\
                                         ${pickle_file}\
                                         ${fasta_file}\
                                         --n-samples 300\
                                         --batch-size ${training_batch_size}\
                                         --seed ${sampling_seed}
    """
}

// nextflow run ... -profile ... -entry ENERGY_ESTIMATE
workflow ENERGY_ESTIMATE{
    // calling process ReestimateEnergiesPosterior

    // some placeholders, they dont really mean anything
    seedID      = Channel.of("my-seed-id")
    modelID     = Channel.of("my-model")
    temperature = Channel.of("my-temperature")

    pickle_file  = Channel.fromPath( params.energy_estimate.pickle_file )
    pytorch_file = Channel.fromPath( params.energy_estimate.pytorch_file )
    fasta_file   = Channel.fromPath( params.energy_estimate.fasta_file )

    input = seedID.combine(
                                    modelID.combine(
                                                        temperature.combine(
                                                                                pickle_file.combine(
                                                                                                        pytorch_file.combine(
                                                                                                                                fasta_file
                                                                                                                            )
                                                                                                    )
                                                                            )
                                                )
                            )

    ReestimateEnergiesPosterior(input)



}

// subworkflow to do posterior sampling + analysiss
workflow Posterior{
    take:
        input_posterior_sampling            // input for posterior sampling: tuple [pickle_file, pytorch_file, model_identifier, seed_fasta_file, first_element]
        temperatures                        // temperatures for sampling
        fasta_seq_to_exclude                // FASTA file with sequences that should be excluded

    main:

        // modify input_posterior_sampling
        input_posterior_sampling_extended = input_posterior_sampling.combine(fasta_seq_to_exclude)
        
        // run posterior sampling
        out_seeded_sampling = SeededSampling(
                                                input_posterior_sampling_extended,
                                                temperatures
                                            )
        // filter sequences
        out_filtering = SeededFiltering(
                                            out_seeded_sampling.raw_to_filter
                                        )
        // reestimate energy by running through encoder
        ReestimateEnergiesPosterior(
                                        out_filtering.output_for_energy_reestimation
                                   )
}

//////////////////////////////////////////////////// PROCESSES RELATED TO PREDICTIVE MODELS //////////////////
process PredictEnergies{
    errorStrategy 'ignore' // in case no sequences

    tag "energy-prediction-using-predictive-model"

    // publish reestimated energies
    publishDir "${resultsDir}/supervised-model-tf/predicted-energies/${modelID}", pattern: "*.{csv,fasta}", mode: 'copy'

    input:
        tuple val(modelID), path(pickle_file), path(pytorch_file), path(fasta_file) // [modelID, pickle file, pytorch file, FASTA with clean sequences]
    
    output:
        path("${fasta_file.baseName}-predicted-energies.fasta"), emit: fasta_with_predicted_energies // FASTA file with reestimated energies, sequences are the same here
        path("${fasta_file.baseName}-predicted-energies.csv"), emit: csv_predicted_energies // CSV with reestimated energies
    
    script:
    """
    downstream.py predictor tf-predict-energy ${fasta_file.baseName}-predicted-energies.fasta\
                                              ${fasta_file.baseName}-predicted-energies.csv\
                                              ${pytorch_file}\
                                              ${pickle_file}\
                                              ${fasta_file}\
                                              --seed ${sampling_seed}
    """
}


////////////////////////////////////////////////////// LATENT SPACE OPTIMIZATION ////////////////////
// SGD optimisation, no constraints
process SGDOptimisation {
    tag "SGD-optimisation-in-latent-space;seed-${opt_seed}"

    // publish results
    publishDir "${resultsDir}/latent-space-sgd-optimisation/seed-${opt_seed}", pattern: "*.{log}", mode: 'copy'
    publishDir "${resultsDir}/latent-space-sgd-optimisation/seed-${opt_seed}", pattern: "best-results.fasta", mode: 'copy'
    publishDir "${resultsDir}/latent-space-sgd-optimisation/seed-${opt_seed}/restarts", pattern: "optimisation-results-*.fasta", mode: 'copy'

    input:
        path(pickle_file)
        path(pytorch_file)
        each opt_seed


    output:
        path("logger-${opt_seed}.log"), emit: optimisation_logger
        path("optimisation-results-*.fasta"), emit: intermediate_results
        path("best-results.fasta"), emit: best_results

    script:
    """
    downstream.py transformer sgd-optimization ${pytorch_file}\
                                               ${pickle_file}\
                                                logger-${opt_seed}.log\
                                                optimisation-results\
                                                best-results.fasta\
                                                --seed ${opt_seed}\
                                                --learning-rate ${params.optimisation.learning_rate}\
                                                --n-restarts ${params.optimisation.n_restarts}\
                                                --delta-f-tol ${params.optimisation.delta_f_tol}\
                                                --max-opt-steps ${params.optimisation.max_opt_steps}
    """
}

// Trust Region optimisation, constraints to lie within Gaussian ball N(0,I)
process TrustRegionOptimisationPrior{
    errorStrategy 'ignore'
    
    tag "trust-region-optimisation-in-latent-space;seed-${opt_seed}"

     // publish results
    publishDir "${resultsDir}/latent-space-trust-region-optimisation/prior/seed-${opt_seed}", pattern: "*.{log,fasta,csv,pickle}", mode: 'copy'
    publishDir "${resultsDir}/latent-space-trust-region-optimisation/prior/seed-${opt_seed}", pattern: "duplicated.detail.txt", mode: 'copy'

    input:
        path(pickle_file)
        path(pytorch_file)
        path(fasta_files_clean_against)
        each opt_seed

    output:
        path("logger-${opt_seed}.log"), emit: optimisation_logger // logger file
        path("all-samples.fasta"), emit: fasta_output_raw // generated output from trust-region-optimization
        path("samples.fasta"), emit: fasta_output_clean // unduplicated output from trust-region-optimization
        path("all-samples.pickle"), emit: pickle_output_raw // generated output from trust-region-optimization
        path("duplicated.detail.txt"), emit: duplicates_info // txt file with information on duplicated sequences
        path("*.csv"), emit: all_csv // output with reestimated energies [MAIN OUTPUT]


    // 1: downstream.py transformer trust-region-optimization: runs trust region optimisation and then samples 2000 (categorical) + 1 (argmax) samples using optimal z, that corresponds to lowest energy
    
    // 2: remove sequences found in ${fasta_files_clean_against}
    
    // WE SKIP FILTERING STEP HERE AND RUN ENERGY REESTIMATION DIRECTLY
    // 3: downstream.py transformer seq2energy: runs energy reestimation (again, using 500 samples)

    script:
    """
    downstream.py transformer trust-region-optimization ${pytorch_file}\
                                                        ${pickle_file}\
                                                        logger-${opt_seed}.log\
                                                        all-samples.fasta\
                                                        all-samples.pickle\
                                                        --max-number-of-steps ${sampling_max_length}\
                                                        --n-samples 2000\
                                                        --seed ${opt_seed}

    cat ${fasta_files_clean_against} all-samples.fasta > joint.fasta

    cat joint.fasta | seqkit rmdup -s -o unduplicated-joint.fasta -d duplicated.fasta -D duplicated.detail.txt

    [ -f duplicated.detail.txt ] && echo "duplicated.detail.txt file exist." || touch duplicated.detail.txt

    cut -f2- duplicated.detail.txt | tr ',' '\n' | tr -d "[:blank:]" > output.duplicated.detail.txt

    miscellaneous.py various remove-sequences output.duplicated.detail.txt all-samples.fasta samples.fasta

    downstream.py transformer seq2energy samples-estimated-average-values.csv\
                                         samples-individual\
                                         ${pytorch_file}\
                                         ${pickle_file}\
                                         samples.fasta\
                                         --n-samples 300\
                                         --batch-size ${training_batch_size}\
                                         --seed ${opt_seed}
    """

}

// Trust Region optimisation, constraints to lie within Gaussian ball N(mu(seed),diag(seed))
process TrustRegionOptimisationSeeded{
    
    errorStrategy 'ignore'
    
    tag "trust-region-optimisation-in-latent-space;seed-sequence:${seed_fasta_file.baseName};seed-${opt_seed}"

     // publish results
    publishDir "${resultsDir}/latent-space-trust-region-optimisation/seeded/${seed_fasta_file.baseName}/seed-${opt_seed}", pattern: "*.{log,fasta,csv,pickle}", mode: 'copy'
    publishDir "${resultsDir}/latent-space-trust-region-optimisation/seeded/seed-${opt_seed}", pattern: "duplicated.detail.txt", mode: 'copy'

    input:
        path(pickle_file)
        path(pytorch_file)
        path(fasta_files_clean_against)
        each seed_fasta_file // FASTA file with a seed sequence
        each opt_seed

    output:
        path("logger-${opt_seed}.log"), emit: optimisation_logger // logger file
        path("all-samples.fasta"), emit: fasta_output_raw // generated output from trust-region-optimization
        path("samples.fasta"), emit: fasta_output_clean // unduplicated output from trust-region-optimization
        path("all-samples.pickle"), emit: pickle_output_raw // generated output from trust-region-optimization
        path("duplicated.detail.txt"), emit: duplicates_info // txt file with information on duplicated sequences
        path("*.csv"), emit: all_csv // output with reestimated energies [MAIN OUTPUT]


    // 1: downstream.py transformer trust-region-optimization: runs trust region optimisation and then samples 500 (categorical) + 1 (argmax) samples using optimal z, that corresponds to lowest energy
    
    // 2: remove sequences found in ${fasta_files_clean_against}
    
    
    // WE SKIP FILTERING STEP HERE AND RUN ENERGY REESTIMATION DIRECTLY
    // 3: downstream.py transformer seq2energy: runs energy reestimation (again, using 500 samples)

    script:
    """
    downstream.py transformer trust-region-optimization ${pytorch_file}\
                                                        ${pickle_file}\
                                                        logger-${opt_seed}.log\
                                                        all-samples.fasta\
                                                        all-samples.pickle\
                                                        --seed-fasta-file ${seed_fasta_file}\
                                                        --max-number-of-steps ${sampling_max_length}\
                                                        --n-samples 2000\
                                                        --seed ${opt_seed}

    cat ${fasta_files_clean_against} all-samples.fasta > joint.fasta

    cat joint.fasta | seqkit rmdup -s -o unduplicated-joint.fasta -d duplicated.fasta -D duplicated.detail.txt

    [ -f duplicated.detail.txt ] && echo "duplicated.detail.txt file exist." || touch duplicated.detail.txt

    cut -f2- duplicated.detail.txt | tr ',' '\n' | tr -d "[:blank:]" > output.duplicated.detail.txt

    miscellaneous.py various remove-sequences output.duplicated.detail.txt all-samples.fasta samples.fasta

    downstream.py transformer seq2energy samples-estimated-average-values.csv\
                                         samples-individual\
                                         ${pytorch_file}\
                                         ${pickle_file}\
                                         samples.fasta\
                                         --n-samples 300\
                                         --batch-size ${training_batch_size}\
                                         --seed ${opt_seed}
    """

}


// workflow that serves as entrypoint
// nextflow run main.nf -profile test -entry OPTIMISATION
workflow OPTIMISATION{
    // input channels
    pickle_file         = Channel.fromPath( params.optimisation.pickle_file )
    pytorch_file        = Channel.fromPath( params.optimisation.pytorch_file )
    optimisation_seeds  = Channel.fromList( params.optimisation.seeds )

    ch_seeds_fasta_file = Channel.fromPath( params.optimisation.seeds_file )
    ch_individual_fasta_files = ch_seeds_fasta_file
                                                    .splitFasta( record: [id: true, text: true ])                                        // obtain a map with keys: id and text, basically each map is a single entry of original FASTA file
                                                    .collectFile() { item -> [ "${item.id.replaceAll("[._|]","-")}.fasta", item.text ] } // save single entry in respective FASTA files (using ID as filename)

    // load databases, aka FASTA files with sequences that you want to exclude (if they are found!) from the samples set
    ch_databases        = Channel.fromPath( params.optimisation.databases )

    // merge databases
    out_merge_fasta         = MergeFasta(ch_databases.collect())
    ch_fasta_seq_to_exclude = out_merge_fasta.merged_file

    // Trust Region Optimisation for prior
    TrustRegionOptimisationPrior(
                                    pickle_file,
                                    pytorch_file,
                                    ch_fasta_seq_to_exclude,
                                    optimisation_seeds
                                )

    // Trust Region Optimisation for posterior
    TrustRegionOptimisationSeeded(
                                    pickle_file,
                                    pytorch_file,
                                    ch_fasta_seq_to_exclude,
                                    ch_individual_fasta_files,
                                    optimisation_seeds
                                 )

    
}

// workflow for TransformerVAE model: training + sampling
workflow MAIN{
    
    /////////////////////////////////////////////// INFORMATION ABOUT PROCESS ////////////////
    Info()
    ////////////////////////////////////////////// PREPROCESSING ////////////////////////////////////////
    // fasta files: both are lists
    ch_fasta_files_to_join      = Channel.fromPath( params.preprocessing.training_list )
    ch_seed_fasta_files_to_join = Channel.fromPath( params.preprocessing.seed_list )

    // run preprocessing step (STEP I)
    out_preprocessing = Preprocessing(
                                        ch_fasta_files_to_join,
                                        ch_seed_fasta_files_to_join
                                     )

    ////////////////////////////////////////////////////// MODEL TRAINING //////////////////////////////
    // TODO: add checkpoints and sequence weights (need to change python code)
     ch_checkpointfile = Channel.fromPath( training_entry_checkpoint )

    // STEP II: model training
    out_train_model_process = TrainModel(
                                            out_preprocessing.train_file,
                                            out_preprocessing.val_file,
                                            ch_checkpointfile
                                        )

    ////////////////////////////////////////////////////// PRIOR ANALYSIS (SAMPLING + PROTEIN ANALYSIS) //////////
    // get necessary outputs from training
    ch_pickle_file             = out_train_model_process.pickle_file
    ch_best_train_pytorch      = out_train_model_process.best_train_pytorch
    ch_best_validation_pytorch = out_train_model_process.best_validation_pytorch
    ch_seeds_fasta_file        = out_preprocessing.seed_file
    
    // generate identifiers manually
    ch_train_identifier        = Channel.of( "best-train-error" )
    ch_validation_identifier   = Channel.of( "best-validation-error" )

    
    // create input for prior sampling (part of Prior workflow)
    ch_input_train_prior_sampling = ch_pickle_file.combine(
                                                                ch_best_train_pytorch.combine(
                                                                                                ch_train_identifier
                                                                                            )
                                                            )
                                                    
                                                    
    
    ch_input_validation_prior_sampling = ch_pickle_file.combine(
                                                                    ch_best_validation_pytorch.combine(
                                                                                                            ch_validation_identifier
                                                                                                        )
                                                                )
    ch_input_prior_sampling = ch_input_train_prior_sampling
                                                            .mix(
                                                                    ch_input_validation_prior_sampling
                                                                )
    // databases: [train-set, validation-set]
    // also will serve as list of FASTA files, which sequences should be removed from any samples!
    ch_databases = out_preprocessing.train_file
                                                .mix(out_preprocessing.val_file)
                                                .collect()

    // pass through MergeFasta to get merged file
    out_merge_fasta         = MergeFasta(ch_databases)
    ch_fasta_seq_to_exclude = out_merge_fasta.merged_file

    // STEP III: prior sampling and analysis
    Prior(
            ch_input_prior_sampling,
            ch_databases,
            ch_fasta_seq_to_exclude
         )

    ///////////////////////////////////////////////////////////// SEEDED ANALYSIS ////////////////////////////////////
    // skip or not argmax sampling of first element
    if (sampling_argmax_first_element == true)
        ch_first_element           = Channel.of( "--skip-first-element" )
    else
        ch_first_element           = Channel.of( "--no-skip-first-element" )
    
    // get multiple fasta files from one
    // fasta file with sequence ID
    ch_individual_fasta_files = ch_seeds_fasta_file
                                                    .splitFasta( record: [id: true, text: true ])                                        // obtain a map with keys: id and text, basically each map is a single entry of original FASTA file
                                                    .collectFile() { item -> [ "${item.id.replaceAll("[._|]","-")}.fasta", item.text ] } // save single entry in respective FASTA files (using ID as filename)

    // create input for seeded sampling
    ch_input_train_seeded_sampling = ch_pickle_file.combine(
                                                                ch_best_train_pytorch.combine(
                                                                                                ch_train_identifier.combine(
                                                                                                                                ch_individual_fasta_files.combine(
                                                                                                                                                                    ch_first_element
                                                                                                                                                                )
                                                                                                                            )
                                                                                            )
                                                            )
    ch_input_validation_seeded_sampling = ch_pickle_file.combine(
                                                                    ch_best_validation_pytorch.combine(
                                                                                                            ch_validation_identifier.combine(
                                                                                                                                                ch_individual_fasta_files.combine(
                                                                                                                                                                                    ch_first_element
                                                                                                                                                                                )
                                                                                                                                            )
                                                                                                        )
                                                                    )
    ch_input_seeded_sampling = ch_input_train_seeded_sampling
                                                            .mix(
                                                                    ch_input_validation_seeded_sampling
                                                                )                  
    
    // temperature input
    ch_temperatures = Channel.fromList( sampling_temperature )

    //call Posterior workflow
    Posterior(
                ch_input_seeded_sampling,
                ch_temperatures,
                ch_fasta_seq_to_exclude
             )
}

// workflow for TransformerVAE model: sampling
// nextflow run main.nf -profile test -entry VAE_SAMPLE
workflow VAE_SAMPLE{
    // create inputs
    ch_pickle_file          = Channel.fromPath( params.sampling.pickle_file )
    ch_best_model_pytorch   = Channel.fromPath( params.sampling.pytorch_file )
    ch_seeds_fasta_file     = Channel.fromPath( params.sampling.seeds_file )
    ch_model_identifier     = Channel.of( "user-provided-model" )
    ch_databases            = Channel.fromPath( params.sampling.databases )
    ch_temperatures         = Channel.fromList( sampling_temperature )

    if (sampling_argmax_first_element == true)
        ch_first_element           = Channel.of( "--skip-first-element" )
    else
        ch_first_element           = Channel.of( "--no-skip-first-element" )

    
    // Prior workflow
    ch_input_prior_sampling = ch_pickle_file.combine(
                                                        ch_best_model_pytorch.combine(
                                                                                        ch_model_identifier
                                                                                     )
                                                    )

    // pass through MergeFasta to get merged file
    out_merge_fasta         = MergeFasta(ch_databases.collect())
    ch_fasta_seq_to_exclude = out_merge_fasta.merged_file

    Prior(
            ch_input_prior_sampling,
            ch_databases.collect(),
            ch_fasta_seq_to_exclude
         )

    
    // Posterior workflow
    
    // get multiple fasta files from one
    // fasta file with sequence ID
    ch_individual_fasta_files = ch_seeds_fasta_file
                                                    .splitFasta( record: [id: true, text: true ])                                        // obtain a map with keys: id and text, basically each map is a single entry of original FASTA file
                                                    .collectFile() { item -> [ "${item.id.replaceAll("[._|]","-")}.fasta", item.text ] } // save single entry in respective FASTA files (using ID as filename)

    // single input here
    ch_input_seeded_sampling = ch_pickle_file.combine(
                                                        ch_best_model_pytorch.combine(
                                                                                        ch_model_identifier.combine(
                                                                                                                        ch_individual_fasta_files.combine(
                                                                                                                                                            ch_first_element
                                                                                                                                                        )
                                                                                                                    )
                                                                                      )
                                                        )

    //call Posterior workflow
    Posterior(
                ch_input_seeded_sampling,
                ch_temperatures,
                ch_fasta_seq_to_exclude
             )

    
}


// workflow for supervised learning: sequence -> energy (using GRU)
// nextflow run main.nf -profile test -entry GRU_SUPERVISED_TRAIN
workflow GRU_SUPERVISED_TRAIN{
    /////////////////////////////////////// PREPROCESSING (same as for TransformerVAE) ///////////////////////////////////
    // fasta files: both are lists
    ch_fasta_files_to_join      = Channel.fromPath( params.preprocessing.training_list )
    ch_seed_fasta_files_to_join = Channel.fromPath( params.preprocessing.seed_list )

    // run preprocessing step (STEP I)
    out_preprocessing = Preprocessing(
                                        ch_fasta_files_to_join,
                                        ch_seed_fasta_files_to_join
                                     )
    /////////////////////////////////////// SUPERVISED MODEL TRAINING /////////////////////////////////////////////////////
    ch_checkpointfile = Channel.fromPath( params.training_gru.entry_checkpoint )
    out_model_training = TrainModelPredictiveGRU(
                                                    out_preprocessing.train_file,
                                                    out_preprocessing.val_file,
                                                    ch_checkpointfile
                                                )
}

// workflow for supervised learning: sequence -> energy (using TF encoder)
// nextflow run main.nf -profile test -entry TF_SUPERVISED_TRAIN
workflow TF_SUPERVISED_TRAIN{
    /////////////////////////////////////// PREPROCESSING (same as for TransformerVAE) ///////////////////////////////////
    // fasta files: both are lists
    ch_fasta_files_to_join      = Channel.fromPath( params.preprocessing.training_list )
    ch_seed_fasta_files_to_join = Channel.fromPath( params.preprocessing.seed_list )

    // run preprocessing step (STEP I)
    out_preprocessing = Preprocessing(
                                        ch_fasta_files_to_join,
                                        ch_seed_fasta_files_to_join
                                     )
    /////////////////////////////////////// SUPERVISED MODEL TRAINING /////////////////////////////////////////////////////
    ch_checkpointfile = Channel.fromPath( params.training_tf.entry_checkpoint )
    
    if (params.training_tf.weighted_sampling == true)
        ch_training_weighted_sampling = Channel.value("--weighted-sampling")
    else
        ch_training_weighted_sampling =  Channel.value("--no-weighted-sampling")


    out_model_training = TrainModelPredictiveTFEncoder(
                                                        out_preprocessing.train_file,
                                                        out_preprocessing.val_file,
                                                        ch_checkpointfile,
                                                        ch_training_weighted_sampling
                                                      )
}


// workflow for predicting energy from sequence (using TF encoder)
// nextflow run main.nf -profile test -entry TF_SUPERVISED_PREDICT
workflow TF_SUPERVISED_PREDICT{
    // calling process PredictEnergies, inputs: [modelID, pickle file, pytorch file, FASTA with clean sequences]

    // some placeholders, they dont really mean anything
    modelID      = Channel.of("my-model")
    pickle_file  = Channel.fromPath( params.predicting_tf.pickle_file )
    pytorch_file = Channel.fromPath( params.predicting_tf.pytorch_file )
    fasta_file   = Channel.fromPath( params.predicting_tf.fasta_file )

    input = modelID.combine( 
                                pickle_file.combine(
                                                        pytorch_file.combine(
                                                                                fasta_file
                                                                            )
                                                    )
                            )

    PredictEnergies(input)
}

// workflow for TransformerVAE model: latent space parameters
// nextflow run main.nf -profile test -entry LATENTVARIABILITY
workflow LATENTVARIABILITY{
    // create channels
    ch_pickle_file      = Channel.fromPath( params.latent.pickle_file )
    ch_pytorch_file     = Channel.fromPath( params.latent.pytorch_file )
    ch_model_identifier = Channel.of( "user-provided-model" )
    ch_fasta_files      = Channel.fromPath( params.latent.fasta_files )

    // cross-product them
    ch_input = ch_pickle_file.combine(
        ch_pytorch_file.combine(
            ch_model_identifier.combine(
                ch_fasta_files
            )
        )
    )

    // call EstimateLatentSpaceVariability
    EstimateLatentSpace(ch_input)
}

// main entry point
// call as follows: nextflow run main.nf -profile test
workflow{
    MAIN()
}