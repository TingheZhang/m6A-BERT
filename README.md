# m6A-BERT-Deg
This repository contains the implementation of our paper ‘Understanding YTHDF2-mediated mRNA Degradation By m6A-BERT-Deg’.

In this repository, we provides resources including: source codes of the m6A-BERT-Deg, usage examples, pre-trained models, fine-tuned models and visulization tool. We are still actively developing the repo, so please kindly report any issues that you encounter.

If you have used m6A-BERT in your research, please kindly cite the our publications.

## 1. Environment setup 
[Conda](https://docs.anaconda.com/anaconda/install/linux/) is recommended for setting up the environment. You can easily install the necessary dependencies by using the following commands:
1. Create the Conda environment from the provided YAML file:
  > conda env create -f m6abert.yml 

    You can customize the environment name by changing the first line of the `m6abert.yml` file. For more details on managing Conda environments, refer to the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).


2. The following packages are necessary for our project: `pytorch=1.10.2`, `captum=0.5.0`, `python=3.6`.

3. After setting up the environment, activate it using the following command:

    ```bash
    conda activate m6abert
    ```
    
   Ensure that you activate the environment before running any scripts or commands related to this project.
   
## 2. Data Processing

Before fine-tuning and prediction, the input sequences should be processed into K-mers. Follow the steps below to preprocess your data:

1. Set the desired K-mer length (K) by exporting the variable:

    ```bash
    export KMER=3 ### Select K from 3 to 6
    ```

2. Specify the paths for raw data and the location to save processed data:

    ```bash
    export RAW_DATA_PATH=YOUR_RAW_DATA_PATH
    export DATA_PATH=THE_PATH_YOU_WANT_TO_SAVE_PROCESS_DATA
    ```

3. Set a seed number for data balancing:

    ```bash
    export SEED=452 ### Choose your desired seed number
    ```

4. Run the following command in a Linux bash environment to get proper input:

    ```bash
    python3 get_input.py --do_val --kmer $KMER --extend_len 250 --task finetune --data_dir $RAW_DATA_PATH --save_dir $DATA_PATH --seed $SEED
    ```

    Ensure that under your `RAW_DATA_PATH`, positive sequences are in `pos.fa`, and negative sequences are in `neg.fa`.

Our input data used in the paper is available in the `data/` directory.

## 3. Fine-tune the model 
m6A-BERT can be easily fine-tuned for downstream analysis. The code can be simply run in Linux bash :
```
export KMER=3 ### select the K from 3 to 6, have to match the K in the data process section 
export RAW_DATA_PATH= YOUR_RAW_DATA_PATH
export DATA_PATH=THE_PATH_YOU_SAVED_PROCESSED_DATA
export MODEL_PATH=THE_PATH_OF_PRETRAINED_MODEL
export OUTPUT_PATH=THE_PATH_TO_SAVE_FINETUNED_MODEL

python3 run_finetune_degradation.py --model_type dna --tokenizer_name=dna$KMER --model_name_or_path $MODEL_PATH \
 --task_name dnaprom --do_train --do_eval --data_dir $DATA_PATH --save_steps 50 --logging_steps 50 \
 --max_seq_length 512 --per_gpu_eval_batch_size=40 --per_gpu_train_batch_size=40 --learning_rate 1e-6 --num_train_epochs 100 \
 --output_dir $OUTPUT_PATH --n_process 10 --hidden_dropout_prob 0.1 --evaluate_during_training --weight_decay 0.01
```
Please adjust the per_gpu_eval_batch_size and per_gpu_train_batch_size based on your GPU memory size.

Our pre-trained m6A-BERT  can be downloaded from [here](https://drive.google.com/drive/folders/1K66vzqkc68hmCZto5Xj-c6ui7ROvj6B3?usp=sharing)
Our fine-tuned m6A-BERT-DEG  can be downloaded from [here](https://drive.google.com/drive/folders/1EKb2KiDRMnHCSlcGFCplpiQ__4G102EL?usp=sharing)


## 4. Make the prediction 
After fine-tuned and obtained m6A-BERT-DEG model, people can make the prediction by using the commends below: 

```
export KMER=3 ### select the K from 3 to 6, have to match the K in the data process section 
export RAW_DATA_PATH= YOUR_RAW_DATA_PATH
export DATA_PATH=THE_PATH_YOU_SAVED_PROCESSED_DATA
export MODEL_PATH=THE_PATH_OF_PRETRAINED_MODEL
export OUTPUT_PATH=THE_PATH_TO_SAVE_FINETUNED_MODEL
export FINUE_TUNED_MODEL_PATH=$OUTPUT_PATH/checkpoint-# ##change # to user selected numbers 

python3 run_predict_degradation.py --model_type dna --tokenizer_name=dna$KMER --model_name_or_path $MODEL_PATH \
 --task_name dnaprom --data_dir $DATA_PATH --save_steps 50 --logging_steps 50 --do_predict \
 --max_seq_length 512 --per_gpu_eval_batch_size=50 --per_gpu_train_batch_size=50 --learning_rate 1e-6 --num_train_epochs 100 \
 --output_dir $FINUE_TUNED_MODEL_PATH --n_process 1  --evaluate_during_training --predict_dir $OUTPUT_PATH
```

## 5. Visulzation 
### 5.1 Calculate attention/attribution scores
After fine-tuned and obtained m6A-BERT-DEG model, people can make the prediction by using the commends below: 

```
export KMER=3 ### select the K from 3 to 6, have to match the K in the data process section 
export RAW_DATA_PATH= YOUR_RAW_DATA_PATH
export DATA_PATH=THE_PATH_YOU_SAVED_PROCESSED_DATA
export FINUE_TUNED_MODEL_PATH=THE_PATH_OF_FINUE_TUNED_MODEL
export MODEL_PATH=THE_PATH_OF_PRETRAINED_MODEL
export OUTPUT_PATH=THE_PATH_TO_SAVE_OUTPUT
export MOTIF_PATH=THE_PATH_TO_SAVE_MOTIF
export DATASET=dev ##select the dataset to visulize, could be train, dev, test
export TASK=attr  ##select the visulzation methods, could be attn (attention weights) or attr (attribution scores)

python3 visualize_all.py --kmer $KMER --model_name_or_path $MODEL_PATH --model_path $FINUE_TUNED_MODEL_PATH --output_dir $OUTPUT_PATH \
--data_dir $DATA_PATH --data_name $DATASET --vis_task $TASK --batch_size 50
```
User can select the dataset to visulze by changing the value of DATASET. 
train means to visulize training set.dev means to visulize validation set. test means to visulize test set.  

### 5.2 Find the significant motifs enriched in positive set or negitives set

```
## commend to find the enriched motif from positive set
python3 visualize_find_motif_pos.py --data_dir $DATA_PATH --npy_dir $OUTPUT_PATH --window_size 24 --min_len 5 \
--pval_cutoff 0.05 --min_n_motif 3 --align_all_ties --save_file_dir $MOTIF_PATH --verbose --data_name $DATASET --do_plot --vis_task $TASK
## commend to find the enriched motif from negivtive set
python3 visualize_find_motif_neg.py --data_dir $DATA_PATH --npy_dir $OUTPUT_PATH --window_size 24 --min_len 5 \
--pval_cutoff 0.05 --min_n_motif 3 --align_all_ties --save_file_dir $MOTIF_PATH --verbose --data_name $DATASET --do_plot --vis_task $TASK
```

if users want to find the motif based on all set (inlcude training set , test set, validation set), they have to calculate attention/attribution scores for all three set one by one
Then they can motif based on all set by changing the value of DATASET :
> export DATASET=all
