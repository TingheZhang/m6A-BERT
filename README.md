# m6A-BERT
This repository includes the implementation of 'Understanding YTHDF2-mediated mRNA Degradation By m6A-BERT-Deg'. 

Please cite our paper if you use the models or codes. The repo is still actively under development, so please kindly report if there is any issue encountered.


## 1. Environment setup 
[Conda](https://docs.anaconda.com/anaconda/install/linux/) is recommanded to set up the enviroment. 
you can simply install the necessary dependacy by using command 

  > conda env create -f m6abert.yml 

you can set your enviroment names by change the first line of the m6abert.yml . Details can be found at [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

The following package is necessary for our project: pytorch=1.10.2, captum=0.5.0,python=3.6
After enviroment setup, use following command to activate this enviroments:
 > conda activate m6abert
## 2.Data process
The sequence should be processed into the K-mer before the fine-tuning and prediction.
You can run the following lines in Linux bash：
```
export KMER=3 ### select the K from 3 to 6
export RAW_DATA_PATH= YOUR_RAW_DATA_PATH
export DATA_PATH=THE_PATH_YOU_WANT_TO_SAVE_PROCESS_DATA
export seed=452 ## select the seed number your like, this seed is used for data balance 
python3 get_input.py --do_val --kmer $KMER --extend_len 250 --task finetune --data_dir $RAW_DATA_PATH --save_dir $DATA_PATH --seed $seed
```
under user's RAW_DATA_PATH, positive sequence should be put into pos.fa; negitive sequence should be put into neg.fa
The data we used in the paper can be found at data/

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
Our input file can be downloaded from [here]()
Our pre-trained m6A-BERT  can be downloaded from [here]()
Our fine-tuned m6A-BERT-DEG  can be downloaded from [here]()


## 4. Make the prediction 
After fine-tuned and obtained m6A-BERT-DEG model, people can make the prediction by using the commends below: 

```
export KMER=3 ### select the K from 3 to 6, have to match the K in the data process section 
export RAW_DATA_PATH= YOUR_RAW_DATA_PATH
export DATA_PATH=THE_PATH_YOU_SAVED_PROCESSED_DATA
export MODEL_PATH=THE_PATH_OF_PRETRAINED_MODEL
export OUTPUT_PATH=THE_PATH_TO_SAVE_FINETUNED_MODEL
export CP=$OUTPUT_PATH/checkpoint-# ##change # to user selected numbers 

python3 run_predict_degradation.py --model_type dna --tokenizer_name=dna$KMER --model_name_or_path $MODEL_PATH \
 --task_name dnaprom --data_dir $DATA_PATH --save_steps 50 --logging_steps 50 --do_predict \
 --max_seq_length 512 --per_gpu_eval_batch_size=50 --per_gpu_train_batch_size=50 --learning_rate 1e-6 --num_train_epochs 100 \
 --output_dir $CP --n_process 1  --evaluate_during_training --predict_dir $OUTPUT_PATH
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
export CP=$OUTPUT_PATH/checkpoint-# ##change # to user selected numbers
export DATASET=all ##select the dataset to visulize, could be all, train, dev, test
export TASK= attr  ##select the visulzation methods, could be attn (attention weights) or attr (attribution scores)

python3 visualize_all.py --kmer $KMER --model_name_or_path $MODEL_PATH --model_path $FINUE_TUNED_MODEL_PATH --output_dir $OUTPUT_PATH \
--data_dir $DATA_PATH --data_name $DATASET --vis_task $TASK --batch_size 50
```
User can select the dataset to visulze by changing the value of DATASET. 
train means to visulize training set.dev means to visulize validation set. test means to visulize test set. all means visulize all dataset, inlcude training set , test set, validation set. 

### 5.2 Calculate attention/attribution scores

```
python3 visualize_find_motif.py --data_dir $DATA_PATH --npy_dir $OUTPUT_PATH --window_size 24 --min_len 5 \
--pval_cutoff 0.05 --min_n_motif 3 --align_all_ties --save_file_dir $MOTIF_PATH --verbose --data_name $DATASET --do_plot --vis_task $TASK
```
