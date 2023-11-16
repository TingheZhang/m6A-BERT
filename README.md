# m6ABERT
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
> export KMER=3 ### select the K from 3 to 6
> export RAW_DATA_PATH= YOUR_RAW_DATA_PATH
> export DATA_PATH=THE_PATH_YOU_WANT_TO_SAVE_PROCESS-DATA
> export seed=52 ## select the seed number your like, this seed is used for data balance 
> python3 get_input.py --do_val --kmer $KMER --extend_len 250 --task finetune --data_dir $RAW_DATA_PATH --save_dir $DATA_PATH --seed $seed

## 3. Fine-tune the model 
G-TEM_pytorch_3l_34.py is the model that has the best performance for our cancer prediction task. The code can sinply run by :
> python G-TEM_pytorch_3l_34.py  --head_num 5 --learning_rate 0.0001 --act_fun leakyrelu --batch_size 16 --epoch 1 --do_val --dropout_rate 0.3 --result_dir model_res/3l/ --model_dir model/3l/

The code attached used 3 attention layers. If you want to increase or decrease the number of layers, you can change the structure at line 252~ line 255 and line 271~ line 286. 

Our input file can be downloaded from [here](https://drive.google.com/file/d/13-Xjqexsi8-ZkZm17vcH6oGIivL2O8XW/view?usp=sharing)

PBMC data and label can be found from [here](https://drive.google.com/file/d/158PAzib3Nq17UMtLMIJwndlX7hqcLaZT/view?usp=sharing), [here](https://drive.google.com/file/d/1gNLyp7b720MFnvQtVLDaXHScIhnPdpR9/view?usp=sharing)

## 3. Compute the attribution score and give the gene importance rank 
To evaluate which gene is more important to predicte specific cancer, we use integer gradient(IG) to compute the attribution score for each test samples. The larger score means more importance. 
To compute the attribution score, it can be easily got by:
> python G-TEM_t_attr_allcancer.py --head_num 5 --learning_rate 0.0001 --act_fun leakyrelu --batch_size 16 --epoch 1 --do_val --dropout_rate 0.3 --result_dir attr_34cancer/ --attr_method ig --model_location model/3l/pytorch_transformer_head_5_lr_0.0001_leakyrelu_epoch0.model --abs 

The outpult files: validation result/ test result ; ablation study result; Gene importance rank for each cancer prediction;  


Notice, the model structure in G-TEM_t_attr_allcancer.py has to match the structure at G-TEM_pytorch_3l_34.py. 

This code can compute the mean and medain of attribuiton score for the all val/test samples. The attribution score can be used to rank the input importance or do enrichment. We have supportted 6 methods to compute the attribution scores: ig ([Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients)), sg_ig (Integrated Gradients with gaussian noise, [smoothgrad_sq](https://captum.ai/api/noise_tunnel.html)),vg_ig(Integrated Gradients with gaussian noise,vargrad) , gb ([Guided Backprop](https://captum.ai/api/guided_backprop.html)), vg_gb (Guided Backprop with with gaussian noise,vargrad),sg_gb (Guided Backprop with with gaussian noise,smoothgrad_sq). You can change the parameter --attr_method to apply them. 

If you want to compute the attribution for validation data, You can add parameter --do_val. Otherwise it will compute the attribution score for test set.

The input gene importance can be also analyzed by aliblition study. We removed the gene with top attibution score batch by batch and test their affection to accruacy.
Significant accrucay dropped means these genes are import to predict this type of cancers. 


## 4. Compute and visulize the attention weights and entropy for each attention layers
To discover the inner relationship at each layer, we can use attention weights and entropy of attention wieghts.  
G-TEM_t_vis.py has two mode 'attn' and 'vis'. 'attn' is used for get the attention weights for each layer.
'vis' for visualizing. The output will be the average entropy for each heah at each layers and related boxplot. 

> python G-TEM_t_vis.py --head_num 5 --learning_rate 0.0001 --act_fun leakyrelu --batch_size 16 --epoch 1 --dropout_rate 0.3 --result_dir model_res_vis_all/  --model_location model/3l/pytorch_transformer_head_5_lr_0.0001_leakyrelu_epoch0.model --task attn

Notice, the model structure in G-TEM_t_vis.py has to match the structure at G-TEM_pytorch_3l_34.py. 

## 5. Compute the attributation of attention weights for each layer
To obtain the hub genes, first we should compute the attribution of attention weights for each head at each layers 
> python G-TEM_t_attr_3l_head.py --head_num 5 --cancer_type 0 --act_fun leakyrelu --result_dir model_res_vis_all/  --model_location model/3l/pytorch_transformer_head_5_lr_0.0001_leakyrelu_epoch0.model

After obtaining these attribution scores, we can generate the coresponsding net
>python3 get_net.py --head_num 5 --cancer_type 0 --result_dir model_res_vis_all/ --net_dir gene_net/ --threshold 0.001


The output are txt files which contain the Query gene and Key gene pairs. To create the net, these files should be feed into [Cytoscape](https://cytoscape.org/). 
