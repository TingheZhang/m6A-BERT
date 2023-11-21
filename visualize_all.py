import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
# from process_pretrain_data import get_kmer_sentence
import os
from copy import deepcopy
from multiprocessing import Pool
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import sys
sys.path.append('src/')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, DNATokenizer,BertForSequenceClassification
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_processors as processors
from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig
from transformers import glue_output_modes as output_modes

# from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerActivation
# from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import logging
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOKEN_ID_GROUP = ["bert", "dnalong", "dnalongcat", "xlnet", "albert"]
logger = logging.getLogger(__name__)


def load_cache_examples(args, task, tokenizer, data_name,evaluate=False):
    ###data_name should be train or test or dev
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(data_name,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        # examples = (
        #     processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        # )
        if args.data_name=='train':
            examples=(processor.get_train_examples(args.data_dir))
        if args.data_name == 'test':
            examples=(processor.get_test_examples(args.data_dir))
        if args.data_name == 'dev':
            examples=(processor.get_dev_examples(args.data_dir))

        # examples = (
        #     processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        # )

        print("finish loading examples")

        # params for convert_examples_to_features
        max_length = args.max_seq_length
        pad_on_left = bool(args.model_type in ["xlnet"])
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0

        if args.n_process == 1:
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=max_length,
                output_mode=output_mode,
                pad_on_left=pad_on_left,  # pad on the left for xlnet
                pad_token=pad_token,
                pad_token_segment_id=pad_token_segment_id, )

        else:
            n_proc = int(args.n_process)
            print("number of processes for converting feature: " + str(n_proc))
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(examples) / n_proc)
            for i in range(1, n_proc + 1):
                if i != n_proc:
                    indexes.append(len_slice * (i))
                else:
                    indexes.append(len(examples))

            results = []

            for i in range(n_proc):
                results.append(p.apply_async(convert_examples_to_features, args=(
                examples[indexes[i]:indexes[i + 1]], tokenizer, max_length, None, label_list, output_mode, pad_on_left,
                pad_token, pad_token_segment_id, True,)))
                print(str(i + 1) + ' processor started !')

            p.close()
            p.join()

            features = []
            for result in results:
                features.extend(result.get())

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def visualize_attn(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.output_dir,)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):

        # evaluate = False if args.visualize_train else True
        pred_dataset = load_cache_examples(args, pred_task, tokenizer, data_name=args.data_name)

        # if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        #     os.makedirs(pred_output_dir)

        # args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.batch_size)
        #
        # # multi-gpu eval
        # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        #     model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset), 2])
        else:
            preds = np.zeros([len(pred_dataset), 3])
        attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])

        for index, batch in enumerate(tqdm(pred_dataloader, desc="Getting attn")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                if args.layer==None:
                    attention = outputs[-1][-1] ### only visulize the last layer attn
                else:
                    attention = outputs[-1][args.layer]
                _, logits = outputs[:2]

                preds[index * batch_size:index * batch_size + len(batch[0]), :] = logits.detach().cpu().numpy()
                attention_scores[index * batch_size:index * batch_size + len(batch[0]), :, :,
                :] = attention.cpu().numpy()

        if args.task_name != "dnasplice":
            probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
        else:
            probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()

        scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

        for index, attention_score in enumerate(attention_scores):
            attn_score = []
            for i in range(1, attention_score.shape[-1] - kmer + 2):
                attn_score.append(float(attention_score[:, 0, i].sum()))

            for i in range(len(attn_score) - 1):
                if attn_score[i + 1] == 0:
                    attn_score[i] = 0
                    break

            # attn_score[0] = 0
            counts = np.zeros([len(attn_score) + kmer - 1])
            real_scores = np.zeros([len(attn_score) + kmer - 1])
            for i, score in enumerate(attn_score):
                for j in range(kmer):
                    counts[i + j] += 1.0
                    real_scores[i + j] += score
            real_scores = real_scores / counts
            real_scores = real_scores / np.linalg.norm(real_scores)

            # print(index)
            # print(real_scores)
            # print(len(real_scores))

            scores[index] = real_scores

    return scores, probs

def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string) - kmer+1:
        sentence += original_string[i:i + kmer] + " "
        i += stride

    return sentence[:-1].strip("\"")
def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def get_attention_dna(model, tokenizer, sentence_a, start, end):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b=None, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention = model(input_ids)[-1]
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 
    attn = format_attention(attention)
    attn_score = []
    for i in range(1, len(tokens)-1):
        attn_score.append(float(attn[start:end+1,:,0,i].sum()))
    return attn_score

def get_real_score(attention_scores, kmer, metric):
    counts = np.zeros([len(attention_scores)+kmer-1])
    real_scores = np.zeros([len(attention_scores)+kmer-1])

    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i+j] += 1.0
                real_scores[i+j] += score

        real_scores = real_scores/counts
    else:
        pass

    return real_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmer",
        default=0,
        type=int,
        help="K-mer",
    )
    parser.add_argument(
        "--model_path",
        default="/home/zhihan/dna/dna-transformers/examples/ft/690/p53-small/TAp73beta/3/",
        type=str,
        help="The path of the finetuned model",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    # parser.add_argument(
    #     "--start_layer",
    #     default=11,
    #     type=int,
    #     help="Which layer to start",
    # )
    # parser.add_argument(
    #     "--end_layer",
    #     default=11,
    #     type=int,
    #     help="which layer to end",
    # )
    parser.add_argument(
        "--metric",
        default="mean",
        type=str,
        help="the metric used for integrate predicted kmer result to real result",
    )
    parser.add_argument(
        "--data_name",
        default=None,
        type=str,
        required=True,
        help="which data set need to be visualized,choose from test, dev or train",
    )
    parser.add_argument(
        "--vis_task",
        default=None,
        type=str,
        required=True,
        help="which methods should be used for visualization, attr for attribution; attn for attention ",
    )
    # parser.add_argument(
    #     "--sequence",
    #     default=None,
    #     type=str,
    #     help="the sequence for visualize",
    # )
    parser.add_argument(
        "--output_dir",
        default="./results/",
        type=str,
        help="the address for saving the visualized map"
    )
    parser.add_argument(
        "--task_name",
        default='dnaprom',
        type=str,
        help="task name ",
    )
    parser.add_argument(
        "--batch_size", default=50, type=int, help="Batch size per GPU/CPU for visualization.",
    )
    parser.add_argument(
        "--model_type",
        default='dna',
        type=str,
        help="Model type selected in the list: " + ", ".join(TOKEN_ID_GROUP),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        type=int,
        help="compute attn/attr for which layer, default is last layer for attn and embedding for attr",)
    args = parser.parse_args()
    # args.model_name_or_path=args.model_path
    args.n_process=1
    args.do_predict = True
    args.overwrite_cache=False
    # args.vis_task='attn'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and calculate attention
    tokenizer_name = 'dna' + str(args.kmer)
    model_path = args.model_path
    # model = BertModel.from_pretrained(model_path, output_attentions=True)

    config = BertConfig.from_pretrained(
        model_path,
        num_labels=2,
        finetuning_task='dnaprom',
        cache_dir=None,
    )
    config.output_attentions = True
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
        cache_dir=None,
    )
    model.to(device)
    model.eval()
    model.zero_grad()
    tokenizer = DNATokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

    pred_dataset = load_cache_examples(args, args.task_name, tokenizer, data_name=args.data_name, evaluate=True)

    # args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)

    if args.vis_task == 'attr':
        def predict(inputs, token_type_ids=None,position_ids=None, attention_mask=None):
            output = model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids)
            return output

        def pos_forward_func(inputs, token_type_ids=None, position_ids=None,attention_mask=None):

            logits = predict(inputs,
                             token_type_ids=token_type_ids,position_ids=position_ids,
                             attention_mask=attention_mask)

            preds = logits[0]
            # out_label_ids = inputs["labels"].detach().cpu().numpy()

            softmax = torch.nn.Softmax(dim=1)

            probs = softmax(preds)
            # return probs.max(1).values.reshape(-1,1)
            return probs

        def summarize_attributions(attributions):
            # attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions.sum(dim=-1)
            # attributions = attributions.T / torch.norm(attributions, dim=1)
            # attributions = attributions.T / torch.linalg.norm(attributions, dim=1)
            # return attributions.T
            return attributions

        def construct_input_ref_pair(input_ids, ref_token_id, sep_token_id, cls_token_id):
            SEP_idx = np.where(input_ids.detach().cpu().numpy()[0] == 3)[0]
            ref_input_ids = [cls_token_id] + [ref_token_id] * int(SEP_idx - 1) + [sep_token_id]+[ref_token_id]*int(input_ids.size(1)-SEP_idx-1)
            # ref_input_ids = [cls_token_id] + [ref_token_id] * (input_ids.size(1) - 2) + [sep_token_id]

            # return input_ids, torch.tensor([ref_input_ids], device=device), len(
            #     input_ids)
            return input_ids, torch.tensor([ref_input_ids], device=device), input_ids.shape[1]

        # def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
        #     seq_len = input_ids.size(1)
        #     token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
        #     ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
        #     return token_type_ids, ref_token_type_ids

        def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
            seq_len = input_ids.size(1)
            token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
            ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
            return token_type_ids, ref_token_type_ids

        def construct_input_ref_pos_id_pair(input_ids):
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
            ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
            return position_ids, ref_position_ids

        def construct_whole_bert_embeddings(input_ids, ref_input_ids, \
                                            token_type_ids=None, ref_token_type_ids=None, \
                                            position_ids=None, ref_position_ids=None):
            input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids,
                                                     position_ids=position_ids)
            ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids,
                                                         position_ids=ref_position_ids)

            return input_embeddings, ref_input_embeddings

        def pos_forward_func_emb(input_emb, attention_mask=None):
            pred = model(inputs_embeds=input_emb, attention_mask=attention_mask, )

            preds = pred[0]
            # out_label_ids = inputs["labels"].detach().cpu().numpy()
            # pred.max(1).values
            softmax = torch.nn.Softmax(dim=1)

            probs = softmax(preds)
            # return probs.max(1).values.reshape(-1,1)
            return probs

        # out_label_ids = None
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=1)

        attributions_list = []

        for batch in tqdm(pred_dataloader, desc="Getting attr"):
            # model1.eval()
            batch = tuple(t.to(args.device) for t in batch)

            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in TOKEN_ID_GROUP else None
            #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            # outputs = model1(**inputs)
            # _, logits = outputs[:2]
            #
            #
            # preds = logits.detach().cpu().numpy()
            # out_label_ids = inputs["labels"].detach().cpu().numpy()
            #
            # softmax = torch.nn.Softmax(dim=1)
            #
            # probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            input_ids = batch[0]
            token_type_ids = batch[2] if args.model_type in TOKEN_ID_GROUP else None
            attention_mask = batch[1]
            labels = batch[3]

            # pos_forward_func(batch[0],batch[2], batch[1])
            #

            ref_token_id = tokenizer.pad_token_id
            sep_token_id = tokenizer.sep_token_id
            cls_token_id = tokenizer.cls_token_id
            input_ids, ref_input_ids, sep_id = construct_input_ref_pair(input_ids, ref_token_id, sep_token_id,
                                                                        cls_token_id)
            token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
            position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
            input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, ref_input_ids,
                                                                                     token_type_ids=token_type_ids,
                                                                                     ref_token_type_ids=ref_token_type_ids,
                                                                                     position_ids=position_ids,
                                                                                     ref_position_ids=ref_position_ids)
            if args.layer==None:
                lig = LayerIntegratedGradients(pos_forward_func, model.bert.embeddings,multiply_by_inputs=True) ###attr on embedding
                attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                                                baselines=ref_input_ids, target=labels,
                                                                # baselines=0, target=labels,
                                                                additional_forward_args=(
                                                                    token_type_ids, position_ids,attention_mask),
                                                                return_convergence_delta=True)
                # lig_ = LayerIntegratedGradients(pos_forward_func, [model.bert.embeddings.word_embeddings,
                #                                                    model.bert.embeddings.token_type_embeddings,
                #                                                    model.bert.embeddings.position_embeddings]) ###attr on embedding
                # attributions_start_, delta_start_ = lig_.attribute(inputs=input_ids,
                #                                                 baselines=0, target=labels,
                #                                                 additional_forward_args=(
                #                                                     token_type_ids, attention_mask),
                #                                                 return_convergence_delta=True)
                # print(attributions_start)

                attributions = summarize_attributions(attributions_start)
                # attributions_ = summarize_attributions(attributions_start_[0])
                attributions_np=abs(attributions.detach().cpu().numpy()) #### use abs, only care importance
                SEP_idx=np.where(input_ids.detach().cpu().numpy()[0]==3)[0]
                attributions_np=np.delete(attributions_np,SEP_idx) #### remove SEP
                attributions_np=np.delete(attributions_np,0) #### remove CLS

                attributions_list.append(attributions_np)
            else:


                # lc = LayerConductance(pos_forward_func_emb, model.bert.encoder.layer[0])
                # layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings,target=labels,
                #                                         additional_forward_args=(attention_mask))

                ## have to use embedding as input  IG for attention weights should change output to attention model.bert.encoder.layer[args.layer].attention
                lig = LayerIntegratedGradients(pos_forward_func_emb, model.bert.encoder.layer[args.layer].output,multiply_by_inputs=True)  ###attr on layers
                attributions_start, delta_start = lig.attribute(inputs=input_embeddings,
                                                                baselines=ref_input_embeddings, target=labels,
                                                                additional_forward_args=(
                                                                    attention_mask),
                                                                return_convergence_delta=True)
                # lig_ = LayerIntegratedGradients(pos_forward_func, [model.bert.embeddings.word_embeddings,
                #                                                    model.bert.embeddings.token_type_embeddings,
                #                                                    model.bert.embeddings.position_embeddings]) ###attr on embedding
                # attributions_start_, delta_start_ = lig_.attribute(inputs=input_ids,
                #                                                 baselines=0, target=labels,
                #                                                 additional_forward_args=(
                #                                                     token_type_ids, attention_mask),
                #                                                 return_convergence_delta=True)
                # print(attributions_start)

                attributions = summarize_attributions(attributions_start)
                # attributions_ = summarize_attributions(attributions_start_[0])
                attributions_np = abs(attributions.detach().cpu().numpy())  #### use abs, only care importance
                SEP_idx = np.where(input_ids.detach().cpu().numpy()[0] == 3)[0]
                attributions_np = np.delete(attributions_np, SEP_idx)  #### remove SEP
                attributions_np = np.delete(attributions_np, 0)  #### remove CLS

                attributions_list.append(attributions_np)
        # print('a')
        attributions_array = np.vstack(attributions_list)

        ####remove CLS and SEP
        kmer=args.kmer
        attr_scores = np.zeros([attributions_array.shape[0], attributions_array.shape[-1]+ kmer - 1])
        for index, attribution_score in enumerate(attributions_array):
            # att_score = []
            # for i in range(1, attribution_score.shape[-1] - kmer + 2):
            #     att_score.append(float(attribution_score[:, 0, i].sum()))

            # for i in range(len(attribution_score) - 1):
            #     attribution_score=abs(attribution_score)
            #     # attribution_score[0]=0
            #     attribution_score=np.delete(attribution_score,0)
            #     if attribution_score[i + 1] == 0:
            #         attribution_score[i] = 0
            #         break

            # attn_score[0] = 0
            # counts = np.zeros([len(attribution_score) + kmer - 1])
            # real_scores = np.zeros([len(attribution_score) + kmer - 1])
            counts = np.zeros([len(attribution_score)+ kmer - 1])
            real_scores = np.zeros([len(attribution_score)+ kmer - 1])
            for i, score in enumerate(attribution_score):
                for j in range(kmer):
                    counts[i + j] += 1.0
                    real_scores[i + j] += score
            real_scores = real_scores / counts
            real_scores = real_scores / np.linalg.norm(real_scores) #####do not apply this norm with attr norm at same time

            # print(index)
            # print(real_scores)
            # print(len(real_scores))

            attr_scores[index] = real_scores

        if args.layer==None:
            np.save(args.output_dir + '/attributions_' + args.data_name + '.npy', attr_scores)
        else:
            np.save(args.output_dir + '/attributions_' + args.data_name + '_layer_'+str(args.layer)+ '.npy', attr_scores)


        # np.save(args.output_dir + '/attributions_'+args.data_name+'.npy', attributions_array)

    if args.vis_task == 'attn':
        # Note that DistributedSampler samples randomly
        # pred_sampler = SequentialSampler(pred_dataset)
        # pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.batch_size)

        scores = None
        all_probs = None
        kmer = args.kmer
        attention_scores, probs = visualize_attn(args, model, tokenizer, kmer=kmer)
        if scores is not None:
            all_probs += probs
            scores += attention_scores
        else:
            all_probs = deepcopy(probs)
            scores = deepcopy(attention_scores)

        # all_probs = all_probs / float(len(args.kmer))
        if args.layer==None:
            np.save(os.path.join(args.output_dir, "atten_"+args.data_name+".npy"), scores)
            np.save(os.path.join(args.output_dir, "pred_results_"+args.data_name+".npy"), all_probs)
        else:
            np.save(os.path.join(args.output_dir, "atten_"+args.data_name+ '_layer_'+str(args.layer)+".npy"), scores)


if __name__ == "__main__":
    main()
    # --kmer 5 --model_name_or_path model_pretrain/5-decay-250-0 --model_path output/5_decay_250 --start_layer 1 --end_layer 2 --metric 'mean' --output_dir output/5_decay_250/result --data_dir data/5_decay_250/52 --data_name test --vis_task attr --batch_size 2
