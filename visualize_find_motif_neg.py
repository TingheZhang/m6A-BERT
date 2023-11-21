#### ::: Modified from DNABERT-viz to find motifs enriched in negitve samples::: ####

import os
import pandas as pd
import numpy as np
import argparse
import motif_utils as utils
import pickle as pl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the sequence+label .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--npy_dir",
        default=None,
        type=str,
        required=True,
        help="Path where the attention scores were saved. Should contain both pred_results.npy and atten.npy",
    )

    parser.add_argument(
        "--window_size",
        default=24,
        type=int,
        help="Specified window size to be final motif length",
    )

    parser.add_argument(
        "--min_len",
        default=5,
        type=int,
        help="Specified minimum length threshold for contiguous region",
    )

    parser.add_argument(
        "--pval_cutoff",
        default=0.005,
        type=float,
        help="Cutoff FDR/p-value to declare statistical significance",
    )

    parser.add_argument(
        "--min_n_motif",
        default=3,
        type=int,
        help="Minimum instance inside motif to be filtered",
    )

    parser.add_argument(
        "--align_all_ties",
        action='store_true',
        help="Whether to keep all best alignments when ties encountered",
    )

    parser.add_argument(
        "--save_file_dir",
        default='.',
        type=str,
        help="Path to save outputs",
    )

    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Verbosity controller",
    )

    parser.add_argument(
        "--return_idx",
        action='store_true',
        help="Whether the indices of the motifs are only returned",
    )
    parser.add_argument(
        "--data_name",
        default=None,
        type=str,
        required=True,
        help="which data set need to be visualized,choose from test, dev or train, or all ",
    )
    parser.add_argument(
        "--do_plot",
        action='store_true',
        help="whether plot the heatmap or not ",
    )
    parser.add_argument(
        "--vis_task",
        default=None,
        type=str,
        required=True,
        help="which methods should be used for visualization, attr for attribution; attn for attention ",
    )
    # TODO: add the conditions
    args = parser.parse_args()
    if not os.path.exists(args.save_file_dir):
            os.makedirs(args.save_file_dir)

    if args.data_name=='all':
        if args.vis_task=='attn':
            att_scores_dev = np.load(os.path.join(args.npy_dir, "atten_dev.npy"))
            att_scores_test = np.load(os.path.join(args.npy_dir, "atten_test.npy"))
            att_scores_train = np.load(os.path.join(args.npy_dir, "atten_train.npy"))
        if args.vis_task=='attr':
            att_scores_dev = np.load(os.path.join(args.npy_dir, "attributions_dev.npy"))
            att_scores_test = np.load(os.path.join(args.npy_dir, "attributions_test.npy"))
            att_scores_train = np.load(os.path.join(args.npy_dir, "attributions_train.npy"))

        # att_scores_all=np.vstack((att_scores_train, att_scores_dev, att_scores_test))
        dev = pd.read_csv(os.path.join(args.data_dir,"dev.tsv"),sep='\t',header=0)
        dev.columns = ['sequence','label']
        dev['seq'] = dev['sequence'].apply(utils.kmer2seq)
        dev_pos = dev[dev['label'] == 1]
        dev_neg = dev[dev['label'] == 0]
        pos_att_scores_dev = att_scores_dev[dev_pos.index.values]
        neg_att_scores_dev = att_scores_dev[dev_neg.index.values]
        #################
        test = pd.read_csv(os.path.join(args.data_dir,"test.tsv"),sep='\t',header=0)
        test.columns = ['sequence','label']
        test['seq'] = test['sequence'].apply(utils.kmer2seq)
        test_pos = test[test['label'] == 1]
        test_neg = test[test['label'] == 0]
        pos_att_scores_test = att_scores_test[test_pos.index.values]
        neg_att_scores_test = att_scores_test[test_neg.index.values]
        ##################
        train = pd.read_csv(os.path.join(args.data_dir,"train.tsv"),sep='\t',header=0)
        train.columns = ['sequence','label']
        train['seq'] = train['sequence'].apply(utils.kmer2seq)
        train_pos = train[train['label'] == 1]
        train_neg = train[train['label'] == 0]
        pos_att_scores_train = att_scores_train[train_pos.index.values]
        neg_att_scores_train = att_scores_train[train_neg.index.values]

        pos_att_scores=np.vstack((pos_att_scores_train, pos_att_scores_dev, pos_att_scores_test))
        neg_att_scores=np.vstack((neg_att_scores_train, neg_att_scores_dev, neg_att_scores_test))

        pos_seq=pd.concat([train_pos['seq'],dev_pos['seq'],test_pos['seq']],ignore_index=True)
        neg_seq=pd.concat([train_neg['seq'],dev_neg['seq'],test_neg['seq']],ignore_index=True)
    else:
        att_scores = np.load(os.path.join(args.npy_dir,"atten_"+args.data_name+".npy"))
        # pred = np.load(os.path.join(args.npy_dir,"pred_results.npy"))
        dev = pd.read_csv(os.path.join(args.data_dir,args.data_name+".tsv"),sep='\t',header=0)
        dev.columns = ['sequence','label']
        dev['seq'] = dev['sequence'].apply(utils.kmer2seq)
        dev_pos = dev[dev['label'] == 1]
        dev_neg = dev[dev['label'] == 0]
        pos_att_scores = att_scores[dev_pos.index.values]
        neg_att_scores = att_scores[dev_neg.index.values]
        assert len(dev_pos) == len(pos_att_scores)
        pos_seq=dev_pos['seq']
        neg_seq=dev_neg['seq']

    if args.do_plot:
        # plot
        scores=np.vstack((pos_att_scores,neg_att_scores))
        sns.set()
        ax = sns.heatmap(scores, cmap='YlGnBu', vmin=0)
        # plt.show()
        plt.savefig(args.save_file_dir+'/heatmap_'+args.data_name+'.png')

    # run motif analysis
    merged_motif_seqs = utils.motif_analysis(neg_seq ,
                                        pos_seq,
                                        neg_att_scores,
                                        window_size = args.window_size,
                                        min_len = args.min_len,
                                        pval_cutoff = args.pval_cutoff,
                                        min_n_motif = args.min_n_motif,
                                        align_all_ties = args.align_all_ties,
                                        save_file_dir = args.save_file_dir,
                                        verbose = args.verbose,
                                        return_idx  = args.return_idx
                                    )
    pl.dump(merged_motif_seqs,open(args.save_file_dir+'/motif_seqs_dict_neg.plk','wb'))


if __name__ == "__main__":
    main()


