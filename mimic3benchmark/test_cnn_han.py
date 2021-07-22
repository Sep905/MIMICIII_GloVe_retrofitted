import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from text_model import CNNHan
import utils
from readers_conll import InHospitalMortalityReader
from preprocessing import Discretizer, Normalizer
import metrics
import common_utils
import numpy as np
import logging
import tempfile
import shutil
import pickle
from datetime import datetime
from sklearn.metrics import brier_score_loss



def eval_model(model, dataset, device, vocab):
    model.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        y_true = []
        predictions = []
        for _, notes, labels,categories in dataset:
            labels = labels.to(device)
            x_notes, sent_seq_len, word_seq_len, categories_input = utils.create_doc_batch(notes,
                    vocab,
                    device,categories)
            logits = model(x_notes, sent_seq_len, word_seq_len,categories_input)
            # [B, S, W, W]
            probs = sigmoid(logits)

            predictions += [p.item() for p in probs]
            y_true += [y.item() for y in labels]

    clf_score = brier_score_loss(y_true, predictions, pos_label=1)
    logging.info("Brier score: %1.3f" % (clf_score))
    results = metrics.print_metrics_binary(y_true, predictions, logging)

    #add the Brier score to the metrics in output
    results["brier"] = clf_score
    return results, predictions, y_true

def main(args):

    args.mode = 'test'
    hidden_size = args.dim
    dropout = args.dropout
    batch_size = args.batch_size
    learning_rate = args.lr
    emb_size = args.emb_size
    best_model = args.best_model
    max_w = args.max_w
    max_s = args.max_s
    dim_categories = args.dim_cat
    vocabulary = args.vocabulary
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")   
    # 1. Get a unique working directory 
    output_dir = args.output_dir
    logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S')
    

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')


    test_reader = InHospitalMortalityReader(dataset_dir=args.data + "/test",
                                         listfile=args.data + '/test_listfile.csv',
                                         notes_dir=args.notes, 
                                         period_length=48.0)

    
    discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

    discretizer_header = discretizer.transform(test_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
        normalizer_state = "/" + normalizer_state
    normalizer.load_params(normalizer_state)

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = discretizer_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl

    # Read data
    logging.info('proc word2vec')
    vocab, weight = utils.Vocabulary.from_data(args.word2vec, vocabulary , emb_size) 
    logging.info('load test')
    test_dataset = utils.MIMICTextDataset(test_reader, 
            discretizer, 
            normalizer, 
            batch_labels=True,
            max_w=max_w,
            max_s=max_s,
            notes_output='doc')
 
    test_dl =  DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.doc_collate)
    #[B, M, feat_size]

   

    random_category = torch.rand(14,dim_categories,device=device)
    random_category = torch.cat( (random_category,torch.zeros((1,dim_categories),device=device)) ,0 )

    # Define the classification model.
    model = CNNHan(vocab_size=vocab.size()+4,
                        label_size=1, #label size = 1 because of the binary nature of the predictions(classification)
                        emb_size=emb_size, 
                        hidden_size=hidden_size,
                        dropout=dropout,
                        model_w2vec=weight,
                        category_size = dim_categories,
                        emb_category = random_category)
  

 
    model.load_state_dict(torch.load(best_model))
    print(model)
    model = model.to(device)

    metrics_results, pred_probs, y_true = eval_model(model,
                                test_dl,
                                device, vocab)
            

    #metrics_results.pkl contains the metrics printed: (accuracy, precision on 0, precision on 1, recall on 0, recall on 1, AUROC, AUPRC, min(+P,Se))
    #modified to contains also brier score and f1

    #pred_probs.pkl will contains the predicted probabilities
    #y_true will contains the true labels

    pickle.dump(metrics_results, open(    output_dir + '/test_metrics.pkl'     , "wb" ) )
    pickle.dump(pred_probs, open( output_dir+ '/test_predprobs.pkl', "wb" ) )
    pickle.dump(y_true, open( output_dir + '/test_ytrue.pkl', "wb" ) )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test')
    parser.description = 'test in hospital mortality classifier'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    common_utils.add_common_arguments(parser)
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=100, help='perfom evaluation and model selection on validation default:100')
    parser.add_argument('--emb_size', type=int, default=128, help='emb_size default:128')
    parser.add_argument('--max_w', type=int, default=1000, help='max word per patient default:1000')
    parser.add_argument('--max_s', type=int, default=1000, help='max sentences per patient default:1000')
    parser.add_argument('--att_proj', type=str, default='softmax', help='projection for self attention layer default:softmax')
    parser.add_argument('--imbalance', dest='imbalance', action='store_true')
    parser.add_argument('--notes', type=str, help='Path to the notes of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/'))
    parser.add_argument('--word2vec', type=str, help='Path to pretrained embeddings (word2vec format)',
                    default=os.path.join(os.path.dirname(__file__), '../../data/'))
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
    parser.add_argument('--best_model', type=str, required=True, help='best model path')
    parser.add_argument('--dim_cat', type=int, default=10, help='emb_size for categories default:10')
    parser.add_argument('--vocabulary', type=int)
    args = parser.parse_args()
    main(args)
