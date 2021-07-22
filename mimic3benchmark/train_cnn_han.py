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


def eval_model(model, dataset, device, vocab):
    model.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        y_true = []
        predictions = []
        for data, notes, labels,categories in dataset:
            data = data.to(device)
            labels = labels.to(device)
            x_notes, sent_seq_len, word_seq_len,categories_input = utils.create_doc_batch(notes, 
                    vocab, 
                    device,categories)
            logits = model(x_notes, sent_seq_len, word_seq_len,categories_input)
            probs = sigmoid(logits)
            predictions += [p.item() for p in probs]
            y_true += [y.item() for y in labels]

    results = metrics.print_metrics_binary(y_true, predictions, logging)
    return results

def main(args):

    args.mode = 'train'
    hidden_size = args.dim
    dropout = args.dropout
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    emb_size = args.emb_size
    dim_categories = args.dim_cat
    vocabulary = args.vocabulary
    max_w = args.max_w
    max_s = args.max_s
    imbalance = args.imbalance
    seed = args.seed
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")   
    
    # 1. Get a unique working directory 
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)  # make sure base_dir exists
    output_dir = tempfile.mkdtemp(prefix=datetime.now().strftime("%y-%m-%d.%Hh%Mm%Ss."), dir=base_dir)
    logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'run.log')),
                logging.StreamHandler()
            ])
    
    logging.info('Workspace: %s', output_dir)
 
    
    if args.small_part:
        args.save_every = 2**30

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')
    
    
    logging.info('proc word2vec')
    
    vocab, weight = utils.Vocabulary.from_data(args.word2vec, vocabulary , emb_size) 




    train_reader = InHospitalMortalityReader(dataset_dir=args.data + "/train",
                                        notes_dir=args.notes,  
                                        listfile=args.data + '/train_listfile.csv',
                                        period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=args.data + "/train",
                                       notes_dir=args.notes,
                                       listfile=args.data + '/val_listfile.csv',
                                       period_length=48.0)


    discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize

    normalizer_state = args.normalizer_state
    if normalizer_state is None: 
        normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
        normalizer_state = "/" + normalizer_state
        #normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        

    normalizer.load_params(normalizer_state)

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = discretizer_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl

    # Read data
    logging.info('read training data')    
    train_dataset = utils.MIMICTextDataset(train_reader, 
            discretizer, 
            normalizer, 
            batch_labels=True,
            max_w=max_w,
            max_s=max_s,
            notes_output='doc')
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.doc_collate)
    logging.info('read val data')
    val_dataset = utils.MIMICTextDataset(val_reader, 
            discretizer, 
            normalizer, 
            batch_labels=True,
            max_w=max_w,
            max_s=max_s,
            notes_output='doc')
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.doc_collate)
    #[B, M, feat_size]
    feat_size = train_dataset.x.shape[-1] 
    if target_repl:
        raise NotImplementedError("target repl not implemented")

    #initialize the random word vectors for the notes(sentences) categories --> each vector has values in (0,1)
    random_category = torch.rand(14,dim_categories,device=device)
    random_category = torch.cat( (random_category,torch.zeros((1,dim_categories),device=device)) ,0 )

    # Define the classification model.

    model = CNNHan(vocab_size=vocab.size(),
                        label_size=1, #label size = 1 because of the binary nature of the predictions(classification)
                        emb_size=emb_size, 
                        hidden_size=hidden_size,
                        dropout=dropout,
                        model_w2vec=weight,
                        category_size = dim_categories,
                        emb_category = random_category)

    model = model.to(device)

    logging.info(args)
    logging.info(model)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if target_repl:
        raise  NotImplementedError("loss for each time: step to be implemented") 
    else:
        criterion = nn.BCEWithLogitsLoss()

    # path to best model save on disk
    best_model = output_dir + '/best_model.pt'
    best_val_auc = 0.

    results = []

    step = 0
    num_batches = 0

    # loop over the epochs
    for epoch_num in range(1, num_epochs+1): 
        print("epoch number: ")
        print(epoch_num)
        print("\n")
        loss_batch = .0
        num_batches = 0
        # loop over mini-batches
        for _, notes, labels, categories in train_dl:   #addedd categories in the trainind dataloader
            labels = labels.to(device)
            x_notes, sent_seq_len, word_seq_len,categories_input = utils.create_doc_batch(notes, 
                    vocab, 
                    device, categories)
            # Model is in training mode (for dropout).
            model.train()
            optimizer.zero_grad()
       
            # run forward
            logits = model(x_notes, sent_seq_len, word_seq_len,categories_input)
            if imbalance:
                logits = logits.squeeze()
            loss = criterion(logits, labels)
            
            loss_batch += loss.item()
            # Backpropagate and update the model weights.
            loss.backward()
            optimizer.step()
        
            num_batches += 1
        
            # Every 100 steps we evaluate the model and report progress.
            if step % args.steps == 0:
                logging.info("epoch (%d) step %d: training loss = %.2f"% 
                 (epoch_num, step, loss_batch/num_batches))
            
            
            step += 1
           
        
        metrics_results = eval_model(model,
                                    val_dl,
                                    device, vocab)
        metrics_results['epoch'] = epoch_num
        results.append(metrics_results)
        if metrics_results['auroc'] > best_val_auc:
            best_val_auc = metrics_results['auroc']
            # save best model in disk
            torch.save(model.state_dict(), best_model)
            logging.info('best model AUC of ROC = %.3f'%(best_val_auc))
            logging.info("Finished epoch %d" % (epoch_num))
            


    pickle.dump(results, open(output_dir + '/metrics.pkl', "wb" ) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train')
    parser.description = 'train in hospital mortality classifier'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    common_utils.add_common_arguments(parser)
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=100, help='perfom evaluation and model selection on validation default:100')
    parser.add_argument('--emb_size', type=int, default=128, help='emb_size default:128')
    parser.add_argument('--max_w', type=int, default=1000, help='max word per patient default:1000')
    parser.add_argument('--max_s', type=int, default=1000, help='max word per patient default:1000')
    parser.add_argument('--imbalance', dest='imbalance', action='store_true')
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
    parser.add_argument('--notes', type=str, help='Path to the notes of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/'))
    parser.add_argument('--word2vec', type=str, help='Path to pretrained embeddings (word2vec format)',
                    default=os.path.join(os.path.dirname(__file__), '../../data/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
    parser.add_argument('--dim_cat', type=int, default=10, help='emb_size for categories default:10')
    parser.add_argument('--vocabulary', type=int)
    args = parser.parse_args()    
    main(args)
