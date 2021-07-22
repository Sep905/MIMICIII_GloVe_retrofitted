import pickle
import sklearn.calibration
import matplotlib.pyplot as plt
import numpy as np
from utils import Vocabulary
from sklearn.manifold import TSNE

def read_test(dir_path):

    test_metrics = pickle.load(open(dir_path + "test_metrics.pkl", "rb"))
    test_predprobs = pickle.load(open(dir_path + "test_predprobs.pkl", "rb"))
    test_ytrue = pickle.load(open(dir_path + "test_ytrue.pkl", "rb"))

    return test_metrics, test_predprobs, test_ytrue


def plots(true_base,pred_base,Label_base,true_retro,pred_retro,Label_retro):

    
    #calibration quantile
    ybaseq, xbaseq = sklearn.calibration.calibration_curve(true_base, pred_base, strategy='quantile', n_bins=10)
    yretroq, xretroq = sklearn.calibration.calibration_curve(true_retro, pred_retro, strategy='quantile', n_bins=10)
    plt.figure(figsize=[10, 8])
    plt.ylim(0., 1.0)
    plt.xlim(0.,1.0)
    plt.plot(xbaseq,ybaseq, marker='o', linestyle="", markersize=7, label=Label_base, color='darkorange')
    plt.plot(xretroq,yretroq, marker='^', linestyle="", markersize=7, label=Label_retro, color='green')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.legend()
    plt.title("Calibration plot",fontsize=18)
    plt.xlabel("Predicted probabilities",fontsize=18)
    plt.ylabel("True probabilities",fontsize=18)
    plt.savefig("calibration.png")
    
    
    
    #ROC
    fprbase,tprbase,thresholdsbase = sklearn.metrics.roc_curve(true_base, pred_base)
    fprretro,tprretro,thresholdsretro = sklearn.metrics.roc_curve(true_retro, pred_retro)
    plt.figure(figsize=[10, 8])
    plt.plot(fprbase, tprbase, linestyle='-.', label=Label_base, color='darkorange')
    plt.plot(fprretro, tprretro, marker='.', label=Label_retro, color='green')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title("ROC curve",fontsize=18)
    plt.xlabel("False Positive rate",fontsize=18)
    plt.ylabel("True Positive rate",fontsize=18)
    plt.savefig("ROC.png")
    
    #PR CURVE
    prebase,recbase,Thresholdsbase = sklearn.metrics.precision_recall_curve(true_base, pred_base)
    preretro,recretro,Thresholdsretro = sklearn.metrics.precision_recall_curve(true_retro, pred_retro)
    plt.figure(figsize=[10, 8])
    plt.plot(recbase, prebase, linestyle='-.', label=Label_base, color='darkorange')
    plt.plot(recretro, preretro, marker='.', label=Label_retro, color='green')
    plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()


def mean_metrics (m1,m2,m3,m4,m5):
    
    meanAUC = np.mean([m1["auroc"],m2["auroc"], m3["auroc"], m4["auroc"], m5["auroc"]])
    meanAUPR = np.mean([m1["auprc"],m2["auprc"], m3["auprc"], m4["auprc"], m5["auprc"]])
    meanACC = np.mean([m1["acc"],m2["acc"], m3["acc"], m4["acc"], m5["acc"]])
    meanf10 = np.mean([m1["f10"],m2["f10"], m3["f10"], m4["f10"], m5["f10"]])
    meanf11 = np.mean([m1["f11"],m2["f11"], m3["f11"], m4["f11"], m5["f11"]])
    meanprec0 = np.mean([m1["prec0"],m2["prec0"], m3["prec0"], m4["prec0"], m5["prec0"]])
    meanprec1 = np.mean([m1["prec1"],m2["prec1"], m3["prec1"], m4["prec1"], m5["prec1"]])
    meanrec0 = np.mean([m1["rec0"],m2["rec0"], m3["rec0"], m4["rec0"], m5["rec0"]])
    meanrec1 = np.mean([m1["rec1"],m2["rec1"], m3["rec1"], m4["rec1"], m5["rec1"]])
    meanBrier = np.mean([m1["brier"],m2["brier"], m3["brier"], m4["brier"], m5["brier"]])
    
    stdAUC = np.std([m1["auroc"],m2["auroc"], m3["auroc"], m4["auroc"], m5["auroc"]])
    stdAUPR = np.std([m1["auprc"],m2["auprc"], m3["auprc"], m4["auprc"], m5["auprc"]])
    stdACC = np.std([m1["acc"],m2["acc"], m3["acc"], m4["acc"], m5["acc"]])
    stdf10 = np.std([m1["f10"],m2["f10"], m3["f10"], m4["f10"], m5["f10"]])
    stdf11 = np.std([m1["f11"],m2["f11"], m3["f11"], m4["f11"], m5["f11"]])
    stdprec0 = np.std([m1["prec0"],m2["prec0"], m3["prec0"], m4["prec0"], m5["prec0"]])
    stdprec1 = np.std([m1["prec1"],m2["prec1"], m3["prec1"], m4["prec1"], m5["prec1"]])
    stdrec0 = np.std([m1["rec0"],m2["rec0"], m3["rec0"], m4["rec0"], m5["rec0"]])
    stdrec1 = np.std([m1["rec1"],m2["rec1"], m3["rec1"], m4["rec1"], m5["rec1"]])
    stdBrier = np.std([m1["brier"],m2["brier"], m3["brier"], m4["brier"], m5["brier"]])
    
    print("------>>>>" + "\033[1m" + "AUCROC: " + str(meanAUC) + " +- " + str(stdAUC) + "\033[0m")
    print("------>>>>" + "\033[1m" + "AUCPR: " + str(meanAUPR) +  " +- " + str(stdAUPR)+ "\033[0m")
    print("ACC: " + str(meanACC) +  " +- " + str(stdACC) )
    print("pre0: " + str(meanprec0) +  " +- " + str(stdf10) )
    print("------>>>>" + "\033[1m" + "pre1 (PPV): " + str(meanprec1) +  " +- " + str(stdf11) + "\033[0m")
    print("------>>>>" + "\033[1m" + "rec0 (SPEC): " + str(meanrec0) +  " +- " + str(stdprec0) + "\033[0m")
    print("------>>>>" + "\033[1m" + "rec1 (SENS): " + str(meanrec1) +  " +- " + str(stdprec1)+ "\033[0m" )
    print("f10: " + str(meanf10) +  " +- " + str(stdrec0) )
    print("f11: " + str(meanf11) +  " +- " +str(stdrec1) )
    print("------>>>>" + "\033[1m" + " Brier score: " + str(meanBrier) + " +- " + str(stdBrier) + "\033[0m")
    


def find_best_run(m1,m2,m3,m4,m5):
    best_auc = 0
    count = 1
    for metric in [m1,m2,m3,m4,m5]:
        if metric["auroc"]>=best_auc:
            best_auc = metric["auroc"]
            best_metric_ind = count
        else:
            continue
        count+=1
    
    return best_metric_ind
        

def tsne_reduction_and_visualization(total_number_of_words_inCorpus,embeddings_dimension):

    bones_entities = ["humerus","rib","femur","scapula","tibia","phalanges","ulna"]
    respiratory_disease_syntomps_entities = ["dyspnea","cough","bronchitis","asthma","tuberculosis","bronchopneumonia","pleuritis"]
    diagnostic_procedures_entities = ["imaging","biopsy","radiography","cystoscopy","screening","immunohistochemistry","palpations"]      
    antibiotics_entities = ["amoxicillin","clarithromycin","sulfamethoxazole","cephalexin","trimethoprim","ceftriaxone","penicillin"]

    bones_glove = np.zeros((len(bones_entities),100))
    respiratory_disease_syntomps_glove = np.zeros((len(respiratory_disease_syntomps_entities),100))
    diagnostic_procedures_glove = np.zeros((len(diagnostic_procedures_entities),100))
    antibiotics_glove = np.zeros((len(antibiotics_entities),100))

    bones_retro = np.zeros((len(bones_entities),100))
    respiratory_disease_syntomps_retro = np.zeros((len(respiratory_disease_syntomps_entities),100))
    diagnostic_procedures_retro = np.zeros((len(diagnostic_procedures_entities),100))
    antibiotics_retro = np.zeros((len(antibiotics_entities),100))



    #loag glove and retro
    vocab_glove, weight_glove = Vocabulary.from_data("embeddings/glove_embeddings.txt", total_number_of_words_inCorpus, embeddings_dimension) 

    vocab_retro, weight_retro = Vocabulary.from_data("embeddings/glove_retrofitted_embeddings.txt", total_number_of_words_inCorpus, embeddings_dimension) 


    wv_glove_dict = {}
    wv_retro_dict = {}
    i = 0
    for word in concept_tui:
            try:
                wv_glove_dict[word] = weight_glove[vocab_glove.word_to_idx[word]]
                wv_retro_dict[word] = weight_retro[vocab_retro.word_to_idx[word]]
                
            except:
                continue




    #glove
    for i in range(len(bones_entities)):
        bones_glove[i]=wv_glove_dict[bones_entities[i]]

    for i in range(len(bones_entities)):
        respiratory_disease_syntomps_glove[i]=wv_glove_dict[respiratory_disease_syntomps_entities[i]]


    for i in range(len(diagnostic_procedures)):
        diagnostic_procedures_glove[i]=wv_glove_dict[diagnostic_procedures_entities[i]]

    for i in range(len(bones_entities)):
        antibiotics_glove[i]=wv_glove_dict[antibiotics_entities[i]]

    all_embedding_glove = np.concatenate((bones_glove  ,  respiratory_disease_syntomps_glove      ,  diagnostic_procedures_glove ,  antibiotics_glove ))


    tsne = TSNE(n_components=2,perplexity=15,init='pca')
    result_gl= tsne.fit_transform(all_embedding_glove)

    plt.figure(figsize=(18, 10))
    plt.scatter(result_gl[:7, 0], result_gl[:7, 1],s=120,label = "Anatomy - bones")
    plt.scatter(result_gl[7:14, 0], result_gl[7:14, 1],s=120, label = "Disorders - respiratory disease & symptoms")
    plt.scatter(result_gl[14:21, 0], result_gl[14:21, 1],s=120, label = "Procedure - diagnostic procedures")
    plt.scatter(result_gl[21:28, 0], result_gl[21:28, 1],s=120, label = "Chemicals & Drugs - antibiotics")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)









    #retrofitting 
    for i in range(len(bones_entities)):
        bones_retro[i]=wv_retro_dict[bones_entities[i]]

    for i in range(len(bones_entities)):
        respiratory_disease_syntomps_retro[i]=wv_retro_dict[respiratory_disease_syntomps_entities[i]]


    for i in range(len(diagnostic_procedures)):
        diagnostic_procedures_retro[i]=wv_retro_dict[diagnostic_procedures_entities[i]]

    for i in range(len(bones_entities)):
        antibiotics_retro[i]=wv_retro_dict[antibiotics_entities[i]]

    all_embedding_retro = np.concatenate((bones_retro  ,  respiratory_disease_syntomps_retro      ,  diagnostic_procedures_retro  ,  antibiotics_retro ))


    tsne = TSNE(n_components=2,perplexity=15,init='pca')
    result_re= tsne.fit_transform(all_embedding_retro)

    plt.figure(figsize=(18, 10))
    plt.scatter(result_re[:7, 0], result_re[:7, 1],s=120,label = "Anatomy - bones")
    plt.scatter(result_re[7:14, 0], result_re[7:14, 1],s=120, label = "Disorders - respiratory disease & symptoms")
    plt.scatter(result_re[14:21, 0], result_re[14:21, 1],s=120, label = "Procedure - diagnostic procedures")
    plt.scatter(result_re[21:28, 0], result_re[21:28, 1],s=120, label = "Chemicals & Drugs - antibiotics")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)