# Pre-trained Text Representations with Knowledge Bases for Mortality Prediction 


![Diagram](img/diagram.PNG?raw=true)


><b>Abstract</b>: 
>It is postulated that free text in clinical notes residing in the Electronic Health Record (EHR) contains information that benefits various clinical tasks. An important example is utilizing such free text to predict patient mortality in the Intensive Care Unit (ICU). Recent neural Natural Language Processing (NLP) approaches automatically learn effective representations of text that are called embeddings. The main idea is that when words appear in the same context then their vector representations (embeddings) will tend to be similar. However, seeking meaning only in text neighbourhoods misses out on other sources of meaning. Specifically, concepts appearing in biomedical knowledge bases can potentially enhance the text representations.

>In this paper, we explore the added predictive value of introducing knowledge into a mortality prediction model in the ICU that relies on free text in the clinical notes. In particular, we use the retrofitting approach to adjust pre-trained word embeddings based on knowledge graphs in the Unified Medical Language System (UMLS). In particular, we use the synonymy relations in the UMLS. Our approach is applied on the MIMIC-III database to predict in-hospital mortality based on the first 48hours of Intensive Care admission. 5% of the words were linked to UMLS concepts. The retrofitted models achieved an AUC-ROC of 0.822 (±0.002 std) and an AUC-PR of 0.451 (±0.005 std). The baseline models achieved an AUC-ROC of 0.815 (±0.003 std), which was statistically significantly worse than the retrofitted model, and an AUC-PR of 0.443 (±0.003 std). This demonstrates the potential predictive improvement of infusing medical knowledge to adjust word embeddings.



Code based on: https://github.com/YerevaNN/mimic3-benchmarks
### Citation

Please cite the following publication: 

G. Albi, M. Rios, R. Bellazzi and A. Abu-Hanna. Pre-trained Text Representations with Knowledge Bases for Mortality predicion. In Workshop on Knowledge Representation for Health Care, KR4HC 2021.

### Requirements

- MIMIC-III clinical notes, available at https://mimic.physionet.org/ after request, as CSV files.
- UMLS release, available at https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html. Then the Metathesaurus tables can be loaded in a DB through  MethamorphoSys installation wizard: https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html.
- ```requirements.txt``` contains the Python requirements.

### Build Dataset

Example to build the dataset in ```prepro_text.sh```.  
In the dataset created each patient is represented by a text file containing the clinical notes relating to its admission. 

## Learn baseline word vectors

- Tokenize the MIMIC-III notes of the training set (creating a unique text file):
```
python mimic3benchmark/glove_reader_conll.py > embeddings/training_notes.txt
```
- Learn Glove embeddings (https://github.com/stanfordnlp/GloVe) using ```embeddings/training_notes.txt``` as input. 
The outputs are:
  - a text file with the learned word vectors e.g. ```embeddings/glove_embeddings.txt```;
  - a vocabulary text file e.g. ```embeddings/vocabulary.txt``` with the unique tokens and their frequencies in the corpus.


## Enriching word embeddings with UMLS 

- <b>Sentences preprocess</b>: Make sentences for each patients clinical notes (one for each line) ```conceptExtraction_entityLinking/make_sentences_reader_conll.py```;
- <b>ScispaCy concept extraction and entity linking</b>: After installing ScispaCy (https://github.com/allenai/scispacy) choose one of the proposed models; then run ScispaCy pre-trained pipeline ```conceptExtraction_entityLinking/scispacy_pipeline.py```. For each patient (text file) the pipeline recognize medical concepts and normalize them to UMLS, and this output is written in ```conceptExtraction_entityLinking/sentences_files```;
- <b>Parsing the ScispaCy output</b>: run ```conceptExtraction_entityLinking/parsing_scispacy_output.py``` to parse the concept extractiong and entity linking output; a dictionary will be created with: 
{ <i>key</i> --> concept name and <i>values</i> CUIs (max 3) associated to the UMLS concept }
and then writted as text file in ```conceptExtraction_entityLinking/map_concept_CUIs.txt```;
- <b>Lexicon creation with PyMedTermino </b>: run ```conceptExtraction_entityLinking/pymedtermino_lexicon_creation.py``` to create a semantic lexicon ```conceptExtraction_entityLinking/lexicon_UMLS_synonyms.txt``` using the concept:CUIs dictionary created; for each concept we retrieve its synonyms leveraging PyMedTermino (https://github.com/MedevaKnowledgeSystems/pymedtermino);
- <b>Enhancing the word vectors using Retrofitting method </b>: Retrofitting (https://github.com/mfaruqui/retrofitting) is the post-processing method used to enhance the baseline word vectors ```embeddings/glove_embeddings.txt``` with the UMLS synonyms semantic lexicon ```conceptExtraction_entityLinking/lexicon_UMLS_synonyms.txt```; the retrofitted word vectors are written in ```embeddings/glove_retrofitted_embeddings.txt``` :
```
python conceptExtraction_entityLinking/retrofit.py -i embeddings/glove_embeddings.txt -l conceptExtraction_entityLinking/lexicon_UMLS_synonyms.txt -n 10 -o embeddings/glove_retrofitted_embeddings.txt
``` 

## In-hospital mortality prediction with Hierarchical CNN (HCNN)
Example to train the model:
```
python model/train_cnn_han.py --dim 128 --emb 100 --timestep 1.0 --dropout 0.2 --batch_size 16 --data mimic3_textdata/in-hospital-mortality --notes mimic3_textdata/train --output_dir results --epoch 30 --lr 2e-4 --word2vec embeddings/glove_embeddings.txt --max_w 25 --max_s 500 --dim_cat 10 --vocabulary 30000
``` 
Note: each <b>training</b> run produces a directory in ```results/...```, containing:
1. ```run.txt``` log text file;
2. ```best_model.pt``` the best model (best set of hyperparameters) from model selection;
3. ```metrics.pkl``` metrics achieved during the training.

Example to test the model:
```
python model/test_cnn_han.py --dim 128 --emb 100 --timestep 1.0 --dropout 0.2 --batch_size 16 --data mimic3_textdata/in-hospital-mortality --notes mimic3_textdata/test  --output_dir results_test --lr 2e-4 --word2vec embeddings/glove_embeddings.txt --max_w 25 --max_s 500 --best_model results/best_model.pt --vocabulary 30000
``` 
Note: a test run needs a ```best_model.pt``` files resulting from training, since its hyperparameters are loaded and tested on the test set. Each <b>testing</b> run produces in ```results_test/...```:
1. ```test_metrics.pkl``` similar to ```metrics.pkl```, these are the metrics achieved during the testing;
2. ```test_predprobs.pkl``` predicted probabilities for each example in the test set;
3. ```test_ytrue.pkl``` true probabilities for each example in the test set;


## Results, plot and visualization 
Scores achieved in AUC-ROC, AUC-PR and Brier-score when using the HCNN for mortality prediction. We compared the MIMIC-III text representaions learned with GloVe and enriched with UMLS synonyms by using Retrofitting in the 48H ICU mortality prediction task:

| Model | AUC-ROC | AUC-PR | Brier-score |
|     :---:       |     :---:      |     :---:      |     :---:      |
| GloVe   | 0.815±0.003     | 0.443±0.003    |  <b>0.088</b>±0.005    |
| Retrofitted GloVe     | <b>0.822</b>±0.002       | <b>0.451</b>±0.003     | 0.089±0.003    |




![ROC curve and calibration plot](plots.png?raw=true)



![TSNE visualization](tsne.png?raw=true)

