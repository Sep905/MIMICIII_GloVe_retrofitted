import os.path
import spacy
import scispacy
from scispacy.umls_linking import UmlsEntityLinker
from collections import OrderedDict


def unified_medical_language_entity_linker(model,document):
   
    doc = model(document)
    
    entity = doc.ents
    entity = [str(item) for item in entity]               # convert each entity tuple to list of strings
    entity = str(OrderedDict.fromkeys(entity))            # returns unique entities only
    entity = model(entity).ents                             # convert unique entities back to '.ents' object

    list_entity = []
    list_cui = [] 
    list_info = []
    list_score = []
    for entity in entity:
        for umls_ent in entity._.umls_ents:
            list_entity.append(str(entity))
            Concept_Id, Score = umls_ent
            list_cui.append(str(Concept_Id))
            list_score.append(str(Score))
            list_info.append(str(linker.umls.cui_to_entity[umls_ent[0]]))
   
    return list_entity,list_cui,list_score,list_info




#load scispacy model 
scilg_nlp = spacy.load("en_core_sci_lg")

#define paths 
read_path = 'conceptExtraction_entityLinking/sentences_files/'
write_path = 'conceptExtraction_entityLinking/scispacy_pipeline_output/'

#load entity linked and add it to the concept extractor pipeline
linker = UmlsEntityLinker(max_entities_per_mention = 3) 
scilg_nlp.add_pipe(linker)

# 100000 is only indicative, indeed each for each path created the relative file (patient) is checked
for i in range(0,100000):
    patient_path_read = read_path +   "patient" +str(i) + ".txt"
    patient_path_write = write_path +   "patient" +str(i) + ".txt"

    if os.path.isfile(patient_path_read):
        #read the patient file
        with open(patient_path_read, 'r',encoding="utf-8") as fr:
    
            document=fr.read()
        fr.close()
    
        #try the entity linking -> the pipeline can preprocess text with a memory limits and fail for some files
        #these files can be splitted manually in two text files and then run the pipeline again 
        try:
            list_ent,list_cui,list_score,list_info =  unified_medical_language_entity_linker(scilg_nlp,document)
        except:
            print('patient' + str(i) + ' fail for memory limits')
            continue
        
        #write the output for the entity linking
        with open(patient_path_write, 'w',encoding="utf-8") as fw:
            for j in range(len(list_ent)):
                fw.write(list_ent[j] + '\n' + list_cui[j] + '\n' + list_score[j] + '\n' + list_info[j] + '\n')
        fw.close()
        
    else:
        continue