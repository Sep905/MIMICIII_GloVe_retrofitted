from __future__ import absolute_import
from mimic3_inhospital_mortality.readers_conll import InHospitalMortalityReader
import os.path

#read the notes
reader = InHospitalMortalityReader(dataset_dir='../mimic3_textdata/in-hospital-mortality/train',    
                              notes_dir='../mimic3_textdata/train',                                 
                              listfile='../mimic3_textdata/in-hospital-mortality/train_listfile.csv')


N = reader.get_number_of_examples()

#for each patient tokenize its notes and write a text file with one sentence in each row
for i in range(N):
    
    
    patient = reader.read_example(i)
    patient_notes = patient['text']
    patient_number = patient['name'].split('_')[0]
    
    base_path = "sentences_files/"
    pat_path = base_path +   "patient" +str(patient_number) + ".txt"
    

    if os.path.isfile(pat_path):
        with open(pat_path, 'a') as fa:
            for doc, sentences in patient_notes.items():
                sentence_write = ''
                for s in sentences:
                    sentence_write = sentence_write + ' '.join(s)
                    fa.write(sentence_write + '\n')
                    sentence_write = ''
        fa.close()
    else:
        
        with open(pat_path, 'w') as fw:
    
            for doc, sentences in patient_notes.items():
                sentence_write = ''
                for s in sentences:
                    sentence_write = sentence_write + ' '.join(s)
                    fw.write(sentence_write + '\n')
                    sentence_write = ''
        fw.close()