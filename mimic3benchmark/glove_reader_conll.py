from __future__ import absolute_import
from mimic3benchmark.readers_conll import InHospitalMortalityReader

reader = InHospitalMortalityReader(dataset_dir='../mimic3benchmark_textdata/in-hospital-mortality/train',
                              notes_dir='../mimic3benchmark_textdata/train',  
                              listfile='../mimic3benchmark_textdata/in-hospital-mortality/train/listfile.csv')

N = reader.get_number_of_examples()
for n in range(N):
    patient = reader.read_example(n)
    patient_notes = patient['text']
    for doc, sentences in patient_notes.items():

        for sentence in sentences:
            print(' '.join(sentence))


