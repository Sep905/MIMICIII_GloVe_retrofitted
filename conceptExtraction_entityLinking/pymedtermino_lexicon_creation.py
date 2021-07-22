import pymedtermino

#setup pymedtermino
pymedtermino.LANGUAGE = "en"
pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = True
pymedtermino.DATA_DIR = 'C:\pymedtermino-master' #specify here the pymedtermino local installation


from pymedtermino import *
from pymedtermino.umls import *


#create the connection to the local DB where UMSL Metathesaurus is installed     
#root and pass are only and example for the user and password where UMLS is locally installed
connect_to_umls_db('localhost', 'root', 'pass', "umls",  "latin1")


#read the concept:cui dict
with open("map_concept_CUIs.txt", 'r',encoding='utf-8') as f:
    cui_dict = {}

  
    while True:
        line = f.readline()
        line = line.split()
        
        if len(line)==1:
            continue
        elif len(line)==2:
            cui_dict[line[0]] = [line[1]]
        elif len(line)==3:
            cui_dict[line[0]] = [line[1],line[2]]
        elif len(line)==4:
            cui_dict[line[0]] = [line[1],line[2],line[3]]
        else:
            break
            
            

f.close()


#write the semantic lexicon 
def write_semantic_lexicon_syn(cui_dict):
    
    print('writing semantic lexicon on : ' + 'synonys')

        with open("conceptExtraction_entityLinking/lexicon_UMLS_synonyms.txt", 'w',encoding="utf-8") as f:

            #some concepts can not have synonyms and in this case the next one (giving the lexicon) is tried
            for key,value in cui_dict.items():
                try:
                    synonyms_list = UMLS_CUI[value[0]].terms
                except:
                    try:
                        synonyms_list = UMLS_CUI[value[1]].terms
                    except:
                        try:
                            synonyms_list = UMLS_CUI[value[2]].terms
                        except:
                            print(key + ' not founded')
                            continue

                line = key + ' '


                for i in range(len(synonyms_list)):

                    limit = len(synonyms_list[i].split())
                    for tok in synonyms_list[i].split():



                        if limit==1:
                            line += tok

                        else:
                            line += tok + '_'
                            limit-=1
                    line+=' '
                line = line[:-1]
                line+='\n'

                try:
                    f.write(line)
                except:
                    print(line)
                    
            
      
    f.close()


write_semantic_lexicon_syn(cui_dict)