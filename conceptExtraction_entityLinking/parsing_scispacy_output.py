import os.path
base_path = 'scispacy_pipeline_output/'


#function to parse the scipacy output(the text files produced)
def parse_output(n):
    
    cui_dict = {}
    for j in range(0,n):
        #select a patient text file path
        pat_path = base_path +   "patient" +str(j) + ".txt"
        
        if os.path.isfile(pat_path):
            
            with open(pat_path, 'r') as f:
                document=f.read().split('\n')
            f.close()
            
            #the output from scispacy follow a precise order (lines):
            # -concept name 
            # -concept CUI  
            # -concept score for the entity linking 
            # -concept definition
            # -concept TUI(s)
            # -concept aliases

            #We are interested in the concept name and CUI (first and second line), that's the motivation for the counters count_concept_name and count_concept_cui.
            #when we find a first or a second line for a concept we keep the lines, otherwise we skip to the next concept output (+8 each counter)

            count_concept_name = 0 
            count_concept_cui = 1
            cui_list = []
            current_concept_name=document[0]
            
            for i in range(len(document)): #extracting the name of the entity
                
                        if i ==count_concept_name:
                            if len(document[i].split())==1 :    #only save the mention with one word 
                                concept_name=document[i]
                                count_concept_name+=8
                                
                                if concept_name!=current_concept_name:
                                        current_concept_name=concept_name
                                        cui_list = []
                            else:
                                count_concept_name+=8
                                count_concept_cui+=8
                                
                        elif i == count_concept_cui: #extracting the cui of the entity
                            cui=document[i]
                            count_concept_cui+=8
         
                            cui_list.append(cui)
                            
            
                        if concept_name in cui_dict.keys():
                            continue
                        else:
                            cui_dict[concept_name] = cui_list
     
    return cui_dict

#parse the output giving a number of patient --> 100000 is used to indicate a large number of patients and to create the path in line 11
#the if in line 13 check if the number is valid (if a patient with the current number exists) 
cui_dict=parse_output(100000)


#to create the concept:CUIs dictionary. 
#Note that the output is preprocessed in order to eliminate symbols (/ \ ' ^ * = ~) contained in their names

with open("map_concept_CUIs.txt", 'w',encoding='utf-8') as f:
    for key,value in cui_dict.items():
     
        if key.find("\\") == -1 and key.find("/") == -1 and key.find("'") == -1 and key.find("=") == -1 and key.find("*") == -1 and  key.find("~")==-1 and key.find("^")== -1 :   #eliminate the concept with \ or more \ in the name for preprocessing
    
            if len(value)==1:
                CUIs = value[0]
                f.write(key + ' ' + CUIs + '\n')
            elif len(value)==2:
                CUIs = value[0] + ' ' + value[1]
                f.write(key + ' ' + CUIs + '\n')
            elif len(value)>2:
                CUIs = value[0] + ' ' + value[1] + ' ' + value[2]
                f.write(key + ' ' + CUIs + '\n')
            else:
                print(key)
        else:
            continue
            
f.close()
        
        