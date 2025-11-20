import torch
import os
from tqdm import tqdm
import random 

def sample_case(seed, data_name, batched_case_list, label_dict, hard_neg_num, bm25_hard_neg_dict):
    random.seed(seed)
    RDIR_sum ='./processed_files/'+data_name+'/train/summary_train_'+data_name+'_txt'
    RDIR_refer_sen = './processed_files/'+data_name+'/train/processed_new'
    RDIR_reasoning = './processed_files/'+data_name+'/train/reasoning'   
    RDIR_fact_ie_path = './processed_files/'+data_name+'/coliee'+data_name+'_ie/coliee'+data_name+'train_sum/result/'
    RDIR_issue_ie_path = './processed_files/'+data_name+'/coliee'+data_name+'_ie/coliee'+data_name+'train_refer/result/' 
    # ## Training

    query_case = []
    pos_case = []
    bm25neg_case = []
    ranneg_case = []

    for x in range(len(batched_case_list)):
        query_name = batched_case_list[x]
        with open(os.path.join(RDIR_sum, query_name), 'r') as f:
            original_sum_text = f.read()
            f.close()
        fact_text = "Legal facts:"+original_sum_text+' '
        with open(os.path.join(RDIR_refer_sen, query_name), 'r') as f:
            original_refer_text = f.read()
            f.close()
        issue_text = "Legal issues:"+original_refer_text 
        # query_promptcase_text = fact_text + issue_text         
        with open(RDIR_fact_ie_path+'named_entity_'+query_name.split('.')[0]+'.csv', "r") as f:
            fact_relation_triplets = f.readlines()
            f.close()
        fact_split_txt = ''
        for line in fact_relation_triplets:
            if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                continue             
            else:
                if '×' in line:
                    line = line.replace('×','')
                a_1 = line[:-1].split(',')
                fact_split_txt += a_1[1]+', '+a_1[2]+', '+a_1[4]+'/n '   
        with open(RDIR_issue_ie_path+'named_entity_'+query_name.split('.')[0]+'.csv', "r") as f:
            issue_relation_triplets = f.readlines()
            f.close()
        issue_split_txt = ''
        for line in issue_relation_triplets:
            if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                continue             
            else:
                if '×' in line:
                    line = line.replace('×','')
                a_1 = line[:-1].split(',')
                issue_split_txt += a_1[1]+', '+a_1[2]+', '+a_1[4]+'/n '  
        with open(os.path.join(RDIR_reasoning, query_name), 'r') as f:
            reasoning = f.read()
            f.close()
        # if reasoning == '':
        #     print('empty reasoning: ', query_name)
        reasoning_text = 'Legal reasoning: "'+reasoning+'". '
        case_txt = fact_text+'Legal fact relation triplets: "'+fact_split_txt+'", '+issue_text+'Legal issue relation triplets: "'+issue_split_txt+'", '+reasoning_text
        # tokenized_query = llm_tokenizer(query_promptcase_text, max_length=512, return_tensors='pt', padding=True, truncation=True).data
        
        query_case += [case_txt]
        
        ###########################################
        ## #pos
        pos_case_name = random.choice(label_dict[query_name])
        with open(os.path.join(RDIR_sum, pos_case_name), 'r') as f:
            original_sum_text = f.read()
            f.close()
        fact_text = "Legal facts:"+original_sum_text+' '
        with open(os.path.join(RDIR_refer_sen, pos_case_name), 'r') as f:
            original_refer_text = f.read()
            f.close()
        issue_text = "Legal issues:"+original_refer_text 
        # pos_promptcase_text = fact_text + issue_text  
        # case_txt = pos_promptcase_text             
        with open(RDIR_fact_ie_path+'named_entity_'+pos_case_name.split('.')[0]+'.csv', "r") as f:
            fact_relation_triplets = f.readlines()
            f.close()
        fact_split_txt = ''
        for line in fact_relation_triplets:
            if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                continue             
            else:
                if '×' in line:
                    line = line.replace('×','')
                a_1 = line[:-1].split(',')
                fact_split_txt += a_1[1]+', '+a_1[2]+', '+a_1[4]+'/n '   
        with open(RDIR_issue_ie_path+'named_entity_'+pos_case_name.split('.')[0]+'.csv', "r") as f:
            issue_relation_triplets = f.readlines()
            f.close()
        issue_split_txt = ''
        for line in issue_relation_triplets:
            if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                continue             
            else:
                if '×' in line:
                    line = line.replace('×','')
                a_1 = line[:-1].split(',')
                issue_split_txt += a_1[1]+', '+a_1[2]+', '+a_1[4]+'/n '   
        with open(os.path.join(RDIR_reasoning, pos_case_name), 'r') as f:
            reasoning = f.read()
            f.close()
        # if reasoning == '':
        #     print('empty reasoning: ', pos_case_name)
        reasoning_text = 'Legal reasoning: "'+reasoning+'". '
        case_txt = fact_text+'Legal fact relation triplets: "'+fact_split_txt+'", '+issue_text+'Legal issue relation triplets: "'+issue_split_txt+'", '+reasoning_text

        pos_case += [case_txt]

        ###########################################
        ## #ran neg
        i = 0
        while i<4400: 
            ran_neg_case = random.choice(os.listdir(RDIR_sum))
            if ran_neg_case not in label_dict[query_name]:
                with open(os.path.join(RDIR_sum, ran_neg_case), 'r') as f:
                    original_sum_text = f.read()
                    f.close()
                fact_text = "Legal facts:"+original_sum_text+' '
                with open(os.path.join(RDIR_refer_sen, ran_neg_case), 'r') as f:
                    original_refer_text = f.read()
                    f.close()
                issue_text = "Legal issues:"+original_refer_text 
                # ranneg_promptcase_text = fact_text + issue_text    
                # case_txt = ranneg_promptcase_text           
                with open(RDIR_fact_ie_path+'named_entity_'+ran_neg_case.split('.')[0]+'.csv', "r") as f:
                    fact_relation_triplets = f.readlines()
                    f.close()
                fact_split_txt = ''
                for line in fact_relation_triplets:
                    if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                        continue             
                    else:
                        if '×' in line:
                            line = line.replace('×','')
                        a_1 = line[:-1].split(',')
                        fact_split_txt += a_1[1]+', '+a_1[2]+', '+a_1[4]+'/n '   
                with open(RDIR_issue_ie_path+'named_entity_'+ran_neg_case.split('.')[0]+'.csv', "r") as f:
                    issue_relation_triplets = f.readlines()
                    f.close()
                issue_split_txt = ''
                for line in issue_relation_triplets:
                    if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                        continue             
                    else:
                        if '×' in line:
                            line = line.replace('×','')
                        a_1 = line[:-1].split(',')
                        issue_split_txt += a_1[1]+', '+a_1[2]+', '+a_1[4]+'/n '   
                with open(os.path.join(RDIR_reasoning, ran_neg_case), 'r') as f:
                    reasoning = f.read()
                    f.close()
                # if reasoning == '':
                #     print('empty reasoning: ', ran_neg_case)
                reasoning_text = 'Legal reasoning: "'+reasoning+'". '
                case_txt = fact_text+'Legal fact relation triplets: "'+fact_split_txt+'", '+issue_text+'Legal issue relation triplets: "'+issue_split_txt+'", '+reasoning_text
            ranneg_case += [case_txt]
            break
        else:
            continue
            # break

        if hard_neg_num != 0:
            for i in range(hard_neg_num):
                bm25_neg_case = random.choice(bm25_hard_neg_dict[query_name])
                with open(os.path.join(RDIR_sum, bm25_neg_case), 'r') as f:
                    original_sum_text = f.read()
                    f.close()
                fact_text = "Legal facts:"+original_sum_text+' '
                with open(os.path.join(RDIR_refer_sen, bm25_neg_case), 'r') as f:
                    original_refer_text = f.read()
                    f.close()
                issue_text = "Legal issues:"+original_refer_text 
                # bm25neg_promptcase_text = fact_text + issue_text    
                # case_txt = bm25neg_promptcase_text           
                with open(RDIR_fact_ie_path+'named_entity_'+bm25_neg_case.split('.')[0]+'.csv', "r") as f:
                    fact_relation_triplets = f.readlines()
                    f.close()
                fact_split_txt = ''
                for line in fact_relation_triplets:
                    if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                        continue             
                    else:
                        if '×' in line:
                            line = line.replace('×','')
                        a_1 = line[:-1].split(',')
                        fact_split_txt += a_1[1]+', '+a_1[2]+', '+a_1[4]+'/n '   
                with open(RDIR_issue_ie_path+'named_entity_'+bm25_neg_case.split('.')[0]+'.csv', "r") as f:
                    issue_relation_triplets = f.readlines()
                    f.close()
                issue_split_txt = ''
                for line in issue_relation_triplets:
                    if line == 'Type,Entity1,Relationship,Type,Entity2\n':
                        continue             
                    else:
                        if '×' in line:
                            line = line.replace('×','')
                        a_1 = line[:-1].split(',')
                        issue_split_txt += a_1[1]+', '+a_1[2]+', '+a_1[4]+'/n '   
                with open(os.path.join(RDIR_reasoning, bm25_neg_case), 'r') as f:
                    reasoning = f.read()
                    f.close()
                # if reasoning == '':
                #     print('empty reasoning: ', bm25_neg_case)
                reasoning_text = 'Legal reasoning: "'+reasoning+'". '
                case_txt = fact_text+'Legal fact relation triplets: "'+fact_split_txt+'", '+issue_text+'Legal issue relation triplets: "'+issue_split_txt+'", '+reasoning_text                       

                bm25neg_case += [case_txt]

    if hard_neg_num != 0:
        case_txt_list = query_case + pos_case + ranneg_case + bm25neg_case
    else:
        case_txt_list = query_case + pos_case + ranneg_case

    return case_txt_list
