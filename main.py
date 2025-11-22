import torch
import json
import os
from tqdm import tqdm
#from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# import vllm
# from vllm import LLM

# import dgl
# from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from peft import LoraConfig, get_peft_model, TaskType

from accelerate import Accelerator 
from accelerate.utils import ProjectConfiguration
# from llm2vec import LLM2Vec

# from data_load import SyntheticDataset, PoolDataset, collate
# from model_casegnn2plus import EUGATGNN, early_stopping

# from train import forward
from train import forward

from torch.utils.tensorboard import SummaryWriter
import time
import logging
import argparse
parser = argparse.ArgumentParser()
## model parameters
parser.add_argument("--in_dim", type=int, default=4096, help="Input_feature_dimension")
parser.add_argument("--h_dim", type=int, default=4096, help="Hidden_feature_dimension")
parser.add_argument("--out_dim", type=int, default=4096, help="Output_feature_dimension")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for embedding / GNN layer ")       
parser.add_argument("--num_head", default=1, type=int, help="Head number of GNN layer ")                            

## training parameters
parser.add_argument("--epoch", type=int, default=200, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-07, help="Learning rate")
parser.add_argument("--wd", default=1e-05, type=float, help="Weight decay if we apply some.")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
parser.add_argument("--temp", type=float, default=0.1, help="Temperature for relu")
parser.add_argument("--ran_neg_num", type=int, default=1, help="Random sampled case number")
parser.add_argument("--hard_neg_num", type=int, default=1, help="Bm25_neg case number")
parser.add_argument('--llm_max_length',type=int, default=2048)
parser.add_argument('--seed',type=str, default=42)

## other parameters
parser.add_argument("--data", type=str, default='2022', help="coliee2022 or coliee2023")

args = parser.parse_args()

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')
logging.warning(args)

def main():   
    torch.cuda.manual_seed_all(42)

    accelerator = Accelerator()
    device = accelerator.device

    ## Load LLM model and tokenizer
    llm_model_name = 'Qwen/Qwen3-Embedding-0.6B'
    base_model = AutoModel.from_pretrained(llm_model_name, token='YOUR_TOKEN', device_map=None, ).to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    ## LORA config
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # embeddings
        inference_mode=False,
        r=4,              # rank
        lora_alpha=32,    # scaling
        lora_dropout=0.1,  # dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 👈 must add this
    )
    
    llm_model = get_peft_model(base_model, peft_config)
    llm_model.print_trainable_parameters()  # sanity check
    
    
    if accelerator.is_main_process:
        log_dir = './training_log/ReaKase_seed'+str(args.seed)+'_llmmaxlength'+str(args.llm_max_length)+'_coliee'+args.data+'_bs'+str(args.batch_size)+'_dp'+str(args.dropout)+'_lr'+str(args.lr)+'_wd'+str(args.wd)+'_t'+str(args.temp)+'_headnum'+str(args.num_head)+'_hardneg'+str(args.hard_neg_num)+'_ranneg'+str(args.ran_neg_num)+'_'+time.strftime("%m%d-%H%M%S")
        print(log_dir)
        writer = SummaryWriter(log_dir)
        try:
            os.makedirs(log_dir)
        except Exception as e:
            print(f"Could not create {log_dir}, skipping. Reason: {e}")                    
    else:
        writer = None


    ## Load training label
    train_labels = {}
    with open('./label/task1_train_labels_'+args.data+'.json', 'r')as f:
        train_labels = json.load(f)
        f.close() 

    bm25_hard_neg_dict = {}
    with open('./label/hard_neg_top50_train_'+args.data+'.json', 'r')as file:
        for line in file.readlines():
            dic = json.loads(line)
            bm25_hard_neg_dict.update(dic)
        file.close() 

    ## Load test label
    test_labels = {}
    with open('./label/task1_test_labels_'+args.data+'.json', 'r')as f:
        test_labels = json.load(f)
        f.close()  

    with open('./label/BM25_coliee'+args.data+'_prediction_dict.json', 'r')as file:
        top_k_list = json.load(file)
        file.close() 

    training_data = list(train_labels.keys())
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    RDIR_sum ='./processed_files/'+args.data+'/test/summary_test_'+args.data+'_txt'
    RDIR_refer_sen = './processed_files/'+args.data+'/test/processed_new'
    RDIR_reasoning = './processed_files/'+args.data+'/test/reasoning'   
    RDIR_fact_ie_path = './processed_files/'+args.data+'/coliee'+args.data+'_ie/coliee'+args.data+'test_sum/result/'
    RDIR_issue_ie_path = './processed_files/'+args.data+'/coliee'+args.data+'_ie/coliee'+args.data+'test_refer/result/' 

    ## Wrap model + dataloader first
    llm_model, train_dataloader = accelerator.prepare(llm_model, train_dataloader)

    ## Create optimizer from the wrapped model's parameters
    llm_optimizer = torch.optim.AdamW(llm_model.parameters(), lr=args.lr)

    ## Wrap the optimizer too (prepare can be called again)
    llm_optimizer = accelerator.prepare(llm_optimizer)
    # register optimizer for checkpointing
    accelerator.register_for_checkpointing(llm_optimizer)  
    

    test_data = os.listdir(RDIR_sum)  

    test_mask = []
    test_query_list = []
    test_query_index_list = []
    for k,v in test_labels.items():
        test_mask_0 = []
        test_query_list.append(k)
        case_index = test_data.index(k)
        test_query_index_list.append(case_index)
        for i in range(len(test_data)):
            case = test_data[i]
            if case in v:
                test_mask_0.append(1)
            else:
                test_mask_0.append(0)  
        test_mask.append(torch.FloatTensor(test_mask_0))
    test_mask = torch.stack(test_mask).to(device)   

    yf_path = './label/test_'+args.data+'_candidate_with_yearfilter.json' 

    test_label_list = []
    test_case_list = []
    for pfile in tqdm(test_data[:]):
        query_name = pfile
        test_label_list.append(query_name)
        with open(os.path.join(RDIR_sum, query_name), 'r') as f:
            original_sum_text = f.read()
            f.close()
        fact_text = 'Legal facts: "'+original_sum_text+'", '
        with open(os.path.join(RDIR_refer_sen, query_name), 'r') as f:
            original_refer_text = f.read()
            f.close()
        issue_text = 'Legal issues: "'+original_refer_text+'", '  
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
        reasoning_text = 'Legal reasoning: "'+reasoning+'". '
        promptcase_text = fact_text+'Legal fact relation triplets: "'+fact_split_txt+'", '+issue_text+'Legal issue relation triplets: "'+issue_split_txt+'", '+reasoning_text
        test_case_list.append(promptcase_text)       
    
    ## save model params and optimizer states every SAVE_EVERY epochs
    SAVE_EVERY = 1  
    OUTPUT_DIR = './saved_model/ReaKase_seed'+str(args.seed)+'_llmmaxlength'+str(args.llm_max_length)+'_coliee'+args.data+'_bs'+str(args.batch_size)+'_dp'+str(args.dropout)+'_lr'+str(args.lr)+'_wd'+str(args.wd)+'_t'+str(args.temp)+'_headnum'+str(args.num_head)+'_hardneg'+str(args.hard_neg_num)+'_ranneg'+str(args.ran_neg_num)
    checkpoint_dir = os.path.join(OUTPUT_DIR, "last")
    if os.path.exists(checkpoint_dir):
        accelerator.print(f"Resuming from checkpoint {checkpoint_dir}")
        accelerator.load_state(checkpoint_dir)
    else:
        accelerator.print("No checkpoint found, training from scratch.")

    step = 0
    for epoch in tqdm(range(args.epoch)):
        print('epoch: ', str(epoch))
        llm_model.train()     
        
        for batched_case_list in tqdm(train_dataloader):
            step += 1
            forward(args.seed, step, args.data, accelerator, writer, llm_model, llm_tokenizer, args.llm_max_length, device, training_data, batched_case_list, train_labels, yf_path, top_k_list, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg_num, mask=None, query_list=None, query_index_list=None, train_flag=True, embedding_saving=False, llm_optimizer=llm_optimizer)        

        llm_model.eval()
        with torch.no_grad():                       
            ndcg = forward(args.seed, step, args.data, accelerator, writer, llm_model, llm_tokenizer, args.llm_max_length, device, test_data, test_case_list, test_labels, yf_path, top_k_list, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg_num, mask=test_mask, query_list=test_query_list, query_index_list=test_query_index_list, train_flag=False, embedding_saving=False, llm_optimizer=llm_optimizer)

        if epoch % SAVE_EVERY == 0:
            accelerator.print(f"Saving checkpoint at epoch {epoch}")
            accelerator.save_state(os.path.join(OUTPUT_DIR, "last"))
            accelerator.save_state(os.path.join(OUTPUT_DIR, f"epoch_{epoch}"))
            accelerator.save_state(os.path.join(OUTPUT_DIR, "last"))

if __name__ == '__main__':
    main()
