import torch
import torch.nn as nn
from torch_metrics import t_metrics, metric, yf_metric, rank, rank_2stage
from data_sampler import sample_case

def forward(seed, step, data_name, accelerator, writer, llm_model, llm_tokenizer, llm_max_length, device, data, dataloader, label_dict, yf_path, top_k_list, epoch, temp, bm25_hard_neg_dict, hard_neg_num, pos_aug, ran_aug, aug_edgedrop, aug_featmask_node, aug_featmask_edge, mask, query_list, query_index_list, train_flag, embedding_saving, llm_optimizer):
    ## Train
    if train_flag:
        loss_model = nn.CrossEntropyLoss()
        batched_case_list = dataloader

        case_txt_list = sample_case(seed, data_name, batched_case_list, label_dict, hard_neg_num, bm25_hard_neg_dict)
        print(len(case_txt_list))
        batch_dict = llm_tokenizer(case_txt_list,padding=True,truncation=True,max_length=llm_max_length,return_tensors="pt")
        batch_dict.to(device)
        outputs = llm_model(**batch_dict)
        case_embedding = outputs.last_hidden_state[:, -1] 

        query_output = case_embedding[:len(batched_case_list),:]
        query_norm = query_output / query_output.norm(dim=1)[:, None]

        ## pos
        pos_output = case_embedding[len(batched_case_list):len(batched_case_list)*2,:]
        pos_norm = pos_output / pos_output.norm(dim=1)[:, None]

        ## ran_neg
        ranneg_output = case_embedding[len(batched_case_list)*2:len(batched_case_list)*3,:]
        ranneg_norm = ranneg_output / ranneg_output.norm(dim=1)[:, None]

        ## bm25_neg
        if hard_neg_num != 0:
            bm25neg_output = case_embedding[len(batched_case_list)*2:len(batched_case_list)*3,:]
            bm25neg_norm = bm25neg_output / bm25neg_output.norm(dim=1)[:, None] 

        # positive logits: l_pos[batch_size, batch_size]:que x pos
        l_pos = torch.mm(query_norm, pos_norm.transpose(0,1))
        # negative logits: l_neg[batch_size, batch_size]: que x que 
        l_neg = torch.mm(query_norm, query_norm.transpose(0,1))
        ## diagonal is the dot product of query and itself
        l_neg.fill_diagonal_(float('-inf'))

        ## random negative logits: l_ran_neg[batch_size, batch_size]: que x ranneg
        l_ranneg = torch.mm(query_norm, ranneg_norm.transpose(0,1))   

        if hard_neg_num != 0:
            l_bm25neg = torch.mm(query_norm, bm25neg_norm.transpose(0,1)) 
            logits = torch.cat([l_pos, l_neg, l_ranneg, l_bm25neg], dim=1).to(device)
        else:    
            logits = torch.cat([l_pos, l_neg, l_ranneg], dim=1)
        
        logits_label_matrix = torch.arange(logits.size(0), device=logits.device)

        loss = loss_model(logits/temp, logits_label_matrix)
        

        max_norm = 1.0  # maximum L2 norm

        # Backward pass must be done first
        accelerator.backward(loss)

        # Optimizer step
        llm_optimizer.step()
        llm_optimizer.zero_grad()

        try:
            writer.add_scalar('Loss/Train', loss.item(), step)
        except:
            pass

                
    else:
        ## Test
        print('Test:')
        llm_model.eval()
        with torch.no_grad():
            test_label_list = data
            test_case_list = dataloader

            case_embedding_list = []
            for i in range(len(test_case_list)):        
                batch_dict = llm_tokenizer(test_case_list[i],padding=True,truncation=True,max_length=llm_max_length,return_tensors="pt",)
                batch_dict.to(device)
                outputs = llm_model(**batch_dict)
                case_embedding = outputs.last_hidden_state[:, -1].squeeze()
                case_embedding_list.append(case_embedding) 
            test_case_embedding = torch.stack(case_embedding_list)

            test_case_embedding_norm = test_case_embedding / test_case_embedding.norm(dim=1)[:, None]
            test_sim_score = torch.mm(test_case_embedding_norm, test_case_embedding_norm.T)
            test_sim_score.fill_diagonal_(float('-inf'))
            test_sim_score = test_sim_score[query_index_list]
                               
            ## Test loss
            nominator = torch.log((torch.exp(test_sim_score / temp) * ((mask == 1) + 1e-24)).mean(dim=1))
            denominator = torch.logsumexp(test_sim_score / temp, dim=1)
            loss = -(nominator - denominator)
            test_loss = loss.mean()
            print("Loss/Test:", test_loss)

            try:
                writer.add_scalar('Loss/Test', test_loss.item(), step)
            except:
                pass
            
            final_pre_dict = rank(test_sim_score, len(test_label_list), query_list, test_label_list)
            
            final_pre_dict_2stage = rank_2stage(final_pre_dict, top_k_list)

            correct_pred, retri_cases, relevant_cases, Micro_pre, Micro_recall, Micro_F, macro_pre, macro_recall, macro_F = metric(5, final_pre_dict, label_dict)
            ndcg_score, mrr_score, map_score, p_score = t_metrics(label_dict, final_pre_dict, 5)

            yf_dict, correct_pred_yf, retri_cases_yf, relevant_cases_yf, Micro_pre_yf, Micro_recall_yf, Micro_F_yf, macro_pre_yf, macro_recall_yf, macro_F_yf = yf_metric(5, yf_path, final_pre_dict, label_dict)                    
            ndcg_score_yf, mrr_score_yf, map_score_yf, p_score_yf = t_metrics(label_dict, yf_dict, 5)

            correct_pred_2stage, retri_cases_2stage, relevant_cases_2stage, Micro_pre_2stage, Micro_recall_2stage, Micro_F_2stage, macro_pre_2stage, macro_recall_2stage, macro_F_2stage = metric(5, final_pre_dict_2stage, label_dict)
            yf_dict_2stage, correct_pred_yf_2stage, retri_cases_yf_2stage, relevant_cases_yf_2stage, Micro_pre_yf_2stage, Micro_recall_yf_2stage, Micro_F_yf_2stage, macro_pre_yf_2stage, macro_recall_yf_2stage, macro_F_yf_2stage = yf_metric(5, yf_path, final_pre_dict_2stage, label_dict)

            ndcg_score_2stage, mrr_score_2stage, map_score_2stage, p_score_2stage = t_metrics(label_dict, final_pre_dict_2stage, 5)
            ndcg_score_yf_2stage, mrr_score_yf_2stage, map_score_yf_2stage, p_score_yf_2stage = t_metrics(label_dict, yf_dict_2stage, 5)
            
            print("Correct Predictions: ", correct_pred)
            print("Retrived Cases: ", retri_cases)
            print("Relevant Cases: ", relevant_cases)
            print("Micro Precision: ", Micro_pre)
            print("Micro Recall: ", Micro_recall)
            print("Micro F1: ", Micro_F)
            print("Macro F1: ", macro_F)
            print("NDCG@5: ", ndcg_score)
            print("MRR@5: ", mrr_score)
            print("MAP: ", map_score)

            print("Correct Predictions yf: ", correct_pred_yf)
            print("Retrived Cases yf: ", retri_cases_yf)
            print("Relevant Cases yf: ", relevant_cases_yf)
            print("Micro Precision yf: ", Micro_pre_yf)
            print("Micro Recall yf: ", Micro_recall_yf)
            print("Micro F1 yf: ", Micro_F_yf)
            print("Macro F1 yf: ", macro_F_yf)
            print("NDCG@5 yf: ", ndcg_score_yf)
            print("MRR@5 yf: ", mrr_score_yf)
            print("MAP yf: ", map_score_yf)
            
            print("2stage"+"\n")
            print("Correct Predictions 2stage: ", correct_pred_2stage)
            print("Retrived Cases 2stage: ", retri_cases_2stage)
            print("Relevant Cases 2stage: ", relevant_cases_2stage)            
            print("Micro Precision 2stage: ", Micro_pre_2stage)
            print("Micro Recall 2stage: ", Micro_recall_2stage)
            print("Micro F-Measure 2stage: ", Micro_F_2stage)
            print("Macro F-Measure 2stage: ", macro_F_2stage)           
            print("NDCG@5 2stage: ", ndcg_score_2stage)
            print("MRR@5 2stage: ", mrr_score_2stage)
            print("MAP 2stage: ", map_score_2stage)

            print("Correct Predictions_yf 2stage: ", str(correct_pred_yf_2stage))
            print("Retrived Cases_yf 2stage: ", str(retri_cases_yf_2stage))
            print("Relevant Cases_yf 2stage: ", str(relevant_cases_yf_2stage))           
            print("Micro Precision_yf 2stage: ", str(Micro_pre_yf_2stage))
            print("Micro Recall_yf 2stage: ", str(Micro_recall_yf_2stage))
            print("Micro F-Measure_yf 2stage: ", str(Micro_F_yf_2stage))
            print("Macro F-Measure_yf 2stage: ", str(macro_F_yf_2stage))
            print("NDCG_yf 2stage: ", str(ndcg_score_yf_2stage))
            print("MRR_yf 2stage: ", str(mrr_score_yf_2stage))
            print("MAP_yf 2stage: ", str(map_score_yf_2stage))

            if writer != None:
                writer.add_scalar("One stage/Correct num", correct_pred, epoch)        
                writer.add_scalar("One stage/Micro Precision", Micro_pre, epoch)
                writer.add_scalar("One stage/Micro Recall", Micro_recall, epoch)
                writer.add_scalar("One stage/Micro F1", Micro_F, epoch)
                writer.add_scalar("One stage/Macro F1", macro_F, epoch)
                writer.add_scalar("One stage/NDCG", ndcg_score, epoch)
                writer.add_scalar("One stage/MRR", mrr_score, epoch)
                writer.add_scalar("One stage/MAP", map_score, epoch)

                writer.add_scalar("One stage yf/Correct num yf", correct_pred_yf, epoch)        
                writer.add_scalar("One stage yf/Micro Precision yf", Micro_pre_yf, epoch)
                writer.add_scalar("One stage yf/Micro Recall yf", Micro_recall_yf, epoch)
                writer.add_scalar("One stage yf/Micro F1 yf", Micro_F_yf, epoch)
                writer.add_scalar("One stage yf/Macro F1 yf", macro_F_yf, epoch)
                writer.add_scalar("One stage yf/NDCG@5 yf", ndcg_score_yf, epoch)
                writer.add_scalar("One stage yf/MRR yf", mrr_score_yf, epoch)
                writer.add_scalar("One stage yf/MAP yf", map_score_yf, epoch)

                writer.add_scalar("Two stage/Correct num", correct_pred_2stage, epoch)        
                writer.add_scalar("Two stage/Micro Precision", Micro_pre_2stage, epoch)
                writer.add_scalar("Two stage/Micro Recall", Micro_recall_2stage, epoch)
                writer.add_scalar("Two stage/Micro F1", Micro_F_2stage, epoch)
                writer.add_scalar("Two stage/Macro F1", macro_F_2stage, epoch)
                writer.add_scalar("Two stage/NDCG", ndcg_score_2stage, epoch)
                writer.add_scalar("Two stage/MRR", mrr_score_2stage, epoch)
                writer.add_scalar("Two stage/MAP", map_score_2stage, epoch)

                writer.add_scalar("Two stage yf/Correct num", correct_pred_yf_2stage, epoch)        
                writer.add_scalar("Two stage yf/Micro Precision", Micro_pre_yf_2stage, epoch)
                writer.add_scalar("Two stage yf/Micro Recall", Micro_recall_yf_2stage, epoch)
                writer.add_scalar("Two stage yf/Micro F1", Micro_F_yf_2stage, epoch)
                writer.add_scalar("Two stage yf/Macro F1", macro_F_yf_2stage, epoch)
                writer.add_scalar("Two stage yf/NDCG", ndcg_score_yf_2stage, epoch)
                writer.add_scalar("Two stage yf/MRR", mrr_score_yf_2stage, epoch)
                writer.add_scalar("Two stage yf/MAP", map_score_yf_2stage, epoch)
    
    if train_flag == False:
        return correct_pred_yf

