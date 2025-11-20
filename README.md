# ReaKase-8B
Official Repository for [ReaKase-8B: Legal Case Retrieval via Knowledge and Reasoning Representations with LLMs](https://arxiv.org/abs/2510.26178), accepted at ADC 2025.

Author: Yanran Tang, Ruihong Qiu, Xue Li, and Zi Huang

![Alt text](images/ReaKase.png)

# Installation
Requirements can be seen in `/requirements.txt`

# Dataset
Datasets can be downloaded from [COLIEE2022](https://sites.ualberta.ca/~rabelo/COLIEE2022/) and [COLIEE2023](https://sites.ualberta.ca/~rabelo/COLIEE2023/). 

The label file `task1_train_labels_2022.json` and `task1_test_labels_2022.json` shoule be put into folder `/label/`, also for COLIEE2023. 

The final project files are as follows:

    ```
    $ ./ReaKase-8B/
    .
    ├── images
    ├── label 
    │   ├── BM25_coliee2022_prediction_dict.json
    │   ├── BM25_coliee2023_prediction_dict.json
    │   ├── hard_neg_top50_train_2022.json
    │   ├── hard_neg_top50_train_2023.json
    │   ├── task1_test_labels_2022.json            
    │   ├── task1_test_labels_2023.json 
    │   ├── task1_train_labels_2022.json 
    │   ├── task1_train_labels_2023.json 
    │   ├── test_2022_candidate_with_yearfilter.json
    │   └── test_2023_candidate_with_yearfilter.json  
    ├── legal_element_extraction
    │   ├── judge_extract.py
    │   └── reasoning_generate.py   
    ├── processed_files 
    ├── run.sh
    ├── main.py
    ├── train.py
    ├── torch_metrics.py
    ├── requirements.txt
    └── README.md          
    ```

# Data Preparation
## 1. Legal Element Extraction
- (a) Legal facts and legal issues.

    Legal facts and legal issues are extracted as in PromptCase, please see the preprocessing section of [PromptCase github repo](https://github.com/yanran-tang/PromptCase?tab=readme-ov-file#preprocessing).
    
    The processed files of `/processed/`, `/processed_new/` and `/summary_test(train)_2022_txt/` can be downloaded here [2022](https://drive.google.com/drive/folders/1vNLUBuw5yfguCoGFMqmTvdq9miMTv3xW?usp=sharing), [2023](https://drive.google.com/drive/folders/1e5CrKWnH0oMTh1DrOk7pvMNiHV1pmxLJ?usp=sharing).

- (b) Legal judgements

    `python judge_extract.py`

    The processed files of `/judgment/` can be downloaded here [2022](https://drive.google.com/drive/folders/1vNLUBuw5yfguCoGFMqmTvdq9miMTv3xW?usp=sharing), [2023](https://drive.google.com/drive/folders/1e5CrKWnH0oMTh1DrOk7pvMNiHV1pmxLJ?usp=sharing).

## 2. Legal Relation Triplets

  - Legal relation triplets are extracted as in CaseGNN, please see the Relation Extraction section of [CaseGNN github repo](https://github.com/yanran-tang/CaseGNN#1-information-extraction).
  - The extracted legal relation triplets can be also downloaded [here](https://drive.google.com/drive/folders/1Ck1KecF28xqsjDZK1fqVGF3BozmSsAb7?usp=sharing).


## 3. Legal Reasoning Generation

  `python reasoning_gen.py`

  The processed files of `/reasoning/` can be downloaded here [2022](https://drive.google.com/drive/folders/1vNLUBuw5yfguCoGFMqmTvdq9miMTv3xW?usp=sharing), [2023](https://drive.google.com/drive/folders/1e5CrKWnH0oMTh1DrOk7pvMNiHV1pmxLJ?usp=sharing).

## 4. Processed files
After all the data processing steps, the folders in processed_files are as follows:
    
    processed_files
    ├── 2022
    │   ├── coliee2022_ie  
    │   ├── test 
    │   │   ├── judgment 
    │   │   ├── processed 
    │   │   ├── processed_now 
    │   │   ├── reasoning 
    │   │   └── summary_test_2022_txt 
    │   └── train  
    │   │   ├── judgment 
    │   │   ├── processed 
    │   │   ├── processed_now 
    │   │   ├── reasoning 
    │   │   └── summary_train_2022_txt  
    └── 2023 
        ├── coliee2023_ie  
        ├── test 
        │   ├── judgment 
        │   ├── processed 
        │   ├── processed_now 
        │   ├── reasoning 
        │   └── summary_test_2023_txt 
        └── train  
            ├── judgment 
            ├── processed 
            ├── processed_now 
            ├── reasoning 
            └── summary_train_2023_txt 

* Make sure that all the files are generated or downloaded before running models.  

# Model Training
## 1. ReaKase-8b Model Training

  `python main.py`

# Cite
If you find this repo useful, please cite
```
@article{ReaKase-8B,
  author       = {Yanran Tang and
                  Ruihong Qiu and
                  Xue Li and
                  Zi Huang},
  title        = {ReaKase-8B: Legal Case Retrieval via Knowledge and Reasoning Representations with LLMs},
  journal      = {CoRR},
  volume       = {abs/2510.26178},
  year         = {2025}
}
```
