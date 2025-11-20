import os
from tqdm import tqdm
from openai import OpenAI
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='2023', help="coliee2022 or coliee2023")
parser.add_argument("--data _type", type=str, default='train', help="train or test")

args = parser.parse_args()

## insert the openai api key
client = OpenAI(api_key="YOUR_API_KEY") 

RDIR_sum ='./processed_files/'+args.data+'/'+args.data_type+'/summary_'+args.data_type+'_'+args.data+'_txt'
RDIR_refer_sen = './processed_files/'+args.data+'/'+args.data_type+'/processed_new' 
RDIR_judgment = './processed_files/'+args.data+'/'+args.data_type+'/judgment' 
WDIR = './processed_files/'+args.data+'/'+args.data_type+'/reasoning' 

files = os.listdir(RDIR_sum)
for pfile in tqdm(files[:]):
    if os.path.exists(os.path.join(WDIR, pfile)):
        # print(pfile, 'already exists')
        pass
    else:
        # print(pfile, 'does not exist')
        with open(os.path.join(RDIR_sum, pfile), 'r') as f:
            original_sum_text = f.read()
            f.close()
        fact_text = 'legal facts: "'+original_sum_text+'", '
        with open(os.path.join(RDIR_refer_sen, pfile), 'r') as f:
            original_refer_text = f.read()
            f.close()
        issue_text = 'legal issues:"'+original_refer_text+'", ' 
        with open(os.path.join(RDIR_judgment, pfile), 'r') as f:
            judgment = f.read()
            f.close()
        # Loop through each line in the file
        response = client.responses.create(
            model="gpt-5",
            input='Given a case with its '+fact_text+'and '+issue_text+'and the final case judgment: "'+judgment+'", please explain how to deduce the final judgment from both legal facts and legal issues in 100 tokens.'
        )
        reasoning_text = response.output_text
        with open(os.path.join(WDIR, pfile), 'w') as file:
            file.write(reasoning_text)
            file.close()
print('finish')