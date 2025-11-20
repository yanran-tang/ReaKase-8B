import os
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='2023', help="coliee2022 or coliee2023")

args = parser.parse_args()

def extract_judgment(text):
    if "\nJUDGMENT\n" in text:
        judgment_text = text.split("\nJUDGMENT\n", 1)[1].strip()
        return judgment_text
    elif "\nORDER\n" in text:
        judgment_text = text.split("\nORDER\n", 1)[1].strip()
        return judgment_text
    elif "Application allowed" in text:
        return "Application allowed"
    elif "Application dismissed" in text:
        return "Application dismissed"
    elif "Motion allowed" in text:
        return "Motion allowed"
    elif "Motion granted" in text:  
        return "Motion granted"
    elif "accordingly dismissed":
        return "accordingly dismissed"
    else:
        return "no judgment found"

def process_folder(folder_path):
    no_judgment_files = []
    longest = 0
    totoal_length = 0
    WDIR = folder_path.split('/processed/')[0] + '/judgment/' 
    os.mkdir(WDIR) if not os.path.exists(WDIR) else None
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        with open(os.path.join(WDIR, filename), 'w', encoding='utf-8') as out_f:
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    judgment = extract_judgment(text)
                    if "Editor:" in judgment:
                        judgment = judgment.split("Editor:", 1)[0].strip()
                    ###calculate the average tokens per judgments and the largest tokens of judgments of files in the folder
                    if judgment == "no judgment found":
                        no_judgment_files.append(filename)
                    else:
                        length = judgment.count(' ') + 1
                        totoal_length += length
                        if length > longest:
                            longest = length
                            longest_file = filename
                        out_f.write(f"{judgment}")
    # Write files with no judgment to a separate file
    print(f"Longest judgment length: {longest}")
    print(f"File with longest judgment: {longest_file}")
    print(f"average judgment length: {totoal_length/ (len(os.listdir(folder_path)) - len(no_judgment_files))}")
    print(f"No JUDGMENT files: {len(no_judgment_files)}")

if __name__ == "__main__":
    train_folder = './processed_files/'+args.data+'/train/processed/'
    test_folder = './processed_files/'+args.data+'/test/processed/'
    process_folder(train_folder)
    process_folder(test_folder)
