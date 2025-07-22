import torch
from kobert_transformers import get_kobert_model, get_tokenizer
import pandas as pd
from tqdm import tqdm
import csv
import os
from config import config

args = config["nikl"]  # Change to "nikl" for the other dataset

def str_to_score_list(str_list):
    score_list = [float(st) for st in str_list.split('#')]
    return score_list

def get_dataset():
    train = pd.read_csv(args['train_dataset_path'],encoding='utf-8-sig')
    valid = pd.read_csv(args['test_dataset_path'],encoding='utf-8-sig')

    if args['train_dataset_path'].startswith('./nikl'):
        train_essays = [essay.split(args['sentence_sep']) for essay in train[args['essay_key']].to_list()]
        valid_essays = [essay.split(args['sentence_sep']) for essay in valid[args['essay_key']].to_list()]
        if args['is_topic_label'] == True:
            for i in range(len(train_essays)):
                train_essay = train_essays[i]
                labled_essay = [train.iloc[i][args['prompt_key']] +'###'+ sent for sent in train_essay]
                train_essays[i] = labled_essay
            for i in range(len(valid_essays)):
                valid_essay = valid_essays[i]
                labled_essay = [valid.iloc[i][args['prompt_key']] + '###' + sent for sent in valid_essay]
                valid_essays[i] = labled_essay
        rubric_names = [f"evaluator{args['evaluator']}_score_{rub}"for rub in args['rubric']]
        return train_essays, valid_essays, train[rubric_names].to_numpy()/len(args['num_range']), valid[rubric_names].to_numpy()/len(args['num_range'])

    elif args['train_dataset_path'].startswith('./aihub_v1'):
        train_scores = train['essay_score_avg'].to_list()
        valid_scores = valid['essay_score_avg'].to_list()

        return [essay.split(args['sentence_sep']) for essay in train['essay'].to_list()] , \
        [essay.split(args['sentence_sep']) for essay in valid['essay'].to_list()],\
        [[round(x/3,2) for x in str_to_score_list(train_score)] for train_score in train_scores], \
        [[round(x/3,2) for x in str_to_score_list(valid_score)] for valid_score in valid_scores]

def embedding(model,tokenizer,essays, is_train = True):
    ff = open(os.path.join(args['emb_file_path'], f"{args['train_dataset_path'].split('/')[1]}_{'train' if is_train else 'valid'}_{'notlabeled' if args['is_topic_label'] == False else 'labeled'}.csv"), 'w', newline='')
    writer_ff = csv.writer(ff)
    for i in tqdm(range(len(essays))):
        inputs = tokenizer.batch_encode_plus(essays[i],max_length=args['max_length'], padding='max_length', truncation=True)
        input_ids = torch.tensor(inputs['input_ids']).cuda()
        attention_mask = torch.tensor(inputs['attention_mask']).cuda()
        out = model(input_ids=input_ids,attention_mask=attention_mask)
        embedded_features = out[0].detach().cpu()[:, 0, :].numpy()
        for i in embedded_features:
            writer_ff.writerow(i)
        torch.cuda.empty_cache()
    ff.close()
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_kobert_model().to(device)
    tokenizer = get_tokenizer()
    train, valid, _, _ = get_dataset()
    embedding(model,tokenizer,train, is_train=True)
    embedding(model,tokenizer,valid, is_train=False)
