import torch
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


label_names = ['O', 'B-MethodName', 'I-MethodName', 'B-HyperparameterName','I-HyperparameterName', 'B-HyperparameterValue','I-HyperparameterValue','B-MetricName','I-MetricName','B-MetricValue','I-MetricValue','B-TaskName','I-TaskName','B-DatasetName','I-DatasetName']
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

# find locations where indices change
def find_change_indices(numbers):
    change_indices = [0]  # Initialize with 0 since the first element is always considered a change
    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1]:
            change_indices.append(i)
    return change_indices


def predictions():
    paragraphs = []
    temp_token = []
    temp_ids = []
    device = 'cuda'
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              add_prefix_space=True)

    test_df = pd.read_csv(args.test_csv)
    for i, row in test_df.iterrows():
        if row['input'] != row['input']:
            paragraphs.append({'tokens':temp_token.copy(), 'ids':temp_ids})
            temp_token = []
            temp_ids = []
        else:
            temp_token.append(row['input'])
            temp_ids.append(row['id'])
    paragraphs.append({'tokens':temp_token.copy(), 'ids':temp_ids})

    #make predictions, TODO huge potential for stuff like beam search and rule based decoding here
    for para in paragraphs:
        tokenized_inputs = tokenizer(
            para["tokens"], truncation=True, is_split_into_words=True, max_length=512
        )
        cindex = find_change_indices(tokenized_inputs.word_ids()[1:-1])
        outputs = model(input_ids=torch.LongTensor(tokenized_inputs['input_ids']).unsqueeze(0).to(device),
            attention_mask=torch.LongTensor(tokenized_inputs['attention_mask']).unsqueeze(0).to(device),
            # token_type_ids=torch.LongTensor(tokenized_inputs['token_type_ids']).unsqueeze(0).to(device)
            )   
        preds = torch.argmax(outputs.logits[:,1:-1,:][:,cindex,],dim=-1).squeeze(0).cpu()
        pred_labels = [id2label[val.item()] for val in preds]
        if len(para['tokens'])!=len(pred_labels):
            print("found truncation")
            print(para['tokens'])
            print("original length", len(para['tokens']))
            print("other length", len(pred_labels))
            pred_labels+=[id2label[0]]*(len(para['tokens']) - len(pred_labels))
        para['pred_labels'] = pred_labels.copy()

    # put ids and pred in a list
    id_list = []
    label_list = []
    for val in paragraphs:
        id_list += val['ids']
        label_list += val['pred_labels']
    assert len(id_list) == len(label_list)
    output_df = pd.DataFrame({'id':id_list,'labels':label_list})
    # left join on test data
    result_df = test_df.merge(output_df, on='id', how='left')
    result_df.drop(columns=['input', 'target'],inplace=True)
    result_df.rename({'labels':'target'}, axis=1,inplace=True)
    result_df.fillna('X', axis=1, inplace=True)
    result_df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='roberta-large')
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)


    args = parser.parse_args()
    predictions()
