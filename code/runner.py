import os
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from torch.optim import AdamW
import sklearn
from seqeval.metrics import classification_report
from collections import Counter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from transformers import get_scheduler
import argparse



label_names = ['O', 'B-MethodName', 'I-MethodName', 'B-HyperparameterName','I-HyperparameterName', 'B-HyperparameterValue','I-HyperparameterValue','B-MetricName','I-MetricName','B-MetricValue','I-MetricValue','B-TaskName','I-TaskName','B-DatasetName','I-DatasetName']
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

def align_labels_with_tokens(labels, word_ids):
    label_all_tokens = True
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx == 0:
            label_ids.append(0)
        elif word_idx != previous_word_idx:
            label_ids.append(labels[word_idx])
        else:
            label_ids.append(labels[word_idx] if label_all_tokens else -100)
        previous_word_idx = word_idx
    # label_ids.append(label_ids)
    return label_ids

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

def parse_file(file_name, split_size=128):

    with open(file_name, 'r') as f:
        raw_file = f.read()
    paragraphs = raw_file.replace('\n\n','\n')
    data = []
    tokens = paragraphs.split('\n')
    for i in range(0, len(tokens), split_size):
        data.append(tokens[i:i+split_size])
    raw_data = []
    for i in range(len(data)):
        tokens = []
        ner_tags = []
        named_tags=[]
        for line in data[i]:
            line = line.split(' ')
            if len(line) <2:
                continue
            try:
                ner_tags.append(label2id[line[-1]])
                tokens.append(line[0])
                named_tags.append(line[-1])
            except:
                print(line)
            
        raw_data.append({'tokens':tokens,'ner_tags': ner_tags, 'named_tags':named_tags})
    return raw_data

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    all_labels = examples["ner_tags"]

    tokenized_inputs["labels"] = align_labels_with_tokens(all_labels, tokenized_inputs.word_ids())
    return tokenized_inputs

def train(args):
    model_checkpoint = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                            add_prefix_space=True,
                                          )
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )
    raw_train_data = []
    train_files = [val for val in os.listdir(args.train_dir) if val.endswith('.conll')]
    for file_name in train_files:
        raw_train_data += parse_file(file_name, args.seq_len)

    tokenized_train_dataset = [tokenize_and_align_labels(tokenizer, val) for val in raw_train_data]
    train_tokens , val_tokens = train_test_split(tokenized_train_dataset, test_size=args.val_split, 
                                                 random_state=args.seed)   

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_tokens,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=4,
    )
    eval_dataloader = DataLoader(
        val_tokens, collate_fn=data_collator, batch_size=args.batch_size
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()


    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader, criterion = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, criterion
    )


    num_train_epochs = args.epochs
    output_dir = args.output_dir
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup,
        num_training_steps=num_training_steps,
    )
    for param in model.parameters():
        param.requires_grad = True

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        print("Epoch:", epoch)
        model.train()
        for batch in train_dataloader:
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            # token_type_ids=batch['token_type_ids']
                            )
            loss = criterion(outputs.logits.permute(0,2,1), batch['labels'])
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            all_preds += true_predictions
            all_labels += true_labels
            # metric.add_batch(predictions=true_predictions, references=true_labels)

        print(classification_report(all_labels, all_preds))

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

def test(args):
    model_checkpoint = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                            add_prefix_space=True,
                                          )
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )
    raw_test_data = []
    test_files = [val for val in os.listdir(args.test_dir) if val.endswith('.conll')]
    for file_name in test_files:
        raw_test_data += parse_file(file_name, args.seq_len)

    tokenized_test_dataset = [tokenize_and_align_labels(tokenizer, val) for val in raw_test_data]
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    test_dataloader = DataLoader(
        tokenized_test_dataset, collate_fn=data_collator, batch_size=args.batch_size
    )


    accelerator = Accelerator()

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    model.eval()
    all_preds = []
    all_labels = []

    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            all_preds += true_predictions
            all_labels += true_labels

        print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    #args for output dir, lr, batch size, epochs, model name, train-dir, test-dir,
    # lr decay strategy

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='roberta-large')
    parser.add_argument('--train_dir', type=str, default='./train')
    parser.add_argument('--test_dir', type=str, default='./test')
    parser.add_argument('--lr_warmup', type=int, default=0)
    parser.add_argument('--lr_scheduler', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.2)

    args = parser.parse_args()
    if args.train:
        train(args)
    else:
        test(args)
