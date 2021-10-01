### library


```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from run_classifier_ import *
from sklearn.model_selection import train_test_split
```

### dataset
- preprocessed data path

```
data_path_1 = './dataset'
data_path_2 = './en_data'
data_path_3 = './ppr_data'
```

### config
- config_setting

```
bert_model = 'bert-base-multilingual-cased'
batch_size = 32
num_train_epochs = 5
learning_rate = 5e-5 
warmup_proportion = 0.1

max_seq_length = 128

output_dir = './output'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### model_class
- get config from init, and train_model
- data path, model_name, batch_size, epochs, learning_rate, max_seq_length, optimizer_config, saved_path, device(gpu) 

```
class bert_model:
    def __init__(path_, bert_model_, batch_size_, num_train_epochs_, learning_rate_, 
                 max_seq_length_, warmup_proportion_, output_dir_, device_):
        self.path = path_
        self.bert_model = bert_model_
        self.batch_size = batch_size_
        self.num_train_epochs = num_train_epochs_
        self.learning_rate = learning_rate_
        self.max_seq_length = max_seq_length_
        self.warmup_proportion = warmup_proportion_
        self.output_dir = output_dir_
        self.device = device_
    
    
    def get_dataexamples(self):
        train_path = os.path.join(self.path, 'train/train.csv')
        val_path = os.path.join(self.path, 'val/val.csv')
        
        train_ = pd.read_csv(train_path)
        val_ = pd.read_csv(val_path)

        train_examples_ = [InputExample('train', row.question, row.answer, row.intent) for row in train_.itertuples()]
        val_examples_ = [InputExample('val', row.question, row.answer, row.intent) for row in val_.itertuples()]
        
    
        return train_examples_, val_examples_, train_

    
    def get_model_tokenizer(self, lst):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        model = BertForSequenceClassification.from_pretrained(self.bert_model,
                      num_labels = len(lst))
        model.to(device)
    
        return model, tokenizer
    
    
    def get_optimizer(self, model_, train_examples_):
        param_optimizer = list(model_.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = int(len(train_examples_) / self.batch_size) * num_train_epochs
        optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=self.learning_rate,
                                     warmup=self.warmup_proportion,
                                     t_total=self.num_train_optimization_steps)
        
        return optimizer
    
    
    
    def get_data(self, data_examples_, label_list_, tokenizer):
        data_features = convert_examples_to_features(
            data_examples_, label_list_, self.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(data_examples_))
        logger.info("  Batch size = %d", self.batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in data_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in data_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in data_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in data_features], dtype=torch.long)
        data_ = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        data_sampler = SequentialSampler(data_)
        dataloader = DataLoader(data_, sampler=data_sampler, batch_size=self.batch_size)

        return data_, dataloader
    
    
    
    def save_model(self, model_):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save a trained model and the associated configuration
        model_to_save = model_.module if hasattr(model_, 'module') else model_  # Only save the model it-self
        output_model_file = os.path_join(self.output_dir, 'langual_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(self.output_dir, 'config.json')
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())


    
    def train_run(self):
        
        train_examples, val_examples, train_ = self.get_dataexamples(self.path)
        
        
        lst = [i for i in train_.intent.value_counts().index]
        
        model, tokenizer = self.get_model_tokenizer(lst)
        
        optimizer = self.get_optimizer(model, train_examples)
        
        train_data, train_dataloader = get_data(train_examples)

        val_data, val_dataloader = get_data(val_examples)


        early_stopping = EarlyStopping(patience = patience, verbose = False)


        for _ in trange(int(num_train_epochs), desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            total_step = len(train_data) // train_batch_size
            ten_percent_step = total_step // 10

            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if torch.cuda.device_count() > 1:
                    loss = loss.mean() 

                optimizer.zero_grad()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()


                if step % ten_percent_step == 0:
                    print("Fininshed: {:.2f}% ({}/{})".format(step/total_step*100, step, total_step))



            model.eval()

            val_loss, val_accuracy = 0, 0
            nb_val_steps, nb_val_examples = 0, 0

            for step, batch in enumerate(val_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.no_grad():
                    tmp_val_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask) 

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_val_accuracy = accuracy(logits, label_ids)

                val_loss += tmp_val_loss.mean().item()
                val_accuracy += tmp_val_accuracy

                nb_val_examples += input_ids.size(0)
                nb_val_steps += 1

            val_loss = val_loss / nb_val_steps
            val_accuracy = val_accuracy / nb_val_examples

            print("val_loss : ", loss)
            print("val_acc : ", val_accuracy)


            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        return model
```

### EarlyStopping
- earlyStopping during training when validation loss doesn't lower than before 

```
class EarlyStopping:
    def __init__(self, patience=2, verbose=False, delta=0, path='./output'):
        """
        Args:
            patience (int): patience time 
                            Default: 2
            verbose (bool): if True, print validation loss update message
                            Default: False
            delta (float): threshold of changed validation loss
                            Default: 0
            path (str): path of saving model
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''if validation loss is reduced, then save model'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
```

### load_model
- load pretrained model 

```
def load_model(dir_):
    # Load a trained model and config that you have fine-tuned
    config = BertConfig(output_config_file)
    model = BertForSequenceClassification(config, num_labels=118)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)  # important to specific device
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    return model
```

### inference_testset
- get inference result from each data (data path)

```
def get_testdata(path_):
    train_path = os.path.join(path_, 'train/train.csv')
    test_path = os.path.join(path_, 'test/test.csv')
    train_ = pd.read_csv(train_path)
    test_ = pd.read_csv(test_path)
    
    label_list = [i for i in train_.intent.value_counts().index]
    
    test_examples_ = [InputExample('test', row.question, row.answer, '배송_날짜_질문') for row in test_.itertuples()]
    sub_ = pd.read_csv(sub_path)
    
    
    return test_examples_, sub_, label_list

def predict(mo_path, data_path, bert_model, max_seq_length = 128, eval_batch_size=32):
    
    model = load_model(mo_path)
    
    test_examples_, sub_, label_list = get_testdata(data_path)
    
    model.to(device)
    eval_examples = test_examples_
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    res = []
    
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

    
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        
        
        
        res.extend(logits.argmax(-1))

        nb_eval_steps += 1


    return res, label_list
```

### average_ensemble
- get average_ensemble result using loaded model
- save result to csv file

```
logit_1, label_list = predict(model_path_1, data_path_1, 'bert-base-multilingual-cased', max_seq_length = 128, eval_batch_size=32)
logit_2, _ = predict(model_path_2, data_path_2, 'bert-base-multilingual-cased', max_seq_length = 128, eval_batch_size=32)
```


```
res = (logit_1 + logit_2)/2
```


```
cat_map = {idx:lab for idx, lab in enumerate(label_list)}
res = [cat_map[c] for c  in res]
```


```
sub_ = pd.read_csv(sub_path)
sub_['conv_num'] = test_['conv_num']
sub_['intent'] = res
sub_
```


```
sub_.to_csv('result_.csv',index=False)
```
