import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

class FFD(nn.Module):
    def __init__(self, hidden_size=1024,padding='do_not_pad', dropout=0.1, language = 'en'):
        super(FFD,self).__init__()
        if language == 'zh':
            self.berttokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert = BertModel.from_pretrained('bert-base-chinese')
        # elif language == 'hi':
        #     berttokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        #     bert = BertModel.from_pretrained('bert-base-chinese')
        elif language == 'en':
            self.berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.berttokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.padding = padding
        # self.berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.FFDN = nn.Sequential(nn.Linear(self.bert.config.hidden_size*2, hidden_size*2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size*2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1)
        )

        # Freeze the BERT part
        for param in self.bert.parameters():
            param.requires_grad = False

    def tokenize(self, text):
        input_ids = self.berttokenizer.encode(text, add_special_tokens=True, padding=self.padding, truncation=True, max_length=256)
        attention_mask = [int(id > 0) for id in input_ids]

        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

        return input_ids, attention_mask

    def forward(self, text):
        input_ids, attention_mask = self.tokenize(text)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:,1:-1,:]

        # FFD
        FFD_hidden_state = torch.cat((torch.zeros((1,hidden_state.shape[2])).to(device), hidden_state[:,0]), dim=1).unsqueeze(1)
        for i in range(1, hidden_state.shape[1]):
            ffd = hidden_state[:,0:i].mean(dim=1)
            ffd = torch.cat((ffd,hidden_state[:,i]), dim=1).unsqueeze(1)
            FFD_hidden_state = torch.cat((FFD_hidden_state,ffd), dim=1)
        FFD = self.FFDN(FFD_hidden_state) 

        return FFD
    

class TRT(nn.Module):
    def __init__(self, hidden_size=1024,padding='do_not_pad', dropout=0.1, language='en'):
        super(TRT,self).__init__()
        if language == 'zh':
            self.berttokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert = BertModel.from_pretrained('bert-base-chinese')
        # elif language == 'hi':
        #     berttokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        #     bert = BertModel.from_pretrained('bert-base-chinese')
        elif language == 'en':
            self.berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.berttokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.padding = padding
        # self.berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.TRTN = nn.Sequential(nn.Linear(self.bert.config.hidden_size*3, hidden_size*2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size*2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1)
        )

        # Freeze the BERT part
        for param in self.bert.parameters():
            param.requires_grad = False

    def tokenize(self, text):
        input_ids = self.berttokenizer.encode(text, add_special_tokens=True, padding=self.padding, truncation=True, max_length=256)
        attention_mask = [int(id > 0) for id in input_ids]

        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

        return input_ids, attention_mask

    def forward(self, text):
        input_ids, attention_mask = self.tokenize(text)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:,1:-1,:]

        # TRT
        TRT_hidden_state = torch.cat((torch.zeros((1,hidden_state.shape[2])).to(device), hidden_state[:,0], hidden_state[:,1:].mean(dim=1)), dim=1).unsqueeze(1)
        for i in range(1, hidden_state.shape[1]-1):
            trt = torch.cat((hidden_state[:,0:i].mean(dim=1),hidden_state[:,i],hidden_state[:,i+1:].mean(dim=1)), dim=1).unsqueeze(1)
            TRT_hidden_state = torch.cat((TRT_hidden_state,trt), dim=1)
        trt = torch.cat((hidden_state[:,0:-1].mean(dim=1),hidden_state[:,-1],torch.zeros((1,hidden_state.shape[2])).to(device)), dim=1).unsqueeze(1)
        TRT_hidden_state = torch.cat((TRT_hidden_state,trt), dim=1)

        TRT = self.TRTN(TRT_hidden_state)

        return TRT