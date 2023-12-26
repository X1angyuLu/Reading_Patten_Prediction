import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def clean_word(word):
    cleaned_word = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', word)
    if not cleaned_word.isalnum():
        return word
    return cleaned_word

class dataset(DataLoader):
    def __init__(self, file_path, language):
        data = pd.read_csv(file_path)
        self.data = data[data['language'] == language]
        self.sentence_ids = self.data['sentence_id'].unique().tolist()


    def __len__(self) -> int:
        return len(self.sentence_ids)
    
    def __getitem__(self, item):
        sentence = self.data[self.data['sentence_id'] == self.sentence_ids[item]]['word'].astype(str).tolist()
        sentence = [clean_word(word) for word in sentence]
        FFDAvg = np.array(self.data[self.data['sentence_id'] == self.sentence_ids[item]]['FFDAvg'])
        FFDStd = np.array(self.data[self.data['sentence_id'] == self.sentence_ids[item]]['FFDStd'])
        TRTAvg = np.array(self.data[self.data['sentence_id'] == self.sentence_ids[item]]['TRTAvg'])
        TRTStd = np.array(self.data[self.data['sentence_id'] == self.sentence_ids[item]]['TRTStd'])

        return (sentence,FFDAvg, FFDStd,TRTAvg,TRTStd)