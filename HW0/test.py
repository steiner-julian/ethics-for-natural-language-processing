from torch.utils.data import Dataset, DataLoader
import gensim.downloader
import torch

w2v = gensim.downloader.load('word2vec-google-news-300')

from gensim.parsing.preprocessing import (preprocess_string,
                                          strip_tags,
                                          strip_punctuation,
                                          strip_multiple_whitespaces,
                                          strip_numeric,
                                          remove_stopwords)


# We pick a subset of the default filters,
# in particular, we do not take
# strip_short() and stem_text().
FILTERS = [strip_punctuation,
           strip_tags,
           strip_multiple_whitespaces,
           strip_numeric,
           remove_stopwords]

# See how the sentece is transformed into tokes (words)
preprocess_string('This is a "short" text!', FILTERS)

import pandas as pd
#Load All_Beauty training, validation and testing split using read_csv from pandas library
#Write your code here
all_beauty_training_data = pd.read_csv(
    filepath_or_buffer='/home/steinerj/Documents/Technische Universität Darmstadt/Ethics for Natural Language Processing/ethics-for-natural-language-processing/HW0/data/All_Beauty_Split_train.csv'
)

all_beauty_test_data = pd.read_csv(
    filepath_or_buffer='/home/steinerj/Documents/Technische Universität Darmstadt/Ethics for Natural Language Processing/ethics-for-natural-language-processing/HW0/data/All_Beauty_Split_test.csv'
)

all_beauty_validation_data = pd.read_csv(
    filepath_or_buffer='/home/steinerj/Documents/Technische Universität Darmstadt/Ethics for Natural Language Processing/ethics-for-natural-language-processing/HW0/data/All_Beauty_Split_val.csv'
)

#Load Gift Card training, validation and testing split using read_csv from pandas library
#Write your code here
gift_cards_training_data = pd.read_csv(
    filepath_or_buffer='/home/steinerj/Documents/Technische Universität Darmstadt/Ethics for Natural Language Processing/ethics-for-natural-language-processing/HW0/data/Gift_Cards_Split_train.csv'
)

gift_cards_test_data = pd.read_csv(
    filepath_or_buffer='/home/steinerj/Documents/Technische Universität Darmstadt/Ethics for Natural Language Processing/ethics-for-natural-language-processing/HW0/data/Gift_Cards_Split_test.csv'
)

gift_cards_validation_data = pd.read_csv(
    filepath_or_buffer='/home/steinerj/Documents/Technische Universität Darmstadt/Ethics for Natural Language Processing/ethics-for-natural-language-processing/HW0/data/Gift_Cards_Split_val.csv'
)

#Concatenate the two splits to get a combined dataframe
#Write your code here
trainig_data = pd.concat([all_beauty_training_data, gift_cards_training_data], ignore_index=True)
test_data = pd.concat([all_beauty_test_data, gift_cards_test_data], ignore_index=True)
validation_data = pd.concat([all_beauty_validation_data, gift_cards_validation_data], ignore_index=True)

import numpy as np

#Creating dataset class
class reviews(Dataset):
    def __init__(self,split):
        super(reviews,self).__init__()

        if split == 'train':
        #Give train dataframe
            df_data = trainig_data
        elif split == 'val':
        #Give validation dataframe
            df_data = validation_data
        elif split == 'test':
        #Give test dataframe
            df_data = test_data
        else:
            raise Exception(f"wrong split: {split}")
        #List for containing (input,output) pairs
        self.data = []
        #Make a list of ratings from df_data
        ratings = df_data['Rating']
        #Make a list of reviews from df_data
        text = df_data['Text']

        for i in range(len(text)):
            #Creating list of tokens for each review
            txt = preprocess_string(text[i], filters=FILTERS)
            #Appending (input,output) pairs to list
            self.data.append([txt,ratings[i]])

    def vectorise(self,txt):
        #Final tensor
        X_tensor = torch.zeros(300)
        l=0
        for i in txt:
          #Check if the word is in the vocabulary of word2vec
          if i in w2v:
            l = l+1
            #If so add it to the X_tensor
            X_tensor = X_tensor + torch.tensor(w2v[i])
        #Divide X_tensor by number of vectors that were added. This is called mean pooling
        X_tensor = X_tensor / l if l > 0 else X_tensor
            
        return X_tensor
    
    def vectorise_test(self,txt):
        #Final tensor
        word_embeddings = []
        l=0
        for i in txt:
          #Check if the word is in the vocabulary of word2vec
          if i in w2v:
            l = l+1
            #If so add it to the X_tensor
            word_embeddings.append(w2v[i])
        
        embeddings_array = torch.tensor(word_embeddings)
        mean_pooled_embedding = torch.mean(embeddings_array, axis=0)
            
        return mean_pooled_embedding

    def __getitem__(self,index):
        sent_tensor = self.vectorise(self.data[index][0])
        test = self.vectorise_test(self.data[index][0])

        eq = torch.equal(sent_tensor, test)

        return [sent_tensor,(self.data[index][1]-1)]
    def __len__(self):
        return len(self.data)
    
import torch

target_classes = ["1", "2", "3", "4", "5"]

train_dataset = reviews("train")
val_dataset = reviews("val")
test_dataset = reviews("test")

train_loader = DataLoader(train_dataset, batch_size=128)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader  = DataLoader(test_dataset, batch_size=128)

for X, Y in train_loader:
    print(X.shape, Y.shape)