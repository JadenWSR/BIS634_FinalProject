import torch
import re
import numpy as np
from transformers import BertTokenizer#version 4.0.1
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels = 3,   
        output_attentions = False,
        output_hidden_states = False,
    )
    #load the pretrained model
    state_dict = torch.load('checkpoint.pth',  map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model,tokenizer
    
def predict_sentiment(model,tokenizer,user_text):
    device = torch.device("cpu")
    #model.eval()
    input_ids = []
    attention_masks = []
    batch_size = 1

    # Data cleaning
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    user_text = url_pattern.sub(r'', user_text)
    user_text = re.sub('\S*@\S*\s?', '', user_text)
    user_text = re.sub('\s+', ' ', user_text)
    user_text = re.sub("\'", "", user_text)
    user_text = re.sub("#", "", user_text)

    encoded_dict = tokenizer.encode_plus(
                        user_text,                      
                        add_special_tokens = True, 
                        max_length = 110,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )
     
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    user_data = TensorDataset(input_ids, attention_masks,torch.tensor([0]))
    dataloader = DataLoader(
            user_data,  
            sampler = RandomSampler(user_data),     #random sampling in training
            batch_size = batch_size 
        )

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():

            result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            return_dict=True)

        logits = result.logits

        logits = logits.numpy()

        labels = np.argmax(logits, axis=1).flatten()
        prob = np.exp(logits)/(np.exp(logits).sum())

    return labels[0],prob[0]

def get_sentiment(prediction,prob):
    l = {0:"negative", 1:"neutral",2:"positive"}
    label = l[prediction]
    prob = list(zip(l.values(),list(prob)))
    return label,prob[0],prob[1],prob[2]

if __name__ == "__main__":
    model,tokenizer = load_model()
    user_text = "I hate Pfizer vaccine"
    prediction,prob = predict_sentiment(model,tokenizer,user_text)
    print(get_sentiment(prediction,prob))