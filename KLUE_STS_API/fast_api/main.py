import torch
from PreOnboarding_AI.model import CustomRegressor
from fastapi import FastAPI
from pydantic import BaseModel
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel

bert_embedding_model = BertModel.from_pretrained('monologg/kobert')
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

app = FastAPI()

if torch.cuda .is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

ckpt = torch.load(('model_4_3e-05_16_fold4.ckpt.epoch3'), map_location = device)

def some_func(s1, s2):
    return '[CLS] ' + s1 + ' [SEP] ' + s2 + ' [SEP]'

class Data(BaseModel):
    sentence: str


model = CustomRegressor(hidden_size=768)
model.load_state_dict(ckpt["model_state_dict"])



@app.post("/")
def similarity(request1: Data, request2: Data):
    request1 = request1.sentence.strip()
    request2 = request2.sentence.strip()
    
 
    data = some_func(request1, request2)


    tensorized_input = tokenizer(
                                data,
                                padding='longest',  # True or 'longest': Pad to the longest sequence in the batch
                                truncation=True,
                                return_tensors='pt',
                                add_special_tokens=False
                                )
    

    with torch.no_grad():
        logits = model(**tensorized_input)

    score = logits.squeeze().cpu().numpy()
    

    return logits, score


