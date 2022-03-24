from flask import  request, render_template, Blueprint
import torch
from PreOnboarding_AI.model import CustomRegressor
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel, BertTokenizer
import numpy as np

main_bp = Blueprint('main', __name__)

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
if torch.cuda .is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# checkpoint
ckpt = torch.load(('model_4_3e-05_16_fold4.ckpt.epoch3'), map_location = device)

def some_func(s1, s2):
    return '[CLS] ' + s1 + ' [SEP] ' + s2 + ' [SEP]'

model = CustomRegressor(hidden_size=768)
model.load_state_dict(ckpt["model_state_dict"])

def prediction(x):
    tensorized_input = tokenizer(
                                x,
                                padding='longest',  # True or 'longest': Pad to the longest sequence in the batch
                                truncation=True,
                                max_length = 512,
                                return_tensors='pt',
                                )
    
    with torch.no_grad():
        logits = model(**tensorized_input)
      
    score = logits.squeeze().cpu().numpy()
    pred = (logits.flatten() >= 3).cpu().numpy()

    return score, pred
@main_bp.route('/', methods =['GET'])
def index():
    """
    index 함수에서는 '/' 엔드포인트로 접속했을 때 'index.html' 파일을
    렌더링 해줍니다.

    'index.html' 파일에서 'users.csv' 파일에 저장된 유저 목록을 보여줄 수 있도록
    유저들을 html 파일에 넘길 수 있어야 합니다.d

    요구사항:
      - HTTP Method: `GET`
      - Endpoint: `/`

    상황별 요구사항:
      - `GET` 요청이 들어오면 `templates/index.html` 파일을 렌더해야 합니다.

    """
    return render_template('model.html')

@main_bp.route('/predict', methods =['POST', 'GET'])
def information():

  sentence1 = request.form.get('sentence1', str)
  sentence2 = request.form.get('sentence2', str)

  dat= f'{sentence1}'
  sentence1 = str(dat)
  dat= f'{sentence2}'
  sentence2 = str(dat)

  
  def some_func(s1, s2):
    return '[CLS] ' + s1 + ' [SEP] ' + s2 + ' [SEP]'

  data = some_func(sentence1, sentence2)
  score, pred = prediction(data)


  return render_template('model.html', label = score, preds = pred)
