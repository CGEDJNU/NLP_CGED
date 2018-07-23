import pickle
import torch
from train import *
from tqdm import tqdm
from sklearn.metrics import classification_report,accuracy_score

EMBEDDING_DIM = 5
HIDDEN_DIM = 4  
word_to_ix_path = '../data/word_to_ix.pkl'
tag_to_ix_path = '../data/tag_to_ix.pkl'

START_TAG = '<START>'
STOP_TAG = '<STOP>'

model_dir = '../data/models/0_5_4_0.2/'
model_path = model_dir+'90-2018-07-22-02-19-20-model.pkl'
test_data_path = '../data/CGED-Test-2016/test_CGED2016.txt'

test_data = get_training_data(test_data_path)
word_to_ix, tag_to_ix = load_word_tag_ix(word_to_ix_path, tag_to_ix_path)

ix_to_tag = {value:key for key, value in tag_to_ix.items()}

# Load model
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model = torch.load(model_path)

# Test seq

all_real = []
all_pred = []

for sample_idx in tqdm (range(len(test_data[:100])) ):
    
    seq_test = prepare_sequence(test_data[sample_idx][0], word_to_ix)
    score,tag_seq = model(seq_test)
    
    predict = [ix_to_tag[ix] for ix in tag_seq]
    real = test_data[sample_idx][1]
    
    all_real.extend(real)
    all_pred.extend(predict)
    
    print('Predict: ', predict)
    print('Real: ', real)
    

target_names = [key for key in tag_to_ix.keys()]
print(classification_report(all_real, all_pred, target_names))
print(accuracy_score(all_real, all_pred))
