import argparse
import pickle
import torch
from train import *
from tqdm import tqdm
from sklearn.metrics import classification_report,accuracy_score

def get_test_data_sids(test_data_sid_path):
    # Get test data sids
    with open(test_data_sid_path, 'r') as sidf:
        lines = sidf.readlines()
    # (sid=[sid_value]),left_off=5, right_off=1
    return [line.split('\t')[0][5:][:-1] for line in lines]


def predict_to_file(sid, predict_res, res_file_path):
    flag = True
    length = len(predict_res)
    
    # Check correct
    if predict_res.count('O') == length:
        with open(res_file_path, 'a+') as f:
            f.write(', '.join([sid, 'correct'])+'\n')
        return
    
    # Check diff errs
    for i in range(length):
        if predict_res[i] == 'O' and flag == False:
            pre_tag = predict_res[i-1].split('-')[-1]
            with open(res_file_path, 'a+') as f:
                f.write(', '.join([sid, str(start), str(end), pre_tag])+'\n')
            flag = True
            continue
        if predict_res[i] != 'O' and flag:
            start = i + 1 
            end = start
            flag = False
        if predict_res[i] != 'O' and flag == False:
            end = i + 1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim')
    parser.add_argument('--hidden_dim')
    parser.add_argument('--checkpoint')
    parser.add_argument('--model_dir')
    args = parser.parse_args()
    EMBEDDING_DIM = int( args.embedding_dim )
    HIDDEN_DIM = int( args.hidden_dim )
    model_checkpoint = int( args.checkpoint )
    model_dir = args.model_dir

    word_to_ix_path = '../data/word_to_ix.pkl'
    tag_to_ix_path = '../data/tag_to_ix.pkl'
    res_file_path = '../data/CGED-Test-2016/CGED16_HSK_Test_Output_.txt'

    START_TAG = '<START>'
    STOP_TAG = '<STOP>'

    model_path = model_dir+str(model_checkpoint)+'-model.pkl'
    test_data_path = '../data/CGED-Test-2016/test_CGED2016.txt'
    test_data_sid_path = '../data/CGED-Test-2016/CGED16_HSK_Test_Input.txt'


    test_data = get_training_data(test_data_path)
    test_data_sids = get_test_data_sids(test_data_sid_path)

    word_to_ix, tag_to_ix = load_word_tag_ix(word_to_ix_path, tag_to_ix_path)

    ix_to_tag = {value:key for key, value in tag_to_ix.items()}

    # Load model
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    model = torch.load(model_path)

    # Test seq

    all_real = []
    all_pred = []
    
    for sample_idx in tqdm (range(len(test_data)) ):
        
        seq_test = prepare_sequence(test_data[sample_idx][0], word_to_ix)
        #with torch.no_grad():
        score,tag_seq = model(seq_test)
        sid = test_data_sids[sample_idx]

        predict = [ix_to_tag[ix] for ix in tag_seq]
        real = test_data[sample_idx][1]

        
        all_real.extend(real)
        all_pred.extend(predict)
        
        # Result store
        #print('Predict: ', predict)
        #print('Real: ', real)
        predict_to_file(sid, predict, res_file_path)
        

    target_names = [key for key in tag_to_ix.keys()]
    print(classification_report(all_real, all_pred, target_names))
    print(accuracy_score(all_real, all_pred))
