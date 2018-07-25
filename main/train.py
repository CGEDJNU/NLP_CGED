import os
import time
import pickle
from tqdm import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils.visualize import *

torch.manual_seed(1)

CUDA_VALID = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_VALID = 1


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def load_word_tag_ix(word_to_ix_path, tag_to_ix_path):
    with open(word_to_ix_path, 'rb') as wordf:
        word_to_ix = pickle.load(wordf)
    with open(tag_to_ix_path, 'rb') as tagf:
        tag_to_ix = pickle.load(tagf)
    return word_to_ix, tag_to_ix

def get_training_data(train_data_path):
    """
        input: 
            ---------------
            抱 B-v O
            怨 B-v O    -> sentence 1
            的 b-u O

            科 B-n O
            技 B-n O    -> sentence 2
            ---------------
    """
    data = open(train_data_path).readlines()
    training_data = []

    list1 = []
    list2 = []
    for i in range(len(data)):
        if data[i] != '\n':
            char, pos, err = data[i].split()
            list1.append(char)
            list2.append(err)
        else:
            training_data.append((list1, list2))
            list1 = []
            list2 = []
    return training_data

def get_training_data_(train_data_path):
    """
        input: 
            ---------------
            我 们 在 回 去 加 拿 大 之 前
            O  O  O  O  O  B-R O O  O  O    -> sentence 1
            我 认 为 所 有 的 人 类
            O  O  O  O  O  O  O  O          -> sentence 2
            ---------------
    """
    training_data=[]
    data= open(train_data_path).readlines()
    for i in range(len(data)):
        data[i] = data[i].replace('\n', ' ');
        if i%2 ==0:
            tup1=(data[i].split(),)
        else :
            tup2=(data[i].split(),)
            tup3 = tup1 + tup2
            training_data.append(tup3)
    return training_data

def get_dict_word_and_tag(train_data_path, test_data_path, word_to_ix_path, tag_to_ix_path):
    training_data = get_training_data(train_data_path)
    test_data = get_training_data(test_data_path) 
    all_data = training_data + test_data
    word_to_ix = {}
    for sentence, tags in all_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"O": 0, "B-R": 1, "I-R": 2,"B-M": 3, "I-M": 4, "B-S": 5, "I-S": 6,"B-W": 7, "I-W": 8, START_TAG: 9, STOP_TAG: 10}

    with open(word_to_ix_path, 'wb') as wordf:
        pickle.dump(word_to_ix, wordf)
    with open(tag_to_ix_path, 'wb') as tagf:
        pickle.dump(tag_to_ix, tagf)

#####################################################################
# Create model


START_TAG = "<START>"
STOP_TAG = "<STOP>"

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag 
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        if CUDA_VALID:
            forward_var = forward_var.cuda()
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        if CUDA_VALID:
            self.hidden = tuple([elem.cuda() for elem in self.hidden])# Default is not cuda
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        
        if CUDA_VALID:
            tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]).cuda(), tags])
            score = autograd.Variable(torch.Tensor([0])).cuda()
        else:
            score = autograd.Variable(torch.Tensor([0]))
            tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):   
        # Move feats from GPU to CPU
        feats = feats.cpu()
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                
                
                # align data
                if CUDA_VALID:
                    next_tag_var = forward_var + self.transitions[next_tag].cpu().view(1, -1)
                else:
                    next_tag_var = forward_var + self.transitions[next_tag]

                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].cpu().view(1,-1)
        #terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#####################################################################

if __name__ == '__main__':

        # Run training
        EMBEDDING_DIM = 10
        HIDDEN_DIM = 10 
        RESUME = 0
        CHECKPOINT = 80
        
        # Make up some training data
        epoch_num = 300
        model_save_interval = 5
        # Ratio of data to use
        train_ratio = 0.8
        val_ratio = 0.8
        
        train_data_path = '../data/CRF-input/train_CGED2016.txt'
        test_data_path = '../data/CGED-Test-2016/test_CGED2016.txt'
        
        word_to_ix_path = '../data/word_to_ix.pkl'
        tag_to_ix_path = '../data/tag_to_ix.pkl'
        
        #get_dict_word_and_tag(train_data_path, test_data_path, word_to_ix_path, tag_to_ix_path)
        experiment_num = 1
        experiment_name = str(experiment_num)+'_'+str(EMBEDDING_DIM)+'_'+str(HIDDEN_DIM)+'_'+str(train_ratio)+'/'
        
        # Visualize
        vis = Visualizer(env=experiment_name)
        
        model_dir = '../data/models/'+experiment_name
        
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        # Load word dict
        word_to_ix, tag_to_ix = load_word_tag_ix(word_to_ix_path, tag_to_ix_path)
        
        # Training data
        training_data = get_training_data(train_data_path)
        # Get subset of training data to verify whether work or not
        training_data = training_data[:int( len(training_data) * train_ratio )]
        
        # Val data
        val_data = get_training_data(test_data_path) 
        val_data = val_data[:int( len(val_data) * val_ratio )]
        
        # Create model
        model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
        
        # Model checkpoint
        if RESUME:
            model_name = str(CHECKPOINT)+'-model.pkl'
            model_path = model_dir+model_name
            model = torch.load(model_path)
        
        if CUDA_VALID:
            model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Check predictions before training
        #precheck_sent = prepare_sequence(training_data[1][0], word_to_ix)
        #precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
        #if torch.cuda.is_available():
        #    precheck_sent = precheck_sent.cuda()

        #print(model(precheck_sent))
        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        for epoch in tqdm ( range(epoch_num) ):  # again, normally you would NOT do 300 epochs, it is toy data
            
            sample_train_num = len(training_data)
            epoch_train_loss = torch.autograd.Variable( torch.FloatTensor([0.0]) )
            # Training loss            
            for sentence, tags in tqdm (training_data):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Variables of word indices.
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = torch.LongTensor([tag_to_ix[t] for t in tags])

                # Step 3. Run our forward pass.
                
                if CUDA_VALID:
                    sentence_in = sentence_in.cuda()
                    targets = targets.cuda()
                    epoch_train_loss = epoch_train_loss.cuda()
                
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)
                epoch_train_loss += neg_log_likelihood
                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                neg_log_likelihood.backward()
                optimizer.step()
            
            avg_epoch_train_loss = epoch_train_loss.data[0] / sample_train_num
            vis.plot('train_logloss', avg_epoch_train_loss)
            
            # Validation loss
            sample_val_num = len(val_data)
            epoch_val_loss = torch.autograd.Variable( torch.FloatTensor([0.0]) )
            for sentence, tags in tqdm(val_data):
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = torch.LongTensor([tag_to_ix[t] for t in tags])
                if CUDA_VALID:
                    sentence_in = sentence_in.cuda()
                    targets = targets.cuda()
                    epoch_val_loss = epoch_val_loss.cuda()
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)
                epoch_val_loss += neg_log_likelihood
            avg_epoch_val_loss = epoch_val_loss.data[0] / sample_val_num
            vis.plot('val_logloss', avg_epoch_val_loss)
            
            # Save model
            if epoch % model_save_interval == 0:
                model_name = str(epoch)+'-model.pkl'
                model_path = model_dir+model_name
                torch.save(model, model_path)

