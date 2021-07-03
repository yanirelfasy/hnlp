import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

NEG = "neg_words"
RARE = "rare_words"
TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    w2v = np.zeros(embedding_dim)
    word_lst = sent.get_leaves()
    word_cntr = 0
    for w in word_lst:
        # word_cntr += 1
        if w.text[0] in word_to_vec:
            word_cntr += 1
            w2v += word_to_vec[w.text[0]]

    if word_cntr > 0:
        return w2v/word_cntr
    else:
        return w2v


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] =1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    dim = len(word_to_ind)
    one_hot = np.zeros(dim)
    word_lst = sent.get_leaves()
    for w in word_lst:
        one_hot += get_one_hot(dim, word_to_ind[w.text[0]])
    return one_hot/len(word_lst)



def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_index_dic = {}
    words_list = list(set(words_list))
    for i,word in enumerate(words_list):
        word_index_dic[word] = i
    return word_index_dic


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    embedded_vec = np.zeros((seq_len, embedding_dim))
    for i, word in enumerate(sent.text):
        if i == seq_len:
            break
        if word in word_to_vec:
            embedded_vec[i] = word_to_vec[word]
    return embedded_vec


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()
        #our addition
        self.sentences[RARE] = self.sentiment_dataset.get_rare_word_test_set()
        #our addition
        self.sentences[NEG] = self.sentiment_dataset.get_neg_word_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, 1)
        return

    def forward(self, text):
        text = text.permute(1, 0, 2)
        _, (h1, _) = self.lstm(text)
        # h1 = nn.ReLU()(h1)
        h1 = torch.cat((h1[0],h1[1]),1)
        h1 = self.dropout(h1)
        h2 = self.linear(h1)
        return h2

    def predict(self, text):
        text = text.permute(1, 0, 2)
        _, (h1, _) = self.lstm(text)
        # h1 = nn.ReLU()(h1)
        h1 = torch.cat((h1[0],h1[1]),1)
        h2 = self.linear(h1)
        return h2


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        # label = torch.sigmoid(self.forward(x))
        label = self.forward(x)
        return label


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    match = 0
    if len(preds) != len(y):
        print('error in lengths!!!')
        return
    for i, val in enumerate(preds):
        if val <= 0.5:
            if y[i] == 0:
                match += 1
        elif val > 0.5:
            if y[i] == 1:
                match += 1
        else:
            print('error in val of y')
    return match/len(y)

def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """

    returned_loss = 0
    accuracy = 0
    batches_count = 0
    for data, correct_labels in data_iterator:
        optimizer.zero_grad()
        predicted_labels = model(data.float()).reshape(len(correct_labels))
        epoch_loss = criterion(predicted_labels, correct_labels)
        returned_loss += epoch_loss.item()
        accuracy += binary_accuracy(nn.Sigmoid()(predicted_labels), correct_labels)
        epoch_loss.backward()
        optimizer.step()
        batches_count += 1
    returned_loss /= batches_count
    accuracy /= batches_count
    return returned_loss, accuracy


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    returned_loss = 0
    accuracy = 0
    batches_count = 0
    for data, correct_labels in data_iterator:
        predicted_labels = model.predict(data.float()).reshape(len(correct_labels))
        epoch_loss = criterion(predicted_labels, correct_labels)
        returned_loss += epoch_loss.item()
        accuracy += binary_accuracy(nn.Sigmoid()(predicted_labels), correct_labels)
        batches_count += 1
    returned_loss /= batches_count
    accuracy /= batches_count
    return returned_loss, accuracy


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    train_error_list = []
    train_accuracy_list = []
    val_error_list = []
    val_accuracy_list = []
    epoch_list = list(range(n_epochs))
    data_iterator_train = data_manager.get_torch_iterator()
    data_iterator_val = data_manager.get_torch_iterator(data_subset=VAL)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        model.train()
        train_error, train_accuracy = train_epoch(model, data_iterator_train, optimizer, criterion)
        train_error_list.append(train_error)
        train_accuracy_list.append(train_accuracy)
        print('training loss and accuracy', train_error, train_accuracy)
        model.eval()
        val_error, val_accuracy = evaluate(model, data_iterator_val, criterion)
        val_error_list.append(val_error)
        val_accuracy_list.append(val_accuracy)
        print('validation loss and accuracy', val_error, val_accuracy)
    loss_plot = pd.DataFrame({'Train': train_error_list, 'Validation': val_error_list}, index=epoch_list)
    accuracy_plot = pd.DataFrame({'Train': train_accuracy_list, 'Validation': val_accuracy_list}, index=epoch_list)
    plot_show1 = loss_plot.plot.line()
    plot_show2 = accuracy_plot.plot.line()
    plt.show()
    return

def run_model_W2V():
    """
    run W2V model. Train it and than run it on TEST and special subsets
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    vocab_size = data_manager.get_input_shape()[0]
    model = LogLinear(vocab_size)

    train_model(model, data_manager, 20, 0.01, 0.)
    data_iterator_TEST = data_manager.get_torch_iterator(data_subset=TEST)
    criterion = nn.BCEWithLogitsLoss()

    #our addition
    data_iterator_NEG =  data_manager.get_torch_iterator(data_subset=NEG)
    #our addition
    data_iterator_Rare =  data_manager.get_torch_iterator(data_subset=RARE)

    val_error, val_accuracy = evaluate(model, data_iterator_TEST, criterion)
    print('Test loss and accuracy', val_error, val_accuracy)

    val_error, val_accuracy = evaluate(model, data_iterator_NEG, criterion)
    print('Neg Test loss and accuracy', val_error, val_accuracy)

    val_error, val_accuracy = evaluate(model, data_iterator_Rare, criterion)
    print('Rare Test loss and accuracy', val_error, val_accuracy)
    val_error, val_accuracy = evaluate(model, data_iterator_NEG, criterion)
    print('Test loss and accuracy', val_error, val_accuracy)

    return

def run_model_one_hot():
    """
    run log-linear with one-hot model. Train it and than run it on TEST and special subsets
    """
    weight_decay = [0., 0.001, 0.0001]
    data_manager = DataManager()
    vocab_size = data_manager.get_input_shape()[0]
    model = LogLinear(vocab_size)
    train_model(model, data_manager, 20, 0.01, 0.0001)

    data_iterator_TEST = data_manager.get_torch_iterator(data_subset=TEST)
    criterion = nn.BCEWithLogitsLoss()

    #our addition
    data_iterator_NEG =  data_manager.get_torch_iterator(data_subset=NEG)
    #our addition
    data_iterator_Rare =  data_manager.get_torch_iterator(data_subset=RARE)

    val_error, val_accuracy = evaluate(model, data_iterator_TEST, criterion)
    print('Test loss and accuracy', val_error, val_accuracy)

    val_error, val_accuracy = evaluate(model, data_iterator_NEG, criterion)
    print('Neg Test loss and accuracy', val_error, val_accuracy)

    val_error, val_accuracy = evaluate(model, data_iterator_Rare, criterion)
    print('Rare Test loss and accuracy', val_error, val_accuracy)

    return

def run_model_LSTM():
    """
    run LSTM model. Train it and than run it on TEST and special subsets
    """

    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(W2V_EMBEDDING_DIM, 100, 1, 0.5)
    train_model(model, data_manager, 4, 0.001, 0.0001)
    model.eval()
    data_iterator_TEST = data_manager.get_torch_iterator(data_subset=TEST)
    criterion = nn.BCEWithLogitsLoss()
    val_error, val_accuracy = evaluate(model, data_iterator_TEST, criterion)
    print('Test loss and accuracy', val_error, val_accuracy)

    #our addition
    data_iterator_NEG =  data_manager.get_torch_iterator(data_subset=NEG)
    #our addition
    data_iterator_Rare =  data_manager.get_torch_iterator(data_subset=RARE)

    val_error, val_accuracy = evaluate(model, data_iterator_NEG, criterion)
    print('Neg Test loss and accuracy', val_error, val_accuracy)

    val_error, val_accuracy = evaluate(model, data_iterator_Rare, criterion)
    print('Rare Test loss and accuracy', val_error, val_accuracy)

    return



def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    weight_decay = [0., 0.001, 0.0001]
    data_manager = DataManager()
    print(data_manager.get_input_shape()[0])
    vocab_size = data_manager.get_input_shape()[0]
    model = LogLinear(vocab_size)
    train_model(model, data_manager, 20, 0.01, 0.)
    return


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    print(data_manager.get_input_shape()[0])
    vocab_size = data_manager.get_input_shape()[0]
    model = LogLinear(vocab_size)
    train_model(model, data_manager, 20, 0.01, 0.)
    return


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    print(data_manager.get_input_shape()[0])
    vocab_size = data_manager.get_input_shape()[0]
    model = LSTM(W2V_EMBEDDING_DIM, 100, 1, 0.5)
    train_model(model, data_manager, 4, 0.001, 0.0001)
    return


if __name__ == '__main__':
    run_model_one_hot()
    run_model_W2V()
    run_model_LSTM()

