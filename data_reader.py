import re
from collections import Counter

# pip install nltk
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords
    
class DataReader():
    def __init__(self,
                data_path = './triples.train.small.tsv',
                data_count = 10000,
                vocab_size = 20000):
        self.data_path = data_path
        self.stopword_list = set([re.sub('[^a-zA-Z0-9 \n\.]', '', x) \
                                  for x in stopwords.words("english") if 'wh' not in x and 'how' not in x])
        
        self.unprocessed_data = []
        self.data = []
        self.unknown_token = 'UNK'
        self.vocab_size = vocab_size
        self.max_doc_len = 100
        self.max_query_len = 15
        self.vocab = []
        self.word2idx_dict = {}
        self.max_data_count = data_count
        
    def preprocess(self, line):
        '''Converts a sentence to list of lowercase word tokens'''
        line = re.sub('[^a-zA-Z0-9 \n\.]', '', line)
        word_list = re.findall('[a-z0-9]{1,}', line.lower())
        word_list = [word for word in word_list if word not in self.stopword_list]
        return word_list


    def read_file(self):
        '''Reads and preprocesses triplets data file from initialized data path'''
        query_word_list, pos_word_list, neg_word_list = [], [], []
        with open(self.data_path, 'r') as fp:
            idx = 0
            for line in fp:
                # MS MARCO contains tab separated lines of query, positive doc and negative doc
                triplet = line.encode('latin1').decode('utf-8').strip().split('\t')
                self.unprocessed_data.append(triplet)
                query_word_list.append(self.preprocess(triplet[0]))
                pos_word_list.append(self.preprocess(triplet[1]))
                neg_word_list.append(self.preprocess(triplet[2]))
                
                idx += 1
                if self.max_data_count is not None and idx == self.max_data_count:
                    break
                    
        return (query_word_list, pos_word_list, neg_word_list)
    
    def create_idx_vectors(self, data, max_len = 15, pad_value = 0):
        '''Creates vector of indices from list of words'''
        out_list = []
        for word_list in data:
            idx_vec = [self.word2idx_dict.get(word, self.word2idx_dict.get(self.unknown_token)) for word in word_list]
            idx_vec = idx_vec[:max_len] + [pad_value] * max(0, (max_len - len(idx_vec)))
            out_list.append(idx_vec)
        return out_list
    
    def load_data(self):
        '''Loads, preprocesses and encodes triplets data'''
        (query_word_list, pos_word_list, neg_word_list) = self.read_file()
        
        unigram_list = [x for y in query_word_list for x in y] \
                        + [x for y in pos_word_list for x in y] \
                        + [x for y in neg_word_list for x in y]
        
        # Create vocabulary
        self.vocab =  [self.unknown_token] + [word for word, freq in Counter(unigram_list).most_common(self.vocab_size - 1)]
        self.word2idx_dict = {word:idx for idx, word in enumerate(self.vocab)}
        
        # Create list of index encodings from list of words
        pad_value = self.word2idx_dict[self.unknown_token]
        query_list = self.create_idx_vectors(query_word_list, max_len = self.max_query_len, pad_value = pad_value)
        pos_list = self.create_idx_vectors(pos_word_list, max_len = self.max_doc_len, pad_value = pad_value)
        neg_list = self.create_idx_vectors(neg_word_list, max_len = self.max_doc_len, pad_value = pad_value)

        self.data = list(zip(query_list, pos_list, neg_list))
        return self.data

