import numpy as np
from collections import Counter

class TFIDF():
    def __init__(self,
                data):
        super(TFIDF, self).__init__()
        self.data = data
        self.N = len(data)
        
        self.tf_dict = {}
        self.df_dict = {}
        self.idf_dict = {}
        
        # Create data structures with tfidf for all documents
        self.create_tf_dict()
        self.create_df_dict()
        self.create_idf_dict()
        self.create_tfidf_dict()
        
    def create_df_dict(self):
        '''Creates dictionary of df values for each word'''
        self.df_dict = {}
        for doc_text in self.data:
            for word in set(doc_text):
                self.df_dict[word] = self.df_dict.get(word, 0) + 1
        return self.df_dict

    def create_idf_dict(self):
        '''Creates dictionary of idf values for each word'''
        self.idf_dict = {}
        maximum = 0
        delta = 0.5
        for word, df in self.df_dict.items():
            idf = np.log((self.N + delta) / (df + delta))
            self.idf_dict[word] = idf
        return self.idf_dict
    
    def create_tf_dict(self):
        self.tf_dict = {}
        for doc_num, doc_text in enumerate(self.data):
            self.tf_dict[doc_num] = {}
            if (len(doc_text) == 0):
                continue
            word_freq_list = Counter(doc_text).most_common()
            # max normalization
            max_freq = word_freq_list[0][1]
            for word, freq in word_freq_list:
                self.tf_dict[doc_num][word] = freq / max_freq  

        return self.tf_dict
    
    def create_tfidf_dict(self):
        '''Creates dictionary of tfidf vectors for each document in data'''
        self.tfidf_dict = {}
        for doc_num, tf_per_doc in self.tf_dict.items():
            self.tfidf_dict[doc_num] = {}
            for word, tf in tf_per_doc.items():
                self.tfidf_dict[doc_num][word] = tf * self.idf_dict.get(word, 0)

        return self.tfidf_dict
    
    def compute_query_tfidf(self, query):
        '''Computes tfidf vector representation for a given (tokenized) query'''
        query_vec =  {}
        word_freq_list = Counter(query).most_common()
        max_freq = word_freq_list[0][1]
        for word, freq in word_freq_list:
            # Get idf from pre-computed data structure
            idf = self.idf_dict.get(word, 0)
            # Compute tf for query (similar to that for a document)
            tf = (0.5 + 0.5 * freq / max_freq)
            query_vec[word] = tf * idf
        return query_vec
    
    def match(self, query_vec, doc_vec):
        '''Computes cosine similarity score (range [-1, 1]) between two vectors'''
        q_norm = np.linalg.norm([score for score in query_vec.values()])
        d_norm = np.linalg.norm([score for score in doc_vec.values()])
        score = 0
        for term, val in query_vec.items():
            score += val * doc_vec.get(term, 0)
        score /= q_norm * d_norm
        return score
