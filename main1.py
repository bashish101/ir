# main file for tfidf based retrieval model

from tfidf_model import TFIDF
from data_reader import DataReader

def main():    
    # Load data  
    max_data_count = 10000
    data_path = './triples.train.small.tsv'
    data_reader = DataReader(data_path = data_path,
                             data_count = max_data_count)
    queries, pos_data, neg_data = data_reader.read_file()
    text_data = data_reader.unprocessed_data
    
    # Create tfidf model for all documents
    combined_data = pos_data + neg_data
    tfidf_model = TFIDF(combined_data)
    
    print('Data loaded successfully!')
    # Test the model
    idx = 5000
    pos_doc_vec = tfidf_model.tfidf_dict[idx]
    neg_doc_vec = tfidf_model.tfidf_dict[idx + len(pos_data)]
    query_vec = tfidf_model.compute_query_tfidf(queries[idx])
    
    pos_score = tfidf_model.match(query_vec, pos_doc_vec)
    neg_score = tfidf_model.match(query_vec, neg_doc_vec)
    print('Query: {0} \n\nPositive Document: {1} \n\nNegative Document: {2}\n\n'.format(*text_data[idx]))
    print('First document score = {0:.2f} \nSecond document score = {1:.2f}'.format(pos_score, neg_score))
          
if __name__ == '__main__':
    main()
