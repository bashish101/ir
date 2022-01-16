import os
import numpy as np

def download(url):
    import urllib
    import zipfile
    filepath = os.path.basename(url)
    filename = os.path.splitext(filepath)[0]
    save_path = os.path.join('./', filename)
    print('Downloading...', end=' ')
    print(url, save_path)
    # Uncomment to download
    urllib.request.urlretrieve(url, save_path)
    print('Done!')
    print('Extracting files...', end=' ')
    print(os.path.abspath(filepath))
    
    with zipfile.ZipFile(filepath, 'r') as fp:
        fp.extractall(save_path)
        
    # os.remove(filepath)
    print('Done!')
    
# Use this function after creating vocabulary to create embedding matrix initialization
def create_embedding_matrix(word2idx_dict, vocab, emb_path = "./glove.6B/glove.6B.300d.txt", n_emb = 300, 
                            init = 'normal', url = 'http://nlp.stanford.edu/data/glove.6B.zip', save_path = './emb_weights.npy'):
    if not os.path.exists(emb_path):
        download(url)
    
    # Initialize matrix
    n_vocab = len(vocab)
    if init == 'ones':
        embed_mat = np.ones((n_vocab, n_emb))
    else:
        embed_mat = np.random.normal(scale = 0.6, size = (n_vocab, n_emb))
    count = 0
    embed_mat[0] = np.zeros(n_emb)
    with open(emb_path) as fp:
        for line in fp:
            word_vec = line.split(' ')
            word = word_vec[0]

            if word not in vocab:
                continue
            count += 1
            word_idx = word2idx_dict[word]
            embed_mat[word_idx] = np.asarray(word_vec[1:], dtype='float32')
            
    print('Found total words matching with vocabulary = ', count)      
    with open(save_path, 'wb') as fp:
        np.save(fp, embed_mat)
    
    print('Embedding matrix saved successfully to location {}'.format(emb_path))
    return embed_mat


