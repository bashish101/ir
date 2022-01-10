import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
class ConvKNRM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_weights = None, 
                 query_max_len = 15,
                 ngram_len = 3, 
                 n_emb = 300, 
                 n_filters = 128,
                 mu = [-0.9, -0.4, 0, 0.4, 0.8, 1.0],
                 sigma = [0.2, 0.2, 0.2, 0.2, 0.2, 0.001]):
        super(ConvKNRM, self).__init__()
        self.vocab_size = vocab_size
        self.ngram_len = ngram_len
        self.query_max_len = query_max_len
        self.eps = 1e-12
        
        # Initilization of embedding weights
        self.embedding = nn.Embedding(self.vocab_size, n_emb)
        self.embedding.load_state_dict({"weight": torch.tensor(emb_weights, device = device)})
        self.embedding.weight.requires_grad = False
        
        # Gaussian Kernel parameters for soft-counting
        self.n_bins = len(mu)
        self.mu = torch.FloatTensor(mu).to(device)
        self.mu = self.mu.view(1, 1, 1, 1, self.n_bins)
        self.sigma = torch.FloatTensor(sigma).to(device)
        self.sigma = self.sigma.view(1, 1, 1, 1, self.n_bins)
                
        # Convolution layers to create ngrams
        self.ngram_conv = nn.ModuleList([nn.Sequential(
                    nn.ConstantPad1d((0, kernel_size - 1), 0),
                    nn.Conv1d(n_emb, n_filters, kernel_size, \
                                     padding = 0, bias = False)
            ) for kernel_size in range(1, self.ngram_len + 1)])
                          
        # Linear layer for computing final score
        self.fc = nn.Linear((self.ngram_len ** 2) * self.n_bins, 1, 1)
        
    def compute_similarity_matrix(self, x, y):
        '''Computes cosine similary matrix'''
        x = F.normalize(x, p = 2, dim = 1, eps = self.eps)
        y = F.normalize(y, p = 2, dim = 1, eps = self.eps)
        x = x.permute(0, 2, 1)
        mat = x.bmm(y)
        return mat
        
    def forward(self, x):
        query, doc = x
        # Create embeddings
        query_emb = self.embedding(query)
        query_emb = query_emb.permute(0, 2, 1)
        
        doc_emb = self.embedding(doc)
        doc_emb = doc_emb.permute(0, 2, 1)
        
        # Compute embeddings for ngrams
        x_list = []
        y_list = []
        for idx in range(self.ngram_len):
            x = self.ngram_conv[idx](query_emb)
            y = self.ngram_conv[idx](doc_emb)
            
            x_list.append(x)
            y_list.append(y)
        
        # Compute cosine similarity matrix for later counting
        mat_list = []
        for idx1 in range(self.ngram_len):
            for idx2 in range(self.ngram_len):
                mat = self.compute_similarity_matrix(x_list[idx1], y_list[idx2])
                mat_list.append(mat)
        x = torch.stack(mat_list, dim = 1)
        mask = x.sum(dim = 3) != 0
        
        # Apply Gaussian kernel for soft-counting
        x = torch.exp((-((x[..., None] - self.mu) ** 2) / (self.sigma ** 2) / 2))
        mask = mask.unsqueeze(3).repeat(1, 1, 1, self.n_bins)
        
        # Sum over all document terms
        x = x.sum(dim = 3)
        
        # Max normalization of term counts
        max_x = torch.amax(x, dim = 2, keepdim = True).detach() + self.eps
        x = 0.5 * x / max_x       # can also use rescaled log
        
        # Set query pad locations to zero value
        x *= mask.float()
        
        # Sum over query dimension
        x = x.sum(dim = 2).flatten(start_dim = 1)
        
        # Combine all counts to get single score
        x = self.fc(x)
        
        # Rescale score to range [-1, 1]
        x = torch.tanh(x)
        
        return x

class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, x, y):
        output = (1 - x + y).clamp(min = 0)
        return output.mean()
