import torch
import torch.nn as nn



class CNNHan(nn.Module):

    def __init__(self, vocab_size, label_size, emb_size, hidden_size, dropout, model_w2vec, category_size, emb_category):

        super().__init__()
        self.emb_size = emb_size
        self.kernel_sizes = [3, 4, 5]
        self.num_kernels = hidden_size

        #embeddings for the category 
        self.embedder_category = nn.Embedding.from_pretrained(emb_category, freeze=False, padding_idx=14)
       
        weights = torch.FloatTensor(model_w2vec)

        self.embedder = nn.Embedding.from_pretrained(weights, freeze=True)


        self.sent_convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.sent_convs.append(torch.nn.Conv1d(
                self.emb_size, self.num_kernels,
                kernel_size))

        self.top_k = 1
        self.hidden_size = len(self.kernel_sizes) * \
                      self.num_kernels * self.top_k

        self.hidden_size_doc_conv = self.hidden_size + category_size  

        self.doc_convs = torch.nn.ModuleList()

        self.doc_convs.append(torch.nn.Conv1d(
                   self.hidden_size_doc_conv , self.num_kernels,
                    3))
        
        self.projection = nn.Linear(self.num_kernels, label_size)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
    

    def forward(self, x, sent_seq_len, word_seq_len,x_category): #x=notes

        # Compute word embeddings
        # [B, S, W, E] 
        (B, S, W) = x.size()

        x = x.long()

        x_embed = self.embedder(x)
        x_embed = self.dropout_layer(x_embed)
        # [B x S, W, hid_size]
        x_embed = x_embed.contiguous().view(-1, W, self.emb_size)

        x_embed = x_embed.transpose(1, 2)

        pooled_outputs = []
        for i, conv in enumerate(self.sent_convs):

            convolution = self.relu(conv(x_embed))

            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)

            pooled_outputs.append(pooled)


        sent_embedding = torch.cat(pooled_outputs, 1)

        sent_embedding = self.dropout_layer(sent_embedding)
        #[B, S, hid_size]
        sent_embedding = sent_embedding.contiguous().view(B, S, self.hidden_size)

        #category embeddings
        categories_embedded = self.embedder_category(x_category)

        sent_embedding = torch.cat((sent_embedding,categories_embedded), 2)

        sent_embedding = sent_embedding.transpose(1, 2)

        doc_pooled_outputs = []
        for i, conv in enumerate(self.doc_convs):
            
            convolution = self.relu(conv(sent_embedding))

            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)

            doc_pooled_outputs.append(pooled)

        doc_outputs =  torch.cat(doc_pooled_outputs, 1)

        doc_outputs = self.dropout_layer(doc_outputs)
        
        logits = self.projection(doc_outputs) #size [B,1] each patient has a logit mortality


        
        return logits





