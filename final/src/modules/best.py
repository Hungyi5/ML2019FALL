import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class Best(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(Best, self).__init__()
        # context
        self.lstm = nn.GRU(dim_embeddings*2, 300, bidirectional=True)
        self.lstm2 = nn.GRU(600, 300, bidirectional=True)
        self.w = Variable(torch.zeros(600, 600).cuda())
        self.u = Variable(torch.zeros(600).cuda())

        # options
        self.lstm3 = nn.GRU(dim_embeddings, 150, bidirectional=True)
        self.lstm4 = nn.GRU(600, 300, bidirectional=True)
        self.w2 = Variable(torch.zeros(300, 300).cuda())
        self.u2 = Variable(torch.zeros(300).cuda())

        self.bilinear = nn.Bilinear(600, 300, 1)

            

    def forward(self, context, context_lens, options, option_lens):
        """

        Args:
            context (batch, padded_len, embed_size): (100, 300, 600)
            context_lens (batch, original_len)
            options (batch, options, padded_len, embed_size): (100, 5, 50, 300)
            option_lens (batch, options, original_len)
        Return:
            logits (batch, options): score of every options
        """

        #context
        context = context.transpose(1, 0)
        context_l, _ = self.lstm(context)

        query_len, batch_size, embed_size = context_l.shape

        '''
        output_reshape = context_l.view(-1, 600)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w))
        attn_hidden_layer = torch.mm(attn_tanh, self.u.view(-1, 1))

        exps = torch.exp(attn_hidden_layer).view(-1, query_len)
        alphas = exps / torch.sum(exps, 1).view(-1, 1)
        alphas_reshape = alphas.view(-1, query_len, 1)

        state = context_l.permute(1, 0, 2)       
        att_context = torch.sum(state * alphas_reshape, 1)
        #(batch_size, 600)
        '''

        context_state = context_l.max(0)[0]
        #(batch_size, 600)

        #options
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):

            option = option.transpose(1, 0)
            option_l, _ = self.lstm3(option)

            query_len, batch_size, embed_size = option_l.size()
            '''
            output_reshape = option_l.view(-1, 300)

            attn_tanh = torch.tanh(torch.mm(output_reshape, self.w2))
            attn_hidden_layer = torch.mm(attn_tanh, self.u2.view(-1, 1))

            exps = torch.exp(attn_hidden_layer).view(-1, query_len)
            alphas = exps / torch.sum(exps, 1).view(-1, 1)
            alphas_reshape = alphas.view(-1, query_len, 1)

            state = option_l.permute(1, 0, 2) 
            att_option = torch.sum(state * alphas_reshape, 1)
            #(batch_size, 300)
            '''
            option_state = option_l.max(0)[0]
            #(batch_size, 300)

            #logit = self.bilinear(att_context, att_option)
            logit = self.bilinear(context_state, option_state)
            logits.append(logit)
   
        logits = torch.stack(logits, 2)
        logits = logits.view(len(logits),-1)
        

        return logits