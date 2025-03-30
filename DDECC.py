"""
Implementation of "Denoising Diffusion Error Correction Codes" (DDECC), in ICLR23
https://arxiv.org/abs/2209.13533
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from Codes import sign_to_bin, bin_to_sign
import numpy as np

############################################################
#   ECCT classes
############################################################
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
        
class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 2)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N>1:
            self.norm2 = LayerNorm(layer.size)
    def forward(self, x, mask,time_emb):
        for idx, layer in enumerate(self.layers,start=1):
            x = layer(x, mask)
            # x = time_emb*x
            if idx == len(self.layers)//2 and len(self.layers)>1:
               x = self.norm2(x) 
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
    
    
############################################################
############################################################
# classes for UNet
# classes for Unet
class Rescale_NN(nn.Module):
  def __init__(self, input_dim , output_dim):
    super(Rescale_NN, self).__init__()
    # Define a fully coonected layer
    self.fc = nn.Linear(input_dim , output_dim)

  def forward(self, x):
    # pass the input through the fully connected layer
    return self.fc(x)


class ShortcutElementwiseMultiply(nn.Module):
    def __init__(self):
        super(ShortcutElementwiseMultiply, self).__init__()

    def forward(self, x1, x2):
        # Ensure the input tensors have the same shape
        assert x1.shape == x2.shape, "Input tensors must have the same shape"

        # Perform element-wise multiplication
        return x1 * x2

class Bottleneck(nn.Module):
  def __init__(self, d_input, d_output):
    super(Bottleneck, self).__init__()


    # define a fully connected linear layer with GELU activation function
    self.fc = nn.Linear(d_input, d_output)


  def forward(self, x):
    # apply linear transformation followed by GELU activation
    x = self.fc(x)
    return F.gelu(x)

############################################################
############################################################

############################################################
#   DDECC
############################################################


class DDECCT(nn.Module):
    def __init__(self, args,device, dropout=0):
        super(DDECCT, self).__init__()
        ####
        self.n_steps = args.N_steps
        self.d_model = args.d_model
        self.sigma = args.sigma
        self.register_buffer('pc_matrix', args.code.pc_matrix.transpose(0, 1).float())
        self.device = device
        #
        betas = torch.linspace(1e-3, 1e-2, self.n_steps)
        betas = betas*0+self.sigma
        self.betas = betas.view(-1,1)
        self.betas_bar =  torch.cumsum(self.betas, 0).view(-1,1)
        self.ema = EMA(0.9,flag_run=True)
        ###
        self.line_search = False
        ###
        code = args.code
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_model*4, dropout)
        
        # Define layers for different sizes (e.g., d_1, d_2, d_3)
        self.d_1 = args.d_model // 2
        
        self.d_2 = 3 * args.d_model // 4
        self.d_3 = 5 * args.d_model // 4
        self.d_4 = 3 * args.d_model // 2
        
        attn_1 = MultiHeadedAttention(args.h, self.d_1)
        ff_1 = PositionwiseFeedForward(self.d_1, self.d_1*4, dropout)

        attn_2 = MultiHeadedAttention(args.h, self.d_2)
        ff_2 = PositionwiseFeedForward(self.d_2, self.d_2*4, dropout)

        attn_3 = MultiHeadedAttention(args.h, self.d_3)
        ff_3 = PositionwiseFeedForward(self.d_3,self.d_3*4, dropout)

        #attn_4 = MultiHeadedAttention(args.h, self.d_3)
        #ff_4 = PositionwiseFeedForward(self.d_3, self.d_3*4, dropout)

        #attn_5 = MultiHeadedAttention(args.h, self.d_2)
        #ff_5 = PositionwiseFeedForward(self.d_2, self.d_2*4, dropout)

        #attn_6 = MultiHeadedAttention(args.h, self.d_1)
        #ff_6 = PositionwiseFeedForward(self.d_1, self.d_1*4, dropout)
        

        
        # Define rescale layers for the different dimensions
        self.rescale1 = Rescale_NN(self.d_model, self.d_1)
        self.rescale2 = Rescale_NN(self.d_1, self.d_2)
        self.rescale3 = Rescale_NN(self.d_2, self.d_3)
        self.rescale4 = Rescale_NN(self.d_4, self.d_3)
        self.rescale5 = Rescale_NN(self.d_3, self.d_2)
        self.rescale6 = Rescale_NN(self.d_2, self.d_1)
        self.rescale7 = Rescale_NN(self.d_1, self.d_model)
        
        # Define the bottleneck layer
        self.bottleneck_layer = Bottleneck(self.d_3, self.d_4)
        
        # Define shortcut connection
        self.shortcut_multiply = ShortcutElementwiseMultiply()

        self.src_embed = torch.nn.Parameter(torch.empty(
            (code.n + code.pc_matrix.size(0), args.d_model)))
        
        #self.decoder = Encoder(EncoderLayer(
            # args.d_model, c(attn), c(ff), dropout), args.N_dec)
           
        # Define the transformer layers for multiple dimensions
        self.transformer_layer1 = Encoder(EncoderLayer(self.d_1, c(attn_1), c(ff_1), dropout),  args.N_dec )
        self.transformer_layer2 = Encoder(EncoderLayer(self.d_2, c(attn_2), c(ff_2), dropout),  args.N_dec )
        self.transformer_layer3 = Encoder(EncoderLayer(self.d_3, c(attn_3), c(ff_3), dropout),  args.N_dec )
        #self.transformer_layer4 = Encoder(EncoderLayer(self.d_3, c(attn_4), c(ff_4), dropout), 1)
        #self.transformer_layer5 = Encoder(EncoderLayer(self.d_2, c(attn_5), c(ff_5), dropout), 1)
        #self.transformer_layer6 = Encoder(EncoderLayer(self.d_1, c(attn_6), c(ff_6), dropout), 1)
        
        
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)
        self.time_embed = nn.Embedding(self.n_steps, args.d_model)
        
        self.get_mask(code)
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, y, time_step):
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long().float(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        time_emb = self.time_embed(time_step).view(-1, 1, self.d_model)
        emb = time_emb * emb
        # emb = self.decoder(emb, self.src_mask,time_emb)
        
        
        # Processing through rescale and transformer layers
        x_1 = self.rescale1(emb)
        x_2 = self.transformer_layer1(x_1, self.src_mask, time_emb)
        x_3 = self.rescale2(x_2)
        x_4 = self.transformer_layer2(x_3, self.src_mask, time_emb)
        x_5 = self.rescale3(x_4)
        x_6 = self.transformer_layer3(x_5, self.src_mask, time_emb)
 
        # Bottleneck layer
        x_7 = self.bottleneck_layer(x_6)
        x_8 = self.rescale4(x_7)
 
        # Upsampling with shortcut multiplication
        x_9 = self.shortcut_multiply(x_6, x_8)
        #x_10 = self.transformer_layer4(x_9, self.src_mask, time_emb)
        x_10 = self.transformer_layer3(x_9, self.src_mask, time_emb)
        x_11 = self.rescale5(x_10)
 
        x_12 = self.shortcut_multiply(x_4, x_11)
        #x_13 = self.transformer_layer5(x_12, self.src_mask , time_emb)
        x_13 = self.transformer_layer2(x_12, self.src_mask , time_emb)
        x_14 = self.rescale6(x_13)
 
        x_15 = self.shortcut_multiply(x_2, x_14)
        #x_16 = self.transformer_layer6(x_15, self.src_mask, time_emb)
        x_16 = self.transformer_layer1(x_15, self.src_mask, time_emb)
        x_17 = self.rescale7(x_16)
        
        # Final output
        return self.out_fc(self.oned_final_embed(x_17).squeeze(-1))

    def p_sample(self, yt):
        #Single sampling from the real p dist.
        sum_syndrome =  (torch.matmul(sign_to_bin(torch.sign(yt.to(self.device))),self.pc_matrix) % 2).round().long().sum(-1)
        #sum_syndrome =  (torch.matmul(sign_to_bin(torch.sign(yt.to(self.device))),
                              #self.pc_matrix.to(yt.device)) % 2).round().long().sum(-1)

        # assert sum_syndrome.max() <= self.pc_matrix.shape[1] and sum_syndrome.min() >= 0
        t = sum_syndrome
        # Model output
        noise_mul_pred = self(yt.to(self.device), sum_syndrome.to(self.device)).cpu()# predicted multiplicative noise
        noise_add_pred = yt-torch.sign(-noise_mul_pred * torch.sign(yt)) #predicted additive noise
        # factor = (torch.sqrt(self.betas_bar[t])*self.betas[t]/(self.betas_bar[t]+self.betas[t])) #theoretical step size
        
        t = t.to(self.betas_bar.device)  # Ensure t is on the same device
        factor = (torch.sqrt(self.betas_bar[t]) * self.betas[t] / (self.betas_bar[t] + self.betas[t]))

        alpha_final = 1
        if self.line_search:
            #Perform Step Sizer Line-search # TODO : perform it on GPU for speed
            alpha = torch.linspace(1,20,20).unsqueeze(0).unsqueeze(0)
            new_synd = (torch.matmul(sign_to_bin(torch.sign(yt.unsqueeze(-1) - alpha*(noise_add_pred*factor).unsqueeze(-1))).permute(0,2,1),self.pc_matrix.cpu()) % 2).round().long().sum(-1)
            alpha_final = alpha.squeeze(0)[:,new_synd.argmin(-1).unsqueeze(-1)].squeeze(0)
        yt_1 = yt - alpha_final*noise_add_pred*factor
        yt_1[t==0] = yt[t==0] # if some codeword has 0 synd. keep it as is
        return (yt_1), t




    def p_sample_loop(self, cur_y):
        #Iterative sampling from the real p dist.
        res = []
        synd_all = []
        for it in range(self.pc_matrix.shape[1]):
            cur_y,curr_synd = self.p_sample(cur_y)
            synd_all.append(curr_synd)
            res.append(cur_y)
        synd_all = torch.stack(synd_all).t().cpu()
        # Chose the biggest iteration that reaches 0 synd.
        aa = (synd_all == 0).int()*2-1 
        idx = torch.arange(aa.shape[1], 0, -1)
        idx_conv = torch.argmax(aa * idx, 1, keepdim=True)
        return cur_y, res, idx_conv.view(-1), synd_all

  #################################  
    def loss(self,x_0):
        t = torch.randint(0, self.n_steps, size=(x_0.shape[0] // 2 + 1,))
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:x_0.shape[0]].long()
        e = torch.randn_like(x_0)
        noise_factor = torch.sqrt(self.betas_bar[t]).to(x_0.device)
        #
        h = torch.from_numpy(np.random.rayleigh(x_0.size(0),x_0.size(1))).float()
        h = 1.
        yt = h*x_0 * 1 + e * noise_factor
        sum_syndrome =  (torch.matmul(sign_to_bin(torch.sign(yt.to(self.device))),
        self.pc_matrix) % 2).sum(-1).long()
        #
        output = self(yt.to(self.device), sum_syndrome.to(self.device))
        z_mul = (yt *x_0)
        return F.binary_cross_entropy_with_logits(output, sign_to_bin(torch.sign(z_mul.to(self.device))))
    #################################
    #################################
    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            mask_size = code.n + code.pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(code.n - code.k):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask
        src_mask = build_mask(code)
        self.register_buffer('src_mask', src_mask)
############################################################
class EMA(object):
    def __init__(self, mu=0.999,flag_run = True):
        self.mu = mu
        self.shadow = {}
        self.flag_run = flag_run

    def register(self, module):
        if self.flag_run:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

    def update(self, module):
        if self.flag_run:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

############################################################
############################################################

if __name__ == '__main__':
    pass
