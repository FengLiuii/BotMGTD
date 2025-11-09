

from statistics import mean
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import dgl
import dgl.function as fn
import dgl.nn.functional as dglF

from params import args
from Utils.Utils import cal_infonce_loss  # kept for compatibility

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Attention-based Relation Fusion Module
# -----------------------------------------------------------------------------
class AttentionBasedRelationFusion(nn.Module):
   
    def __init__(self, hidden_dim, num_relations, num_heads=8, dropout=0.1, ffn_expansion=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ffn_expansion = ffn_expansion
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim)
        )
        
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        
        importance_hidden = getattr(args, 'importance_hidden_ratio', 2)  
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // importance_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // importance_hidden, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, relation_embeddings, relation_types):
        """
        Args:
            relation_embeddings: [batch_size, num_relations, hidden_dim]
            relation_types: [num_relations] 关系类型标识
        Returns:
            fused_embedding: [batch_size, hidden_dim]
            attention_weights: [batch_size, num_relations, num_relations]
            importance_weights: [batch_size, num_relations]
        """
        batch_size, num_relations, hidden_dim = relation_embeddings.shape
        
        
        rel_emb = self.relation_embedding(relation_types)  # [num_relations, hidden_dim]
        enhanced_embeddings = relation_embeddings + rel_emb.unsqueeze(0)  # [batch_size, num_relations, hidden_dim]
        
        
        attended, attention_weights = self.attention(
            enhanced_embeddings, enhanced_embeddings, enhanced_embeddings
        )
        
        
        attended = self.layer_norm1(attended + enhanced_embeddings)
        
       
        ffn_output = self.ffn(attended)
        ffn_output = self.layer_norm2(ffn_output + attended)
        
        
        importance_scores = self.importance_predictor(ffn_output).squeeze(-1)  # [batch_size, num_relations]
        importance_weights = F.softmax(importance_scores, dim=-1)
        
        
        fused_embedding = torch.sum(ffn_output * importance_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        return fused_embedding, attention_weights, importance_weights



class UUGCNLayerGraphTransformer(nn.Module):
    def __init__(self, in_feats, out_feats, weight=True, bias=True, activation=None, dropout=0.1, use_edge_feat=False):
        """
        Single-head Graph-Transformer-style layer with optional edge features.
        """
        super(UUGCNLayerGraphTransformer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        self.use_edge_feat = use_edge_feat

        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
           
            init_method = getattr(args, 'weight_init', 'xavier_uniform')
            if init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(self.u_w)
            elif init_method == 'xavier_normal':
                nn.init.xavier_normal_(self.u_w)
            else:
                nn.init.xavier_uniform_(self.u_w)

        self._activation = activation

        # Q, K, V projections
        self.W_q = nn.Linear(out_feats, out_feats)
        self.W_k = nn.Linear(out_feats, out_feats)
        self.W_v = nn.Linear(out_feats, out_feats)

        self.sqrt_d = math.sqrt(out_feats)
        self.layer_norm = nn.LayerNorm(out_feats)
        self.dropout = nn.Dropout(dropout)
        self.residual = getattr(args, 'use_residual', True)  

        if self.use_edge_feat:
            self.edge_mlp = nn.Linear(out_feats, 1)

    def forward(self, graph, u_f, edge_feat=None):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)

            q = self.W_q(u_f)
            k = self.W_k(u_f)
            v = self.W_v(u_f)

            graph.ndata['q'] = q
            graph.ndata['k'] = k
            graph.ndata['v'] = v

            def compute_attention(edges):
                score = (edges.dst['q'] * edges.src['k']).sum(dim=-1) / self.sqrt_d
                if self.use_edge_feat and edge_feat is not None:
                    edge_score = self.edge_mlp(edge_feat).squeeze(-1)
                    score += edge_score
                return {'score': score}

            graph.apply_edges(compute_attention)
            graph.edata['a'] = dglF.edge_softmax(graph, graph.edata['score'])

            graph.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'h_new'))
            rst = graph.ndata['h_new']

            if self.residual:
                rst = rst + u_f

            rst = self.layer_norm(self.dropout(rst))

            if self._activation is not None:
                rst = self._activation(rst)
            return rst


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        
        self.drop = nn.Dropout(getattr(args, 'denoise_dropout', dropout))
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(device)
            if beta_fixed:
                self.betas[0] = 0.0001
            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = [1 - alpha_bar[0]]
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas)

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]]).to(device)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(device)]).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # sqrt(alphas) == sqrt(1 - betas)
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(1.0 - self.betas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).to(device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            x_t = model_mean
        return x_t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)
        model_variance = self._extract_into_tensor(self.posterior_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output
                      + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        return model_mean, model_log_variance

    def training_losses(self, model, targetEmbeds, x_start):
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(targetEmbeds, ts, noise) if self.noise_scale != 0 else x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        diff_loss = weight * mse
        return diff_loss, model_output

    def training_losses2(self, model, targetEmbeds, x_start, batch):
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise) if self.noise_scale != 0 else x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        diff_loss = (weight * mse)[batch]
        return diff_loss, model_output

    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

# -----------------------------------------------------------------------------
# Improved BotMGTD with Attention-based fusion
# -----------------------------------------------------------------------------
class BotMGTD(nn.Module):
    """
    Improved BotMGTD with Attention-based fusion instead of GRU
    Changes vs previous version:
      1) Strict target/aux separation:
         - target graph = he_adjs[0] (encoded by self.main_layers)
         - auxiliary graphs = he_adjs[1:] (each encoded by self.helayers; then attention-fused)
      2) Fusion changes to Attention:
         - Aux internal fusion: Multi-head attention over auxiliary embeddings
         - Final fusion: Concatenation + MLP fusion
    """
    def __init__(self, f_dim):
        super(BotMGTD, self).__init__()

        out_dims = eval(args.dims) + [args.latdim]
        in_dims = out_dims[::-1]

        self.user_denoise_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm)
        self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps)

        self.act = nn.LeakyReLU(0.5, inplace=True)

        # number of auxiliary groups (used for he_adjs[1:])
        self.num_adj_groups = getattr(args, 'num_adj_groups', 2)
        self.helayers = nn.ModuleList()
        for _ in range(self.num_adj_groups):
            group = nn.ModuleList()
            for _layer in range(args.uugt_layer):
                group.append(
                    UUGCNLayerGraphTransformer(args.latdim, args.latdim, weight=False, bias=False, activation=self.act)
                )
            self.helayers.append(group)

        # main_layers for target_embedding (always he_adjs[0])
        self.main_layers = nn.ModuleList()
        for _ in range(args.uugt_layer):
            self.main_layers.append(
                UUGCNLayerGraphTransformer(args.latdim, args.latdim, weight=False, bias=False, activation=self.act)
            )

        # feature projection to latent
        self.transform_layer = nn.Linear(f_dim, args.latdim, bias=True)
        nn.init.xavier_normal_(self.transform_layer.weight, gain=1.414)

        # -------- Attention-based fusion modules --------
        
        self.max_relations = getattr(args, 'max_relations', 10)  
        self.attention_heads = getattr(args, 'attention_heads', args.head)  
        self.attention_dropout = getattr(args, 'attention_dropout', args.dropRate)  
        
        self.relation_fusion = AttentionBasedRelationFusion(
            args.latdim, self.max_relations, 
            num_heads=self.attention_heads, 
            dropout=self.attention_dropout
        )
        
        
        self.final_fusion = nn.Sequential(
            nn.Linear(args.latdim * 2, args.latdim),
            nn.ReLU(),
            nn.Dropout(self.attention_dropout),
            nn.Linear(args.latdim, args.latdim)
        )

        # classifier
        self.dense = nn.Linear(args.latdim, 2)

    def process_group(self, module_list, he_adjs, idx, initial):
        """
        Apply a group of GT layers on he_adjs[idx] with residual-sum over depths.
        """
        g = he_adjs[idx] if idx < len(he_adjs) else he_adjs[-1]
        outputs = [initial]
        for layer in module_list:
            out = layer(g, outputs[-1])
            out = F.normalize(out, p=2, dim=1)
            outputs.append(out)
        return sum(outputs)

    def forward(self, he_adjs, feature_tensor, is_training=True):
        """
        he_adjs: list of DGLGraphs
        For MGTAB: [Follower, Friend, Mention, Reply, Quote, Other]
        - First fusion: Friend + Mention + Reply + Quote + Other
        - Second fusion: Friend + first_fused_result
        - Target: Follower (he_adjs[0])
        """
        # project raw features to latent
        embed = self.transform_layer(feature_tensor)

        # target embedding from Follower graph (he_adjs[0])
        target_embedding = self.process_group(self.main_layers, he_adjs, 0, embed)

        # First fusion: Friend + Mention + Reply + Quote + Other (he_adjs[1:])
        first_aux_list = []
        first_attention_weights = None
        first_importance_weights = None
        
        if len(he_adjs) > 1:
            
            for i in range(1, len(he_adjs)):
                group = self.helayers[i - 1] if (i - 1) < len(self.helayers) else self.helayers[-1]
                aux_i = self.process_group(group, he_adjs, i, embed)
                first_aux_list.append(aux_i)

            # First attention fusion
            if len(first_aux_list) > 0:
                first_aux_stack = torch.stack(first_aux_list, dim=1)
                first_relation_types = torch.arange(len(first_aux_list), device=first_aux_stack.device)
                
                first_fused, first_attention_weights, first_importance_weights = self.relation_fusion(
                    first_aux_stack, first_relation_types
                )
            else:
                first_fused = torch.zeros_like(target_embedding)
        else:
            first_fused = torch.zeros_like(target_embedding)

        # Second fusion: Follower + first_fused_result
        second_attention_weights = None
        second_importance_weights = None
        
        if len(he_adjs) > 0:  
            
            # Second attention fusion: Follower + first_fused
            second_input = torch.stack([target_embedding, first_fused], dim=1)  # [batch, 2, hidden_dim]
            second_relation_types = torch.tensor([0, 1], device=second_input.device)  # [Follower, First_Fused]
            
            aux_fused, second_attention_weights, second_importance_weights = self.relation_fusion(
                second_input, second_relation_types
            )
        else:
            aux_fused = first_fused

        return aux_fused, target_embedding, (first_attention_weights, first_importance_weights), (second_attention_weights, second_importance_weights)

    def _final_fusion(self, target_embedding, aux_refined):
        """
        Final fusion by concatenation + MLP
        """
        fused = torch.cat([target_embedding, aux_refined], dim=-1)
        fused = self.final_fusion(fused)
        return fused

    def cal_loss(self, ancs, label, he_adjs, initial_feature):
        """
        Training loss:
          - diffusion loss (predict target from corrupted aux_fused)
          - classification loss over attention-fused embeddings
        """
        aux_fused, target_embedding, first_attention, second_attention = self.forward(he_adjs, initial_feature)

        # diffusion: denoise aux toward target
        diff_loss, diff_embeddings = self.diffusion_model.training_losses2(
            self.user_denoise_model, target_embedding, aux_fused, ancs
        )
        diff_loss = diff_loss.mean()

        # final fusion (concatenation + MLP)
        all_embeddings = self._final_fusion(target_embedding, diff_embeddings)

        scores = self.dense(all_embeddings)
        scores = F.log_softmax(scores, dim=1)

        batch_u = scores[ancs]
        batch_label = torch.argmax(label[ancs], dim=-1)
        nll_loss = F.nll_loss(batch_u, batch_label)
        return nll_loss, diff_loss

    def get_embeds(self, ancs, label, he_adjs, initial_feature):
        aux_fused, target_embedding, _, _ = self.forward(he_adjs, initial_feature)
        diff_embeddings = self.diffusion_model.p_sample(self.user_denoise_model, aux_fused, args.sampling_steps)

        # final fusion during inference
        all_embeddings = self._final_fusion(target_embedding, diff_embeddings)
        return all_embeddings[ancs]

    def get_allembeds(self, he_adjs, initial_feature, label=None, save_path=None):
        """
        Compute all-node embeddings and scores; optionally save labels and embeddings.
        """
        aux_fused, target_embedding, first_attention, second_attention = self.forward(he_adjs, initial_feature)
        diff_embeddings = self.diffusion_model.p_sample(self.user_denoise_model, aux_fused, args.sampling_steps)

        # final fusion during inference
        all_embeddings = self._final_fusion(target_embedding, diff_embeddings)
        scores = self.dense(all_embeddings)

        if label is not None and save_path is not None:
            torch.save({
                'labels': label, 
                'all_embeddings': all_embeddings,
                'first_attention': first_attention,
                'second_attention': second_attention
            }, save_path)

        return all_embeddings, scores

    def get_attention_analysis(self, he_adjs, initial_feature):
       
        with torch.no_grad():
            aux_fused, target_embedding, first_attention, second_attention = self.forward(he_adjs, initial_feature)
            
            first_attention_weights, first_importance_weights = first_attention
            second_attention_weights, second_importance_weights = second_attention
            
            return {
                'first_fusion': {
                    'attention_weights': first_attention_weights.cpu().numpy() if first_attention_weights is not None else None,
                    'importance_weights': first_importance_weights.cpu().numpy() if first_importance_weights is not None else None,
                    'relation_names': ['Friend', 'Mention', 'Reply', 'Quote', 'Other']
                },
                'second_fusion': {
                    'attention_weights': second_attention_weights.cpu().numpy() if second_attention_weights is not None else None,
                    'importance_weights': second_importance_weights.cpu().numpy() if second_importance_weights is not None else None,
                    'relation_names': ['Follower', 'First_Fused']
                },
                'target_embedding': target_embedding,
                'aux_fused': aux_fused
            }
