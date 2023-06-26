# Modified from https://github.com/KaiyangZhou/CoOp

import torch.nn as nn
import clip
import torch
from collections import OrderedDict
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class PromptLearner(nn.Module):
    def __init__(self, clip_model, 
                 train_classnames, val_classnames, val_on_train_classnames=None, 
                 use_neg_classname=False,
                 n_ctx=8, device="cuda",
                 mode="single", cocoop=False):
        super().__init__()
        self.mode = mode
        self.cocoop = cocoop
        assert self.mode in ["single", "dual"]
        
        n_train_cls = len(train_classnames)
        n_val_cls = len(val_classnames)
        if val_on_train_classnames is not None:
            n_val_on_train_cls = len(val_on_train_classnames)
        else:
            n_val_on_train_cls = 0
        if use_neg_classname: # first half of classnames are positive, second half are negative
            n_train_cls = n_train_cls // 2
            n_val_cls = n_val_cls // 2
            n_val_on_train_cls = n_val_on_train_cls // 2
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        
        prompt_prefix = " ".join(["X"] * n_ctx)  
        ctx_vectors = torch.empty(n_ctx, ctx_dim).to(device)
        nn.init.normal_(ctx_vectors, std=0.02) 
        self.ctx = nn.Parameter(ctx_vectors)
        if mode == "dual":
            # positive and negative prompts
            ctx_vectors = torch.empty(n_ctx, ctx_dim).to(device)
            nn.init.normal_(ctx_vectors, std=0.02) 
            self.ctx_neg = nn.Parameter(ctx_vectors)
        else:
            self.ctx_neg = None
            
        self.use_neg_classname = use_neg_classname
        
        train_classnames = [name.replace("_", " ") for name in train_classnames]
        train_prompts = [prompt_prefix + " " + name + "." for name in train_classnames]
        val_classnames = [name.replace("_", " ") for name in val_classnames]
        val_prompts = [prompt_prefix + " " + name + "." for name in val_classnames]
        if val_on_train_classnames is not None:
            val_on_train_classnames = [name.replace("_", " ") for name in val_on_train_classnames]
            val_on_train_prompts = [prompt_prefix + " " + name + "." for name in val_on_train_classnames]
        else:
            val_on_train_classnames = None
            val_on_train_prompts = None
        
        train_tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in train_prompts]).to(device)  # (n_train_cls, n_tkn)
        val_tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in val_prompts]).to(device)  # (n_val_cls, n_tkn)
        if val_on_train_classnames is not None:
            val_on_train_tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in val_on_train_prompts]).to(device)  # (n_val_on_train_cls, n_tkn)
        else:
            val_on_train_tokenized_prompts = None
        
        with torch.no_grad():
            train_embedding = clip_model.token_embedding(train_tokenized_prompts).type(dtype)
            val_embedding = clip_model.token_embedding(val_tokenized_prompts).type(dtype)
            if val_on_train_tokenized_prompts is not None:
                val_on_train_embedding = clip_model.token_embedding(val_on_train_tokenized_prompts).type(dtype)
            else:
                val_on_train_embedding = None

        if self.cocoop:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
            ])).to(device)
            if mode == "dual":
                self.meta_net_neg = nn.Sequential(OrderedDict([
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
                ])).to(device)
            else:
                self.meta_net_neg = None
        else:
            self.meta_net = None # prompt is input-image-agnostic
            self.meta_net_neg = None

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("train_token_prefix", train_embedding[:, :1, :])  # SOS
        self.register_buffer("train_token_suffix", train_embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("val_token_prefix", val_embedding[:, :1, :])  # SOS
        self.register_buffer("val_token_suffix", val_embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        if val_on_train_embedding is not None:
            self.register_buffer("val_on_train_token_prefix", val_on_train_embedding[:, :1, :])
            self.register_buffer("val_on_train_token_suffix", val_on_train_embedding[:, 1 + n_ctx :, :])

        self.n_train_cls = n_train_cls
        self.n_val_cls = n_val_cls
        self.n_val_on_train_cls = n_val_on_train_cls
        self.n_ctx = n_ctx
        self.train_tokenized_prompts = train_tokenized_prompts  # torch.Tensor
        self.val_tokenized_prompts = val_tokenized_prompts  # torch.Tensor
        self.val_on_train_tokenized_prompts = val_on_train_tokenized_prompts  # torch.Tensor
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, mode='train'):
        if mode == 'train':
            prefix = self.train_token_prefix
            suffix = self.train_token_suffix
            n_cls = self.n_train_cls
        elif mode == 'test':
            prefix = self.val_token_prefix
            suffix = self.val_token_suffix
            n_cls = self.n_val_cls
        elif mode == 'val_on_train':
            prefix = self.val_on_train_token_prefix
            suffix = self.val_on_train_token_suffix
            n_cls = self.n_val_on_train_cls
        else:
            raise NotImplementedError()
        
        ctx = self.ctx
        if self.cocoop:
            bias = self.meta_net(im_features).unsqueeze(1)  # (batch, 1, ctx_dim)
            ctx = ctx.unsqueeze(0) + bias  # (batch, n_ctx, ctx_dim)
        else:
            ctx = ctx.unsqueeze(0) # (1, n_ctx, ctx_dim)
        ctx_neg = self.ctx_neg
        if ctx_neg is not None:
            if self.cocoop:
                bias_neg = self.meta_net_neg(im_features).unsqueeze(1)  # (batch, 1, ctx_dim)
                ctx_neg = ctx_neg.unsqueeze(0) + bias_neg  # (batch, n_ctx, ctx_dim)
            else:
                ctx_neg = ctx_neg.unsqueeze(0) # (1, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_i in ctx:
            ctx_i = ctx_i.unsqueeze(0).expand(n_cls, -1, -1)  # (n_cls, n_ctx, ctx_dim)
            if self.use_neg_classname:
                pts_i = self.construct_prompts(ctx_i, prefix[:n_cls], suffix[:n_cls])
            else:
                pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts) # [d1, n_cls, n_tkn, ctx_dim] where d1 = 1 if cocoop=False; d1 = batch_size if cocoop=True
        
        prompts_neg = []
        if self.mode == "dual":
            for ctx_neg_i in ctx_neg:
                ctx_neg_i = ctx_neg_i.unsqueeze(0).expand(n_cls, -1, -1)  # (n_cls, n_ctx, ctx_dim)
                if self.use_neg_classname:
                    pts_neg_i = self.construct_prompts(ctx_neg_i, prefix[n_cls:], suffix[n_cls:])
                else:
                    pts_neg_i = self.construct_prompts(ctx_neg_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
                prompts_neg.append(pts_neg_i)
            prompts_neg = torch.stack(prompts_neg)
        
        if len(prompts_neg) > 0:
            return prompts, prompts_neg
        else:        
            return prompts
    
    

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x