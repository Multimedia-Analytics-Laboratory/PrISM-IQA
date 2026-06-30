import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip


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


class PromptLearner(nn.Module):
    def __init__(self, clip_model, attributes, n_ctx=16, specific=True):
        super().__init__()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        n_attr = len(attributes)
        all_attributes = attributes

        # random initialization
        if specific:
            print("Initializing specific context embeddings")
            ctx_vectors = torch.empty(n_attr, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        else:
            print(f"Initializing a generic context embedding")
            ctx_vectors = torch.empty(1, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        all_attributes = [name.replace("_", " ") for name in all_attributes]
        prompts = [prompt_prefix + " " + name + "." for name in all_attributes]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_attr = n_attr
        self.spec = specific
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        specific = self.spec

        if specific:
            prompts = torch.cat(
                [
                    prefix,    # (n_attr, 1, dim)
                    self.ctx,  # (n_attr, n_ctx, dim)
                    suffix,    # (n_attr, *, dim)
                ],
                dim=1,
            )
        else:
            ctx = self.ctx.expand(self.n_attr, -1, -1)
            prompts = torch.cat(
                [
                    prefix,  # (n_attr, 1, dim)
                    ctx,     # (n_attr, n_ctx, dim)
                    suffix,  # (n_attr, *, dim)
                ],
                dim=1,
            )

        return prompts

def inv_softplus(x: torch.Tensor) -> torch.Tensor:
# softplus^{-1}(x) = log(exp(x) - 1)
    return torch.log(torch.expm1(x))

class CustomCLIP(nn.Module):
    def __init__(self, clip_model, attributes, n_ctx=16, n_binary=3, specific=True):
        super().__init__()
        self.prompt_learner = PromptLearner(clip_model, attributes, n_ctx=n_ctx, specific=specific)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        #self.logit_scale = clip_model.logit_scale
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.dtype = clip_model.dtype
        # self.delta = nn.Parameter(
        #     torch.log(torch.full((len(attributes), n_binary), 0.01))
        # )
        # self.base = nn.Parameter(torch.zeros(len(attributes), 1))

        A = len(attributes)
        # ===== scheme A: thresholds in cosine domain =====
        t1, t2, t3 = -0.15, 0.0, 0.15
        gap12 = t2 - t1  # 0.4
        gap23 = t3 - t2  # 0.4

        self.tau1 = nn.Parameter(torch.full((A, 1), t1))
        init_gaps = torch.tensor([gap12, gap23], dtype=torch.float32).view(1, 2).repeat(A, 1)
        self.tau_gap_raw = nn.Parameter(inv_softplus(init_gaps))  # so softplus(raw)=gap

    # def forward(self, image):
    #     image_features = self.image_encoder(image.type(self.dtype))
    #
    #     prompts = self.prompt_learner()
    #     tokenized_prompts = self.tokenized_prompts
    #     text_features = self.text_encoder(prompts, tokenized_prompts)
    #
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #
    #     logit_scale = self.logit_scale.exp()
    #     logits = logit_scale * image_features @ text_features.t()
    #
    #     # add
    #     delta = self.delta.exp()
    #     delta_cum = torch.cumsum(delta, dim=1)
    #     # logits_1 = logits.unsqueeze(-1) - delta_cum
    #     # logits_1 = logits_1.reshape(logits_1.shape[0], -1)
    #     gamma = self.base + delta_cum  # (A,3) 每个 attribute 整体平移
    #     logits_1 = logits.unsqueeze(-1) - gamma
    #     logits_1 = logits_1.reshape(logits_1.shape[0], -1)
    #     return logits_1

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        alpha = self.logit_scale.exp()  # scalar
        #alpha = 10
        cos = image_features @ text_features.t()  # (B, A)

        gaps = F.softplus(self.tau_gap_raw) + 1e-4  # (A,2) > 0
        tau2 = self.tau1 + gaps[:, 0:1]
        tau3 = tau2 + gaps[:, 1:2]
        tau = torch.cat([self.tau1, tau2, tau3], dim=1)  # (A,3)

        # optional: 防止tau跑飞（tanh单调，不破坏有序）
        # tau = torch.tanh(tau)

        logits_1 = alpha * (cos.unsqueeze(-1) - tau)  # (B,A,3)
        logits_1 = logits_1.reshape(logits_1.shape[0], -1)  # (B, 3A=21)
        return logits_1
