# import torch, pdb
# import torch.nn as nn
# import numpy as np
# from typing import Optional
#
#
# class pLoss_all_fidelity(nn.Module):
#     def __init__(self, states, alpha=0.8, gamma=2):
#         super(pLoss_all_fidelity, self).__init__()
#         self.register_buffer("legal_state", states.float())
#         self.my_loss = fidelity_loss_soft(gamma=gamma, alpha=alpha)
#
#     def crf(self, p):
#         s = self.legal_state
#         potential = s @ p.T  # s: n_state*n_node, p: n_batch*n_node
#         max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
#         J = torch.exp(potential - max_sf)  # J: n_state*n_batch
#         z_ = torch.sum(J, dim=0)
#
#         p_margin = torch.zeros_like(p)
#         for i in range(p.shape[1]):
#             p_margin[:, i] = torch.sum(J[s[:, i] > 0, :], dim=0) / z_
#
#         # add (only for 4 levels)
#         p_margin = p_margin.reshape(-1, p_margin.shape[1] // 3, 3)
#         p_margin_2 = p_margin.new_zeros(p_margin.shape[0], p_margin.shape[1], 4)
#         p_margin_2[..., 0] = 1.0 - p_margin[..., 0]
#         p_margin_2[..., 1] = p_margin[..., 0] - p_margin[..., 1]
#         p_margin_2[..., 2] = p_margin[..., 1] - p_margin[..., 2]
#         p_margin_2[..., 3] = p_margin[..., 2]
#
#         return p_margin_2
#
#     def forward(self, p, g):
#         g = g.float()
#         p_margin = self.crf(p)
#         all_loss = self.my_loss(p_margin, g)
#
#         return all_loss, p_margin
#
#
# class fidelity_loss_soft(torch.nn.Module):
#     def __init__(self, gamma: float = 2.5, alpha: Optional[torch.Tensor] = None):
#         super(fidelity_loss_soft, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.eps = 1e-8
#         """
#         muti-attribute, muti-class
#         p: (B, A, C)  A=6 attributes, C=4 classes
#         g: (B, A, C)  soft labels, sum over C = 1
#         gamma: >= 1
#         alpha: 原始逆频率，做均值归一化（mean = 1）
#         - None: no class balancing
#         - (C,): per-class weights shared across attributes
#         - (A, C): per-attribute per-class weights
#         """
#
#     def forward(self, p, g):
#         # p_hat = F.softmax(logits, dim=-1)                  # (B, 6, 4)
#         # p = target_dist
#
#         B, A, C = p.shape
#         assert g.shape == (B, A, C)
#
#         bc = torch.sum(torch.sqrt(p * g + self.eps), dim=-1)  # (B, A)
#         fid = 1.0 - bc
#
#         # focal modulation
#         focal_term = fid ** self.gamma
#
#         # alpha-balanced weighting (soft-label expectation)
#         if self.alpha is not None:
#             if self.alpha.dim() == 1:
#                 assert self.alpha.shape == (C,)
#                 alpha_eff = torch.sum(g * self.alpha.view(1, 1, C), dim=-1)  # (B, A)
#             elif self.alpha.dim() == 2:
#                 assert self.alpha.shape == (A, C)
#                 alpha_eff = torch.sum(g * self.alpha.view(1, A, C), dim=-1)  # (B, A)
#             else:
#                 raise ValueError("alpha must be None, shape (C,), or shape (A, C).")
#
#             loss = alpha_eff * focal_term
#         else:
#             loss = focal_term
#
#         return loss.mean()
#
#
#
# class Fidelity_Loss_binary(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.esp = 1e-8
#
#     def forward(self, p, g):
#         loss = 1 - (torch.sqrt(p * g + self.esp) + torch.sqrt((1 - p) * (1 - g) + self.esp))
#
#         return torch.mean(loss)
#
#
# class FocalFidelityLoss(torch.nn.Module):
#     def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
#         super(FocalFidelityLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#         self.esp = 1e-8
#
#     def forward(self, p, g):
#         g = g.view(-1, 1)
#         p = p.view(-1, 1)
#
#         fidelity = torch.sqrt(p * g + self.esp) + torch.sqrt((1 - p) * (1 - g) + self.esp)
#         p_t = p * g + (1 - p) * (1 - g)
#         alpha_t = self.alpha * g + (1 - self.alpha) * (1 - g)
#         loss = (1 - fidelity) * alpha_t * (1 - p_t) ** self.gamma
#         # loss = loss.sum(dim=1)
#         return loss.mean()
#
#
# # utils for pLoss
# def _check_abnornal(z_):
#     if np.inf in z_:
#         pdb.set_trace()
#         idx = z_ == np.inf
#         id_num = [i for i, v in enumerate(list(idx)) if v == True]
#     else:
#         id_num = [-1]
#     return id_num
#

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class pLoss_all_fidelity(nn.Module):
    def __init__(self, states, alpha=0.8, gamma=2):
        super(pLoss_all_fidelity, self).__init__()
        self.num_levels = 4
        self.num_binary = self.num_levels - 1  # 3
        self.factorized = isinstance(states, dict) and states.get("mode") == "factorized"

        if self.factorized:
            self.num_attrs = int(states["num_attrs"])
            self.num_bits = int(states["num_bits"])
            self.num_groups = len(states["local_states"])

            for group_idx in range(self.num_groups):
                self.register_buffer(
                    f"component_attr_idx_{group_idx}",
                    torch.tensor(states["component_attr_indices"][group_idx], dtype=torch.long),
                )
                self.register_buffer(
                    f"component_bit_idx_{group_idx}",
                    torch.tensor(states["component_bit_indices"][group_idx], dtype=torch.long),
                )
                self.register_buffer(
                    f"component_state_{group_idx}",
                    states["local_states"][group_idx].float(),
                )
                self.register_buffer(
                    f"component_level_{group_idx}",
                    states["local_levels"][group_idx].long(),
                )

            self.register_buffer(
                "mutex_attr_group_id",
                torch.tensor(states["mutex_attr_group_id"], dtype=torch.long),
            )
        else:
            self.register_buffer("legal_state", states.float())
            S, K = self.legal_state.shape
            self.num_attrs = K // self.num_binary
            self.num_bits = K
            state_levels = self.legal_state.view(S, self.num_attrs, 3).sum(dim=-1).long()
            self.register_buffer("state_levels", state_levels)

    def _validate_inputs(self, p: torch.Tensor, g: Optional[torch.Tensor] = None):
        if p.dim() != 2:
            raise ValueError(f"expected p to have shape (B, K), got {tuple(p.shape)}")
        if p.shape[1] != self.num_bits:
            raise ValueError(
                f"CRF expects {self.num_bits} binary logits, but got {p.shape[1]}. "
                "Please sync your label/group specification with the model output."
            )
        if g is not None:
            if g.dim() != 3:
                raise ValueError(f"expected g to have shape (B, A, 4), got {tuple(g.shape)}")
            if g.shape[1] != self.num_attrs or g.shape[2] != self.num_levels:
                raise ValueError(
                    f"CRF expects g to have shape (B, {self.num_attrs}, {self.num_levels}), "
                    f"but got {tuple(g.shape)}."
                )

    def _iter_factor_groups(self):
        for group_idx in range(self.num_groups):
            yield (
                getattr(self, f"component_attr_idx_{group_idx}"),
                getattr(self, f"component_bit_idx_{group_idx}"),
                getattr(self, f"component_state_{group_idx}"),
                getattr(self, f"component_level_{group_idx}"),
            )

    def _crf_global(self, p: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax((self.legal_state @ p.T).T, dim=1)  # (B,S)
        p_margin_2 = p.new_zeros(p.shape[0], self.num_attrs, self.num_levels)
        for level in range(self.num_levels):
            mask = (self.state_levels == level).to(probs.dtype)  # (S,A)
            p_margin_2[..., level] = probs @ mask
        return p_margin_2

    def _crf_factorized(self, p: torch.Tensor) -> torch.Tensor:
        p_margin_2 = p.new_zeros(p.shape[0], self.num_attrs, self.num_levels)

        for attr_idx, bit_idx, local_state, local_levels in self._iter_factor_groups():
            probs = torch.softmax((local_state @ p.index_select(1, bit_idx).T).T, dim=1)  # (B,Sg)
            for level in range(self.num_levels):
                mask = (local_levels == level).to(probs.dtype)  # (Sg,Ag)
                p_margin_2[:, attr_idx, level] = probs @ mask

        return p_margin_2

    def crf(self, p):
        """
        精确推断：
        - 老实现：对全局 legal states 做归一化
        - 新实现：若图由多个互不相连的小 group 组成，则按 group 分解精确计算
        """
        p = p.float()
        self._validate_inputs(p)
        if self.factorized:
            return self._crf_factorized(p)
        return self._crf_global(p)

    def _map_decode_bits_global(self, p: torch.Tensor) -> torch.Tensor:
        best_idx = torch.argmax(self.legal_state @ p.T, dim=0)
        return self.legal_state[best_idx]

    def _map_decode_bits_factorized(self, p: torch.Tensor) -> torch.Tensor:
        bits = p.new_zeros(p.shape[0], self.num_bits)
        for _, bit_idx, local_state, _ in self._iter_factor_groups():
            best_idx = torch.argmax(local_state @ p.index_select(1, bit_idx).T, dim=0)
            bits[:, bit_idx] = local_state[best_idx]
        return bits

    @torch.no_grad()
    def map_decode_bits(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: (B, K)  每个binary节点取1的logit (即你CustomCLIP输出的logits_1)
        return: bits (B, K)  取到的MAP合法state的0/1向量
        """
        p = p.float()
        self._validate_inputs(p)
        if self.factorized:
            return self._map_decode_bits_factorized(p)
        return self._map_decode_bits_global(p)

    def _map_decode_levels_global(self, p: torch.Tensor) -> torch.Tensor:
        best_idx = torch.argmax(self.legal_state @ p.T, dim=0)
        return self.state_levels[best_idx].to(torch.int64)

    def _map_decode_levels_factorized(self, p: torch.Tensor) -> torch.Tensor:
        levels = torch.zeros(p.shape[0], self.num_attrs, dtype=torch.int64, device=p.device)
        for attr_idx, bit_idx, local_state, local_levels in self._iter_factor_groups():
            best_idx = torch.argmax(local_state @ p.index_select(1, bit_idx).T, dim=0)
            levels[:, attr_idx] = local_levels[best_idx]
        return levels

    @torch.no_grad()
    def map_decode_levels(self, p: torch.Tensor) -> torch.Tensor:
        """
        return: levels (B, A) in {0,1,2,3}
        """
        p = p.float()
        self._validate_inputs(p)
        if self.factorized:
            return self._map_decode_levels_factorized(p)
        return self._map_decode_levels_global(p)

    @torch.no_grad()
    def infer_map(self, p):
        return self.map_decode_levels(p)

    # ===== 新增 helper 1：one-hot ordinal -> cumulative bits =====
    def _ordinal_onehot_to_cum_bits(self, g: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        g: (B,A,4) one-hot，表示 y in {1..4}（你的数据里 index 0..3，所以要 +1）
        return: bits (B, A*3) 对应 b_{j,k}=I[y_j > k], k=1..3
        """
        # y in {1..4}
        y = torch.argmax(g, dim=-1) + 1  # (B,A)

        ks = torch.arange(1, self.num_binary + 1, device=g.device).view(1, 1, -1)  # (1,1,3)
        bits = (y.unsqueeze(-1) > ks).to(dtype)  # (B,A,3)
        return bits.reshape(g.shape[0], -1)       # (B,K)

    # ===== 新增 helper 2：把 bits 变成 constrained mask =====
    def _mask_states_equal_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """
        bits: (B,K) with 0/1
        return: mask (B,S) where True means that legal_state[s] == bits[b]
        """
        # legal_state: (S,K) -> (1,S,K)
        # bits: (B,K) -> (B,1,K)
        return (self.legal_state.unsqueeze(0) == bits.unsqueeze(1)).all(dim=-1)

    # （可选）兼容你另一个 test.py 的 criterion.infer(...) 写法
    @torch.no_grad()
    def infer(self, p):
        return self.crf(p)

    def _get_attr_group_id(self, device: torch.device, num_attrs: int) -> torch.Tensor:
        if self.factorized:
            return self.mutex_attr_group_id.to(device)

        if num_attrs == 7:
            return torch.tensor([0, 0, 1, 1, 2, 2, 2], device=device, dtype=torch.long)

        return torch.arange(num_attrs, device=device, dtype=torch.long)

    def _weakify_zero_labels(self, g_onehot: torch.Tensor) -> torch.Tensor:
        """
        g_onehot: (B,A,4) one-hot (强标签)
        return:   g_adm  : (B,A,4) 0/1 admissible mask (弱标签集合)
          - y>0: {y}
          - y==0:
              * 若同互斥组内存在某个>0: {0} (强制0)
              * 否则若其他组存在>0: {0,1}
              * 否则: {0}
        """
        B, A, C = g_onehot.shape
        assert C == 4

        y = torch.argmax(g_onehot, dim=-1)  # (B,A) in {0,1,2,3}
        pos = (y > 0)

        gid = self._get_attr_group_id(y.device, A)
        G = int(gid.max().item()) + 1

        # 每个样本每个组是否存在正例
        group_pos = torch.stack([pos[:, gid == g].any(dim=1) for g in range(G)], dim=1)  # (B,G)
        same_group_has_pos = group_pos.gather(1, gid.view(1, A).expand(B, A))  # (B,A)
        any_pos = pos.any(dim=1, keepdim=True)  # (B,1)

        y0 = (y == 0)
        allow1 = y0 & (~same_group_has_pos) & any_pos  # (B,A)

        # 先按强标签初始化
        g_adm = torch.zeros_like(g_onehot, dtype=torch.float32)
        g_adm.scatter_(2, y.unsqueeze(-1), 1.0)

        # 对所有 y==0：允许 level 0
        g_adm[..., 0][y0] = 1.0

        # 对 allow1：额外允许 level 1
        g_adm[..., 1][allow1] = 1.0

        return g_adm

    def _allowed_local_states(self, g_bool: torch.Tensor, attr_idx: torch.Tensor, local_levels: torch.Tensor) -> torch.Tensor:
        group_mask = g_bool.index_select(1, attr_idx)  # (B,Ag,4)
        num_states = local_levels.shape[0]

        allowed = torch.gather(
            group_mask.unsqueeze(1).expand(-1, num_states, -1, -1),
            dim=3,
            index=local_levels.unsqueeze(0).expand(group_mask.shape[0], -1, -1).unsqueeze(-1),
        ).squeeze(-1)  # (B,Sg,Ag)

        return allowed.all(dim=-1)  # (B,Sg)

    def _apply_observed_mask(self, g_adm: torch.Tensor, observed_mask: Optional[torch.Tensor]):
        if observed_mask is None:
            return g_adm
        observed_mask = observed_mask.to(device=g_adm.device, dtype=torch.bool)
        if observed_mask.shape != g_adm.shape[:2]:
            raise ValueError(
                f"observed_mask should have shape {tuple(g_adm.shape[:2])}, "
                f"got {tuple(observed_mask.shape)}"
            )
        return torch.where(observed_mask.unsqueeze(-1), g_adm, torch.ones_like(g_adm))

    def _forward_global(self, p: torch.Tensor, g: torch.Tensor, observed_mask: Optional[torch.Tensor] = None):
        s = self.legal_state.float()  # (S,K)
        potential = s @ p.T  # (S,B)
        logZ = torch.logsumexp(potential, dim=0)  # (B,)

        g_adm = self._weakify_zero_labels(g)
        g_adm = self._apply_observed_mask(g_adm, observed_mask)
        g_bool = (g_adm > 0.5)

        mask = torch.gather(
            g_bool.unsqueeze(1).expand(-1, self.state_levels.shape[0], -1, -1),
            dim=3,
            index=self.state_levels.unsqueeze(0).expand(g.shape[0], -1, -1).unsqueeze(-1),
        ).squeeze(-1).all(dim=-1)  # (B,S)
        cnt = mask.any(dim=1)

        pot_masked = potential.T.masked_fill(~mask, -1e9)
        logZ_C = torch.logsumexp(pot_masked, dim=1)
        logZ_C = torch.where(cnt, logZ_C, torch.zeros_like(logZ_C))

        nll = torch.where(cnt, logZ - logZ_C, torch.zeros_like(logZ))
        loss = nll.mean()

        return loss, self.crf(p)

    def _forward_factorized(self, p: torch.Tensor, g: torch.Tensor, observed_mask: Optional[torch.Tensor] = None):
        g_adm = self._weakify_zero_labels(g)
        g_adm = self._apply_observed_mask(g_adm, observed_mask)
        g_bool = (g_adm > 0.5)

        total_logZ = p.new_zeros(p.shape[0])
        total_logZ_C = p.new_zeros(p.shape[0])
        valid = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)

        for attr_idx, bit_idx, local_state, local_levels in self._iter_factor_groups():
            potential = local_state @ p.index_select(1, bit_idx).T  # (Sg,B)
            total_logZ = total_logZ + torch.logsumexp(potential, dim=0)

            allowed = self._allowed_local_states(g_bool, attr_idx, local_levels)  # (B,Sg)
            has_allowed = allowed.any(dim=1)
            valid = valid & has_allowed

            local_logZ_C = torch.logsumexp(
                potential.T.masked_fill(~allowed, -1e9),
                dim=1,
            )
            local_logZ_C = torch.where(has_allowed, local_logZ_C, torch.zeros_like(local_logZ_C))
            total_logZ_C = total_logZ_C + local_logZ_C

        nll = torch.where(valid, total_logZ - total_logZ_C, torch.zeros_like(total_logZ))
        loss = nll.mean()

        return loss, self.crf(p)

    # ======= 需要“替换”的 forward：改成 marginal likelihood loss =======
    def forward(self, p, g, observed_mask: Optional[torch.Tensor] = None):
        """
        p: (B, K)            cumulative-binary logits
        g: (B, A, 4)         one-hot ordinal labels
        observed_mask: (B,A) True where the label is observed/applicable
        """
        g = g.float()
        p = p.float()
        self._validate_inputs(p, g)

        if self.factorized:
            return self._forward_factorized(p, g, observed_mask)
        return self._forward_global(p, g, observed_mask)


# 下面旧 loss 可以留着（不影响），也可以删掉
class fidelity_loss_soft(torch.nn.Module):
    def __init__(self, gamma: float = 2.5, alpha: Optional[torch.Tensor] = None):
        super(fidelity_loss_soft, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-8

    def forward(self, p, g):
        B, A, C = p.shape
        assert g.shape == (B, A, C)
        bc = torch.sum(torch.sqrt(p * g + self.eps), dim=-1)
        fid = 1.0 - bc
        focal_term = fid ** self.gamma
        if self.alpha is not None:
            if self.alpha.dim() == 1:
                alpha_eff = torch.sum(g * self.alpha.view(1, 1, C), dim=-1)
            elif self.alpha.dim() == 2:
                alpha_eff = torch.sum(g * self.alpha.view(1, A, C), dim=-1)
            else:
                raise ValueError("alpha must be None, shape (C,), or shape (A, C).")
            loss = alpha_eff * focal_term
        else:
            loss = focal_term
        return loss.mean()
