import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


from .config import AllQuantizeConfigs
from .train_parameter import FakeQuantizer


class RotationQuantLinear(nn.Module):
    def __init__(self, config: AllQuantizeConfigs, linear: nn.Linear, num_bits=8, rotation_pos="none", R_pre = None, R_post = None):
        super().__init__()
        self.config = config
        self.num_bits = num_bits
        self.rotation_pos = rotation_pos
        self.R_pre = R_pre
        self.R_post = R_post

        self.linear = linear

        self.actQuant = FakeQuantizer(copy.deepcopy(self.config.activation))
        self.weightQuant = FakeQuantizer(copy.deepcopy(self.config.weight))
        self.biasQuant = FakeQuantizer(copy.deepcopy(self.config.bias))


    def forward(self, x):
        w = self.linear.weight
        b = self.linear.bias if self.linear.bias is not None else None

        if self.rotation_pos in ["pre", "around"]:
            assert w.shape[1] % self.R_pre.weight.shape[0] == 0, "Input dim should be multiple of R_pre dim"
            num_blocks = w.shape[1] // self.R_pre.weight.shape[0]

            w_dtype = w.dtype
            w_device = w.device
            w = w.view(w.shape[0], num_blocks, self.R_pre.weight.shape[0])
            w = (w.to(self.R_pre.weight.dtype) @ self.R_pre.weight.to(device=w_device)).to(dtype=w_dtype)   
            w = w.view(w.shape[0], num_blocks * self.R_pre.weight.shape[0])

        if self.rotation_pos in ["post", "around"]:
            assert w.shape[0] % self.R_post.weight.shape[0] == 0, "Output dim(weight) should be multiple of R_post dim"
            num_blocks = w.shape[0] // self.R_post.weight.shape[0]

            w_dtype = w.dtype
            w_device = w.device
            w = w.T
            w = w.view(w.shape[0], num_blocks, self.R_post.weight.shape[0])
            w = (w.to(self.R_post.weight.dtype) @ self.R_post.weight.to(device=w_device)).to(dtype=w_dtype)
            w = w.view(w.shape[0], num_blocks * self.R_post.weight.shape[0])
            w = w.T
            if b is not None:
                assert b.shape[0] % self.R_post.weight.shape[0] == 0, "Output dim(bias) should be multiple of R_post dim"
                b_dtype = b.dtype
                b_device = b.device
                b = (b.to(self.R_post.weight.dtype).view(num_blocks, -1) @ self.R_post.weight.to(device=b_device)).to(dtype=b_dtype)
                b = b.view(-1)

        x_q, w_q, b_q = self.allQuant(x, w, b)  # FakeQuant

        # x_q = x
        # w_q = w
        # b_q = b

        y = F.linear(x_q, w_q, b_q)
        return y
    

    def allQuant(self, x, w, b):
        # Activation:
        x_q = self.actQuant(x)

        # Weight:
        w_q = self.weightQuant(w)

        # bias:
        if b is not None:
            b_q = self.biasQuant(b)
        else:
            b_q = None
            
        return x_q, w_q, b_q
    


class RotationEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, rotation_pos="none", R_pre = None, R_post = None):
        super().__init__()
        self.embedding = embedding
        self.rotation_pos = rotation_pos
        self.R_pre = R_pre
        self.R_post = R_post

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: LongTensor [batch_size, seq_len]
        return: FloatTensor [batch_size, seq_len, hidden_size]
        """
        # Get the original embedding
        embeds = self.embedding(input_ids)  # [B, L, D]

        assert self.rotation_pos not in ["pre", "around"], "An error occurred in the rotation position of the embedding layer."
            
        if self.rotation_pos in ["post"]:
            assert embeds.shape[-1] == self.R_post.weight.shape[0], "R should be same size as dim of output activation"
            embeds_dtype = embeds.dtype
            embeds_device = embeds.device
            embeds = (embeds.to(dtype=self.R_post.weight.dtype) @ self.R_post.weight.to(device=embeds_device)).to(dtype=embeds_dtype)
        return embeds
