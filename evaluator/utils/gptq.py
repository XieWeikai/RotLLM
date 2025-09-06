import torch
import math
import time
import copy
import logging
import pprint
import torch.nn as nn
from tqdm import tqdm


from utils.utils import cleanup_memory
from train.config import QuantizeConfig
from .weight_quant import WeightQuantizer


def find_qlayers(module, layers=[nn.Linear, nn.Embedding], name: str = ""):
    if type(module) in [nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        Scale = self.layer.weight.data.clone()
        Scale = Scale.float()
        W_int = self.layer.weight.data.clone()
        W_int = W_int.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            W_int1 = torch.zeros_like(W1)
            Scale1 = torch.zeros_like(W1).to(Scale.dtype)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q, int_weight, scale = self.quantizer.fake_quantize(w.unsqueeze(1))
                Q1[:, i] = q.flatten()
                q = q.flatten()
                W_int1[:, i] = int_weight.flatten()
                Scale1[:, i] = scale.flatten()

                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W_int[:, i1:i2] = W_int1
            Scale[:, i1:i2] = Scale1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")

            pprint.pprint(
                self.quantizer.config.num_bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        cleanup_memory(verbos=True)


@torch.no_grad()
def gptq_fwrd(model, dataloader, wConfig: QuantizeConfig):
    """
    From GPTQ repo
    """
    logging.info("-----GPTQ Quantization-----")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    dev = model.device

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (wConfig.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]

    quantizers = {}
    sequential = [
        [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
        ],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    for i in tqdm(range(len(layers)), desc="(GPTQ Quant.) Layers"):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i]

        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                wQuantConfig = copy.deepcopy(wConfig)

                if "lm_head" in name:
                    wQuantConfig.num_bits = 16
                if wQuantConfig.int8_down_proj and "down_proj" in name:
                    wQuantConfig.num_bits = 8

                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer(wQuantConfig)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(wConfig.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=wConfig.percdamp,
                    groupsize=wConfig.groupsize,
                    actorder=wConfig.act_order,
                    static_groups=False,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(wConfig.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]

        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers