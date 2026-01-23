import torch
from tqdm import tqdm


@torch.no_grad()
def evaluator(model, testenc, seqlen, args):
    model.eval()
    dev = model.device

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i].cpu()
        del layer

    torch.cuda.empty_cache()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // seqlen  # The tail is truncated.
    input_ids = (
        input_ids[:, : nsamples * seqlen].view(nsamples, seqlen).to(dev)
    )  # (nsamples, seqlen)

    batch_size = args.bsz
    input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)


    inps = [0] * nbatches
    cache = {"i": 0, "kwargs": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            # ✅ 关键：传 Qwen2 需要的属性
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    
    outs = [0] * nbatches
    kwargs = cache["kwargs"]

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        for j in range(nbatches):
            outs[j] = layer(
                inps[j],
                **kwargs
            )
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    del outs
    torch.cuda.empty_cache()

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(nbatches), desc="(Eval) PPL"):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        lm_logits = lm_logits.cpu()
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:].cpu()
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    return ppl.item()