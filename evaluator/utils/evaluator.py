import torch
from tqdm import tqdm

@torch.no_grad()
def evaluator(model, testenc, seqlen, args):
    model.eval()
    dev = "cuda"

    use_cache = model.config.use_cache
    model.config.use_cache = False

    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // seqlen  
    input_ids = input_ids[:, : nsamples * seqlen].view(nsamples, seqlen).to(dev)

    batch_size = args.bsz
    input_batches = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_batches)

    model = model.to(dev)

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(nbatches), desc="(Eval) Forward"):
        batch = input_batches[i]
        outputs = model(batch)
        lm_logits = outputs.logits.detach()
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    model.config.use_cache = use_cache
    return ppl.item()

