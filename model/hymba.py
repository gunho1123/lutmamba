import sys
import time
import math
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn

from quant_methods.gptq import *
from quant_methods.shiftaddllm import *
from modelutils import *
from parsers import parse_args

from quantizers.quant import *
from quant_methods.quant_model_bcq import quant_model
from quantizers.bcq_quant.quantizer import BCQuantizer
from lut_gemm.kernel import load_shiftaddllm_weight

from transformers import AutoTokenizer
from model.modeling_hymba import HymbaForCausalLM

def get_hymba(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    from transformers import AutoConfig

    # config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    # config.use_mamba_kernels=False
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto', trust_remote_code=True)
    # model = AutoModelForCausalLM.from_config(config)
    model.seqlen = 2048
    return model

@torch.no_grad()
def hymba_sequential(model, dataloader, dev):
    print('Starting Hymba quantization ...')

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False

    def _get(obj, path_list):
        for p in path_list:
            cur = obj
            ok = True
            for name in p.split("."):
                if not hasattr(cur, name):
                    ok = False
                    break
                cur = getattr(cur, name)
            if ok:
                return cur, p
        return None, None

    # Hymba 모델의 레이어 구조 탐색
    layers, layers_prefix = _get(model, [
        "backbone.layers",
        "model.layers", 
        "layers",
    ])
    if layers is None:
        raise RuntimeError("Could not locate layers for Hymba model.")

    embeddings, emb_path = _get(model, [
        "backbone.embeddings",
        "model.embed_tokens",
        "embed_tokens",
    ])

    final_norm, norm_path = _get(model, [
        "backbone.norm_f",
        "backbone.norm",
        "model.norm",
        "norm",
    ])

    if embeddings is not None:
        setattr(eval("model." + emb_path), "", None)  # noop to keep linter calm
        cur = model
        for k in emb_path.split(".")[:-1]:
            cur = getattr(cur, k)
        setattr(cur, emb_path.split(".")[-1], getattr(cur, emb_path.split(".")[-1]).to(dev))
    if final_norm is not None:
        cur = model
        for k in norm_path.split(".")[:-1]:
            cur = getattr(cur, k)
        setattr(cur, norm_path.split(".")[-1], getattr(cur, norm_path.split(".")[-1]).to(dev))

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen+model.config.num_memory_tokens, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, hidden_states, *f_args, **f_kwargs):
            # print("================")
            # print(hidden_states.shape)
            # print("================")
            inps[cache['i']] = hidden_states
            cache['i'] += 1
            cache['attention_mask'] = f_kwargs.get('attention_mask', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model_inputs = {"input_ids": batch[0].to(dev)}
            if isinstance(batch, (list, tuple)) and len(batch) > 1:
                model_inputs["attention_mask"] = batch[1].to(dev)
            model(**model_inputs)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if embeddings is not None:
        cur = model
        for k in emb_path.split(".")[:-1]:
            cur = getattr(cur, k)
        setattr(cur, emb_path.split(".")[-1], getattr(cur, emb_path.split(".")[-1]).cpu())
    if final_norm is not None:
        cur = model
        for k in norm_path.split(".")[:-1]:
            cur = getattr(cur, k)
        setattr(cur, norm_path.split(".")[-1], getattr(cur, norm_path.split(".")[-1]).cpu())
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    quant_config_dict = None
    if args.quant_config:
        import json
        with open(args.quant_config, "r") as f:
            quant_config_dict = json.load(f)
        print(f"quant_config: {quant_config_dict}")

    print('Ready.')

    quantizers = {}

    # ----- Hymba 레이어별 처리 -----
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        # Hymba는 Transformer와 Mamba를 결합한 하이브리드 구조
        full = {n: m for n, m in find_layers(layer).items() if isinstance(m, nn.Linear)}

        if args.true_sequential:
            # Hymba의 하이브리드 구조를 고려한 그룹핑
            # Transformer attention 관련
            group1 = [n for n in full if any(x in n for x in ["q_proj", "k_proj", "v_proj", "o_proj"])]
            # Mamba mixer 관련  
            group2 = [n for n in full if "mixer" in n and n.endswith("in_proj")]
            group3 = [n for n in full if "mixer" in n and n.endswith("out_proj")]
            # 기타 MLP 관련
            group4 = [n for n in full if any(x in n for x in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2"])]
            others = [n for n in full if n not in group1 + group2 + group3 + group4]
            
            sequential = []
            if group1: sequential.append(group1)
            if group2: sequential.append(group2)
            if group3: sequential.append(group3)
            if group4: sequential.append(group4)
            if others: sequential.append(others)
            if not sequential:
                sequential = [list(full.keys())]
        else:
            sequential = [list(full.keys())]

        key_prefix = layers_prefix

        for names in sequential:
            subset = {n: full[n] for n in names}

            quant_method = {}
            for name in subset:
                if args.gptq or args.lut_eval:
                    quant_method[name] = GPTQ(subset[name])
                else:
                    quant_method[name] = ShiftAddLLM(subset[name])

                if quant_config_dict is not None:
                    key1 = f"model.layers.{i}.{name}"
                    key2 = f"{key_prefix}.{i}.{name}"
                    if key1 in quant_config_dict:
                        wbits = quant_config_dict[key1]["bits"]
                    elif key2 in quant_config_dict:
                        wbits = quant_config_dict[key2]["bits"]
                    else:
                        wbits = args.wbits
                else:
                    wbits = args.wbits

                if args.gptq:
                    quant_method[name].quantizer = Quantizer()
                    quant_method[name].quantizer.configure(
                        wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
                    )
                else:
                    quant_method[name].quantizer = BCQuantizer(
                        subset[name].weight.data.size(),
                        groupsize=args.groupsize,
                        wbits=wbits,
                        rounds=args.bcq_round,
                        use_bst=args.use_bst,
                        apot_nums=args.apot_nums,
                    )

            def add_batch(name):
                def hook_fn(_, inp, out):
                    x = inp[0].data
                    y = out.data if torch.is_tensor(out) else out[0].data
                    quant_method[name].add_batch(x, y)
                return hook_fn

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                hs = inps[j].unsqueeze(0)
                ret = layer(hs)  # Hymba block forward
                out = ret[0] if isinstance(ret, (list, tuple)) else ret
                outs[j] = out

            for h in handles:
                h.remove()
            for name in subset:
                quant_method[name].post_batch()

            for name in subset:
                # Hymba의 Mamba 부분과 Transformer 부분을 구분하여 처리
                if any(x in name for x in ["dt_proj", "mixer"]) or "mixer" in name:
                    print(" ====== Hymba Mamba Layer ", i, name, " ====== ")
                    quant_method[name].preproc(
                        preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                        preproc_rescale=args.pre_rescale,
                        preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra
                    )
                    quant_method[name].fasterquant(
                        args, model_name=str(args.model).split("/")[-1], layer_name=f"{i}.{name}"
                    )

                    save_key = f"{key_prefix}.{i}.{name}"
                    quantizers[save_key] = quant_method[name].quantizer
                    quant_method[name].free()
                else:
                    print(" ====== SKIP Hymba Layer ", i, name, " ====== ")

            for j in range(args.nsamples):
                hs = inps[j].unsqueeze(0)
                ret = layer(hs)
                out = ret[0] if isinstance(ret, (list, tuple)) else ret
                outs[j] = out

            layers[i] = layer.cpu()
            del layer
            del quant_method
            torch.cuda.empty_cache()

            inps, outs = outs, inps

    if use_cache is not None:
        model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def hymba_eval(model, testenc, dev):
    print('Evaluating (Hymba) ...')

    # testenc: BatchEncoding or tensor
    input_ids = getattr(testenc, "input_ids", testenc)
    if input_ids.dim() == 2:
        nsamples = input_ids.size(1) // model.seqlen
    else:
        nsamples = input_ids.numel() // model.seqlen

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False

    def _find_first(obj, paths):
        for p in paths:
            cur = obj
            ok = True
            for name in p.split("."):
                if not hasattr(cur, name):
                    ok = False
                    break
                cur = getattr(cur, name)
            if ok:
                return cur, p
        return None, None

    def _resolve(obj, path):
        cur = obj
        for k in path.split("."):
            cur = getattr(cur, k)
        return cur

    # Hymba 모델의 레이어 구조 탐색
    layers, layers_path = _find_first(model, ["backbone.layers", "model.layers", "layers"])
    if layers is None:
        raise RuntimeError("Could not locate layers for Hymba model.")

    embeddings, emb_path = _find_first(model, ["backbone.embeddings", "model.embed_tokens", "embed_tokens"])
    final_norm, norm_path = _find_first(model, ["backbone.norm_f", "backbone.norm", "model.norm", "norm"])
    lm_head, head_path = _find_first(model, ["lm_head", "model.lm_head"])
    if lm_head is None:
        raise RuntimeError("Could not locate lm_head for Hymba model.")

    if embeddings is not None:
        embeddings.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    inps = torch.zeros((nsamples, model.seqlen, hidden_size), dtype=dtype, device=dev)

    cache = {'i': 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, hidden_states, *args, **kwargs):
            inps[cache['i']] = hidden_states
            cache['i'] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        sl = slice(i * model.seqlen, (i + 1) * model.seqlen)
        batch = input_ids[:, sl].to(dev) if input_ids.dim() == 2 else input_ids[sl].unsqueeze(0).to(dev)
        try:
            model(input_ids=batch)
        except ValueError:
            pass

    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if embeddings is not None:
        embeddings.to('cpu')
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            hs = inps[j].unsqueeze(0)      # (1, T, H)
            ret = layer(hs)                # Tensor or (Tensor, ...)
            out = ret[0] if isinstance(ret, (tuple, list)) else ret
            outs[j] = out
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps  # swap

    norm_mod = _resolve(model, norm_path) if final_norm is not None and norm_path is not None else None
    head_mod = _resolve(model, head_path)
    if norm_mod is not None:
        norm_mod.to(dev)
    head_mod.to(dev)

    input_ids = input_ids.to(dev)
    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    vocab_dtype = getattr(getattr(head_mod, "weight", None), "dtype", None)

    for i in range(nsamples):
        hs = inps[i].unsqueeze(0)          # (1, T, H)
        if norm_mod is not None:
            hs = norm_mod(hs)
        if vocab_dtype is not None and hs.dtype != vocab_dtype:
            hs = hs.to(vocab_dtype)

        lm_logits = head_mod(hs)           # (1, T, V)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        sl = slice(i * model.seqlen, (i + 1) * model.seqlen)
        shift_labels = (input_ids[:, sl] if input_ids.dim() == 2 else input_ids[sl].unsqueeze(0))[:, 1:]
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        nlls.append(loss.float() * model.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    if use_cache is not None:
        model.config.use_cache = use_cache


if __name__ == '__main__':
    from datautils import *
    args = parse_args()

    if args.temp_storage is not None:
        os.makedirs(args.temp_storage, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = get_hymba(args.model)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    model.eval().cuda()
    print(model)

    if args.load_temp_storage is not None:
        assert args.block_quant, "temp_storage only work for blockwise (i.e lat. method) quantization"
        load_shiftaddllm_weight(model, args.load_temp_storage, model_name=str(args.model).split("/")[-1],
                                wbits=args.wbits, groupsize=args.groupsize)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    
    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        if args.bcq:
            print("quantizing with bcq")
            model = quant_model(model, qbits=args.wbits, group_size=args.groupsize)
        else:
            quantizers = hymba_sequential(model, dataloader, DEV)
        print("full quantization time: ",time.time() - tick)
    
    if args.save:
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)

    datasets = ['wikitext2'] 
    # datasets = ['wikitext2', 'ptb'] 
    # if args.new_eval:
    #     datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        hymba_eval(model, testloader, DEV)


