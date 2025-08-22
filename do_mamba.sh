model_name_or_path=state-spaces/mamba-370m-hf


CUDA_VISIBLE_DEVICES=0 python model/mamba.py \
    $model_name_or_path \
    --wbits 4 \
    --groupsize 128 \
    --gptq \
    --save ${model_name_or_path}_q${wbits}_g${groupsize}_test

# CUDA_VISIBLE_DEVICES=0 python lmeval.py \
#     --model hf \
#     --model_args pretrained=${model_name_or_path}_q${wbits}_g${groupsize}_test,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda
