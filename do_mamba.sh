
model_name_or_path=state-spaces/mamba-370m-hf

wbits=3
groupsize=128
output_dir=${model_name_or_path}_q${wbits}_g${groupsize}_per_oc_test
mkdir $output_dir
GPU=0

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --save $output_dir
    # --save $output_dir > $output_dir/log.txt

# CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
#     --model hf \
#     --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda > $output_dir/wiki.txt

# wbits=3
# groupsize=128
# output_dir=${model_name_or_path}_q${wbits}_g${groupsize}_per_ic_test
# mkdir $output_dir
# GPU=1

# CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
#     $model_name_or_path \
#     --wbits $wbits \
#     --groupsize $groupsize \
#     --gptq \
#     --per_ic \
#     --save $output_dir > $output_dir/log.txt

# CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
#     --model hf \
#     --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda > $output_dir/wiki.txt








# CUDA_VISIBLE_DEVICES=0 python model/mamba.py \
#     $model_name_or_path \
#     --wbits 3 \
#     --groupsize 128 \
#     --gptq \
#     --save ${model_name_or_path}_q${wbits}_g${groupsize}_row

# CUDA_VISIBLE_DEVICES=0 python lmeval.py \
#     --model hf \
#     --model_args pretrained=${model_name_or_path}_q${wbits}_g${groupsize}_test,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda

# CUDA_VISIBLE_DEVICES=0 python model/mamba.py \
#     $model_name_or_path \
#     --wbits 4 \
#     --groupsize 128 \
#     --gptq \
#     --per_ic \
#     --save ${model_name_or_path}_q${wbits}_g${groupsize}_col

# CUDA_VISIBLE_DEVICES=0 python lmeval.py \
#     --model hf \
#     --model_args pretrained=${model_name_or_path}_q${wbits}_g${groupsize}_test,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda

# CUDA_VISIBLE_DEVICES=0 python model/mamba.py \
#     $model_name_or_path \
#     --wbits 3 \
#     --groupsize 128 \
#     --gptq \
#     --per_ic \
#     --save ${model_name_or_path}_q${wbits}_g${groupsize}_col

# CUDA_VISIBLE_DEVICES=0 python lmeval.py \
#     --model hf \
#     --model_args pretrained=${model_name_or_path}_q${wbits}_g${groupsize}_test,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda