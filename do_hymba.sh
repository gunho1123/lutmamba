
model_name_or_path=nvidia/Hymba-1.5B-Base

wbits=3
groupsize=256

GPU=0

output_dir=${model_name_or_path}_q${wbits}_g${groupsize}_x_proj
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/hymba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --save $output_dir > $output_dir/log.txt

CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,trust_remote_code=True,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt

# output_dir=${model_name_or_path}_q${wbits}_g${groupsize}_per_ic
# mkdir $output_dir

# CUDA_VISIBLE_DEVICES=$GPU python model/hymba.py \
#     $model_name_or_path \
#     --wbits $wbits \
#     --groupsize $groupsize \
#     --gptq \
#     --per_ic \
#     --save $output_dir > $output_dir/log.txt

# CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
#     --model hf \
#     --model_args pretrained=$output_dir,trust_remote_code=True,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda > $output_dir/wiki.txt
