
model_name_or_path=nvidia/Hymba-1.5B-Base

wbits=3
groupsize=128
output_dir=${model_name_or_path}_q${wbits}_g${groupsize}_per_oc
mkdir $output_dir
GPU=0

CUDA_VISIBLE_DEVICES=$GPU python model/hymba.py \
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
