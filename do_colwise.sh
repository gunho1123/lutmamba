# model_name_or_path=state-spaces/mamba-370m-hf
# model_name_or_path=state-spaces/mamba-790m-hf
# model_name_or_path=state-spaces/mamba-1.4b-hf
# model_name_or_path=state-spaces/mamba-2.8b-hf


GPU=0

wbits=3
model_name_or_path=state-spaces/mamba-1.4b-hf

groupsize=128
output_dir=${model_name_or_path}_gptq_q${wbits}_g${groupsize}_v2
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --per_ic \
    --save $output_dir > $output_dir/log.txt # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt

groupsize=256
output_dir=${model_name_or_path}_gptq_q${wbits}_g${groupsize}_v2
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --per_ic \
    --save $output_dir > $output_dir/log.txt # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt

groupsize=512
output_dir=${model_name_or_path}_gptq_q${wbits}_g${groupsize}_v2
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --per_ic \
    --save $output_dir > $output_dir/log.txt # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt


groupsize=1024
output_dir=${model_name_or_path}_gptq_q${wbits}_g${groupsize}_v2
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --per_ic \
    --save $output_dir > $output_dir/log.txt # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt

wbits=4

groupsize=128
output_dir=${model_name_or_path}_gptq_q${wbits}_g${groupsize}_v2
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --per_ic \
    --save $output_dir > $output_dir/log.txt # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt

groupsize=256
output_dir=${model_name_or_path}_gptq_q${wbits}_g${groupsize}_v2
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --per_ic \
    --save $output_dir > $output_dir/log.txt # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt

groupsize=512
output_dir=${model_name_or_path}_gptq_q${wbits}_g${groupsize}_v2
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --per_ic \
    --save $output_dir > $output_dir/log.txt # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt


groupsize=1024
output_dir=${model_name_or_path}_gptq_q${wbits}_g${groupsize}_v2
mkdir $output_dir

CUDA_VISIBLE_DEVICES=$GPU python model/mamba.py \
    $model_name_or_path \
    --wbits $wbits \
    --groupsize $groupsize \
    --gptq \
    --per_ic \
    --save $output_dir > $output_dir/log.txt # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=$GPU python lmeval.py \
    --model hf \
    --model_args pretrained=$output_dir,dtype=float16,parallelize=True \
    --tasks wikitext \
    --batch_size 8 \
    --num_fewshot 0 \
    --device cuda > $output_dir/wiki.txt
