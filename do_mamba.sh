model_name_or_path=state-spaces/mamba-370m-hf


CUDA_VISIBLE_DEVICES=0 python model/mamba.py \
    $model_name_or_path \
    --wbits 4 \
    --groupsize 128 \
    --acc \
    --save ${model_name_or_path}_q${wbits}_g${groupsize} \
    --bcq_round 20 # bcq_round 20 works too, bigger - slower - maybe better
