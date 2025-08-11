CUDA_VISIBLE_DEVICES=0 python lmeval.py \
    --model hf \
    --model_args pretrained=/mnt/dongsoo-lee2/gunho/ShiftAddLLM/temp_3,dtype=float16,parallelize=True \
    --tasks mmlu \
    --batch_size 8 \
    --num_fewshot 5 \
    --device cuda


# CUDA_VISIBLE_DEVICES=0 python lmeval.py \
#     --model hf \
#     --model_args pretrained=/mnt/dongsoo-lee2/gunho/ShiftAddLLM/temp_3,dtype=float16,parallelize=True \
#     --tasks wikitext \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --device cuda