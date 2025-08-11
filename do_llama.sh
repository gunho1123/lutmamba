# CUDA_VISIBLE_DEVICES=0 python model/llama.py \
#     meta-llama/Llama-3.2-1B \
#     --wbits 2 \
#     --groupsize 128 \
#     --lat \
#     --bcq_round 20 # bcq_round 20 works too, bigger - slower - maybe better
    

# CUDA_VISIBLE_DEVICES=0 python model/llama.py \
#     meta-llama/Llama-3.2-1B \
#     --wbits 3 \
#     --groupsize 128 \
#     --lat \
#     --temp_storage packed_1B_q3_g128_r0 \
#     --save 1B_q3_g128_r0_model.bin \
#     --bcq_round 0 # bcq_round 20 works too, bigger - slower - maybe better


# CUDA_VISIBLE_DEVICES=0 python model/llama.py \
#     meta-llama/Llama-3.2-1B \
#     --wbits 3 \
#     --groupsize -1 \
#     --gptq # bcq_round 20 works too, bigger - slower - maybe better


CUDA_VISIBLE_DEVICES=0 python model/llama.py \
    meta-llama/Llama-3.2-1B \
    --wbits 2 \
    --groupsize 128 \
    --lat \
    --save temp_1B_q2_g128_debug \
    --bcq_round 20 # bcq_round 20 works too, bigger - slower - maybe better
