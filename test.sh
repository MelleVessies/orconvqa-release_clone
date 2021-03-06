#!/bin/zsh
python3 train_pipeline.py \
    --no_cuda=1 \
    --train_file="data/preprocessed/test.txt" \
    --dev_file="data/preprocessed/dev.txt" \
    --test_file="data/preprocessed/test.txt" \
    --orig_dev_file="data/quac_format/dev.txt"\
    --orig_test_file="data/quac_format/test.txt"\
    --qrels="data/new_qrels.txt"\
    --blocks_path="data/all_blocks.txt"\
    --passage_reps_path="data/new_preps.pkl"\
    --passage_ids_path="data/new_pids.pkl"\
    --output_dir="./output"\
    --load_small=False  \
    --history_num=6 \
    --do_train=False\
    --do_eval=False\
    --do_test=True\
    --per_gpu_train_batch_size=2\
    --per_gpu_eval_batch_size=4\
    --learning_rate=5e-5\
    --num_train_epochs=3.0\
    --logging_steps=5\
    --save_steps=5000\
    --overwrite_output_dir=False\
    --eval_all_checkpoints=True \
    --fp16=False \
    --retriever_cache_dir="./cache" \
    --retrieve_checkpoint="data/pipeline_checkpoint/checkpoint-45000/retriever" \
    --retrieve_tokenizer_dir="data/retriever_checkpoint/" \
    --top_k_for_retriever=100 \
    --use_retriever_prob=True \
    --reader_cache_dir="./cache" \
    --qa_loss_factor=1.0 \
    --retrieval_loss_factor=1.0 \
    --top_k_for_reader=5 \
    --include_first_for_retriever=True \
    --use_rerank_prob=True \
    --early_loss=True \
    --max_answer_length=40
