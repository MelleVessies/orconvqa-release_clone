#!/bin/zsh
#--load_small
#   set to True to load a small amount of data only for testing purposes)`
#--history_num
#   how many history turns to prepend
#--retriever_cache_dir
#   optional
#--top_k_for_retriever=100
#   use how many retrieved passages to update the question encoder in the retriever
#--top_k_for_reader=5
#   retrieve how many passages for reader
#--use_retriever_prob=True
#   use retriever score in overall score
#--use_rerank_prob=True
#   use reranker score in overall score
#--use_retriever_prob=True
#   use retriever score in overall score
#--early_loss=True
#   fine tune the question encoder in the retriever
#--reader_cache_dir=path_to_huggingface_bert_cache
#   optional
python3 train_pipeline.py \
    --train_file="./data/preprocessed/test.txt" \
    --dev_file="./data/preprocessed/dev.txt" \
    --test_file="./data/preprocessed/test.txt" \
    --orig_dev_file="./data/quac_format/dev.txt"\
    --orig_test_file="./data/quac_format/test.txt"\
    --qrels="./data/qrels.txt"\
    --blocks_path=path_to_all_blocks_txt\
    --passage_reps_path="./data/passage_reps.pkl"\
    --passage_ids_path="./data/passage_ids.pkl"\
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
    --retrieve_checkpoint="./data/retriever_checkpoint/checkpoint-5917" \
    --retrieve_tokenizer_dir="./data/retriever_checkpoint/" \
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