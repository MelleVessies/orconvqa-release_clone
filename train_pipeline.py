#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

from stdargparser import StdArgparser
import logging
import os
import random
import glob
import timeit
import json
import linecache
import faiss
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
import pytrec_eval
import scipy as sp
from copy import copy
import joblib

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, AlbertConfig, AlbertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import (LazyQuacDatasetGlobal, RawResult, 
                   write_predictions, write_final_predictions, 
                   get_retrieval_metrics, gen_reader_features)
from retriever_utils import RetrieverDataset
from modeling import Pipeline, AlbertForRetrieverOnlyPositivePassage, BertForOrconvqaGlobal
from scorer import quac_eval


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, retriever_tokenizer, reader_tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # TODO fix args override, realy dude?
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        # TODO fix args override, realy dude?
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    # TODO fix args override, realy dude? staaaphhh
    args.warmup_steps = int(t_total * args.warmup_portion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # TODO dont do this here
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # TODO fair enough, should go through future resource-manager class
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    retriever_tr_loss, retriever_logging_loss = 0.0, 0.0
    reader_tr_loss, reader_logging_loss = 0.0, 0.0
    qa_tr_loss, qa_logging_loss = 0.0, 0.0
    rerank_tr_loss, rerank_logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])

    # TODO fun, eval doesnt even run on python 3.6 (only 3.7), I think this comment might be old
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.eval() # we first get query representations in eval mode

            # TODO casting to numpy and back to list seems inefficient
            qids = np.asarray(batch['qid']).reshape(-1).tolist()
            # print('qids', qids)
            question_texts = np.asarray(
                batch['question_text']).reshape(-1).tolist()
            # print('question_texts', question_texts)
            answer_texts = np.asarray(
                batch['answer_text']).reshape(-1).tolist()
            # print('answer_texts', answer_texts)
            answer_starts = np.asarray(
                batch['answer_start']).reshape(-1).tolist()
            # print('answer_starts', answer_starts)
            query_reps = gen_query_reps(args, model, batch)
                
            retrieval_results = retrieve(args, qids, qid_to_idx, query_reps,
                                         passage_ids, passage_id_to_idx, passage_reps,
                                         qrels, qrels_sparse_matrix,
                                         gpu_index, include_positive_passage=True)

            #TODO might be nice to create some kind of RetrievalResults class so we dont need this indexing
            passage_reps_for_retriever = retrieval_results['passage_reps_for_retriever']
            labels_for_retriever = retrieval_results['labels_for_retriever']

            pids_for_reader = retrieval_results['pids_for_reader']
            # print(pids_for_reader)
            passages_for_reader = retrieval_results['passages_for_reader']
            labels_for_reader = retrieval_results['labels_for_reader']

            model.train()
            
            inputs = {'query_input_ids': batch['query_input_ids'].to(args.device),
                      'query_attention_mask': batch['query_attention_mask'].to(args.device),
                      'query_token_type_ids': batch['query_token_type_ids'].to(args.device),
                      'passage_rep': torch.from_numpy(passage_reps_for_retriever).to(args.device),
                      'retrieval_label': torch.from_numpy(labels_for_retriever).to(args.device)}
            retriever_outputs = model.retriever(**inputs)
            # model outputs are always tuple in transformers (see doc)
            retriever_loss = retriever_outputs[0]

            reader_batch = gen_reader_features(qids, question_texts, answer_texts, answer_starts,
                                        pids_for_reader, passages_for_reader, labels_for_reader,
                                        reader_tokenizer, args.reader_max_seq_length, is_training=True)

            reader_batch = {k: v.to(args.device) for k, v in reader_batch.items()}
            inputs = {'input_ids':       reader_batch['input_ids'],
                      'attention_mask':  reader_batch['input_mask'],
                      'token_type_ids':  reader_batch['segment_ids'],
                      'start_positions': reader_batch['start_position'],
                      'end_positions':   reader_batch['end_position'],
                      'retrieval_label': reader_batch['retrieval_label']}
            reader_outputs = model.reader(**inputs)
            reader_loss, qa_loss, rerank_loss = reader_outputs[0:3]

            loss = retriever_loss + reader_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                retriever_loss = retriever_loss.mean()
                reader_loss = reader_loss.mean()
                qa_loss = qa_loss.mean()
                rerank_loss = rerank_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                retriever_loss = retriever_loss / args.gradient_accumulation_steps
                reader_loss = reader_loss / args.gradient_accumulation_steps
                qa_loss = qa_loss / args.gradient_accumulation_steps
                rerank_loss = rerank_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            retriever_tr_loss += retriever_loss.item()
            reader_tr_loss += reader_loss.item()
            qa_tr_loss += qa_loss.item()
            rerank_tr_loss += rerank_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'retriever_loss', (retriever_tr_loss - retriever_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'reader_loss', (reader_tr_loss - reader_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'qa_loss', (qa_tr_loss - qa_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'rerank_loss', (rerank_tr_loss - rerank_logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss
                    retriever_logging_loss = retriever_tr_loss
                    reader_logging_loss = reader_tr_loss
                    qa_logging_loss = qa_tr_loss
                    rerank_logging_loss = rerank_tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    retriever_model_dir = os.path.join(output_dir, 'retriever')
                    reader_model_dir = os.path.join(output_dir, 'reader')
                    if not os.path.exists(retriever_model_dir):
                        os.makedirs(retriever_model_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if not os.path.exists(reader_model_dir):
                        os.makedirs(reader_model_dir)

                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    retriever_model_to_save = model_to_save.retriever
                    retriever_model_to_save.save_pretrained(
                        retriever_model_dir)
                    reader_model_to_save = model_to_save.reader
                    reader_model_to_save.save_pretrained(reader_model_dir)

                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))

                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# In[5]:


def evaluate(args, model, retriever_tokenizer, reader_tokenizer, prefix=""):
    if prefix == 'test':
        eval_file = args.test_file
        orig_eval_file = args.orig_test_file
    else:
        eval_file = args.dev_file
        orig_eval_file = args.orig_dev_file
    pytrec_eval_evaluator = evaluator

    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    DatasetClass = RetrieverDataset
    dataset = DatasetClass(eval_file, retriever_tokenizer,
                           args.load_small, args.history_num,
                           query_max_seq_length=args.retriever_query_max_seq_length,
                           is_pretraining=args.is_pretraining,
                           given_query=True,
                           given_passage=False, 
                           include_first_for_retriever=args.include_first_for_retriever)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    predict_dir = os.path.join(args.output_dir, 'predictions')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(
    #     dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    retriever_run_dict, rarank_run_dict = {}, {}
    examples, features = {}, {}
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        # TODO casting to numpy and back to list seems inefficient
        qids = np.asarray(batch['qid']).reshape(-1).tolist()
        # print(qids)
        question_texts = np.asarray(
            batch['question_text']).reshape(-1).tolist()
        answer_texts = np.asarray(
            batch['answer_text']).reshape(-1).tolist()
        answer_starts = np.asarray(
            batch['answer_start']).reshape(-1).tolist()
        query_reps = gen_query_reps(args, model, batch)
        retrieval_results = retrieve(args, qids, qid_to_idx, query_reps,
                                     passage_ids, passage_id_to_idx, passage_reps,
                                     qrels, qrels_sparse_matrix,
                                     gpu_index, include_positive_passage=False)
        retriever_probs = retrieval_results['retriever_probs']
        # print('retriever_probs before', retriever_probs)
        pids_for_reader = retrieval_results['pids_for_reader']
        passages_for_reader = retrieval_results['passages_for_reader']
        labels_for_reader = retrieval_results['labels_for_reader']

        reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts,
                                                                           answer_starts, pids_for_reader,
                                                                           passages_for_reader, labels_for_reader,
                                                                           reader_tokenizer,
                                                                           args.reader_max_seq_length,
                                                                           is_training=False)
        example_ids = reader_batch['example_id']
        # print('example_ids', example_ids)
        examples.update(batch_examples)
        features.update(batch_features)
        reader_batch = {k: v.to(args.device)
                        for k, v in reader_batch.items() if k != 'example_id'}
        with torch.no_grad():
            inputs = {'input_ids':      reader_batch['input_ids'],
                      'attention_mask': reader_batch['input_mask'],
                      'token_type_ids': reader_batch['segment_ids']}
            outputs = model.reader(**inputs)
        
        retriever_probs = retriever_probs.reshape(-1).tolist()
        # print('retriever_probs after', retriever_probs)
        for i, example_id in enumerate(example_ids):
            result = RawResult(unique_id=example_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               retrieval_logits=to_list(outputs[2][i]), 
                               retriever_prob=retriever_probs[i])
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                evalTime, evalTime / len(dataset))

    output_prediction_file = os.path.join(
        predict_dir, "instance_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(
        predict_dir, "instance_nbest_predictions_{}.json".format(prefix))
    output_final_prediction_file = os.path.join(
        predict_dir, "final_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            predict_dir, "instance_null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
                                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                        args.version_2_with_negative, args.null_score_diff_threshold)
    write_final_predictions(all_predictions, output_final_prediction_file, 
                            use_rerank_prob=args.use_rerank_prob, 
                            use_retriever_prob=args.use_retriever_prob)
    eval_metrics = quac_eval(
        orig_eval_file, output_final_prediction_file)
    rerank_metrics = get_retrieval_metrics(
        pytrec_eval_evaluator, all_predictions, eval_retriever_probs=True)
    eval_metrics.update(rerank_metrics)

    metrics_file = os.path.join(
        predict_dir, "metrics_{}.json".format(prefix))
    with open(metrics_file, 'w') as fout:
        json.dump(eval_metrics, fout)

    return eval_metrics


# In[6]:


def gen_query_reps(args, model, batch):
    model.eval()
    batch = {k: v.to(args.device) for k, v in batch.items() 
             if k not in ['example_id', 'qid', 'question_text', 'answer_text', 'answer_start']}
    with torch.no_grad():
        inputs = {}
        inputs['query_input_ids'] = batch['query_input_ids']
        inputs['query_attention_mask'] = batch['query_attention_mask']
        inputs['query_token_type_ids'] = batch['query_token_type_ids']
        outputs = model.retriever(**inputs)
        query_reps = outputs[0]

    return query_reps


# In[7]:


def retrieve(args, qids, qid_to_idx, query_reps,
             passage_ids, passage_id_to_idx, passage_reps,
             qrels, qrels_sparse_matrix,
             gpu_index, include_positive_passage=False):
    query_reps = query_reps.detach().cpu().numpy()
    D, I = gpu_index.search(query_reps, args.top_k_for_retriever)

    pidx_for_retriever = np.copy(I)
    qidx = [qid_to_idx[qid] for qid in qids]
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_retriever, axis=1)
    labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
    # print('labels_for_retriever before', labels_for_retriever)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_retriever)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = passage_id_to_idx[positive_pid]
                    pidx_for_retriever[i][-1] = positive_pidx
        labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
        # print('labels_for_retriever after', labels_for_retriever)
        assert np.sum(labels_for_retriever) >= len(labels_for_retriever)
    pids_for_retriever = passage_ids[pidx_for_retriever]
    passage_reps_for_retriever = passage_reps[pidx_for_retriever]

    scores = D[:, :args.top_k_for_reader]
    retriever_probs = sp.special.softmax(scores, axis=1)
    pidx_for_reader = I[:, :args.top_k_for_reader]
    # print('pidx_for_reader', pidx_for_reader)
    # print('qids', qids)
    # print('qidx', qidx)
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_reader, axis=1)
    # print('qidx_expanded', qidx_expanded)
    
    labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
    # print('labels_for_reader before', labels_for_reader)
    # print('labels_for_reader before', labels_for_reader)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_reader)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = passage_id_to_idx[positive_pid]
                    pidx_for_reader[i][-1] = positive_pidx
        labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
        # print('labels_for_reader after', labels_for_reader)
        assert np.sum(labels_for_reader) >= len(labels_for_reader)
    # print('labels_for_reader after', labels_for_reader)
    pids_for_reader = passage_ids[pidx_for_reader]
    # print('pids_for_reader', pids_for_reader)
    passages_for_reader = get_passages(pidx_for_reader, args)
    # we do not need to modify scores and probs matrices because they will only be
    # needed at evaluation, where include_positive_passage will be false

    return {'qidx': qidx,
            'pidx_for_retriever': pidx_for_retriever,
            'pids_for_retriever': pids_for_retriever,
            'passage_reps_for_retriever': passage_reps_for_retriever,
            'labels_for_retriever': labels_for_retriever,
            'retriever_probs': retriever_probs,
            'pidx_for_reader': pidx_for_reader,
            'pids_for_reader': pids_for_reader,
            'passages_for_reader': passages_for_reader, 
            'labels_for_reader': labels_for_reader}


# In[8]:


def get_passage(i, args):
    line = linecache.getline(args.blocks_path, i + 1)
    line = json.loads(line.strip())
    return line['text']
get_passages = np.vectorize(get_passage)

def load_pickle(fname, logger):
    logger.info(f'loading pickle file: {fname}')
    if not os.path.isfile(fname):
        logger.error(f'Failed to open {fname}, file not found')
    with open(fname, 'rb') as handle:
        return joblib.load(handle)

def load_json(fname, logger):
    logger.info(f'loading json file {fname}')
    if not os.path.isfile(fname):
        logger.error(f'Failed to open {fname}, file not found')
    with open(args.qrels) as handle:
        return json.load(handle)

#TODO combine with resource manager
def construct_faiss_index(passage_reps, proj_size, no_cuda, logger):
    logger.info('constructing passage faiss_index')

    index = faiss.IndexFlatIP(proj_size)
    index.add(passage_reps)

    if torch.cuda.is_available() and not no_cuda:
        faiss_res = faiss.StandardGpuResources()
        if torch.cuda.device_count() > 1:
            # run faiss on last gpu if more than 1 is available
            gpuId = torch.cuda.device_count() - 1
            index = faiss.index_cpu_to_gpu(faiss_res, gpuId, index)
        else:
            # otherwise use the only available one
            index = faiss.index_cpu_to_gpu(faiss_res, 0, index)

    return index

def create_inv_passage_id_index(passage_ids):
    # TODO this seems like a slow way to do this
    passage_id_to_idx = {}
    for i, pid in enumerate(passage_ids):
        passage_id_to_idx[pid] = i
    return passage_id_to_idx

# TODO rename, also returns inverse quid index
def create_qrel_sparse_matrix(qrels, passage_id_to_idx):
    # TODO no loops?
    qrels_data, qrels_row_idx, qrels_col_idx = [], [], []
    qid_to_idx = {}
    for i, (qid, v) in enumerate(qrels.items()):
        qid_to_idx[qid] = i
        for pid in v.keys():
            qrels_data.append(1)
            qrels_row_idx.append(i)
            qrels_col_idx.append(passage_id_to_idx[pid])

    qrels_sparse_matrix = sp.sparse.csr_matrix(
        (qrels_data, (qrels_row_idx, qrels_col_idx)))

    return qrels_sparse_matrix, qid_to_idx


#####################
# CODE START
####################


logger = logging.getLogger(__name__)

ALL_MODELS = list(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
    'retriever': (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer),
}

argparser = StdArgparser()
args = argparser.get_parsed()


# TODO fix everything going through single output dir (better of with multiple subfiles per run unless continue flag is set)
if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

# TODO DONT OVERWRITE ARGUMENTS
args.retriever_tokenizer_dir = os.path.join(args.output_dir, 'retriever')
args.reader_tokenizer_dir = os.path.join(args.output_dir, 'reader')


# Setup distant debugging if needed
# TODO remove? Seems outside of the scope of this project
# if args.server_ip and args.server_port:
#     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
#     import ptvsd
#     print("Waiting for debugger attach")
#     ptvsd.enable_attach(
#         address=(args.server_ip, args.server_port), redirect_output=True)
#     ptvsd.wait_for_attach()

# Setup CUDA, GPU & distributed training
# we now only support joint training on a single card
# we will request two cards, one for torch and the other one for faiss
# TODO create general resource manager class to assign GPU space, this code seems pretty bad (P.S. look at fais index creation)
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    args.n_gpu = 1
    # torch.cuda.set_device(0)
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
               args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()


model = Pipeline()

args.retriever_model_type = args.retriever_model_type.lower()
retriever_config_class, retriever_model_class, retriever_tokenizer_class = MODEL_CLASSES['retriever']
retriever_config = retriever_config_class.from_pretrained(args.retrieve_checkpoint)

# load pretrained retriever
retriever_tokenizer = retriever_tokenizer_class.from_pretrained(args.retrieve_tokenizer_dir)
retriever_model = retriever_model_class.from_pretrained(args.retrieve_checkpoint, force_download=True)

model.retriever = retriever_model
# do not need and do not tune passage encoder
model.retriever.passage_encoder = None
model.retriever.passage_proj = None

args.reader_model_type = args.reader_model_type.lower()
reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
reader_config = reader_config_class.from_pretrained(args.reader_config_name if args.reader_config_name else args.reader_model_name_or_path,
                                                    cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
reader_config.num_qa_labels = 2
# this not used for BertForOrconvqaGlobal
reader_config.num_retrieval_labels = 2
reader_config.qa_loss_factor = args.qa_loss_factor
reader_config.retrieval_loss_factor = args.retrieval_loss_factor

reader_tokenizer = reader_tokenizer_class.from_pretrained(args.reader_tokenizer_name if args.reader_tokenizer_name else args.reader_model_name_or_path,
                                                          do_lower_case=args.do_lower_case,
                                                          cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
reader_model = reader_model_class.from_pretrained(args.reader_model_name_or_path,
                                                  from_tf=bool(
                                                      '.ckpt' in args.reader_model_name_or_path),
                                                  config=reader_config,
                                                  cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)

model.reader = reader_model

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

# TODO What? assign again? Isnt GPU space already assigned?
model.to(args.device)

logger.info("Training/evaluation parameters %s", args)

# Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
# Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
# remove the need for this code, but it is still valid.
# TODO do we need this?
if args.fp16:
    try:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


passage_ids = load_pickle(args.passage_ids_path, logger)
passage_reps = load_pickle(args.passage_reps_path, logger)


# TODO reading large .plk files for Mac
'''
# GIGURU & MELLE: Other way to read pkl files, because they are too large
# https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
Start replacement
logger.info(f'loading passage ids from {args.passage_ids_path}')
bytes_in = bytearray(0)
max_bytes = 2**31 - 1
input_size = os.path.getsize(args.passage_ids_path)
with open(args.passage_ids_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
passage_ids = pkl.loads(bytes_in)

logger.info(f'loading passage reps from {args.passage_reps_path}')
input_size = os.path.getsize(args.passage_reps_path)
with open(args.passage_reps_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
passage_reps = pkl.loads(bytes_in)
#end replacement
'''


#TODO change this var name, not allways a GPU index, can also be CPU based faiss index
gpu_index = construct_faiss_index(passage_reps, args.proj_size, args.no_cuda, logger)

# logger.info(f'loading all blocks from {args.blocks_path}')
# with open(args.blocks_path, 'rb') as handle:
#     blocks_array = pkl.load(handle)


qrels = load_json(args.qrels, logger)
passage_id_to_idx = create_inv_passage_id_index(passage_ids)
qrels_sparse_matrix, qid_to_idx = create_qrel_sparse_matrix(qrels, passage_id_to_idx)


evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'recall'})


# In[10]:


# Training
if args.do_train:
    DatasetClass = RetrieverDataset
    train_dataset = DatasetClass(args.train_file, retriever_tokenizer,
                                 args.load_small, args.history_num,
                                 query_max_seq_length=args.retriever_query_max_seq_length,
                                 is_pretraining=args.is_pretraining,
                                 given_query=True,
                                 given_passage=False, 
                                 include_first_for_retriever=args.include_first_for_retriever)
    global_step, tr_loss = train(
        args, train_dataset, model, retriever_tokenizer, reader_tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

# Save the trained model and the tokenizer
if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    # TODO should be easy to move to own function
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    if not os.path.exists(args.retriever_tokenizer_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.retriever_tokenizer_dir)
    if not os.path.exists(args.reader_tokenizer_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.reader_tokenizer_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    final_checkpoint_output_dir = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(global_step))

    # TODO fine, but maybe move to function?
    final_retriever_model_dir = os.path.join(
        final_checkpoint_output_dir, 'retriever')
    final_reader_model_dir = os.path.join(
        final_checkpoint_output_dir, 'reader')

    # TODO should be easy to move to own function
    if not os.path.exists(final_checkpoint_output_dir):
        os.makedirs(final_checkpoint_output_dir)
    if not os.path.exists(final_retriever_model_dir):
        os.makedirs(final_retriever_model_dir)
    if not os.path.exists(final_reader_model_dir):
        os.makedirs(final_reader_model_dir)

    # save retriever
    retriever_model_to_save = model_to_save.retriever
    retriever_model_to_save.save_pretrained(final_retriever_model_dir)

    #save reader
    reader_model_to_save = model_to_save.reader
    reader_model_to_save.save_pretrained(final_reader_model_dir)

    # save reader and retriever tokenizers
    # TODO dont use args - think about file structure (saving over input seems wrong)
    retriever_tokenizer.save_pretrained(args.retriever_tokenizer_dir)
    reader_tokenizer.save_pretrained(args.reader_tokenizer_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(
        final_checkpoint_output_dir, 'training_args.bin'))

    # TODO didnt we just save these?
    # Load a trained model and vocabulary that you have fine-tuned
    model = Pipeline()

    model.retriever = retriever_model_class.from_pretrained(
        final_retriever_model_dir, force_download=True)
    model.retriever.passage_encoder = None
    model.retriever.passage_proj = None

    model.reader = reader_model_class.from_pretrained(
        final_reader_model_dir, force_download=True)

    retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
        args.retriever_tokenizer_dir, do_lower_case=args.do_lower_case)
    reader_tokenizer = reader_tokenizer_class.from_pretrained(
        args.reader_tokenizer_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)


# In[11]:


# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory

results = {}
max_f1 = 0.0
best_metrics = {}
if args.do_eval and args.local_rank in [-1, 0]:
    retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
        args.retriever_tokenizer_dir, do_lower_case=args.do_lower_case)
    reader_tokenizer = reader_tokenizer_class.from_pretrained(
        args.reader_tokenizer_dir, do_lower_case=args.do_lower_case)
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = sorted(list(os.path.dirname(os.path.dirname(c)) for c in
                                      glob.glob(args.output_dir + '/*/retriever/' + WEIGHTS_NAME, recursive=False)))
#         logging.getLogger("transformers.modeling_utils").setLevel(
#             logging.WARN)  # Reduce model loading logs

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        # Reload the model
        global_step = checkpoint.split(
            '-')[-1] if len(checkpoint) > 1 else ""
        print(global_step, 'global_step')
        model = Pipeline()
        model.retriever = retriever_model_class.from_pretrained(
            os.path.join(checkpoint, 'retriever'), force_download=True)
        model.retriever.passage_encoder = None
        model.retriever.passage_proj = None
        model.reader = reader_model_class.from_pretrained(
            os.path.join(checkpoint, 'reader'), force_download=True)
        model.to(args.device)

        # Evaluate
        result = evaluate(args, model, retriever_tokenizer,
                          reader_tokenizer, prefix=global_step)
        if result['f1'] > max_f1:
            max_f1 = result['f1']
            best_metrics = copy(result)
            best_metrics['global_step'] = global_step

        for key, value in result.items():
            tb_writer.add_scalar(
                'eval_{}'.format(key), value, global_step)

        result = dict((k + ('_{}'.format(global_step) if global_step else ''), v)
                      for k, v in result.items())
        results.update(result)

    best_metrics_file = os.path.join(
        args.output_dir, 'predictions', 'best_metrics.json')
    with open(best_metrics_file, 'w') as fout:
        json.dump(best_metrics, fout)

    all_results_file = os.path.join(
        args.output_dir, 'predictions', 'all_results.json')
    with open(all_results_file, 'w') as fout:
        json.dump(results, fout)

    logger.info("Results: {}".format(results))
    logger.info("best metrics: {}".format(best_metrics))


# In[12]:


if args.do_test and args.local_rank in [-1, 0]:
    # TODO actual fix
    args.retriever_tokenizer_dir = "./data/pipeline_checkpoint/retriever"
    args.reader_tokenizer_dir = "./data/pipeline_checkpoint/reader"

    if args.do_eval:
        best_global_step = best_metrics['global_step'] 
    else:
        best_global_step = args.best_global_step
        retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
            args.retriever_tokenizer_dir, do_lower_case=args.do_lower_case)
        reader_tokenizer = reader_tokenizer_class.from_pretrained(
            args.reader_tokenizer_dir, do_lower_case=args.do_lower_case)

    best_checkpoint = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(best_global_step))
    logger.info("Test the best checkpoint: %s", best_checkpoint)

    #TODO actual fix
    best_checkpoint = './data/pipeline_checkpoint/checkpoint-45000'

    model = Pipeline()
    model.retriever = retriever_model_class.from_pretrained(
        os.path.join(best_checkpoint, 'retriever'), force_download=True)
    model.retriever.passage_encoder = None
    model.retriever.passage_proj = None
    model.reader = reader_model_class.from_pretrained(
        os.path.join(best_checkpoint, 'reader'), force_download=True)
    model.to(args.device)

    # Evaluate
    result = evaluate(args, model, retriever_tokenizer,
                      reader_tokenizer, prefix='test')

    test_metrics_file = os.path.join(
        args.output_dir, 'predictions', 'test_metrics.json')
    with open(test_metrics_file, 'w') as fout:
        json.dump(result, fout)

    logger.info("Test Result: {}".format(result))


# In[ ]:




