# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import glob
import random
import sys
import json
import pickle as pkl
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import Dataset, load_dataset, load_metric
import copy

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from modeling_ddie import MyModel

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    train_dbid_file: Optional[str] = field(
        default=None, metadata={"help": "A npy file containing the training DrugBank idx data."}
    )
    validation_dbid_file: Optional[str] = field(
        default=None, metadata={"help": "A npy file containing the validation DrugBank idx data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
             ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    dropout_ratio: Optional[float] = field(
        default=.0,
        metadata={"help": ""}
    )

    save_model_epoch: Optional[int] = field(
        default=0, 
    )

    parameter_averaging: bool = field(
        default=False,
    )
    selected_attention_mask: bool = field(
        default=False,
    )
    sharing_position_ids: bool = field(
        default=False,
    )

    kg_emb_file: str = field(
        default='',
    )
    baseline: bool = field(
        default=False,
    )
    use_cls_rep: bool = field(
        default=False,
    )
    use_mention_rep: bool = field(
        default=False,
    )
    use_kg_rep: bool = field(
        default=False,
    )
    freeze_embeddings: bool = field(
        default=False,
    )
    combination_method: str = field(
        default='loss',
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.max_seq_length = data_args.max_seq_length

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    elif data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
            "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)


    # Removing labels and logits npy files in case of overwrite_output_dir
    labels_npy_file = os.path.join(training_args.output_dir, 'labels.npy')
    if os.path.exists(labels_npy_file):
        os.remove(labels_npy_file)
    label_to_id_file = os.path.join(training_args.output_dir, 'label_to_id.json')
    if os.path.exists(label_to_id_file):
        os.remove(label_to_id_file)
    for logits_npy_file in glob.glob(os.path.join(training_args.output_dir, 'logits-epoch:*.npy')):
        os.remove(logits_npy_file)
    
    # 
    assert not (model_args.baseline and model_args.use_kg_rep), "Baseline method cant use KG embeddings"
    assert not (model_args.baseline and model_args.selected_attention_mask), "Baseline method cant use KG embeddings"
    # 
    assert model_args.use_cls_rep or model_args.use_mention_rep or model_args.use_kg_rep
    #
    assert model_args.combination_method in ('loss', 'cat')

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        #output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    e1_token = 'DRUG1'.lower() if tokenizer.do_lower_case else 'DRUG1'
    e2_token = 'DRUG2'.lower() if tokenizer.do_lower_case else 'DRUG2'
    tokenizer.add_tokens(e1_token)
    tokenizer.add_tokens(e2_token)

    #model = AutoModelForSequenceClassification.from_pretrained(
    #    model_args.model_name_or_path,
    #    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #    config=config,
    #    cache_dir=model_args.cache_dir,
    #)
    model_args.device = training_args.device
    #model = MyModel(model_args, config)
    model = MyModel.from_pretrained(model_args.model_name_or_path, config=config)
    model.my_init(model_args)


    # Save Model Arguments
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'model_args.pkl'), 'wb') as f:
        pkl.dump(model_args, f)


    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            #if len(non_label_column_names) >= 2:
            #    sentence1_key, sentence2_key = non_label_column_names[:2]
            #else:
            #    sentence1_key, sentence2_key = non_label_column_names[0], None

            # Change for our model
            sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    e1_id, e2_id = tokenizer.convert_tokens_to_ids((e1_token, e2_token))
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    print('E1: {}, E2: {}, SEP: {}, PAD: {}'.format(e1_id, e2_id, sep_id, pad_id))
    max_seq_len = model_args.max_seq_length

    ## 
    kg_emb_table = np.load(model_args.kg_emb_file)
    UNK_idx = len(kg_emb_table)
    ## 
    kg_id_files = {'train': data_args.train_dbid_file, 'validation': data_args.validation_dbid_file}
    train_and_eval_dataset = []
    for tr_or_ev in ('train', 'validation'):
        data_len = datasets[tr_or_ev].num_rows
        data_dict = datasets[tr_or_ev].to_dict()    

        kg_ids = np.load(kg_id_files[tr_or_ev])

        all_input_ids = []
        all_token_type_ids = []
        all_position_ids = []
        all_entity_position_ids = []
        selected_am = torch.zeros((data_len, max_seq_len, max_seq_len), dtype=torch.int64)
        normal_am = []
        all_inputs = []

        msl_error_cnt = 0

        for idx, input_ids in enumerate(data_dict['input_ids']):

            sep_pos = input_ids.index(sep_id)
            sep_pos = min(sep_pos, max_seq_len-3)

            kg1_pos = sep_pos + 1
            kg2_pos = sep_pos + 2

            # 
            input_ids_ = [x for x in input_ids]
            token_type_ids_ = [x for x in data_dict['token_type_ids'][idx]]
            position_ids_ = list(range(max_seq_len))

            if not model_args.baseline:
                input_ids_[sep_pos] = sep_id
                input_ids_[kg1_pos] = kg_ids[idx][0] + config.vocab_size
                input_ids_[kg2_pos] = kg_ids[idx][1] + config.vocab_size

            e1_pos, e2_pos = sep_pos, sep_pos
            if e1_id in input_ids:
                e1_pos = input_ids.index(e1_id)
            if e2_id in input_ids:
                e2_pos = input_ids.index(e2_id)
    
            flg = False
            e1_pos, e2_pos = kg1_pos, kg2_pos
            if e1_id in input_ids:
                e1_pos = input_ids.index(e1_id)
            else:
                flg = True

            if e2_id in input_ids:
                e2_pos = input_ids.index(e2_id)
            else:
                flg = True

            if flg:
                msl_error_cnt += 1
   

            # # Direct
            # input_ids_[e1_pos] = kg_ids[idx][0] + config.vocab_size
            # input_ids_[e2_pos] = kg_ids[idx][1] + config.vocab_size

            # Sharing Position IDs
            if model_args.sharing_position_ids:
                position_ids_[kg1_pos] = e1_pos
                position_ids_[kg2_pos] = e2_pos

            # Selected Attention Mask
            #selected_am[idx, :sep_pos+1, :sep_pos+1] = 1
            #selected_am[idx, sep_pos+1, :sep_pos+3] = 1
            #selected_am[idx, sep_pos+2, :sep_pos+3] = 1
            #selected_am[idx, e1_pos, sep_pos+1] = 1
            #selected_am[idx, e2_pos, sep_pos+2] = 1

            attention_mask = [1 for _ in range(sep_pos+1)] + [0]*(max_seq_len-(sep_pos+1))
            attention_mask_ = [1 for _ in range(sep_pos+3)] + [0]*(max_seq_len-(sep_pos+3))
            if kg_ids[idx][0] == UNK_idx: attention_mask_[kg1_pos] = 0
            if kg_ids[idx][1] == UNK_idx: attention_mask_[kg2_pos] = 0

            if model_args.baseline:
                normal_am.append(attention_mask)
            else:
                normal_am.append(attention_mask_)

            all_input_ids.append(input_ids_)
            all_token_type_ids.append(token_type_ids_)
            all_position_ids.append(position_ids_)
            all_entity_position_ids.append([e1_pos, e2_pos, kg1_pos, kg2_pos])

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            all_inputs.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'input_ids_w_kg': input_ids_,
                'attention_mask_w_kg': attention_mask_,
                'position_ids_w_kg': position_ids_,
            })

        #print(tr_or_ev, msl_error_cnt)

        if model_args.selected_attention_mask:
            all_attention_mask = selected_am
        else:
            all_attention_mask = normal_am
        new_data_dict = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'token_type_ids': all_token_type_ids,
            'position_ids': all_position_ids,
            'entity_position_ids': all_entity_position_ids,
            'label': data_dict['label']
        }

        dataset_ = Dataset.from_dict(new_data_dict)
        train_and_eval_dataset.append(dataset_)

        # Save valid inputs
        if tr_or_ev == 'validation':
            with open(os.path.join(training_args.output_dir, 'dev_inputs.pkl'), 'wb') as f:
                pkl.dump(all_inputs, f)

    train_dataset = train_and_eval_dataset[0]
    eval_dataset = train_and_eval_dataset[1]
    if model_args.parameter_averaging:
        eval_dataset = eval_dataset.add_column('evaluate', [True for _ in range(eval_dataset.num_rows)])

    if data_args.task_name is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        pass
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(logits) if is_regression else np.argmax(logits, axis=1)

        # Saving logits and labels
        labels_npy_file = os.path.join(training_args.output_dir, 'labels.npy')
        if not os.path.exists(labels_npy_file):
            np.save(labels_npy_file, p.label_ids)
        label_to_id_file = os.path.join(training_args.output_dir, 'label_to_id.json')
        if not os.path.exists(label_to_id_file):
            with open(label_to_id_file, 'w') as f:
                json.dump(label_to_id, f)
        epoch_ = 0
        while True:
            epoch_ += 1
            logits_npy_file = os.path.join(training_args.output_dir, 'logits-epoch:{}.npy'.format(epoch_))
            if not os.path.exists(logits_npy_file):
                np.save(logits_npy_file, logits)

                # Saving model
                if epoch_ == model_args.save_model_epoch:
                    config.save_pretrained(os.path.join(training_args.output_dir))
                    tokenizer.save_pretrained(os.path.join(training_args.output_dir))
                    torch.save(model.state_dict(), 
                               os.path.join(training_args.output_dir, 'full_model.bin'))
                break

        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            #return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
            from sklearn.metrics import precision_recall_fscore_support
            labels = [v for k, v in label_to_id.items() if k != 'negative']
            p,r,f,s = precision_recall_fscore_support(y_true=p.label_ids, y_pred=preds, labels=labels, average='micro')
            return {"Precision": p, 'Recall': r, 'MicroF':f,}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        #trainer.save_model()  # Saves the tokenizer too for easy upload

    eval_results = {}
    # # Evaluation
    # eval_results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         eval_datasets.append(datasets["validation_mismatched"])

    #     for eval_dataset, task in zip(eval_datasets, tasks):
    #         eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    #         output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
    #         if trainer.is_world_process_zero():
    #             with open(output_eval_file, "w") as writer:
    #                 logger.info(f"***** Eval results {task} *****")
    #                 for key, value in eval_result.items():
    #                     if key == 'eval_logits': continue
    #                     logger.info(f"  {key} = {value}")
    #                     writer.write(f"{key} = {value}\n")
    #             #np.save(os.path.join(training_args.output_dir, 'logits.npy'), eval_result['eval_logits'])
    #         eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
    return eval_results


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
