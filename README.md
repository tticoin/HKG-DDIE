# HKG-DDIE
The implementation of "Integrating heterogeneous knowledge graphs into DDI extraction from the literature"

## Requirements
```
pip install -r requirements.txt
```
In addition to these pacages, please install ```torch``` according to your CUDA version.

## Preparing model weights
Please download our pre-trained heterogeneous KG embedding vectors and our DDI extraction model full-parameters from [here](https://github.com/tticoin/HKG-DDIE/releases).
```
unzip weights.zip
```

## DDI extraction
```
python main.py \
  --train_file ./inputs/train.csv \
  --validation_file ./inputs/dev.csv \
  --train_dbid_file ./inputs/train_id.npy \
  --validation_dbid_file ./inputs/dev_id.npy \
  --kg_emb_file ./weights/PharmaHKG_DistMult_entity.npy \
  --do_train \
  --do_eval \
  --use_cls_rep \
  --use_mention_rep \
  --save_strategy no \
  --logging_strategy no \
  --num_train_epochs 10 \
  --learning_rate 5e-05 \
  --fp16 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 256 \
  --dropout_ratio 0.5 \
  --weight_decay 8 \
  --lr_scheduler_type linear \
  --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --parameter_averaging \
  --sharing_position_ids \
  --freeze_embeddings \
  --combination_method cat \
  --output_dir ./outputs/foo
```
or you can

## Acknolwedgement
This work was supported by JSPS KAKENHI Grant Number JP20K11962.
