set -xe

BS=32
LR=5e-2
HS=32
LEN=32
INITIAL_F=0
END_F=0
EPOCH=25

accelerate launch --config_file brain.yaml train.py \
  --data_dir data \
  --do_train \
  --output_dir en_checkpoints_len${LEN}_sparse \
  --hidden_size $HS \
  --train_batch_size $BS \
  --max_seq_length $LEN \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --num_neg_samples 400 \
  --initial_file_number $INITIAL_F \
  --end_file_number $END_F \
  --num_workers 8 \
  --fp16 \
  --run_name "BS${BS}_LR${LR}_HS${HS}_LEN${LEN}_f${INITIAL_F}_EPOCH${EPOCH}" \
  --vocab_path vocab_wiki_4k_en.json \
  --train_full data \
  --sparse \
  --use_frequency \
  --use_bpe \
  --bpe_tokenizer wiki_bpe_tokenizer_4000_bytelevel.json