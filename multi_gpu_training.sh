#nohup bash /home/jknafou/Projects/ArmaSuisse/multi_gpu_training.sh &
source /home/jknafou/.virtualenvs/ArmaSuisse/bin/activate

export WD=$HOME/Projects/ArmaSuisse
export GPU_DIR=$WD/GPU
rm $GPU_DIR/*

TOTAL_GPU=$(lspci | grep -ci nvidia)
for ((i=0;i<TOTAL_GPU;i++)); do
  touch $GPU_DIR/$i
done

#MODELS=("/data/user/knafou/Thesis/TransBERT-bio-fr-base-tokenizer=camembert-base" "/data/user/knafou/Thesis/TransBERT-bio-fr-base" "camembert-base" "flaubert/flaubert_base_cased" "Dr-BERT/DrBERT-7GB" "almanach/camembert-bio-base")
MODELS=('roberta-base' 'xlm-roberta-base' 'roberta-large' 'xlm-roberta-large' 'distilbert-base-uncased')
for MODEL in "${MODELS[@]}"; do
  for K_FOLD in {0..9}; do
    while true; do
      sleep 5
      I_GPU="$(ls $GPU_DIR/ | head -n 1)"
      if [[ -n "$I_GPU" ]]; then
        rm $GPU_DIR/$I_GPU && nohup python /home/jknafou/Projects/ArmaSuisse/ArmaSuisse/training.py --K_FOLD=$K_FOLD --MODEL=$MODEL --DEVICE_ID=$I_GPU &> $MODEL\_$K_FOLD.out && touch $GPU_DIR/$I_GPU &
        break
      else
        sleep 10
      fi
    done
  done
done