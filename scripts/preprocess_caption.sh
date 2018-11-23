. ./CONFIG

python preprocess_caption.py \
    --caption_root ${FLOWERS_CAPTION_ROOT} \
    --fasttext_model ${FASTTEXT_MODEL} \
    --max_nwords 50

python preprocess_caption.py \
    --caption_root ${BIRDS_CAPTION_ROOT} \
    --fasttext_model ${FASTTEXT_MODEL} \
    --max_nwords 50
