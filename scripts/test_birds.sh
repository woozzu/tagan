. ./CONFIG

python test.py \
    --img_root ./test/birds \
    --text_file ./test/text_birds.txt \
    --fasttext_model ${FASTTEXT_MODEL} \
    --generator_model ./models/birds_G.pth \
    --output_root ./test/result_birds
