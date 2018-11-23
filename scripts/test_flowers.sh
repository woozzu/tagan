. ./CONFIG

python test.py \
    --img_root ./test/flowers \
    --text_file ./test/text_flowers.txt \
    --fasttext_model ${FASTTEXT_MODEL} \
    --generator_model ./models/flowers_G.pth \
    --output_root ./test/result_flowers
