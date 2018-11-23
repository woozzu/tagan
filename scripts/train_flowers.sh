. ./CONFIG

python train.py \
    --img_root ${FLOWERS_IMG_ROOT} \
    --caption_root ${FLOWERS_CAPTION_ROOT} \
    --trainclasses_file trainvalclasses.txt \
    --save_filename_G ./models/flowers_G.pth \
    --save_filename_D ./models/flowers_D.pth \
    --lambda_cond_loss 10 \
    --lambda_recon_loss 0.2
