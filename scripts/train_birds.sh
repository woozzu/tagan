. ./CONFIG

python train.py \
    --img_root ${BIRDS_IMG_ROOT} \
    --caption_root ${BIRDS_CAPTION_ROOT} \
    --trainclasses_file trainvalclasses.txt \
    --save_filename_G ./models/birds_G.pth \
    --save_filename_D ./models/birds_D.pth \
    --lambda_cond_loss 10 \
    --lambda_recon_loss 0.2
