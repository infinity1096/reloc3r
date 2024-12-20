CUDA_VISIBLE_DEVICES=0 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --ckpt "checkpoints/Reloc3r-512.pth" \
    --test_dataset "ScanNet1500(resolution=(512,384), seed=777)" \

