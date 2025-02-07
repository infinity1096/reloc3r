
# Running evaluation for scannet1500

CUDA_VISIBLE_DEVICES=0 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --ckpt "checkpoints/Reloc3r-512.pth" \
    --test_dataset "ScanNet1500(resolution=(512,384), seed=777)" \


# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --ckpt "checkpoints/Reloc3r-224.pth" \
#     --test_dataset "ScanNet1500(resolution=(224,224), seed=777)" \

# Running evaluation for megadepth1500

CUDA_VISIBLE_DEVICES=0 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --ckpt "checkpoints/Reloc3r-512.pth" \
    --test_dataset "MegaDepth_valid(resolution=(512,384), seed=777)" \

