CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --ckpt "checkpoints/Reloc3r-512.pth" \
    --scene "GreatCourt" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --ckpt "checkpoints/Reloc3r-512.pth" \
    --scene "KingsCollege" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --ckpt "checkpoints/Reloc3r-512.pth" \
    --scene "OldHospital" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --ckpt "checkpoints/Reloc3r-512.pth" \
    --scene "ShopFacade" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --ckpt "checkpoints/Reloc3r-512.pth" \
    --scene "StMarysChurch" \
    --topk 10 \


# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --ckpt "checkpoints/Reloc3r-224.pth" \
#     --resolution "(224,224)" \
#     --scene "GreatCourt" \
#     --topk 10 \

# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --ckpt "checkpoints/Reloc3r-224.pth" \
#     --resolution "(224,224)" \
#     --scene "KingsCollege" \
#     --topk 10 \

# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --ckpt "checkpoints/Reloc3r-224.pth" \
#     --resolution "(224,224)" \
#     --scene "OldHospital" \
#     --topk 10 \

# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --ckpt "checkpoints/Reloc3r-224.pth" \
#     --resolution "(224,224)" \
#     --scene "ShopFacade" \
#     --topk 10 \

# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --ckpt "checkpoints/Reloc3r-224.pth" \
#     --resolution "(224,224)" \
#     --scene "StMarysChurch" \
#     --topk 10 \

