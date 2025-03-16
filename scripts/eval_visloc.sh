CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --scene "GreatCourt" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --scene "KingsCollege" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --scene "OldHospital" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --scene "ShopFacade" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --scene "StMarysChurch" \
    --topk 10 \


# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --resolution "(224,224)" \
#     --scene "GreatCourt" \
#     --topk 10 \

# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --resolution "(224,224)" \
#     --scene "KingsCollege" \
#     --topk 10 \

# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --resolution "(224,224)" \
#     --scene "OldHospital" \
#     --topk 10 \

# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --resolution "(224,224)" \
#     --scene "ShopFacade" \
#     --topk 10 \

# CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
#     --model "Reloc3rRelpose(img_size=224)" \
#     --resolution "(224,224)" \
#     --scene "StMarysChurch" \
#     --topk 10 \

