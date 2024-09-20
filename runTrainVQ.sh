### basic info
imagenet_path="/wangzedong/data/ImageNet"
codebook_path="/wangzedong/zly/VQGAN-LC/vqgan-gpt-lc/codebook-100K.pth"

# ### MergeVQ, ViT decoder, codebook k=1024 dim=256, bs64 x 4gpu x accum4 = bs1024
# exp_name="mergevq_vit_b_16_k1024_vitdec_m10_cls05_lr5e_4_ep100"
# pretrained="/wangzedong/zly/.cache/timm/mae_pretrain_vit_base.pth"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node 4 --master_port=12346 training_vqgan.py \
#     --batch_size 64 \
#     --accum_iter 4 \
#     --image_size 256 \
#     --epochs 100 \
#     --warmup_epochs 5 \
#     --lr 5e-4 \
#     --n_class 1000 \
#     --imagenet_path $imagenet_path \
#     --num_workers 12 \
#     --vq_config_path vqgan_configs/vq_f16_vit_b_vit.yaml \
#     --output_dir "train_logs_vq/$exp_name" \
#     --log_dir "train_logs_vq/$exp_name" \
#     --pretrained $pretrained \
#     --disc_start 50000 \
#     --n_vision_words 1024 \
#     --local_embedding_path $codebook_path \
#     --quantizer_type "org" --tuning_codebook 2 \
#     --use_cblinear 0 \
#     --rate_cls 0.5 --merge_ratio 10 \
#     --embed_dim 768

### MergeVQ, CNN decoder, codebook k=1024 dim=256, bs32 x 4gpu x accum8 = bs1024
#exp_name="mergevq_vit_b_16_k1024_cnndec_m10_cls01_lr5e_4_ep100"
exp_name="VVQ-GAN_lr5e_4_ep100"
pretrained="/wangzedong/zly/.cache/timm/mae_pretrain_vit_base.pth"
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 --master_port=52346 training_vqgan.py \
    --batch_size 32 \
    --accum_iter 8 \
    --image_size 256 \
    --epochs 100 \
    --warmup_epochs 5 \
    --lr 5e-4 \
    --n_class 1000 \
    --imagenet_path $imagenet_path \
    --num_workers 12 \
    --vq_config_path vqgan_configs/vq_f16_vit_b_cnn.yaml \
    --output_dir "train_logs_vq/$exp_name" \
    --log_dir "train_logs_vq/$exp_name" \
    --pretrained $pretrained \
    --disc_start 50000 \
    --n_vision_words 1024 \
    --local_embedding_path $codebook_path \
    --quantizer_type "org" --tuning_codebook 2 \
    --use_cblinear 0 \
    --rate_cls 0 --merge_ratio 0 \
    --embed_dim 768

# ### MergeVQ, ViT decoder, codebook k=1024 dim=256, bs64 x 4gpu x accum4 = bs1024
# exp_name="mergevq_vit_l_16_k1024_cnndec_m10_cls01_lr5e_4_ep100"
# pretrained="/wangzedong/zly/.cache/timm/mae_pretrain_vit_large.pth"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node 4 --master_port=12346 training_vqgan.py \
#     --batch_size 64 \
#     --accum_iter 4 \
#     --image_size 256 \
#     --epochs 100 \
#     --warmup_epochs 5 \
#     --lr 5e-4 \
#     --n_class 1000 \
#     --imagenet_path $imagenet_path \
#     --num_workers 12 \
#     --vq_config_path vqgan_configs/vq_f16_vit_l_vit.yaml \
#     --output_dir "train_logs_vq/$exp_name" \
#     --log_dir "train_logs_vq/$exp_name" \
#     --pretrained $pretrained \
#     --disc_start 50000 \
#     --n_vision_words 1024 \
#     --local_embedding_path $codebook_path \
#     --quantizer_type "org" --tuning_codebook 2 \
#     --use_cblinear 0 \
#     --rate_cls 0.1 --merge_ratio 10 \
#     --embed_dim 1024
