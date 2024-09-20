imagenet_path="/wangzedong/data/ImageNet"
codebook_path="/wangzedong/zly/VQGAN-LC/vqgan-gpt-lc/codebook-100K.pth"
vq_path="/wangzedong/zly/VQGAN-LC/mergevqhc/train_logs_vq/mergevq_vit_b_16_k1024_cnndec_m10_cls01_lr5e_4_ep100/vqgan_checkpoint-last.pth"

###Eval Reconstruction
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=15301 eval_reconstruction.py \
        --batch_size 8 \
        --image_size 256 \
        --lr 9e-3 \
        --n_class 1000 \
        --imagenet_path $imagenet_path \
        --vq_config_path /wangzedong/zly/VQGAN-LC/mergevqhc/vqgan_configs/vq_f16_vit_b_cnn.yaml \
        --output_dir "log_eval_recons/vqgan_lc_100K_f16" \
        --log_dir "log_eval_recons/vqgan_lc_100K_f16" \
        --quantizer_type "org" \
        --local_embedding_path $codebook_path \
        --stage_1_ckpt $vq_path \
        --tuning_codebook 2 \
        --embed_dim 768 \
        --n_vision_words 1024 \
        --use_cblinear 0 \
        --dataset "imagenet" \
        --merge_ratio 10 \
        --rate_cls 0.1
