New-Item -ItemType Directory -Path .\runs -Force

$env:CUDA_VISIBLE_DEVICES = "0"

python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 main.py `
  --num_steps 1000 `
  --backbone "ViT-L/14-lnpre" `
  --backbone_type clip `
  --backbone_path "C:\Users\man_t\OneDrive\Desktop\Senior Project\signVLM-main\CLIP_weights\ViT-L\ViT-L-14.pt" `
  --decoder_num_layers 4 `
  --decoder_qkv_dim 1024 `
  --decoder_num_heads 16 `
  --num_classes 100 `
  --checkpoint_dir "runs/wlasl_100_vitB16_32f_dec4x1024_FT" `
  --auto_resume `
  --frames_available 1 `
  --train_data_root "C:\Users\man_t\OneDrive\Desktop\Senior Project\preprocessing\train" `
  --val_data_root "C:\Users\man_t\OneDrive\Desktop\Senior Project\preprocessing\test" `
  --train_list_path "C:\Users\man_t\OneDrive\Desktop\Senior Project\signVLM-main\WLASL_train100.txt" `
  --val_list_path "C:\Users\man_t\OneDrive\Desktop\Senior Project\signVLM-main\WLASL_test100.txt" `
  --n_shots 5 `
  --batch_size 16 `
  --batch_split 1 `
  --num_workers 16 `
  --num_frames 24 `
  --sampling_rate 4 `
  --num_spatial_views 3 `
  --num_temporal_views 1 `
  2>&1 | Tee-Object -FilePath "runs\train-$(Get-Date -Format yyyyMMdd_HHmmss).log"