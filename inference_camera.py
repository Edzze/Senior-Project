#!/usr/bin/env python3
import argparse
import cv2
import os

import torch
import torch.nn.functional as F

from model import EVLTransformer


def parse_args():
    parser = argparse.ArgumentParser(description='Webcam inference for signVLM model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved checkpoint .pth file')
    parser.add_argument('--backbone_path', type=str, required=True,
                        help='Path to the CLIP backbone weights file')
    parser.add_argument('--backbone_type', type=str, default='clip',
                        help='Backbone type for weight loader')
    parser.add_argument('--backbone', type=str, default='ViT-L/14-lnpre',
                        help='Backbone name used by the model')
    parser.add_argument('--num_frames', type=int, default=24,
                        help='Number of frames to use for each prediction')
    parser.add_argument('--sampling_rate', type=int, default=4,
                        help='Temporal stride for frame sampling')
    parser.add_argument('--spatial_size', type=int, default=224,
                        help='Spatial input size expected by the model')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Number of output classes')
    parser.add_argument('--decoder_num_layers', type=int, default=4)
    parser.add_argument('--decoder_qkv_dim', type=int, default=1024)
    parser.add_argument('--decoder_num_heads', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def center_crop_video(frames: torch.Tensor, spatial_size: int) -> torch.Tensor:
    # frames shape: C, T, H, W
    _, _, H, W = frames.shape
    if H < W:
        new_H = spatial_size
        new_W = int(W / H * spatial_size)
    else:
        new_W = spatial_size
        new_H = int(H / W * spatial_size)

    frames = F.interpolate(frames.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False)
    frames = frames[0]

    h_st = (new_H - spatial_size) // 2
    w_st = (new_W - spatial_size) // 2
    frames = frames[:, :, h_st:h_st + spatial_size, w_st:w_st + spatial_size]
    return frames


def preprocess_frames(frames, args):
    # frames: list of HxWx3 uint8 BGR images from OpenCV
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    tensor = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in rgb])
    tensor = tensor / 255.0
    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.permute(1, 0, 2, 3)  # C, T, H, W
    tensor = center_crop_video(tensor, args.spatial_size)
    return tensor.unsqueeze(0)


def sample_frames(buffer, args):
    seg_len = (args.num_frames - 1) * args.sampling_rate + 1
    if len(buffer) < seg_len:
        if len(buffer) == 0:
            return None
        buffer = buffer + [buffer[-1]] * (seg_len - len(buffer))
    elif len(buffer) > seg_len:
        buffer = buffer[-seg_len:]

    indices = list(range(0, len(buffer), args.sampling_rate))[: args.num_frames]
    if len(indices) < args.num_frames:
        indices = indices + [indices[-1]] * (args.num_frames - len(indices))
    return [buffer[i] for i in indices]


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = EVLTransformer(
        backbone_name=args.backbone,
        backbone_type=args.backbone_type,
        backbone_path=args.backbone_path,
        backbone_mode='freeze_fp16',
        decoder_num_layers=args.decoder_num_layers,
        decoder_qkv_dim=args.decoder_qkv_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_factor=4.0,
        num_classes=args.num_classes,
        enable_temporal_conv=True,
        enable_temporal_pos_embed=True,
        enable_temporal_cross_attention=True,
        cls_dropout=0.5,
        decoder_mlp_dropout=0.5,
        num_frames=args.num_frames,
    )
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam.')

    frame_buffer = []
    print('Press q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)
        if len(frame_buffer) > 128:
            frame_buffer.pop(0)

        sampled = sample_frames(frame_buffer, args)
        if sampled is not None:
            with torch.no_grad():
                input_tensor = preprocess_frames(sampled, args).to(device)
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=-1)
                topk = torch.topk(probs, k=5, dim=-1)
                top_indices = topk.indices[0].cpu().tolist()
                top_scores = topk.values[0].cpu().tolist()

            label_text = ', '.join([f'{idx}:{score:.2f}' for idx, score in zip(top_indices, top_scores)])
            cv2.putText(frame, f'Prediction: {label_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
