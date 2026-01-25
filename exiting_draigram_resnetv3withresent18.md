# RT-DETRv3 ResNet-18-VD Architecture (ASCII)

```
RT-DETRv3 (ResNet-18-VD, full layer detail)

Input Image
└─ ResNet-18-VD Backbone (BasicBlock, [2,2,2,2])
   ├─ Stem (ResNet-D variant)
   │  ├─ Conv3x3 s2 -> BN -> ReLU  (ch_in/2)
   │  ├─ Conv3x3 s1 -> BN -> ReLU  (ch_in/2)
   │  └─ Conv3x3 s1 -> BN -> ReLU  (ch_in)
   │  └─ MaxPool 3x3 s2
   ├─ Stage2 (res2, stride 1, 2 blocks)
   │  ├─ Block1:
   │  │  ├─ 3x3 conv s1 -> BN -> ReLU
   │  │  ├─ 3x3 conv s1 -> BN
   │  │  └─ Shortcut: identity
   │  └─ Block2: same (shortcut identity)
   ├─ Stage3 (res3, stride 2, 2 blocks)  --> C3
   │  ├─ Block1:
   │  │  ├─ 3x3 conv s2 -> BN -> ReLU
   │  │  ├─ 3x3 conv s1 -> BN
   │  │  └─ Shortcut: AvgPool2D s2 + 1x1 conv s1 (ResNet-D)
   │  └─ Block2: stride 1, shortcut identity
   ├─ Stage4 (res4, stride 2, 2 blocks)  --> C4
   │  ├─ Block1: 3x3 s2 -> 3x3 s1, shortcut AvgPool2D + 1x1
   │  └─ Block2: stride 1, shortcut identity
   └─ Stage5 (res5, stride 2, 2 blocks)  --> C5
      ├─ Block1: 3x3 s2 -> 3x3 s1, shortcut AvgPool2D + 1x1
      └─ Block2: stride 1, shortcut identity

C3/C4/C5
└─ HybridEncoder (hidden_dim=256, expansion=0.5)
   ├─ Input Projections (per level)
   │  └─ 1x1 conv -> BN (C3,C4,C5 → 256)
   ├─ Transformer Encoder (use_encoder_idx=[2], 1 layer)
   │  └─ TransformerLayer:
   │     ├─ MultiHeadAttention (self-attn)
   │     ├─ Add + LayerNorm
   │     ├─ FFN: Linear -> GELU -> Linear
   │     └─ Add + LayerNorm
   ├─ Top-down FPN
   │  ├─ Lateral 1x1 BaseConv
   │  ├─ Upsample x2
   │  └─ CSPRepLayer (x3 RepVggBlock, expansion=0.5)
   └─ Bottom-up PAN
      ├─ Downsample BaseConv 3x3 s2
      └─ CSPRepLayer (x3 RepVggBlock, expansion=0.5)
   └─ Output: P3/P4/P5 (stride 8/16/32)

P3/P4/P5
└─ RTDETRTransformerv3 (num_decoder_layers=3, nhead=8)
   ├─ Input Proj (per level)
   │  ├─ 1x1 conv -> BN (for existing levels)
   │  └─ 3x3 conv s2 -> BN (if extra levels)
   ├─ Encoder Output Heads (per group)
   │  ├─ enc_output: Linear -> LayerNorm
   │  ├─ enc_score_head: Linear (→ num_classes)
   │  └─ enc_bbox_head: MLP (3-layer, → 4)
   ├─ Query Pos Head
   │  └─ MLP: 4 → 2*hidden_dim → hidden_dim
   ├─ Denoising (training)
   │  ├─ denoising_class_embed (Embedding)
   │  ├─ num_noises=3, num_noise_queries=[300,300,300]
   │  └─ num_noise_denoising=100
   └─ Transformer Decoder x3
      └─ Each DecoderLayer:
         ├─ Self-Attn: MultiHeadAttention
         ├─ Add + LayerNorm
         ├─ Cross-Attn: PPMSDeformableAttention
         ├─ Add + LayerNorm
         ├─ FFN: Linear -> ReLU -> Linear
         └─ Add + LayerNorm
   ├─ Decoder Heads (per layer)
   │  ├─ dec_score_head: Linear (→ num_classes)
   │  └─ dec_bbox_head: MLP (3-layer, → 4)
   └─ Outputs: decoder boxes/logits + encoder top-k boxes/logits

Decoder outputs
├─ DINOv3Head (loss + aggregation of enc/dec outputs)
│  └─ Uses transformer heads above for class/box predictions
├─ PPYOLOEHead (Aux O2M branch, from encoder features)
│  ├─ Stem (per level): ESEAttn
│  │  ├─ 1x1 Conv (attention)
│  │  └─ ConvBNLayer or RepVggBlock
│  ├─ pred_cls: 3x3 conv → num_classes
│  ├─ pred_reg: 3x3 conv → 4*reg_channels
│  └─ proj_conv: 1x1 (DFL projection)
└─ DETRPostProcess
   └─ Top-k=300 + NMS → Final Detections
```
