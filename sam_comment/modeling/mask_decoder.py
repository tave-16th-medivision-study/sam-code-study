# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 이 파일은 SAM(Segment Anything Model)의 "mask decoder" 부분 구현이다.
# 역할 요약:
# - 이미지 인코더가 만든 image_embeddings와 프롬프트 임베딩(점/박스/마스크)을 받아
#   트랜스포머를 통해 여러 개의 마스크와 각 마스크의 품질(IOU 점수)을 예측한다.
# - 마스크는 "업샘플된 피처맵"과 "마스크 토큰으로 만든 하이퍼네트워크(MLP) 가중치"의
#   행렬곱으로 생성된다(=동적 컨볼루션/하이퍼네트 방식).

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,          # 트랜스포머의 채널 차원(C)
        transformer: nn.Module,        # 마스크 예측에 사용할 트랜스포머 모듈
        num_multimask_outputs: int = 3,# 다중 후보 마스크를 낼 때의 마스크 개수(한 번에 3개 등)
        activation: Type[nn.Module] = nn.GELU,   # 업샘플 블록에 사용할 활성화 함수
        iou_head_depth: int = 3,       # IOU(품질) 예측 MLP 깊이
        iou_head_hidden_dim: int = 256,# IOU 예측 MLP의 은닉 차원
    ) -> None:
        """
        이미지/프롬프트 임베딩을 입력받아 트랜스포머로 마스크를 예측한다.

        Arguments:
          transformer_dim (int): 트랜스포머 채널 차원
          transformer (nn.Module): 마스크 예측용 트랜스포머
          num_multimask_outputs (int): 다중 마스크 출력 시 생성할 후보 마스크 수
          activation (nn.Module): 업샘플 시 사용할 활성화 함수 타입
          iou_head_depth (int): 마스크 품질(IOU) 예측용 MLP 깊이
          iou_head_hidden_dim (int): IOU 예측 MLP 은닉 차원
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # (1) 출력용 토큰들
        # - iou_token: IOU 점수(마스크 품질) 예측에 사용되는 특수 토큰 1개
        # - mask_tokens: 실제 마스크를 생성할 토큰들 (단일 출력용 1개 + 멀티마스크용 n개)
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # (2) 트랜스포머 출력 피처를 마스크 해상도로 복원하기 위한 업샘플 경로
        # ConvTranspose2d 2단으로 H,W를 4배 업샘플(upscale)한다.
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        # (3) 하이퍼네트워크 MLP
        # 각 마스크 토큰으로부터 "동적 필터(또는 1x1 conv weight)"를 생성한다.
        # 여기서 출력 차원 transformer_dim // 8은 업샘플된 피처의 채널 수와 맞춘다.
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # (4) IOU(마스크 품질) 예측 헤드
        # iou_token의 최종 임베딩을 입력으로 받아 각 마스크에 대한 품질 점수를 낸다.
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,        # [B, C, H, W] 이미지 인코더 출력
        image_pe: torch.Tensor,                # [B, C, H, W] 포지셔널 인코딩(이미지 크기와 동일)
        sparse_prompt_embeddings: torch.Tensor,# [B, Ns, C] 점/박스 등 희소 프롬프트 임베딩
        dense_prompt_embeddings: torch.Tensor, # [B, C, H, W] 마스크 등 밀집 프롬프트 임베딩
        multimask_output: bool,                # True면 다중 마스크(후보) 출력, False면 단일 마스크
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        마스크와 IOU(품질) 점수를 예측한다.
        Returns:
          masks: [B, M, H', W']  (M은 1 또는 num_multimask_outputs)
          iou_pred: [B, M]
        """
        # 트랜스포머로 전체 마스크 후보와 IOU 점수를 먼저 얻는다.
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 단일/다중 출력에 맞게 마스크/IOU 슬라이스 선택
        # - 인덱스 0: "단일" 마스크용 토큰 결과
        # - 인덱스 1~: 멀티마스크 후보용 토큰 결과
        if multimask_output:
            mask_slice = slice(1, None)  # 멀티마스크 후보들만
        else:
            mask_slice = slice(0, 1)     # 단일 마스크만
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """트랜스포머로 마스크와 IOU(품질)를 직접 예측한다."""

        # (A) 출력용 토큰(=쿼리) 준비: [1 + num_mask_tokens, C]
        # 첫 토큰은 iou_token, 이후는 마스크 토큰들
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # 배치 크기만큼 확장: [B, 1+num_mask_tokens, C]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        # (B) 트랜스포머에 넣을 토큰 시퀀스 구성
        # 토큰 = [출력용 토큰들] + [희소 프롬프트 토큰들(점/박스)]
        # shape: [B, (1+num_mask_tokens)+Ns, C]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # (C) 배치 확장 및 입력 피처 준비
        # 이미지 피처에 "밀집 프롬프트(마스크 등)"를 더해 조건을 주입한다.
        # ※ 주의: 아래 repeat_interleave는 구현 의도에 따라 과도한 배치 확장이 될 수 있다.
        #   일반적으로 image_embeddings와 tokens의 배치는 동일 B여야 하며,
        #   보통은 repeat_interleave가 필요 없다. 이 코드는 tokens.shape[0](=B)로
        #   B배 반복하므로 [B*B, ...]가 되어버린다.
        #   만약 의도치 않은 배치 증폭이 발생한다면 이 줄을 제거하고
        #   단순히 src = image_embeddings 로 두고 아래에서 dense_prompt_embeddings를 더하는 편이 맞다.
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape  # 트랜스포머 내부 연산을 위한 크기 기록

        # (D) 트랜스포머 실행
        # 반환:
        # - hs: 최종 토큰(hidden states), shape [b, T, C]
        # - src: 인코더(또는 디코더) 피처 시퀀스, shape [b, HW, C] 등(구현에 따라 다름)
        hs, src = self.transformer(src, pos_src, tokens)

        # 토큰별 출력 분리
        # hs[:, 0, :]       -> iou_token 결과(품질 예측용)
        # hs[:, 1:1+M, :]   -> 마스크 토큰 결과(M = num_mask_tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # (E) 공간 피처 복원(업샘플) 및 마스크 생성
        # src: [b, HW, C] 형태를 [b, C, H, W]로 복원(구현에 따라 transpose+view)
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)  # [b, C', H', W'] (C' = transformer_dim//8)

        # 각 마스크 토큰을 하이퍼네트워크 MLP에 통과시켜 동적 가중치 생성
        # hyper_in: [b, M, C']  (M = num_mask_tokens)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 업샘플 피처를 [b, C', H'*W']로 펼친 뒤,
        # 마스크별 동적 가중치(hyper_in)와 행렬곱 -> [b, M, H'*W'] -> [b, M, H', W']
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # (F) IOU(품질) 예측
        # iou_token_out을 입력으로 각 마스크별 품질 점수 예측: [b, M]
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# 참고: MaskFormer 코드에서 가져와 약간 수정된 MLP
# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,        # 입력 차원
        hidden_dim: int,       # 은닉 차원
        output_dim: int,       # 출력 차원
        num_layers: int,       # 레이어 수
        sigmoid_output: bool = False, # 최종 출력에 시그모이드 적용 여부(옵션)
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 예: input_dim -> hidden_dim -> ... -> output_dim
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        # 마지막 레이어 직전까지 ReLU, 마지막 레이어는 선형 출력
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            # 주의: F.sigmoid는 deprecated. torch.sigmoid(x) 사용 권장.
            x = F.sigmoid(x)
        return x
