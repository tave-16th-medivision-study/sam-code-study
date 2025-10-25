# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 이 파일은 SAM(Segment Anything Model)의 최상위 모듈인 Sam 클래스를 구현한다.
# 주요 구성요소:
#  - ImageEncoderViT: 이미지를 고정 크기 임베딩(피처맵)으로 변환
#  - PromptEncoder: 점/박스/저해상도 마스크 등의 프롬프트를 임베딩으로 변환
#  - MaskDecoder: 이미지/프롬프트 임베딩을 결합하여 마스크와 IOU(품질) 예측
# 파이프라인 개요:
#  (입력 이미지, 프롬프트) → 전처리/정규화 → ImageEncoder → PromptEncoder
#   → MaskDecoder → (저해상도 마스크/IOU) → 후처리(원본 크기로 보간) → 이진 마스크

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    # 마스크 이진화 임계값(> threshold → True). 보통 외부에서 조정 가능
    mask_threshold: float = 0.0
    # 입력 이미지 색상 포맷 정보(참고용)
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        # 이미지 정규화를 위한 평균/표준편차(이미지넷 통계에 맞춘 값)
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM: 이미지와 입력 프롬프트로부터 객체 마스크를 예측한다.

        Arguments:
          image_encoder (ImageEncoderViT): 이미지 → 이미지 임베딩(backbone)
          prompt_encoder (PromptEncoder): 점/박스/마스크 등 프롬프트 → 임베딩
          mask_decoder (MaskDecoder): 임베딩 결합 → 마스크/IOU 예측
          pixel_mean/pixel_std: 입력 이미지 정규화 파라미터(채널별)
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        # 정규화 상수는 학습 파라미터가 아니므로 buffer로 등록
        # shape: [C, 1, 1]로 만들어 브로드캐스팅 연산이 쉽게 함
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        # Sam 모듈이 어느 디바이스에 있는지 간단히 얻기 위한 헬퍼
        return self.pixel_mean.device

    @torch.no_grad()  # 추론 전용 경로: 자동 미분 비활성화로 메모리/속도 이점
    def forward(
        self,
        batched_input: List[Dict[str, Any]],  # 배치 단위 입력(이미지 + 선택적 프롬프트들)
        multimask_output: bool,               # True: 다중 후보 마스크 출력, False: 단일
    ) -> List[Dict[str, torch.Tensor]]:
        """
        이미지와 프롬프트로부터 엔드투엔드 마스크 예측.

        batched_input의 각 원소(dict) 예:
          'image': 3xHxW 텐서(이미 인코더 입력 크기로 리사이즈/변환된 상태 가정)
          'original_size': (H_orig, W_orig) 원본 이미지 크기
          'point_coords': BxNx2 점 좌표 (이미 모델 입력 프레임으로 변환됨)
          'point_labels': BxN 점 라벨(포어그라운드/백그라운드 등)
          'boxes': Bx4 박스 좌표
          'mask_inputs': Bx1xH'xW' 저해상도 마스크 입력(이전 단계 로짓 등)

        Returns: 입력 이미지 개수만큼의 dict 리스트
          'masks': BxCxH_origxW_orig 이진 마스크(Threshold 적용됨)
          'iou_predictions': BxC 각 마스크의 품질(IOU) 예측
          'low_res_logits': BxCx256x256 저해상도 로짓(다음 단계 입력으로 활용 가능)
        """
        # 1) 전처리(정규화+패딩)를 모든 배치에 적용 후 스택 → [N, 3, S, S]
        #    S = image_encoder.img_size (정사각 패딩)
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        # 2) 이미지 인코더: [N, 3, S, S] → [N, C, Hf, Wf] (피처맵/임베딩)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        # 3) 배치의 각 이미지에 대해 프롬프트 인코딩 → 마스크 디코딩 수행
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # (선택) 점 프롬프트가 있으면 (coords, labels) 튜플 구성
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            # 4) 프롬프트 인코더: 점/박스/마스크 입력 → (sparse, dense) 임베딩
            #    - sparse_embeddings: 점/박스 등 희소 토큰 임베딩 [B, Ns, C]
            #    - dense_embeddings : 마스크 등 밀집 임베딩 [B, C, Hf, Wf]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            # 5) 마스크 디코더: 이미지/프롬프트 임베딩 결합 → 저해상도 마스크/IOU
            #    curr_embedding: [C, Hf, Wf] → 배치 차원 추가 [1, C, Hf, Wf]
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),   # 포지셔널 인코딩
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            # 6) 후처리: 패딩 제거 + 원본 크기로 보간
            #    low_res_masks: [B, C, Hf', Wf'] → [B, C, H_orig, W_orig]
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],  # 패딩 전 입력 크기(H_in, W_in)
                original_size=image_record["original_size"],  # 최종 복원 크기(H_orig, W_orig)
            )

            # 7) 임계값 적용으로 이진화 (True/False 마스크)
            masks = masks > self.mask_threshold

            # 8) 현재 이미지에 대한 결과 dict 적재
            outputs.append(
                {
                    "masks": masks,                       # [B, C, H_orig, W_orig] (bool)
                    "iou_predictions": iou_predictions,   # [B, C]
                    "low_res_logits": low_res_masks,      # [B, C, 256, 256]
                }
            )

        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,                 # [B, C, Hm, Wm] (마스크 디코더 출력 저해상도)
        input_size: Tuple[int, ...],         # 패딩 전 인코더 입력 크기(H_in, W_in)
        original_size: Tuple[int, ...],      # 원본 이미지 크기(H_orig, W_orig)
    ) -> torch.Tensor:
        """
        패딩을 제거하고(인코더 입력 크기 기준) 원본 이미지 크기로 업샘플.

        처리 순서:
          (1) 디코더 표준 해상도(예: 256x256)로 일단 보간
          (2) 패딩 이전의 인코더 입력 크기(input_size)에 맞춰 패딩 영역을 잘라냄
          (3) 최종적으로 원본 크기(original_size)로 보간
        """
        # (1) 디코더 표준 해상도(SxS)로 맞춤 (S = image_encoder.img_size)
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # (2) 패딩 제거: 상단/좌측 기준으로 input_size 범위만 남김
        masks = masks[..., : input_size[0], : input_size[1]]
        # (3) 원본 크기로 보간
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """픽셀 정규화 후, 인코더 입력 크기(SxS)에 맞춰 우측/하단 패딩."""
        # 정규화: 색상 평균/표준편차로 채널별 스케일링 (브로드캐스팅)
        x = (x - self.pixel_mean) / self.pixel_std

        # 패딩: 입력을 정사각형(SxS)으로 만들기 위해 우/하단에 0 패딩
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        # pad의 인자는 (왼, 오, 위, 아래) 순서가 아니라 (왼, 오, 위, 아래) → (0, padw, 0, padh)
        x = F.pad(x, (0, padw, 0, padh))
        return x

