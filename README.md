# SAM(Segment Anything Model) Code Study

본 리포지토리는 Meta AI의 "Segment Anything" (ICCV 2023) 논문의 공식 코드를 분석하고, 핵심 아키텍처의 작동 원리를 파악하기 위한 MediVision 팀의 스터디 기록입니다. 

This repository is a code study focused on analyzing the official implementation of the "Segment Anything" (ICCV 2023) paper by Meta AI. The primary goal is to understand the core architecture and the interplay between its main components.


## 1. 원본 자료 (Original Resources)

* **Paper:** [Segment Anything (arXiv:2304.02643)](https://arxiv.org/abs/2304.02643)
* **Original GitHub:** [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

## 2. 분석 범위 (Scope of Study)

본 스터디는 SAM 모델을 구성하는 3가지 핵심 컴포넌트(Image Encoder, Prompt Encoder, Mask Decoder)와 이를 통합하는 메인 `SAM` 모델에 중점을 둡니다.

The study focuses on the four key files that define the SAM architecture:

1.  `image_encoder.py`
2.  `prompt_encoder.py`
3.  `mask_decoder.py`
4.  `modeling/sam.py` (Main model assembler)

## 3. 핵심 아키텍처 분석 (Core Architecture Analysis)

SAM은 "무거운" 이미지 인코더와 "가벼운" 프롬프트 인코더/마스크 디코더로 분리된 것이 핵심입니다. 이미지 인코딩은 한 번만 수행하고, 프롬프트에 따른 마스크 생성은 실시간으로 처리할 수 있습니다.

The core idea of SAM is the separation of a "heavy" **Image Encoder** and "lightweight" **Prompt Encoder** and **Mask Decoder**. This allows for one-time feature extraction of the image, followed by real-time, prompt-based mask generation.

### `image_encoder.py`

* **역할 (Role):**
    * 이미지 전체를 입력받아 고차원의 이미지 임베딩(image embedding)을 생성합니다.
    * Takes a full image as input and generates a high-dimensional image embedding.
* **핵심 구조 (Core Architecture):**
    * **Vision Transformer (ViT):** MAsked Autoencoder (MAE) 방식으로 사전 훈련된 표준 ViT (ViT-H/L/B)를 사용합니다.
    * It uses a standard **Vision Transformer (ViT)**, pre-trained using the Masked Autoencoder (MAE) methodology.
* **주요 특징 (Key Points):**
    * **Heavy Component:** SAM 아키텍처에서 가장 무겁고 연산량이 많은 부분입니다.
    * **One-time Execution:** `sam.set_image()` 메서드를 통해 이미지당 **단 한 번만** 실행됩니다. 생성된 임베딩은 `MaskDecoder`에서 반복적으로 재사용되어 효율성을 극대화합니다.
    * This is the heaviest part of the SAM architecture. It is designed to run **only once** per image (via `sam.set_image()`). The resulting embedding is cached and reused for all subsequent prompts, enabling high efficiency.

### `prompt_encoder.py`

* **역할 (Role):**
    * 사용자의 다양한 프롬프트(points, boxes, text, masks)를 입력받아, `MaskDecoder`가 이해할 수 있는 256-dim의 '프롬프트 토큰'으로 인코딩합니다.
    * Encodes various user prompts (points, boxes, text, masks) into 256-dimensional 'prompt tokens' that the `MaskDecoder` can understand.
* **핵심 구조 (Core Architecture):**
    * **Points & Boxes:** 간단한 positional encoding을 사용하여 256-dim 피처로 변환합니다. `(N, 2)` 좌표가 `(N, 256)` 토큰이 됩니다.
    * **Masks:** Convolutional layers를 거쳐 이미지 임베딩과 동일한 공간 해상도(spatial resolution)로 다운샘플링하고, 256-dim 피처로 변환합니다.
    * **Text:** [CLIP](https://github.com/openai/CLIP)의 텍스트 인코더를 사용하여 256-dim 피처로 인코딩합니다. (현재 public checkpoint에는 텍스트 인코더 가중치가 포함되어 있지 않습니다.)
* **주요 특징 (Key Points):**
    * **Promptable Interface:** SAM을 'Promptable'하게 만드는 핵심 모듈입니다.
    * **Lightweight:** 모든 인코딩 과정이 매우 가벼운 연산(lightweight operations)으로 구성되어 실시간 프롬프팅이 가능합니다.
    * This is the core module that makes SAM "promptable." All encoding processes are extremely lightweight, allowing for real-time user interaction.

### `mask_decoder.py`

* **역할 (Role):**
    * 인코딩된 이미지 피처(from `ImageEncoder`)와 프롬프트 토큰(from `PromptEncoder`)을 입력받아, 실제 segmentation mask를 생성합니다.
    * Takes the encoded image features and prompt tokens as input to generate the final segmentation masks.
* **핵심 구조 (Core Architecture):**
    * **2-way Transformer Decoder:** 2방향(two-way) 트랜스포머 디코더 레이어를 사용합니다.
        1.  **Token-to-Image Attention:** 프롬프트 토큰이 이미지 피처를 attend (프롬프트가 이미지에서 필요한 정보를 가져옴).
        2.  **Image-to-Token Attention:** 이미지 피처가 프롬프트 토큰을 attend (이미지 피처가 프롬프트 정보를 참조하여 스스로를 업데이트).
* **주요 특징 (Key Points):**
    * **Extremely Lightweight:** 매우 가벼워서 브라우저에서도 CPU로 수십 ms 내에 실행될 수 있습니다. (e.g., ~50ms)
    * **Ambiguity Resolution (모호성 해결):** 프롬프트가 모호할 경우(e.g., 셔츠를 입은 사람의 '점' -> 셔츠? 사람? 점?)를 대비해, **3개의 마스크(whole, part, subpart)를** 동시에 출력합니다.
    * **IoU Head:** 생성된 마스크의 품질(IoU)을 예측하는 작은 MLP 헤드가 함께 학습됩니다. 이 IoU 점수를 사용해 가장 품질이 좋은 마스크를 선택할 수 있습니다.
    * This decoder is extremely fast. Its key feature is its ability to resolve ambiguity by outputting **three masks** (whole, part, subpart) for a single prompt, along with a predicted **IoU score** for each mask.

### `modeling/sam.py`

* **역할 (Role):**
    * 위 3가지 컴포넌트(`ImageEncoder`, `PromptEncoder`, `MaskDecoder`)를 하나로 조립(assemble)하는 메인 `Sam` 클래스를 정의합니다.
    * Defines the main `Sam` class, which assembles the three components.
* **주요 기능 (Main Functions):**
    * `__init__`: 모델 가중치(weights)를 로드하고 3가지 서브모듈을 초기화합니다.
    * `forward`: End-to-End 추론 로직을 정의합니다. (이미지 인코딩 -> 프롬프트 인코딩 -> 마스크 디코딩)
    * `postprocess_masks`: `MaskDecoder`에서 나온 low-resolution 마스크(e.g., 256x256)를 원본 이미지 크기로 업샘플링(upsampling)하고, binary mask로 변환하는 후처리 로직을 담당합니다.


## 4. 주요 학습 내용 및 소감 (Key Takeaways)

* SAM의 아키텍처가 어떻게 '실시간성'과 '정확성'이라는 두 가지 목표를 잡았는지 명확히 이해할 수 있었습니다. 특히 `ImageEncoder`를 분리하고 `MaskDecoder`를 2-way-attention 기반의 경량 디코더로 설계한 부분이 인상적이었습니다.
* 프롬프트 인코더가 서로 다른 모달리티(좌표, 박스, 마스크)를 동일한 256-dim 벡터 공간으로 투영(projection)하는 방식이 흥미로웠습니다.

