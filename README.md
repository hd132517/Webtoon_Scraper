# 네이버 웹툰 컷 자동 추출기

이 프로젝트는 **네이버 웹툰 "나이트런"의 특정 회차 혹은 여러 회차를 자동으로 다운로드하고**, 각 이미지를 컷 단위로 정확히 분할하여 저장하는 파이프라인입니다. 웹툰 데이터를 연구하거나, 학습용 데이터셋을 수집하려는 목적 등에 활용될 수 있습니다.

---

## 📦 주요 기능

- **회차 자동 수집**
  - 웹툰 목록 페이지에서 회차별 URL을 자동 추출합니다.
  - 예고편 제거 및 중복 방지 기능 포함

- **이미지 다운로드**
  - Selenium과 BeautifulSoup을 이용해 웹툰 본문에서 이미지를 추출하고 저장합니다.
  - Referer 우회 및 User-Agent 설정으로 이미지 다운로드 차단을 회피

- **컷 분할 알고리즘**
  - 이미지 이진화 및 블러 처리 후, 수직/수평 방향으로 컷을 분할합니다.
  - 컷 내 여백 제거 및 불필요한 조각 제거
  - 재귀적으로 쪼개는 방식으로 복잡한 구성의 컷까지 분할

- **UI 메뉴**
  - 회차 단일 다운로드
  - 회차 범위 다운로드
  - 페이지 범위 다운로드 (다른 목록 페이지에서도 다운로드 가능)

---

## 🛠️ 사용 방법

### 1. 필수 패키지 설치
```bash
pip install -r requirements.txt
```
> 필요한 주요 패키지:
> - selenium
> - opencv-python
> - beautifulsoup4
> - requests
> - webdriver-manager


### 2. 실행
```bash
python main.py
```

실행 시 다음과 같은 메뉴가 표시됩니다:
```
0. 특정 회차만 다운로드
1. 특정 회차부터 특정 회차까지 다운로드
2. 특정 페이지부터 특정 페이지까지 다운로드
```
메뉴에 따라 회차 범위 또는 페이지 범위를 입력하면 자동으로 이미지 다운로드 → 컷 분할 → 폴더 저장이 진행됩니다.


---

## 📂 저장 구조 예시
```
project/
├── raw_images/                  # 원본 이미지 저장 폴더
├── cut_panels/
│   ├── 회차제목/                # 컷 단위 이미지 저장
│   │   ├── example_panel_01.jpg
│   │   ├── example_panel_02.jpg
│   │   └── ...
└── main.py                     # 본 프로젝트의 실행 파일
```

---

## 📌 참고 사항
- Selenium이 Chrome 브라우저를 사용하므로, Chrome이 설치되어 있어야 합니다.
- `webdriver_manager`를 통해 자동으로 크롬 드라이버를 설치합니다.
- 웹툰 이미지 URL이 비공개로 전환되는 경우, Referer 우회 등의 처리가 필요합니다.

---

## 🤝 기여 및 피드백
이 프로젝트는 개인 연구 및 실험을 위한 목적으로 개발되었습니다. 기능 개선이나 에러 보고는 언제든지 환영합니다!


---

## 📄 라이선스
본 프로젝트는 MIT 라이선스를 따릅니다.

