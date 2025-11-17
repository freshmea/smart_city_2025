# FastAPI & Tableau 통합 수업 커리큘럼 (4시간)

## 수업 개요

- **대상**: 웹 개발 및 데이터 시각화에 관심있는 개발자
- **목표**: FastAPI로 데이터 API를 구축하고 Tableau로 시각화하는 통합 시스템 구축
- **총 시간**: 4시간 (각 1시간씩 4개 세션)

---

## 1시간차: FastAPI 기초와 환경 설정

### 학습 목표

- FastAPI의 특징과 장점 이해
- 개발 환경 구축
- 기본 API 엔드포인트 작성

### 강의 내용 (45분)

1. **FastAPI 소개** (10분)
   - FastAPI vs Flask vs Django
   - 자동 문서화 (Swagger UI)
   - 타입 힌트와 Pydantic

2. **환경 설정** (15분)

   ```bash
   pip install fastapi uvicorn pandas sqlalchemy
   ```

3. **첫 번째 API 만들기** (20분)

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### 실습 (15분)

- 기본 API 서버 실행
- Swagger UI 확인 (http://localhost:8000/docs)

---

## 2시간차: FastAPI로 데이터 API 구축

### 학습 목표

- 데이터베이스 연결
- CRUD API 구현
- 데이터 모델링 (Pydantic)

### 강의 내용 (40분)

1. **데이터 모델 정의** (15분)

```python
from pydantic import BaseModel
from datetime import datetime

class SalesData(BaseModel):
    id: int
    product_name: str
    sales_amount: float
    sale_date: datetime
```

1. **데이터베이스 연결** (15분)
   - SQLAlchemy 설정
   - 샘플 데이터 생성

2. **API 엔드포인트 구현** (10분)

   ```python
   @app.get("/api/sales")
   async def get_sales():
       # 판매 데이터 반환

   @app.get("/api/sales/summary")
   async def get_sales_summary():
       # 집계 데이터 반환
   ```

### 실습 (20분)

- 판매 데이터 API 구현
- 집계 데이터 API 구현
- API 테스트

---

## 3시간차: Tableau 기초와 데이터 연결

### 학습 목표

- Tableau 인터페이스 이해
- 웹 데이터 커넥터 사용
- 기본 차트 작성

### 강의 내용 (35분)

1. **Tableau 소개** (10분)
   - Tableau Public vs Desktop
   - 데이터 연결 방식
   - 워크시트와 대시보드

2. **웹 데이터 커넥터 설정** (15분)
   - FastAPI 엔드포인트를 데이터 소스로 연결
   - JSON 데이터 파싱
   - 데이터 타입 설정

3. **기본 시각화** (10분)
   - 막대 차트
   - 라인 차트
   - 파이 차트

### 실습 (25분)

- FastAPI 데이터와 Tableau 연결
- 판매 데이터 시각화
- 시간별 트렌드 차트 작성

---

## 4시간차: 실시간 대시보드 구축 및 통합

### 학습 목표

- 실시간 데이터 업데이트 구현
- 인터랙티브 대시보드 구축
- 배포 및 공유

### 강의 내용 (30분)

1. **실시간 데이터 업데이트** (15분)

    ```python
    from fastapi import BackgroundTasks

    @app.post("/api/sales/refresh")
    async def refresh_data(background_tasks: BackgroundTasks):
        background_tasks.add_task(update_sales_data)
        return {"status": "updating"}
    ```

1. **대시보드 구성** (15분)
   - 필터와 파라미터 활용
   - 대시보드 레이아웃 구성
   - 액션과 상호작용 설정

### 실습 (25분)

- 종합 대시보드 구축
- 실시간 데이터 테스트
- 대시보드 퍼블리싱

### 마무리 (5분)

- 프로젝트 리뷰
- 추가 학습 리소스 소개
- Q&A

---

## 실습 프로젝트: 스마트 시티 대시보드

### 프로젝트 구조

```text
smart_city_dashboard/
├── main.py              # FastAPI 메인 애플리케이션
├── models.py            # 데이터 모델
├── database.py          # 데이터베이스 연결
├── data/
│   └── sample_data.csv  # 샘플 데이터
└── tableau/
    └── dashboard.twbx   # Tableau 워크북
```

### 구현할 기능

1. **데이터 수집 API**
   - 교통량 데이터
   - 에너지 사용량
   - 인구 통계

2. **집계 API**
   - 시간별/일별/월별 통계
   - 지역별 비교
   - 트렌드 분석

3. **Tableau 대시보드**
   - 실시간 모니터링
   - 지역 맵 시각화
   - KPI 지표

---

## 준비물 및 사전 요구사항

### 소프트웨어

- Python 3.8+
- Tableau Public (무료) 또는 Tableau Desktop
- 웹 브라우저
- 코드 에디터 (VS Code 권장)

### 사전 지식

- Python 기초 문법
- HTTP/REST API 기본 개념
- 데이터 시각화 기초 개념

---

## 추가 학습 리소스

### FastAPI

- [공식 문서](https://fastapi.tiangolo.com/)
- [FastAPI 튜토리얼](https://fastapi.tiangolo.com/tutorial/)

### Tableau

- [Tableau 학습 리소스](https://www.tableau.com/learn)
- [Tableau Public 갤러리](https://public.tableau.com/gallery)

### 통합 프로젝트 아이디어

- IoT 센서 데이터 대시보드
- 전자상거래 분석 시스템
- 소셜 미디어 트렌드 모니터링
