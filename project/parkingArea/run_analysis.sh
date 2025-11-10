#!/bin/bash

# UV 환경에서 주차장 분석 실행 스크립트

echo "🚗 주차장 탑뷰 이미지 분석 시스템"
echo "================================="

# UV 의존성 설치 및 동기화
echo "📦 의존성 설치 중..."
uv sync

echo ""
echo "분석 방법을 선택하세요:"
echo "1) OpenCV 고급 분석 (main.py)"
echo "2) 간단한 분석 (simple_parking.py)"
echo "3) 둘 다 실행"

read -p "선택 (1/2/3): " choice

case $choice in
    1)
        echo "🔍 OpenCV 고급 분석 실행 중..."
        uv run main.py
        ;;
    2)
        echo "🔍 간단한 분석 실행 중..."
        uv run simple_parking.py
        ;;
    3)
        echo "🔍 OpenCV 고급 분석 실행 중..."
        uv run main.py
        echo ""
        echo "🔍 간단한 분석 실행 중..."
        uv run simple_parking.py
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "✅ 분석 완료!"
echo "📁 결과 파일들:"
echo "   - parking_detection_result.jpg (OpenCV 결과)"
echo "   - parking_comparison.png (OpenCV 비교)"
echo "   - output_auto/simple_parking_result.jpg (자동 감지)"
echo "   - output_manual/simple_parking_result.jpg (수동 격자)"
echo "   - parking_analysis_comparison.png (전체 비교)"