#!/bin/bash

# 고급 주차장 분석 시스템 통합 실행 스크립트 (Perspective Transform 포함)

echo "🚗 고급 주차장 탑뷰 분석 시스템 v2.0"
echo "=================================================="
echo ""
echo "다음 중 실행할 분석 방법을 선택하세요:"
echo ""
echo "1) 기본 분석 (main.py) - OpenCV 기본 방법"
echo "2) 간단한 분석 (simple_parking.py) - PIL 기반 경량"
echo "3) 최적화된 분석 (optimized_parking.py) - YOLO + Perspective Transform"
echo "4) Perspective 특화 (perspective_parking.py) - 순수 Perspective Transform"
echo "5) 실제 주차장 특화 (realworld_parking.py) - 고급 YOLO + 색상 분석"
echo "6) 모든 방법 비교 실행"
echo "7) 결과 분석 및 통계"
echo ""

read -p "선택 (1-7): " choice

case $choice in
    1)
        echo "🔍 기본 OpenCV 분석 실행..."
        uv run main.py
        ;;
    2)
        echo "🔍 간단한 PIL 기반 분석 실행..."
        uv run simple_parking.py
        ;;
    3)
        echo "🔍 최적화된 YOLO + Perspective Transform 분석 실행..."
        timeout 180 uv run optimized_parking.py
        ;;
    4)
        echo "🔍 Perspective Transform 특화 분석 실행..."
        timeout 180 uv run perspective_parking.py
        ;;
    5)
        echo "🔍 실제 주차장 특화 고급 분석 실행..."
        timeout 180 uv run realworld_parking.py
        ;;
    6)
        echo "🔍 모든 방법 비교 실행 중..."
        echo ""

        echo "1/5: 기본 분석..."
        uv run main.py > main_output.log 2>&1
        [ $? -eq 0 ] && echo "✅ 기본 분석 완료" || echo "⚠️ 기본 분석 실패"

        echo "2/5: 간단한 분석..."
        uv run simple_parking.py > simple_output.log 2>&1
        [ $? -eq 0 ] && echo "✅ 간단한 분석 완료" || echo "⚠️ 간단한 분석 실패"

        echo "3/5: 최적화된 분석..."
        timeout 180 uv run optimized_parking.py > optimized_output.log 2>&1
        [ $? -eq 0 ] && echo "✅ 최적화된 분석 완료" || echo "⚠️ 최적화된 분석 타임아웃/실패"

        echo "4/5: Perspective Transform 분석..."
        timeout 180 uv run perspective_parking.py > perspective_output.log 2>&1
        [ $? -eq 0 ] && echo "✅ Perspective Transform 분석 완료" || echo "⚠️ Perspective Transform 분석 타임아웃/실패"

        echo "5/5: 고급 분석..."
        timeout 180 uv run realworld_parking.py > realworld_output.log 2>&1
        [ $? -eq 0 ] && echo "✅ 고급 분석 완료" || echo "⚠️ 고급 분석 타임아웃/실패"

        echo ""
        echo "🎉 모든 분석 완료!"
        ;;
    7)
        echo "📊 결과 분석 중..."

        echo ""
        echo "=== 생성된 결과 파일들 ==="
        echo "🖼️  이미지 결과:"
        ls -la *.jpg *.png 2>/dev/null | head -15

        echo ""
        echo "📁 디렉토리별 결과:"
        for dir in output_auto output_manual optimized_results optimized_perspective_results perspective_results realworld_results advanced_results; do
            if [ -d "$dir" ]; then
                file_count=$(ls "$dir" 2>/dev/null | wc -l)
                echo "   $dir/: ${file_count}개 파일"

                # JSON 파일이 있으면 주요 통계 표시
                json_file=$(find "$dir" -name "*.json" | head -1)
                if [ -n "$json_file" ] && command -v jq &> /dev/null; then
                    total_spots=$(jq -r '.statistics.total_spots // "N/A"' "$json_file" 2>/dev/null)
                    occupied=$(jq -r '.statistics.occupied_spots // "N/A"' "$json_file" 2>/dev/null)
                    vehicles=$(jq -r '.statistics.vehicles_detected // "N/A"' "$json_file" 2>/dev/null)
                    echo "      📊 주차구역: ${total_spots}개, 점유: ${occupied}개, 차량: ${vehicles}대"
                fi
            fi
        done

        echo ""
        echo "📈 분석 방법별 성능 비교:"

        # 각 방법의 결과 파일 확인 및 요약
        echo "   방법                     | 주차구역 | 점유 | 차량 | 특징"
        echo "   -------------------------|----------|------|------|----------"

        # 기본 분석
        if [ -f "parking_detection_info.txt" ]; then
            spots=$(grep "총 주차 영역 수" parking_detection_info.txt 2>/dev/null | cut -d: -f2 | tr -d ' ' | head -1)
            echo "   기본 OpenCV              | $spots      | -    | -    | 기본 CV"
        fi

        # 최적화된 분석 (Perspective Transform)
        if [ -f "optimized_perspective_results/optimized_analysis.json" ]; then
            spots=$(jq -r '.statistics.total_spots // "N/A"' optimized_perspective_results/optimized_analysis.json 2>/dev/null)
            occupied=$(jq -r '.statistics.occupied_spots // "N/A"' optimized_perspective_results/optimized_analysis.json 2>/dev/null)
            vehicles=$(jq -r '.statistics.vehicles_detected // "N/A"' optimized_perspective_results/optimized_analysis.json 2>/dev/null)
            perspective=$(jq -r '.statistics.perspective_corrected // false' optimized_perspective_results/optimized_analysis.json 2>/dev/null)
            echo "   YOLO + Perspective       | $spots       | $occupied    | $vehicles    | PT: $perspective"
        fi

        # Perspective Transform 특화
        if [ -f "perspective_results/perspective_analysis.json" ]; then
            spots=$(jq -r '.statistics.total_spots // "N/A"' perspective_results/perspective_analysis.json 2>/dev/null)
            occupied=$(jq -r '.statistics.occupied_spots // "N/A"' perspective_results/perspective_analysis.json 2>/dev/null)
            vehicles=$(jq -r '.statistics.vehicles_detected // "N/A"' perspective_results/perspective_analysis.json 2>/dev/null)
            echo "   Perspective 특화         | $spots       | $occupied    | $vehicles    | 순수 PT"
        fi

        # 고급 분석
        if [ -f "realworld_results/realworld_analysis.json" ]; then
            spots=$(jq -r '.statistics.total_spots // "N/A"' realworld_results/realworld_analysis.json 2>/dev/null)
            occupied=$(jq -r '.statistics.occupied_spots // "N/A"' realworld_results/realworld_analysis.json 2>/dev/null)
            vehicles=$(jq -r '.statistics.vehicles_detected // "N/A"' realworld_results/realworld_analysis.json 2>/dev/null)
            confidence=$(jq -r '.statistics.confidence_avg // "N/A"' realworld_results/realworld_analysis.json 2>/dev/null)
            echo "   실세계 특화              | $spots        | $occupied     | $vehicles     | 고급"
        fi

        echo ""
        echo "🎯 권장 사용법:"
        echo "   - 빠른 테스트: 간단한 분석 (2번)"
        echo "   - 균일한 격자: YOLO + Perspective (3번)"
        echo "   - 높은 정확도: 실세계 특화 (5번)"
        echo "   - 차량 크기 균일성 중요: Perspective 특화 (4번)"
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "✅ 실행 완료!"
echo ""
echo "📁 결과 확인:"
echo "   - 이미지 결과: *.jpg, *.png 파일들"
echo "   - JSON 데이터: */분석이름_analysis.json"
echo "   - 로그 파일: *_output.log (6번 선택 시)"
echo ""
echo "💡 Tip: 6번으로 모든 방법을 비교해보세요!"