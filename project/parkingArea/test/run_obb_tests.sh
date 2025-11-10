#!/bin/bash

# YOLO-OBB 테스트 도구 실행 스크립트

echo "🔍 YOLO-OBB 테스트 도구 모음"
echo "==============================="
echo ""
echo "다음 중 실행할 테스트 도구를 선택하세요:"
echo ""
echo "1) 상세 OBB 모니터링 (obb_monitor.py)"
echo "   - 여러 신뢰도 레벨 테스트"
echo "   - 상세한 차량 크기 분석"
echo "   - 통계 차트 생성"
echo ""
echo "2) 빠른 OBB 뷰어 (quick_obb_viewer.py)"
echo "   - 간단한 시각화"
echo "   - 빠른 결과 확인"
echo ""
echo "3) 실시간 모니터링 (realtime_obb_monitor.py)"
echo "   - 웹캠 실시간 감지"
echo "   - 이미지 파일 대화형 모니터링"
echo ""

read -p "선택 (1-3): " choice

case $choice in
    1)
        echo "🔍 상세 OBB 모니터링 실행..."
        cd /home/aa/smart_city_2025/project/parkingArea/test
        uv run obb_monitor.py
        ;;
    2)
        echo "🔍 빠른 OBB 뷰어 실행..."
        cd /home/aa/smart_city_2025/project/parkingArea/test
        uv run quick_obb_viewer.py
        ;;
    3)
        echo "🔍 실시간 모니터링 실행..."
        cd /home/aa/smart_city_2025/project/parkingArea/test
        uv run realtime_obb_monitor.py
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "✅ 테스트 완료!"
echo ""
echo "📁 생성된 파일들:"
ls -la *.png *.jpg 2>/dev/null | head -10

echo ""
echo "💡 다른 테스트를 실행하려면 스크립트를 다시 실행하세요."