#!/bin/bash

# UV 환경용 개발 도구 스크립트

echo "🛠️  주차장 분석 개발 도구 (UV 환경)"
echo "===================================="

function show_help() {
    echo "사용 가능한 명령어:"
    echo "  setup     - 개발 환경 설정"
    echo "  run       - 분석 실행"
    echo "  test      - 테스트 실행"
    echo "  lint      - 코드 품질 검사"
    echo "  clean     - 임시 파일 정리"
    echo "  info      - 환경 정보 표시"
    echo "  help      - 도움말 표시"
}

function setup_env() {
    echo "📦 개발 환경 설정 중..."

    # UV로 의존성 설치
    uv sync

    # 개발용 추가 패키지 설치
    echo "🔧 개발 도구 설치 중..."
    uv add --dev black flake8 pytest

    echo "✅ 개발 환경 설정 완료!"
}

function run_analysis() {
    echo "🚗 주차장 분석 실행..."
    ./run_analysis.sh
}

function run_tests() {
    echo "🧪 테스트 실행 중..."
    uv run python -m pytest -v
}

function lint_code() {
    echo "🔍 코드 품질 검사 중..."

    echo "📋 Black 포맷팅 검사..."
    uv run black --check *.py

    echo "📋 Flake8 린팅..."
    uv run flake8 *.py --max-line-length=88 --extend-ignore=E203,W503
}

function clean_files() {
    echo "🧹 임시 파일 정리 중..."

    # Python 캐시 파일 삭제
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -name "*.pyc" -delete 2>/dev/null
    find . -name "*.pyo" -delete 2>/dev/null

    # 임시 결과 파일 삭제
    rm -f *.png *.jpg parking_detection_info.txt 2>/dev/null
    rm -rf output_auto output_manual 2>/dev/null

    echo "✅ 정리 완료!"
}

function show_info() {
    echo "📊 환경 정보"
    echo "============"

    echo "📍 프로젝트 디렉토리: $(pwd)"
    echo "🐍 Python 버전: $(uv run python --version)"
    echo "📦 UV 버전: $(uv --version)"

    echo ""
    echo "📦 설치된 패키지:"
    uv pip list

    echo ""
    echo "📁 프로젝트 파일:"
    ls -la *.py *.md *.toml 2>/dev/null
}

# 메인 로직
case "${1:-help}" in
    setup)
        setup_env
        ;;
    run)
        run_analysis
        ;;
    test)
        run_tests
        ;;
    lint)
        lint_code
        ;;
    clean)
        clean_files
        ;;
    info)
        show_info
        ;;
    help|*)
        show_help
        ;;
esac