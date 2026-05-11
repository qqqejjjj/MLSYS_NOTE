#!/bin/bash
# 启动 vLLM Kernel 教学文档
# 用法：bash start.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_FILE="$SCRIPT_DIR/kernels_learning.html"

if [ ! -f "$HTML_FILE" ]; then
    echo "错误：找不到 $HTML_FILE"
    exit 1
fi

PORT=8421

if command -v python3 &>/dev/null; then
    echo "================================================"
    echo " vLLM GPU Kernel 全景教学文档"
    echo "================================================"
    echo " 正在启动本地服务器..."
    echo " 请在浏览器打开："
    echo "   http://localhost:${PORT}/kernels_learning.html"
    echo ""
    echo " 按 Ctrl+C 停止服务器"
    echo "================================================"
    cd "$SCRIPT_DIR"
    python3 -m http.server $PORT &
    SERVER_PID=$!
    sleep 1

    # 尝试自动打开浏览器
    if command -v xdg-open &>/dev/null; then
        xdg-open "http://localhost:${PORT}/kernels_learning.html" 2>/dev/null &
    elif command -v open &>/dev/null; then
        open "http://localhost:${PORT}/kernels_learning.html"
    elif command -v google-chrome &>/dev/null; then
        google-chrome "http://localhost:${PORT}/kernels_learning.html" 2>/dev/null &
    elif command -v chromium-browser &>/dev/null; then
        chromium-browser "http://localhost:${PORT}/kernels_learning.html" 2>/dev/null &
    else
        echo "提示：请手动在浏览器打开 http://localhost:${PORT}/kernels_learning.html"
    fi

    wait $SERVER_PID
else
    echo "警告：未找到 python3，将尝试直接打开文件（CDN 资源可能无法加载）"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$HTML_FILE"
    elif command -v open &>/dev/null; then
        open "$HTML_FILE"
    else
        echo "请手动打开文件：$HTML_FILE"
    fi
fi
