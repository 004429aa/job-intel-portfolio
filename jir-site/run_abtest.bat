@echo off
chcp 65001 >nul
cd /d "E:\Aaaaaaa.Item\Item--Job Information Report\Jir-Site"

echo 正在启动 A/B 测试，请稍候...
D:\python\pythonpreservation\python.exe jir_site_api.py --abtest "有哪些和机器视觉相关的岗位"

echo.
echo =========================================
echo 执行结束！如果上方有红字报错，请检查配置。
echo 如果没有报错，请去文件夹查看 RAG评估.md
echo =========================================
pause