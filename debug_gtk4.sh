#!/bin/bash

# GTK4调试脚本 - 诊断PyGObject和GTK4环境问题

echo "=== GTK4环境诊断脚本 ==="
echo "时间: $(date)"
echo ""

# 1. 检查系统GTK4包
echo "1. 检查系统GTK4包安装..."
dpkg -l | grep -E "(python3-gi|libgtk-4|libadwaita|girepository)" | head -10

echo ""
echo "2. 检查Python路径..."
echo "系统Python路径: $(which python3)"
echo "虚拟环境Python路径: $(ls -la /opt/bamboo-cut/venv/bin/python 2>/dev/null || echo '虚拟环境不存在')"

echo ""
echo "3. 测试系统Python的gi模块..."
python3 -c "
try:
    import gi
    print('✓ gi模块可用')
    gi.require_version('Gtk', '4.0')
    from gi.repository import Gtk
    print('✓ GTK4可用，版本:', Gtk.get_major_version(), '.', Gtk.get_minor_version())
    try:
        gi.require_version('Adw', '1')
        from gi.repository import Adw
        print('✓ Adwaita可用')
    except Exception as e:
        print('⚠ Adwaita不可用:', e)
except Exception as e:
    print('✗ gi模块错误:', e)
"

echo ""
echo "4. 测试虚拟环境Python的gi模块..."
if [ -f "/opt/bamboo-cut/venv/bin/python" ]; then
    /opt/bamboo-cut/venv/bin/python -c "
import sys
print('虚拟环境Python路径:', sys.executable)
print('sys.path前5项:')
for i, path in enumerate(sys.path[:5]):
    print(f'  {i}: {path}')

try:
    import gi
    print('✓ gi模块可用')
    gi.require_version('Gtk', '4.0')
    from gi.repository import Gtk
    print('✓ GTK4可用，版本:', Gtk.get_major_version(), '.', Gtk.get_minor_version())
    try:
        gi.require_version('Adw', '1')
        from gi.repository import Adw
        print('✓ Adwaita可用')
    except Exception as e:
        print('⚠ Adwaita不可用:', e)
except Exception as e:
    print('✗ gi模块错误:', e)
    print('建议：重新创建虚拟环境')
    print('rm -rf /opt/bamboo-cut/venv')
    print('python3 -m venv /opt/bamboo-cut/venv --system-site-packages')
"
else
    echo "✗ 虚拟环境不存在"
fi

echo ""
echo "5. 检查关键路径..."
echo "Python包路径:"
ls -la /usr/lib/python3/dist-packages/gi* 2>/dev/null || echo "  gi包不存在"

echo ""
echo "类型库路径:"
ls -la /usr/lib/*/girepository-1.0/Gtk-4.0.typelib 2>/dev/null || echo "  GTK-4.0.typelib不存在"
ls -la /usr/lib/*/girepository-1.0/Adw-1.typelib 2>/dev/null || echo "  Adw-1.typelib不存在"

echo ""
echo "6. 检查systemd服务状态..."
systemctl is-active bamboo-python-gtk4.service 2>/dev/null || echo "服务未运行"
echo "最近的服务日志:"
journalctl -u bamboo-python-gtk4.service --no-pager -n 5 2>/dev/null || echo "无法获取日志"

echo ""
echo "=== 诊断完成 ==="