"""统一字体加载：解决 pygame 界面（含图表/HUD）中文字符无法显示的问题。

语义分割等实验的图表/HUD 会渲染大量中文标签，一旦所在机器没有中文字体，
原先 ``SysFont("microsoftyahei", ...)`` 会**静默**退回到不含中文字形的默认
字体（不会抛异常），导致中文全部渲染成方块。这里统一封装字体解析，按优先级：

  1. 系统字体回退链  microsoftyahei / simhei / pingfang
     （用 ``pygame.font.match_font`` 校验真实字体文件存在，确认真能渲染中文后才采用）
  2. 仓库随包分发的开源中文字体（local_runner/fonts/ 下的 ttf/otf/ttc）
     作为不依赖对方系统的兜底，分发时需随包带上该目录
  3. pygame 内置默认字体（无中文字形，仅最后兜底）

用法：``load_font(16)`` / ``load_font(26, bold=True)``
"""
from __future__ import annotations

import os

import pygame

# 系统字体回退链：按顺序尝试，任一真实存在即采用
_SYSTEM_CANDIDATES = ("microsoftyahei", "simhei", "pingfang")

# 随包分发的开源字体目录（相对本文件，分发时连同目录一起拷贝）
_FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

_SUPPORTED_EXTS = (".ttf", ".otf", ".ttc")

# 字体对象缓存：同一 (size, bold) 只初始化一次
_cache: dict = {}


def _bundled_path():
    """返回随包开源字体的第一个可用文件路径；无则返回 None。"""
    if not os.path.isdir(_FONTS_DIR):
        return None
    for fn in sorted(os.listdir(_FONTS_DIR)):
        if fn.lower().endswith(_SUPPORTED_EXTS):
            p = os.path.join(_FONTS_DIR, fn)
            if os.path.isfile(p):
                return p
    return None


def load_font(size, bold=False):
    """返回一个已初始化的 pygame 字体对象（优先保证能渲染中文）。"""
    # 幂等初始化字体模块：本函数可能在 pygame.init() 之前被调用（例如综合驾驶
    # 规划阶段 map_picker 先于显示循环创建字体），先确保 SDL_ttf 就绪。
    pygame.font.init()
    key = (size, bold)
    if key in _cache:
        return _cache[key]

    font = None

    # 1. 系统字体回退链
    # 注意：match_font 会触发 pygame 的 Windows 注册表字体枚举，个别机器的
    # Font 注册表项可能是非字符串（整数）数据，pygame 2.6.1 枚举时会抛
    # TypeError（expected str... not int）。这里 try/except 兜住，改用随包字体。
    for name in _SYSTEM_CANDIDATES:
        try:
            # match_font 校验到真实字体文件才采用，避免 SysFont 静默退回无中文默认字体
            if pygame.font.match_font(name):
                font = pygame.font.SysFont(name, size, bold=bold)
                break
        except Exception:
            continue

    # 2. 随包开源字体兜底
    if font is None:
        bundled = _bundled_path()
        if bundled:
            try:
                font = pygame.font.Font(bundled, size)
            except Exception:
                font = None

    # 3. pygame 默认字体兜底
    if font is None:
        font = pygame.font.Font(None, size)

    _cache[key] = font
    return font


def clear_font_cache() -> None:
    """清空字体缓存。

    pygame.font 的 Font 对象底层持有 SDL_ttf 资源，一旦调用 ``pygame.quit()``
    这些资源就会被释放，但本模块的 _cache 仍会保留这些已失效对象。跨 quit/re-init
    （例如综合驾驶的规划窗口 → 驾驶窗口切换）继续复用会指向已释放内存，
    渲染时导致原生崩溃（黑屏闪退）。因此在每次 ``pygame.init()`` 后必须清空缓存，
    以便在全新的渲染上下文中重建字体对象。
    """
    _cache.clear()