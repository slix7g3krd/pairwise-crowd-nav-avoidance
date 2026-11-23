# pairwise-crowd-nav-avoidance

## 领域

游戏、图形与多媒体

## 主题

群体寻路避让

## 说明

避让阻尼、导航版本与门容量。

初始状态下 `gameplay/crowd_nav_avoidance.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
