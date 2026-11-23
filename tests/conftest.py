"""pytest 全局配置。

练习文件里还没写的函数会 ``raise NotImplementedError``。这里把这类测试标记为
SKIPPED（显示"练习未完成"），这样没写完的练习不会刷出一堆红色报错；
一旦你开始实现，其他任何错误都会正常显示为 FAILED。

查看还剩哪些练习：``pytest -rs``
"""

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if (rep.when in ('setup', 'call') and call.excinfo is not None
            and call.excinfo.errisinstance(NotImplementedError)):
        rep.outcome = 'skipped'
        rep.longrepr = (str(item.path), item.location[1] or 0,
                        f'练习未完成: {call.excinfo.value}')


@pytest.fixture(autouse=True)
def _rdet_default_scope():
    """所有测试默认在 rdet scope 下运行，保证 type='AnchorGenerator' 等构建出的是我们自己的实现。

    和 mmdet 对比的测试里，构建 mmdet 参照模块时会用 helpers.scope('mmdet') 临时切换。
    """
    try:
        from rdet.registry import HAS_MMDET
    except ImportError:          # 没装 torch 时连 rdet 都导入不了，mini_mm 的测试不需要 scope
        HAS_MMDET = False
    if not HAS_MMDET:
        yield
        return
    from mmengine.registry import DefaultScope
    with DefaultScope.overwrite_default_scope('rdet'):
        yield
