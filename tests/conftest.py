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
