r"""
用于检测 Python 包的文档是否符合规范的脚本。

用法 ``python doc_utils.py <path>``

样例 ``python doc_utils.py ../../fastDemo``

.. csv-table::
   :header: "错误代号", "错误类型"

   0, "项目结构错误"
   1, "模块缺少 __doc__"
   2, "模块缺少 __all__"
   3, "__all__ 中导出的函数/类不应以下划线开头"
   4, "__all__ 中没有导出全部定义的函数/类等"
   5, "__all__ 中存在没有定义的函数/类"
   6, "函数/类中缺少 __doc__"
   7, "类的方法中缺少 __doc__"
   
"""

__all__ = [
    "check",
    "check_module",
    "check_obj"
]

from typing import List, Any
import inspect
import importlib
import sys
import os


class ModuleType:
    __name__: str
    __all__: List[str]


def _colored_string(string: str, color: str or int) -> str:
    r"""在终端中显示一串有颜色的文字

    :param string: 在终端中显示的文字
    :param color: 文字的颜色
    :return:
    """
    if isinstance(color, str):
        color = {
            "black": 30,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "purple": 35,
            "cyan": 36,
            "white": 37
        }[color]
    return "\033[%dm%s\033[0m" % (color, string)


def _alert(code: int, msg: str, color: str = 'red'):
    print(_colored_string("[ERROR-{}] {}".format(code, msg.strip()), color))


def check(path: str):
    r"""检查该项目目录下的所实现的包内的文档

    :param path: 项目目录
    :return:
    """
    path = os.path.abspath(path)
    print("Package path:", path)
    package_name = str(path.split(os.sep)[-1])
    print("Package name:", package_name)
    if not os.path.isdir(os.path.join(path, package_name)):
        _alert(0, "Package structure is wrong.")
        return
    sys.path.insert(0, path)
    importlib.import_module(package_name)
    module = sys.modules[package_name]
    check_module(module, package_name)


def check_module(module: ModuleType, base_name: str):
    r"""递归检查每个模块中对象是否有文档
    
    :param module: 模块对象
    :param base_name: 根模块的名称
    :return:
    """
    print("\n[M]", module.__name__)
    print([e for e in dir(module) if not e.startswith("_")])
    if module.__doc__ is None:
        _alert(1, f"""Module '{module.__name__}' don't have __doc__""")
    if "__all__" not in dir(module):
        _alert(2, f"""'{module.__name__}' don't have __all__""")
    else:
        set_all = set(module.__all__)
        for name, obj in inspect.getmembers(module):
            if inspect.ismodule(obj) and obj.__name__.startswith(base_name):
                check_module(obj, base_name)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if name.startswith("_"):
                    continue
                if name not in set_all:
                    _alert(4, f"""'{obj.__name__}' not in __all__ of '{module.__name__}' """)
                else:
                    check_obj(obj, module.__name__)
                    set_all.remove(name)
        for obj_name in set_all:
            if obj_name.startswith("_"):
                _alert(3, f"""'{obj_name}' in '{module.__name__}' should not start with '_'""")
            else:
                _alert(5, f""" '{obj_name}' in __all__ of '{module.__name__}' does not exist""")

    print("\n")


def check_obj(checked_obj: Any, module_name: str):
    r"""检查某个函数或者类的文档
    
    .. todo:
        
        增加对函数的注释中是否介绍了参数的检查
    
    :param checked_obj: 检查是否有文档的函数或者类
    :param module_name: 函数或者类所在地模块
    :return:
    """
    if inspect.isclass(checked_obj):
        for name, obj in inspect.getmembers(checked_obj):
            if inspect.isfunction(obj) and not obj.__name__.startswith("_"):
                if obj.__doc__ is None:
                    _alert(7, f""" '{checked_obj.__name__}.{obj.__name__}' in '{module_name}' does not have __doc__""")
    elif checked_obj.__doc__ is None:
        _alert(6, f""" '{checked_obj.__name__}' in '{module_name}' does not have __doc__""")


if __name__ == "__main__":
    check(sys.argv[1])
