import os

from pathlib import Path
from typing import Union, Dict
from fastNLP.io.file_utils import get_cache_path, unzip_file
from transformers.utils import cached_file


def check_dataloader_paths(paths:Union[str, Dict[str, str]])->Dict[str, str]:
    """
    检查传入dataloader的文件的合法性。如果为合法路径，将返回至少包含'train'这个key的dict。类似于下面的结果
    {
        'train': '/some/path/to/', # 一定包含，建词表应该在这上面建立，剩下的其它文件应该只需要处理并index。
        'test': 'xxx' # 可能有，也可能没有
        ...
    }
    如果paths为不合法的，将直接进行raise相应的错误

    :param paths: 路径. 可以为一个文件路径(则认为该文件就是train的文件); 可以为一个文件目录，将在该目录下寻找train(文件名
        中包含train这个字段), test.txt, dev.txt; 可以为一个dict, 则key是用户自定义的某个文件的名称，value是这个文件的路径。
    :return:
    """
    if isinstance(paths, str):
        if os.path.isfile(paths):
            return {'train': paths}
        elif os.path.isdir(paths):
            filenames = os.listdir(paths)
            files = {}
            for filename in filenames:
                path_pair = None
                if 'train' in filename:
                    path_pair = ('train', filename)
                if 'dev' in filename:
                    if path_pair:
                        raise Exception("File:{} in {} contains both `{}` and `dev`.".format(filename, paths, path_pair[0]))
                    path_pair = ('dev', filename)
                if 'test' in filename:
                    if path_pair:
                        raise Exception("File:{} in {} contains both `{}` and `test`.".format(filename, paths, path_pair[0]))
                    path_pair = ('test', filename)
                if path_pair:
                    if path_pair[0] in files:
                        raise RuntimeError(f"Multiple file under {paths} have '{path_pair[0]}' in their filename.")
                    files[path_pair[0]] = os.path.join(paths, path_pair[1])
            return files
        else:
            raise FileNotFoundError(f"{paths} is not a valid file path.")

    elif isinstance(paths, dict):
        if paths:
            if 'train' not in paths:
                raise KeyError("You have to include `train` in your dict.")
            for key, value in paths.items():
                if isinstance(key, str) and isinstance(value, str):
                    if not os.path.isfile(value):
                        raise TypeError(f"{value} is not a valid file.")
                else:
                    raise TypeError("All keys and values in paths should be str.")
            return paths
        else:
            raise ValueError("Empty paths is not allowed.")
    else:
        raise TypeError(f"paths only supports str and dict. not {type(paths)}.")

def get_tokenizer():
    try:
        import spacy
        spacy.prefer_gpu()
        en = spacy.load('en')
        print('use spacy tokenizer')
        return lambda x: [w.text for w in en.tokenizer(x)]
    except Exception as e:
        print('use raw tokenizer')
        return lambda x: x.split()
    
# 返回本地缓存的模型目录路径
# 若本地无缓存，从huggingface中下载并解压
# 修改自 fastNLP.io.file_utils.cached_path, transformers.utils.cached_file
def hf_cached_path(model_url: str, cache_sub_dir: str):
    cache_dir = os.path.join(Path(get_cache_path()), cache_sub_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # model_name 为 fasthan_base 或 fasthan_large
    model_name = model_url.split("/")[-1]
    target_path = os.path.join(cache_dir, model_name)

    if model_name not in os.listdir(cache_dir):
        # 若本地不存在缓存, 从huggingface中下载
        zipped_file = cached_file(model_url, model_name+".zip")
        unzip_file(zipped_file, cache_dir)
    return target_path
