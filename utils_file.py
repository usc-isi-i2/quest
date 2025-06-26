import json
import os
from pathlib import Path
import tarfile
from typing import Any, Dict, List, Union
import zipfile

from loguru import logger


def read_json(filepath: Union[str, Path]) -> Dict[Any, Any]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(filepath: Union[str, Path]) -> List[Dict[Any, Any]]:
    if not os.path.exists(filepath):
        return []

    json_lines = []
    with open(filepath, encoding="utf-8") as f:
        while True:
            file_line = f.readline().strip()
            if not file_line:
                break
            json_lines.append(json.loads(file_line))
    return json_lines


def read_txt(filepath: Union[str, Path]) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().rstrip().split("\n")


def write_jsonl(data: List[Dict[Any, Any]], filepath: Union[str, Path]) -> None:
    dirname = Path(filepath).absolute().parent
    create_dir(dirname)
    with open(filepath, "w", encoding="utf-8") as f:
        for datum in data:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")


def write_json(data: Union[Dict[Any, Any], List[Dict[Any, Any]]], filepath: Union[str, Path]) -> None:
    dirname = Path(filepath).absolute().parent
    create_dir(dirname)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def append_jsonl(data: Union[Dict[Any, Any], List[Dict[Any, Any]]], filepath: Union[str, Path]) -> None:
    dirname = Path(filepath).absolute().parent
    create_dir(dirname)

    if isinstance(data, dict):
        data = [data]

    with open(filepath, "a", encoding="utf-8") as f:
        for datum in data:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")


def create_dir(dirname: Path) -> None:
    if not dirname.exists():
        dirname.mkdir(parents=True)


def unzip(zippath: Union[str, Path], unzippath: Union[str, Path]) -> Any:
    if os.path.exists(unzippath):
        logger.info(f"{zippath} is already unzipped in {unzippath}")
        return

    logger.info(f"Unzipping {zippath} to {unzippath} ...")
    return zipfile.ZipFile(zippath).extractall(unzippath)


def untar(tarpath: Union[str, Path], untarpath: Union[str, Path]) -> Any:
    if os.path.exists(untarpath):
        logger.info(f"{tarpath} is already untarred in {untarpath}")
        return

    logger.info(f"Untarring {tarpath} to {untarpath} ...")
    return tarfile.open(tarpath).extractall(untarpath)
