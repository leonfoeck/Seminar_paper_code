import dataclasses
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Generator, Iterable


def _load_file(file_path: Path) -> Generator[Dict[str, str], None, None]:
    with file_path.open("r", encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)


@dataclasses.dataclass(frozen=True)
class Content:
    labels: str
    tokens: str
    subtokens: str


def _load_content(entry: Dict[str, str]) -> Content:
    labels = _clean_and_normalize_entry(" ".join(entry["subLabels"]))
    tokens = _clean_and_normalize_entry(" ".join(entry["tokens"][0]))
    sub_tokens = _clean_and_normalize_entry(" ".join(entry["subTokens"][0]))
    return Content(labels, tokens, sub_tokens)


def _clean_and_normalize_entry(entry: str) -> str:
    escape_characters = ["\n", "\t", "\r"]
    for escape_character in escape_characters:
        entry = entry.replace(escape_character, " ")
    return entry + "\n"


def _export_content(
        sublabel_path: Path,
        code_path: Path,
        subtoken_path: Path,
        export_content: Iterable[Content],
) -> None:
    with sublabel_path.open("w", encoding='utf-8') as sublabel_file, code_path.open("w",encoding='utf-8') as code_file, subtoken_path.open("w", encoding='utf-8') as subtoken_file:
        for content in export_content:
            print(sublabel_file)
            sublabel_file.write(content.labels)
            code_file.write(content.tokens)
            subtoken_file.write(content.subtokens)


def _arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--train-file", type=Path,
                        default="/root/Big_Datasets/training.jsonl")
    parser.add_argument("--val-file", type=Path,
                        default="/root/Big_Datasets/validation.jsonl")
    parser.add_argument("--test-file", type=Path,
                        default="/root/Big_Datasets/test.jsonl")

    parser.add_argument("--out-train-dir", type=Path, default="/root/neuralcodesum/data/scratch/train")
    parser.add_argument("--out-val-dir", type=Path, default="/root/neuralcodesum/data/scratch/dev")
    parser.add_argument("--out-test-dir", type=Path, default="/root/neuralcodesum/data/scratch/test")

    parser.add_argument("--code-file-name", type=Path, default="code.original")
    parser.add_argument("--subtoken-file-name", type=Path, default="code.original_subtoken")
    parser.add_argument("--sublabels-file-name", type=Path, default="javadoc.original")

    return parser


def _main(arguments: List[str]) -> int:
    args = _arg_parser().parse_args(arguments[1:])
    files = [
        (args.train_file, args.out_train_dir),
        (args.val_file, args.out_val_dir),
        (args.test_file, args.out_test_dir),
    ]

    code_file_name = args.code_file_name
    subtoken_file_name = args.subtoken_file_name
    sublabels_file_name = args.sublabels_file_name

    for file, out_dir in files:
        os.makedirs(out_dir, exist_ok=True)
        data = (_load_content(d) for d in _load_file(file))
        _export_content(
            sublabel_path=out_dir / sublabels_file_name,
            code_path=out_dir / code_file_name,
            subtoken_path=out_dir / subtoken_file_name,
            export_content=data,
        )

    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
