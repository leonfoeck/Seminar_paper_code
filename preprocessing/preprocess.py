import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple



def _load_file(file_path: Path) -> List[Dict[str, str]]:
    with open(file_path, 'r') as file:
        json_dict = [json.loads(line) for line in file.readlines()]
        return json_dict


def _load_content(file_content: List[Dict[str, str]]) -> Tuple[List[str], List[str], List[str]]:
    labels = [_clean_and_normalize_entry(entry['subLabels']) for entry in file_content]
    tokens = [_clean_and_normalize_entry(entry['tokens'][0]) for entry in file_content]
    sub_tokens = [_clean_and_normalize_entry(entry['subTokens'][0])  for entry in file_content]
    return labels, tokens, sub_tokens


def _clean_and_normalize_entry(entry: str) -> str:
    escape_characters = ['\n', '\t', '\r']
    for escape_character in escape_characters:
        entry = entry.replace(escape_character, ' ')
    return entry + '\n'


def _export_content(export_dir: Path, export_file: Path, export_content: List[str]) -> None:
    with open(export_dir/export_file, 'w') as file:
        file.writelines(export_content)


def _arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--train-file", type=Path, default='data-preprocessed/java-med/training.txt')
    parser.add_argument("--val-file", type=Path, default='data-preprocessed/java-med/validation.txt')
    parser.add_argument("--test-file", type=Path, default='data-preprocessed/java-med/test.txt')

    parser.add_argument("--out-train-dir", type=Path, default='data/java/train')
    parser.add_argument("--out-val-dir", type=Path, default='data/java/dev')
    parser.add_argument("--out-test-dir", type=Path, default='data/java/test')

    parser.add_argument("--code-file-name", type=Path, default='code.original')
    parser.add_argument("--subtoken-file-name", type=Path, default='code.original_subtoken')
    parser.add_argument("--sublabels-file-name", type=Path, default='javadoc.original')

    return parser


def _main(arguments: List[str]) -> int:
    args = _arg_parser().parse_args(arguments[1:])
    files = [(args.train_file, args.out_train_dir), (args.val_file, args.out_val_dir), (args.test_file, args.out_test_dir)]

    code_file_name = args.code_file_name
    subtoken_file_name = args.subtoken_file_name
    sublabels_file_name = args.sublabels_file_name

    for (file, out_dir) in files:
        json_dict_list = _load_file(file)
        labels, tokens, subtokens = _load_content(json_dict_list)
        _export_content(out_dir, sublabels_file_name, labels)
        _export_content(out_dir, code_file_name, tokens)
        _export_content(out_dir, subtoken_file_name, subtokens)

    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))



