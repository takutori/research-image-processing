from __future__ import annotations

import os
import shutil
from pathlib import Path


def list_dataset_dirs(data_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in data_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
    )


def prompt_dataset_choice(dataset_dirs: list[Path]) -> Path | None:
    print("data 配下の削除対象フォルダを選んでください。")
    for index, dataset_dir in enumerate(dataset_dirs, start=1):
        print(f"{index}. {dataset_dir.name}")

    raw_choice = input("番号を入力して Enter を押してください: ").strip()
    if not raw_choice.isdigit():
        print("数字で入力してください。")
        return None

    selected_index = int(raw_choice)
    if not 1 <= selected_index <= len(dataset_dirs):
        print("有効な番号を入力してください。")
        return None

    return dataset_dirs[selected_index - 1]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    host_data_dir = os.environ.get("HOST_DATA_DIR")
    data_root = Path(host_data_dir).expanduser() if host_data_dir else project_root / "data"

    if not data_root.exists():
        print(f"data ディレクトリが見つかりません: {data_root}")
        return

    dataset_dirs = list_dataset_dirs(data_root)
    if not dataset_dirs:
        print("削除対象のデータセットフォルダがありません。")
        return

    selected_dir = prompt_dataset_choice(dataset_dirs)
    if selected_dir is None:
        return

    try:
        display_path = selected_dir.relative_to(project_root)
    except ValueError:
        display_path = selected_dir

    confirmation = input(f"{display_path} を本当に削除しますか？ [Y/N]: ").strip()
    if confirmation != "Y":
        print("削除を中止しました。")
        return

    shutil.rmtree(selected_dir)
    print(f"削除しました: {display_path}")


if __name__ == "__main__":
    main()
