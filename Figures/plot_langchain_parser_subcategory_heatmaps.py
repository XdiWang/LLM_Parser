from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_DIR = PROJECT_ROOT / "Figures"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures" / "output" / "langchain"
ATTACK_SUBCATEGORY_ORDER = [
    "White text",
    "Zero Size",
    "Double Layer PDF",
    "TRM",
    "Out-of-bound",
    "OCG",
    "Misaligned",
    "Zero Width",
]
PARSER_COLUMNS = [
    "PyPDF",
    "PDFMiner",
    "PDFPlumber",
    "PyMuPDF",
    "PyMuPDF4LLM",
    "PyPDFium2",
    "OpenDataLoader",
    "Unstructured",
    "Docling",
]
SOFT_REDS_CMAP = LinearSegmentedColormap.from_list(
    "soft_reference_reds",
    ["#fff5f0", "#fdd0c2", "#fc9272", "#ef3b2c", "#99000d"],
)
CSV_SPECS = [
    (
        "heatmap_s_rate.csv",
        "LangChain Parsers by Attack Subcategory - Injection Parsing Success Rate",
        "heatmap_s_rate.png",
        "Attack Success Rate",
    ),
    (
        "heatmap_c_recovery.csv",
        "LangChain Parsers by Attack Subcategory - Injection Recovery Completeness",
        "heatmap_c_recovery.png",
        "Injection Recovery Completeness",
    ),
]
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "SimHei",
    "SimSun",
    "NSimSun",
    "KaiTi",
    "FangSong",
    "PingFang SC",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
]


def configure_matplotlib_fonts() -> str | None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CHINESE_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    plt.rcParams["axes.unicode_minus"] = False
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="直接根据两个 LangChain CSV 绘制 8 种攻击子类热图")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="存放 heatmap_*.csv 的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="LangChain 热图输出目录",
    )
    parser.add_argument("--dpi", type=int, default=240, help="图片 DPI")
    return parser.parse_args()


def choose_figure_size(row_count: int, col_count: int) -> tuple[float, float]:
    width = max(8.8, col_count * 1.0 + 2.6)
    height = max(5.4, row_count * 0.72 + 1.8)
    return width, height


def load_direct_heatmap_csv(csv_path: Path) -> tuple[list[str], list[str], list[list[float]]]:
    row_labels: list[str] = []
    values: list[list[float]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        expected_columns = ["attack_subcategory", *PARSER_COLUMNS]
        if fieldnames != expected_columns:
            raise ValueError(
                f"CSV 列不匹配: {csv_path}，期望列为 {expected_columns}，实际为 {fieldnames}"
            )

        row_map: dict[str, dict[str, str]] = {}
        for row in reader:
            row_label = (row.get("attack_subcategory") or "").strip()
            if row_label not in ATTACK_SUBCATEGORY_ORDER:
                continue
            row_map[row_label] = row

    for row_label in ATTACK_SUBCATEGORY_ORDER:
        row = row_map.get(row_label)
        if row is None:
            continue
        row_labels.append(row_label)
        current_row: list[float] = []
        for parser_name in PARSER_COLUMNS:
            raw_value = (row.get(parser_name) or "").strip()
            current_row.append(float(raw_value) if raw_value else math.nan)
        values.append(current_row)

    return row_labels, PARSER_COLUMNS, values


def draw_heatmap(
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float]],
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig_width, fig_height = choose_figure_size(len(row_labels), len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = SOFT_REDS_CMAP.copy()
    cmap.set_bad(color="#d9d9d9")

    image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=14, pad=14)

    ax.set_xticks([x - 0.5 for x in range(1, len(col_labels))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(row_labels))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx, row_values in enumerate(values):
        for col_idx, value in enumerate(row_values):
            label = "N/A" if math.isnan(value) else f"{value:.3f}"
            text_color = "#1f1f1f" if math.isnan(value) or value < 0.6 else "white"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=9, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, rotation=90)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configured_font = configure_matplotlib_fonts()
    if configured_font:
        print(f"已启用中文字体: {configured_font}")
    else:
        print("未检测到可用中文字体，图中中文可能仍然显示异常")

    for csv_name, title, output_name, colorbar_label in CSV_SPECS:
        csv_path = args.csv_dir / csv_name
        if not csv_path.exists():
            print(f"未找到 CSV: {csv_path}")
            return 1

        row_labels, col_labels, values = load_direct_heatmap_csv(csv_path)
        if not row_labels or not col_labels:
            print(f"CSV 中未找到有效数据: {csv_path}")
            return 1

        output_path = args.output_dir / output_name
        draw_heatmap(row_labels, col_labels, values, title, colorbar_label, output_path, args.dpi)
        print(f"已生成热图: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
