import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import glob

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def read_tsv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        # tabulate pads header cells to the widest value in the column, so the
        # same logical column arrives as 'Operation   ' in one file and
        # 'Operation      ' in another depending on which mask names that run
        # emitted. Canonicalise on read: a name resolved from one file's columns
        # is then a valid key in every other file's dataframe.
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

METHOD_LABELS = {
    'old_flashmaskv3': 'FlashMask V3 B.O.',
    'flashmaskv3': 'FlashMask V3',
    'flashmaskv1': 'FlashMask V1',
    'flexattention': 'FlexAttention',
    'fa4_mask_mod': 'FA4 mask_mod',
    'flashmaskv4': 'FlashMask V4 (KV split)',
    'flashmaskv4kvshared': 'FlashMask V4 (KV shared)',
    'mlasparseattn': 'MLA Sparse Attn (+mask)',
    'tilelangswamqa': 'TileLang SWA MQA (+mask)',
}

COLORS = ['#39CFC5', '#FF7D5E', '#6A5ACD', '#FFA500', '#32CD32', '#FF1493', '#00CED1', '#FFD700']


def plot_radar_grid(rows, save_path, methods,
                    suptitle=None,
                    show_value_labels=True,
                    show_percent_label=True,
                    show_second_label=False):
    """Plot a grid of radar charts.

    rows: list of dicts, one per row (e.g. fwd / bwd / total), each of the form
          {'row_label': str, 'categories': {category_name: data, ...}}
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager as fm
    import matplotlib.gridspec as gridspec

    font_prop = fm.FontProperties()
    plt.rcParams['axes.unicode_minus'] = False

    colors = ['#39CFC5', '#FF7D5E', '#6A5ACD', '#FFA500', '#32CD32', '#FF1493', '#00CED1', '#FFD700']

    num_rows = len(rows)
    num_cols = max(len(r['categories']) for r in rows)
    fig = plt.figure(figsize=(6 * num_cols, 6 * num_rows))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols)
    axs = []

    def replace_space_after_second_word(s):
        if len(s) <= 15:
            return s
        words = s.split(' ')
        if len(words) < 3:
            return s
        return ' '.join(words[:2]) + '\n' + ' '.join(words[2:])

    for r_idx, row in enumerate(rows):
        row_label = row['row_label']
        for c_idx, (category, data) in enumerate(row['categories'].items()):
            labels = [replace_space_after_second_word(x) for x in data['labels']]
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]

            method_data = {}
            for method in methods:
                values = data[method][:]
                values += values[:1]
                method_data[method] = values

            ax = fig.add_subplot(gs[r_idx, c_idx], polar=True)
            axs.append(ax)

            for i, method in enumerate(methods):
                values = method_data[method]
                color = colors[i % len(colors)]
                ax.plot(angles, values, color=color, linewidth=2, label=method, marker='o')
                ax.fill(angles, values, color=color, alpha=0.20)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=9)
            # category / metric title
            ax.set_title(f"{category} {data['xlabel']}", size=8, fontproperties=font_prop, y=1.12)
            # fwd / bwd / total label: light color + bold
            ax.text(0.5, 1.22, row_label.upper(), transform=ax.transAxes,
                    ha='center', va='center', color='#9AA7B8', fontweight='bold',
                    fontsize=15, fontproperties=font_prop)
            ax.set_yticklabels([])

            max_r = max(max(values) for values in method_data.values())

            base_offset = max_r * 0.10
            outer_offset = max_r * 0.13
            perc_offset = max_r * 0.20
            angle_offset = np.deg2rad(7)

            for i_method, method in enumerate(methods):
                values = method_data[method]
                color = colors[i_method % len(colors)]
                for i in range(num_vars):
                    angle = angles[i]
                    val = values[i]
                    if show_value_labels:
                        if len(methods) == 1:
                            ax.text(angle, val + outer_offset, f'{val:.3f}',
                                    color=color, fontsize=9, ha='center', va='center',
                                    fontproperties=font_prop,
                                    bbox=dict(boxstyle="round,pad=0.18", fc="w", ec=color, lw=0.6, alpha=0.85))
                        else:
                            if i_method == 0:
                                # baseline (place on the opposite angular side from the percentage)
                                ax.text(angle + angle_offset, val - base_offset, f'{val:.1f}',
                                        color=color, fontsize=9, ha='center', va='center',
                                        fontproperties=font_prop,
                                        bbox=dict(boxstyle="round,pad=0.18", fc="w", ec=color, lw=0.6, alpha=0.85))
                            elif show_second_label:
                                # compare
                                ax.text(angle + angle_offset, val + outer_offset, f'{val:.1f}',
                                        color=color, fontsize=9, ha='center', va='center',
                                        fontproperties=font_prop,
                                        bbox=dict(boxstyle="round,pad=0.18", fc="w", ec=color, lw=0.6, alpha=0.85))

            # draw percentage (when len methods >= 2)
            if show_percent_label and len(methods) >= 2:
                baseline_method = methods[0]
                baseline_values = method_data[baseline_method]
                for i in range(num_vars):
                    angle = angles[i]
                    bval = baseline_values[i]
                    for method_idx in range(1, len(methods)):
                        compare_method = methods[method_idx]
                        fval = method_data[compare_method][i]
                        inc = (fval / bval - 1) * 100 if bval != 0 else np.nan
                        sign = "+" if not np.isnan(inc) and inc >= 0 else ""
                        if not np.isnan(inc):
                            method_perc_offset = perc_offset + (method_idx - 1) * (max_r * 0.08)
                            method_color = colors[method_idx % len(colors)]
                            ax.text(angle - angle_offset, fval + method_perc_offset, f'{sign}{inc:.1f}%',
                                    color=method_color, fontsize=9, ha='center', va='center',
                                    fontproperties=font_prop, fontweight='bold',
                                    bbox=dict(boxstyle="round,pad=0.19", fc="#f7f7f7", ec=method_color, lw=0.7, alpha=0.7))

    handles, legend_labels = axs[0].get_legend_handles_labels()
    legend_labels = [METHOD_LABELS.get(label, label) for label in legend_labels]

    fig.legend(
        handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.955),
        ncol=min(len(methods), 4),
        prop=font_prop.copy().set_size(12), frameon=False
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontproperties=font_prop, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path + '.pdf', dpi=300, format='pdf')
    plt.close(fig)


def plot_bar_grid(rows, save_path, methods,
                  suptitle=None,
                  show_value_labels=True,
                  show_percent_label=True):
    """Grouped bar version of plot_radar_grid, same ``rows`` structure.

    A radar with two or three axes degenerates into a line or a triangle and is
    hard to read; grouped bars stay legible for any number of Operations, which
    is the usual case when only a couple of mask patterns are comparable across
    methods.
    """
    font_prop = fm.FontProperties()
    plt.rcParams['axes.unicode_minus'] = False

    num_rows = len(rows)
    num_cols = max(len(r['categories']) for r in rows)
    fig = plt.figure(figsize=(1.9 * max(3, num_cols * 3), 4.2 * num_rows))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols)
    axs = []

    for r_idx, row in enumerate(rows):
        row_label = row['row_label']
        for c_idx, (category, data) in enumerate(row['categories'].items()):
            labels = data['labels']
            num_vars = len(labels)
            x = np.arange(num_vars)
            width = 0.8 / max(1, len(methods))

            ax = fig.add_subplot(gs[r_idx, c_idx])
            axs.append(ax)

            all_vals = [v for m in methods for v in data[m]]
            max_v = max(all_vals) if all_vals else 1.0

            for i, method in enumerate(methods):
                values = data[method]
                color = COLORS[i % len(COLORS)]
                pos = x - 0.4 + width * (i + 0.5)
                ax.bar(pos, values, width=width * 0.92, color=color,
                       label=method, edgecolor='white', linewidth=0.6)
                if show_value_labels:
                    for xi, val in zip(pos, values):
                        ax.text(xi, val + max_v * 0.015, f'{val:.1f}',
                                ha='center', va='bottom', fontsize=8, color=color,
                                fontproperties=font_prop)
                if show_percent_label and i > 0:
                    base = data[methods[0]]
                    for xi, val, bval in zip(pos, values, base):
                        if bval == 0:
                            continue
                        inc = (val / bval - 1) * 100
                        ax.text(xi, val + max_v * 0.075, f'{inc:+.1f}%',
                                ha='center', va='bottom', fontsize=8.5, color=color,
                                fontweight='bold', fontproperties=font_prop,
                                bbox=dict(boxstyle="round,pad=0.16", fc="#f7f7f7",
                                          ec=color, lw=0.6, alpha=0.75))

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, rotation=18, ha='right')
            ax.set_ylabel(data['xlabel'], fontsize=9, fontproperties=font_prop)
            ax.set_ylim(0, max_v * 1.22)
            ax.set_title(f"{category}", size=9, fontproperties=font_prop)
            ax.text(0.01, 1.06, row_label.upper(), transform=ax.transAxes,
                    ha='left', va='center', color='#9AA7B8', fontweight='bold',
                    fontsize=13, fontproperties=font_prop)
            ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)

    handles, legend_labels = axs[0].get_legend_handles_labels()
    legend_labels = [METHOD_LABELS.get(label, label) for label in legend_labels]
    fig.legend(
        handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
        ncol=min(len(methods), 4),
        prop=font_prop.copy().set_size(12), frameon=False
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontproperties=font_prop, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path + '.pdf', dpi=300, format='pdf')
    plt.close(fig)



def compute_improvement_range(rows, methods):
    """Collect per-point improvement (%) of the last method over the baseline (methods[0]).

    Returns (records, a, b) where records is a list of dicts, a = min%, b = max%.
    """
    if len(methods) < 2:
        return [], None, None
    baseline = methods[0]
    compare = methods[-1]
    records = []
    for row in rows:
        for category, data in row['categories'].items():
            bvals = data[baseline]
            cvals = data[compare]
            for label, bval, cval in zip(data['labels'], bvals, cvals):
                if bval == 0:
                    continue
                inc = (cval / bval - 1) * 100
                records.append({
                    'kernel': row['row_label'],
                    'category': category,
                    'op': label,
                    'baseline': bval,
                    'compare': cval,
                    'inc_pct': inc,
                })
    if not records:
        return records, None, None
    incs = [r['inc_pct'] for r in records]
    return records, min(incs), max(incs)

def get_column_name(df, target, strip=True, startswith=False):
    for col in df.columns:
        col_cmp = col.strip() if strip else col
        target_cmp = target.strip() if strip else target
        if col_cmp == target_cmp:
            return col
        if startswith and col_cmp.startswith(target_cmp):
            return col
    raise KeyError(f"No column found for '{target}' (strip={strip}, startswith={startswith})")

def parse_head_dims(head_dims):
    """['576,512', 128] -> [(576, 512), (128, 128)].

    A bare head dim keeps the old derivation (dv == d, except 192 -> 128); the
    'd,dv' form is what the MLA-shaped runs need (576/512), where dv cannot be
    derived from d.
    """
    pairs = []
    for item in head_dims:
        text = str(item)
        if ',' in text:
            d_str, dv_str = text.split(',', 1)
            pairs.append((int(d_str), int(dv_str)))
        else:
            d = int(text)
            pairs.append((d, 128 if d == 192 else d))
    return pairs


def main(methods: list = ["flashmaskv1", "flashmaskv3"],
         head_dims: list = [128, 192, 256],
         seqlens: list = [4096, 8192, 32768, 65536, 131072],
         chart: str = "radar"):
    plt.rcParams['font.family'] = "DejaVu Sans"
    print("Drawing radar plot with : ", methods)

    root_dir = '.'
    kernels = ["fwd", "bwd", "total"]
    kernel_xlabel = {
        "fwd": 'Fwd Speed (TFLOPs/s)',
        "bwd": 'Bwd Speed (TFLOPs/s)',
        "total": 'Total Speed (TFLOPs/s)',
        "fwd_time": 'FW Time (ms)',
        "bwd_time": 'BW Time (ms)',
        "total_time": 'TOTAL Time (ms)',
        "sparsity": 'Sparsity',
    }
    kernel_metric_col = {
        "fwd": 'FW TFLOPs/s',
        "bwd": 'BW TFLOPs/s',
        "total": 'TOTAL TFLOPs/s',
        "fwd_time": 'FW Time (ms)',
        "bwd_time": 'BW Time (ms)',
        "total_time": 'TOTAL Time (ms)',
        "sparsity": 'Sparsity',
    }

    for dtype in ['bf16']:
        for headdim, headdim_v in parse_head_dims(head_dims):
            rows = []
            # for kernel in ["fwd", "bwd", "total", "fwd_time", "bwd_time", "total_time", "sparsity"]:
            for kernel in kernels:
                categories = {}
                for seqlen in seqlens:
                    method_to_df = {}
                    metric = None
                    non_numeric_column = None
                    for method in methods:
                        filenames = glob.glob(f'{root_dir}/{dtype}/{method}_*{seqlen}_*_{headdim}_{headdim_v}*.csv')
                        dataframes = []
                        for file_path in filenames:
                            df = read_tsv_to_dataframe(file_path)
                            if df is not None:
                                dataframes.append(df)

                        print(f"Method {method}, files: {filenames}")
                        if not dataframes:
                            print(f"Warning: No data found for method {method}, sequence length {seqlen}")
                            continue

                        df = dataframes[0]
                        non_numeric_column = get_column_name(df, 'Operation')
                        metric = get_column_name(df, kernel_metric_col[kernel])

                        # Average across the samples of this seqlen by Operation
                        # NAME, not by row position: --dedup_static_masks makes
                        # benchmark_flashmask.py emit the S-only masks (Full,
                        # Causal, ...) on the first sample only, so the files of
                        # one seqlen no longer share a row count or a row order.
                        # A positional mean would blend different masks together
                        # and then label them from the first file.
                        frames = []
                        for d in dataframes:
                            frames.append(pd.DataFrame({
                                non_numeric_column: d[get_column_name(d, 'Operation')].astype(str).str.strip(),
                                metric: pd.to_numeric(d[get_column_name(d, kernel_metric_col[kernel])], errors='coerce'),
                            }))
                        combined_data = pd.concat(frames, axis=0, ignore_index=True)
                        grouped = combined_data.groupby(non_numeric_column, sort=False)
                        mean_df = grouped[metric].mean().reset_index()
                        counts = grouped[metric].count()
                        if counts.nunique() > 1:
                            print(f"Note: Method {method} seqlen {seqlen} averaged an "
                                  f"uneven number of samples per Operation "
                                  f"(expected with --dedup_static_masks): "
                                  f"{counts.to_dict()}")
                        method_to_df[method] = mean_df
                        print('='*20)
                        print(f"Method {method} data:")
                        print(mean_df)

                    if not method_to_df:
                        print(f"Error: No data found for sequence length {seqlen}")
                        continue

                    one_item = {}
                    # Align categories by Operation name across methods, since
                    # different benchmarks produce different mask category sets
                    # (e.g. flashmask has "Document Mask" but fa4_mask_mod has
                    # "Global Sliding Window"). Positional alignment would silently
                    # mismatch values. Build op -> metric maps and keep only the
                    # categories shared by all present methods.
                    op2val = {}
                    for method, mean_df in method_to_df.items():
                        op2val[method] = {
                            op.strip(): val
                            for op, val in zip(
                                mean_df[non_numeric_column].tolist(),
                                mean_df[metric].tolist(),
                            )
                        }

                    first_method = list(method_to_df.keys())[0]
                    first_labels = [op.strip() for op in method_to_df[first_method][non_numeric_column].tolist()]
                    common_labels = [
                        lbl for lbl in first_labels
                        if all(lbl in op2val[m] for m in method_to_df)
                    ]

                    for method in method_to_df:
                        dropped = [lbl for lbl in op2val[method] if lbl not in common_labels]
                        if dropped:
                            print(f"Note: Method {method} categories dropped (not shared by all methods): {dropped}")

                    one_item['labels'] = common_labels

                    for method in methods:
                        if method in method_to_df:
                            one_item[method] = [op2val[method][lbl] for lbl in common_labels]
                        else:
                            print(f"Warning: Method {method} not found in data, using zeros")
                            one_item[method] = [0] * len(common_labels)

                    one_item['xlabel'] = kernel_xlabel[kernel]

                    categories[f'Sequence length {seqlen // 1024}K, head dim {headdim}'] = one_item

                if categories:
                    rows.append({'row_label': kernel, 'categories': categories})
                else:
                    print(f"Warning: No categories data for {dtype}_{headdim}_{kernel}")

            if not rows:
                print(f"Warning: No data for {dtype}_{headdim}, skip figure")
                continue

            # --- output improvement percentage: total only ---
            records, a, b = compute_improvement_range(rows, methods)
            suptitle = None
            total_records = [rec for rec in records if rec['kernel'] == 'total']
            if total_records:
                total_incs = [rec['inc_pct'] for rec in total_records]
                a, b = min(total_incs), max(total_incs)
                print('#' * 40)
                print(f"[headdim={headdim}] TOTAL improvement of '{methods[-1]}' over baseline '{methods[0]}':")
                for rec in total_records:
                    print(f"  {rec['category']} | {rec['op']:<24} {rec['inc_pct']:+.1f}%")
                print(f"  total improvement range: {a:+.1f}% ~ {b:+.1f}%")
                print('#' * 40)
                suptitle = (f"{methods[0]} vs {methods[-1]}  |  dtype={dtype}, head dim={headdim}  |  "
                            f"total improvement range: {a:+.1f}% ~ {b:+.1f}%")

            methods_str = "_vs_".join(methods)
            kind = 'radar' if chart == 'radar' else 'bar'
            save_path = f'{root_dir}/fig/{methods_str}_{dtype}_{headdim}_{headdim_v}_fwd_bwd_total_{kind}'
            plot_fn = plot_radar_grid if chart == 'radar' else plot_bar_grid
            plot_fn(rows, save_path, methods,
                    suptitle=suptitle,
                    show_value_labels=True, show_percent_label=True)
            print(f"Saved figure: {save_path}(.pdf)")

if __name__ == "__main__":
    from jsonargparse import ArgumentParser
    parser = ArgumentParser(description="Run specific examples or all examples.")

    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        default=["flexattention", "flashmaskv3"],
        help="List of methods to compare (e.g., flashmaskv1 flashmaskv3 flexattention)"
    )

    parser.add_argument(
        "--head_dims",
        type=str,
        nargs='+',
        default=["128", "192", "256", "576,512"],
        help="Head dims to plot. '128' derives dv from d (dv == d, 192 -> 128); "
        "'576,512' gives d and dv explicitly, which is required for the "
        "MLA-shaped runs where dv cannot be derived."
    )

    parser.add_argument(
        "--seqlens",
        type=int,
        nargs='+',
        default=[4096, 8192, 32768, 65536, 131072],
        help="Sequence lengths to plot, one column each."
    )

    parser.add_argument(
        "--chart",
        type=str,
        default="radar",
        help="'radar' (default) or 'bar'. Prefer 'bar' when only two or three "
        "Operations are shared across methods -- a 2-axis radar is just a line."
    )

    args = parser.parse_args()
    main(**vars(args))
