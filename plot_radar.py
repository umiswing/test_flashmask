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
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

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
    method_name_mapping = {
        'old_flashmaskv3': 'FlashMask V3 B.O.',
        'flashmaskv3': 'FlashMask V3',
        'flashmaskv1': 'FlashMask V1',
        'flexattention': 'FlexAttention',
        'fa4_mask_mod': 'FA4 mask_mod',
        'flashmaskv4': 'FlashMask V4',
    }
    legend_labels = [method_name_mapping.get(label, label) for label in legend_labels]

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

def main(methods: list = ["flashmaskv1", "flashmaskv3"]):
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
        for headdim in [128, 192, 256]:
            headdim_v = 128 if headdim == 192 else headdim
            rows = []
            # for kernel in ["fwd", "bwd", "total", "fwd_time", "bwd_time", "total_time", "sparsity"]:
            for kernel in kernels:
                categories = {}
                for seqlen in [4096, 8192, 32768, 131072]:
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
                        columns_to_average = [metric]

                        aligned_dataframes = [d[columns_to_average] for d in dataframes]
                        combined_data = pd.concat(aligned_dataframes, axis=0, keys=range(len(dataframes)))
                        mean_df = combined_data.groupby(level=1).mean()
                        mean_df[non_numeric_column] = dataframes[0][non_numeric_column]
                        mean_df = mean_df[[non_numeric_column] + columns_to_average]
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
            save_path = f'{root_dir}/fig/{methods_str}_{dtype}_{headdim}_fwd_bwd_total'
            plot_radar_grid(rows, save_path, methods,
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

    args = parser.parse_args()
    main(**vars(args))
