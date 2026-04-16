import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import glob

def read_tsv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_radar(categories, save_path, methods,
               show_value_labels=True,
               show_percent_label=True,
               show_second_label=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager as fm
    import matplotlib.gridspec as gridspec

    font_prop = fm.FontProperties()
    plt.rcParams['axes.unicode_minus'] = False

    colors = ['#39CFC5', '#FF7D5E', '#6A5ACD', '#FFA500', '#32CD32', '#FF1493', '#00CED1', '#FFD700']
    
    num_rows = 1
    num_cols = len(categories)
    fig = plt.figure(figsize=(6 * num_cols, 6))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols)
    axs = []

    for idx, (category, data) in enumerate(categories.items()):
        labels = data['labels']

        def replace_space_after_second_word(s):
            if len(s) <= 15:
                return s
            words = s.split(' ')
            if len(words) < 3:
                return s
            return ' '.join(words[:2]) + '\n' + ' '.join(words[2:])

        labels = [replace_space_after_second_word(x) for x in labels]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        method_data = {}
        for method in methods:
            values = data[method][:]
            values += values[:1]
            method_data[method] = values

        ax = fig.add_subplot(gs[0, idx], polar=True)
        axs.append(ax)

        for i, method in enumerate(methods):
            values = method_data[method]
            color = colors[i % len(colors)]
            ax.plot(angles, values, color=color, linewidth=2, label=method, marker='o')
            ax.fill(angles, values, color=color, alpha=0.20)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(category + f" {data['xlabel']}",
                      size=7, fontproperties=font_prop, y=1.12)
        ax.set_yticklabels([])

        max_r = max(max(values) for values in method_data.values())
        
        base_offset = max_r * 0.09
        outer_offset = max_r * 0.13
        perc_offset = max_r * 0.18
        angle_offset = np.deg2rad(3)

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
                            # baseline
                            ax.text(angle - angle_offset, val - base_offset, f'{val:.1f}',
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
        handles, legend_labels, loc='upper center', ncol=min(len(methods), 4),
        prop=font_prop.copy().set_size(12), frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path+'.pdf', dpi=300, format='pdf')
    plt.show()

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
    plt.rcParams['font.family'] = "Liberation Mono"
    print("Drawing radar plot with : ", methods)
    
    root_dir = '.'
    # for kernel in ["fwd", "bwd", "total", "fwd_time", "bwd_time", "total_time", "sparsity"]:
    for kernel in ["fwd", "bwd", "total"]:
        for dtype in ['bf16']:
            for headdim in [64, 128, 192, 256]:
                headdim_v = 128 if headdim == 192 else headdim
                categories = {}
                for seqlen in [8192, 32768, 131072]:
                    method_to_df = {}
                    for method in methods:
                        filenames = glob.glob(f'{root_dir}/{dtype}/{method}_*{seqlen}_*_{headdim}_{headdim_v}*.csv')
                        dataframes = []
                        for file_path in filenames:
                            df = read_tsv_to_dataframe(file_path)
                            df.columns = df.columns.str.strip()
                            # d=192, dv=128: fa4_mask_mod only supports Full & Causal
                            # (mask_mod is not None is not supported), so truncate all
                            # methods to the first 2 rows to keep labels aligned.
                            if headdim == 192 and any(m.startswith('fa4_mask_mod') for m in methods):
                                df = df[:2]
                            if df is not None:
                                dataframes.append(df)

                        print(f"Method {method}, files: {filenames}")
                        non_numeric_column = get_column_name(df, 'Operation')

                        if kernel == "fwd":
                            metric = get_column_name(df, 'FW TFLOPs/s')
                        elif kernel == "bwd":
                            metric = get_column_name(df, 'BW TFLOPs/s')
                        elif kernel == "total":
                            metric = get_column_name(df, 'TOTAL TFLOPs/s')
                        elif kernel == "fwd_time":
                            metric = get_column_name(df, 'FW Time (ms)')
                        elif kernel == "bwd_time":
                            metric = get_column_name(df, 'BW Time (ms)')
                        elif kernel == "total_time":
                            metric = get_column_name(df, 'TOTAL Time (ms)')
                        elif kernel == "sparsity":
                            metric = get_column_name(df, 'Sparsity')
                        else:
                            raise ValueError(f"kernel must be fwd or bwd, but got {kernel}")

                        # columns_to_average = [metric, '  Sparsity']
                        columns_to_average = [metric]
        
                        if not dataframes:
                            print(f"Warning: No data found for method {method}, sequence length {seqlen}")
                            continue
                            
                        aligned_dataframes = [df[columns_to_average] for df in dataframes]
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
                    first_method = list(method_to_df.keys())[0]
                    labels = method_to_df[first_method][non_numeric_column].tolist()
                    labels = [label.strip() for label in labels]
                    one_item['labels'] = labels
                    
                    for method in methods:
                        if method in method_to_df:
                            one_item[method] = method_to_df[method][metric].tolist()
                        else:
                            print(f"Warning: Method {method} not found in data, using zeros")
                            one_item[method] = [0] * len(labels)
                    
                    if kernel == "fwd":
                        one_item['xlabel'] = 'Fwd Speed (TFLOPs/s)'
                    elif kernel == "bwd":
                        one_item['xlabel'] = 'Bwd Speed (TFLOPs/s)'
                    elif kernel == "total":
                        one_item['xlabel'] = 'Total Speed (TFLOPs/s)'
                    elif kernel == "fwd_time":
                        one_item['xlabel'] = 'FW Time (ms)'
                    elif kernel == "bwd_time":
                        one_item['xlabel'] = 'BW Time (ms)'
                    elif kernel == "total_time":
                        one_item['xlabel'] = 'TOTAL Time (ms)'
                    elif kernel == "sparsity":
                        one_item['xlabel'] = 'Sparsity'
                    else:
                        raise ValueError(f"kernel must be fwd or bwd, but got {kernel}")

                    categories[f'Sequence length {seqlen//1024}K, head dim {headdim}'] = one_item
                
                if categories:
                    methods_str = "_vs_".join(methods)
                    plot_radar(categories, f'{root_dir}/fig/{methods_str}_{dtype}_{headdim}_{kernel}', methods, 
                              show_value_labels=True, show_percent_label=True)
                else:
                    print(f"Warning: No categories data for {dtype}_{headdim}_{kernel}")

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
