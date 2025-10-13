import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import glob

# 读取 TSV 格式的数据到 DataFrame
def read_tsv_to_dataframe(file_path):
    try:
        # 使用 pandas 的 read_csv 函数，并指定分隔符为制表符
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_radar(categories, save_path, baseline_key,
               show_baseline_label=True,
               show_flashmaskv3_label=True,
               show_percent_label=True):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager as fm
    import matplotlib.gridspec as gridspec

    font_prop = fm.FontProperties()
    plt.rcParams['axes.unicode_minus'] = False

    num_rows = 1
    num_cols = len(categories)
    fig = plt.figure(figsize=(6 * num_cols, 6))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols)
    axs = []

    colors = ['#39CFC5', '#FF7D5E']

    for idx, (category, data) in enumerate(categories.items()):
        labels = data['labels']
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合曲线

        baseline = data[baseline_key][:]
        flashmaskv3 = data['flashmaskv3'][:]
        baseline += baseline[:1]
        flashmaskv3 += flashmaskv3[:1]

        ax = fig.add_subplot(gs[0, idx], polar=True)
        axs.append(ax)

        ax.plot(angles, baseline, color=colors[0], linewidth=2, label=baseline_key, marker='o')
        ax.fill(angles, baseline, color=colors[0], alpha=0.20)

        ax.plot(angles, flashmaskv3, color=colors[1], linewidth=2, label='flashmaskv3', marker='o')
        ax.fill(angles, flashmaskv3, color=colors[1], alpha=0.20)

        ax.set_xticks(angles[:-1])
        #ax.set_xticklabels(labels, fontsize=7, fontproperties=font_prop.copy().set_size(7))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(category + f" {data['xlabel']}", size=7, fontproperties=font_prop, y=1.12)
        ax.set_yticklabels([])

        max_r = max(max(baseline), max(flashmaskv3))
        base_offset = max_r * 0.09
        outer_offset = max_r * 0.13
        perc_offset = max_r * 0.18

        # 角度错位：±10度
        angle_offset = np.deg2rad(3)

        for i in range(num_vars):
            angle = angles[i]
            bval = baseline[i]
            fval = flashmaskv3[i]
            inc = (fval / bval - 1) * 100 if bval != 0 else np.nan
            sign = "+" if not np.isnan(inc) and inc >= 0 else ""

            # baseline数值：点的内侧，角度左移
            if show_baseline_label:
                ax.text(angle - angle_offset, bval - base_offset, f'{bval:.1f}',
                        color=colors[0], fontsize=7, ha='center', va='center',
                        fontproperties=font_prop,
                        bbox=dict(boxstyle="round,pad=0.18", fc="w", ec=colors[0], lw=0.6, alpha=0.85))

            # flashmaskv3数值：点的外侧，角度右移
            if show_flashmaskv3_label:
                ax.text(angle + angle_offset, fval + outer_offset, f'{fval:.1f}',
                        color=colors[1], fontsize=7, ha='center', va='center',
                        fontproperties=font_prop,
                        bbox=dict(boxstyle="round,pad=0.18", fc="w", ec=colors[1], lw=0.6, alpha=0.85))

            # 提升百分比：更外侧，居中
            if show_percent_label and not np.isnan(inc):
                ax.text(angle - angle_offset, fval + perc_offset, f'{sign}{inc:.1f}%',
                        color='#111', fontsize=7, ha='center', va='center',
                        fontproperties=font_prop, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.19", fc="#f7f7f7", ec='#111', lw=0.7, alpha=0.7))

    handles, legend_labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc='upper center', ncol=2,
        prop=font_prop.copy().set_size(7), frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path+'.pdf', dpi=300, format='pdf')
    plt.show()

def plot_bar(categories, save_path, baseline_key):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import font_manager as fm

    colors = ['#39CFC5', '#FF7D5E']
    font_prop = fm.FontProperties()
    plt.rcParams['axes.unicode_minus'] = False

    num_rows = 1
    num_cols = 3

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols)
    axs = []
    bar_height = 0.8

    for idx, (category, data) in enumerate(categories.items()):
        row = idx // num_cols
        col = idx % num_cols
        ax = fig.add_subplot(gs[row, col])
        axs.append(ax)

        labels = data['labels']
        baseline = data[baseline_key]
        flashmaskv3 = data['flashmaskv3']
        x = np.arange(len(labels))
        increments = [(fm - fa) / fa * 100 for fa, fm in zip(baseline, flashmaskv3)]

        # FlexAttention
        if baseline_key == 'flashmaskv1':
            ax.barh(x, baseline, bar_height, label='FlashMask V1', color=colors[0])
        elif baseline_key == 'flexattention':
            ax.barh(x, baseline, bar_height, label='Flex Attention', color=colors[0])
        elif baseline_key == 'old_flashmaskv3':
            ax.barh(x, baseline, bar_height, label='Old FlashMask V3', color=colors[0])
        else:
            raise ValueError(f"baselinekey must be flashmaskv1 or flexattention, got {baseline_key}")

        for j in range(len(labels)):
            ax.text(
                baseline[j] - max(baseline)*0.01, x[j],
                f'{baseline[j]:.1f}',
                va='center', ha='right', fontsize=12, color='white',
                fontproperties=font_prop
            )

        # FlashMask 增量
        ax.barh(
            # x, [flashmaskv3[j]-baseline[j] for j in range(len(labels))],
            x, [max(0, flashmaskv3[j]-baseline[j]) for j in range(len(labels))],
            bar_height, left=baseline, label='FlashMask V3', color=colors[1]
        )

        # 增量外部数值
        for j in range(len(labels)):
            increment = increments[j]
            sign = '+' if increment >= 0 else ''
            ax.text(
                max(baseline[j] + max(baseline)*0.005, flashmaskv3[j] + max(flashmaskv3)*0.005), x[j],
                f'{flashmaskv3[j]:.1f} ({sign}{increment:.1f}%)',
                va='center', ha='left', fontsize=12, color='black',
                fontproperties=font_prop
            )

        # Y轴
        ax.set_yticks(x)
        if idx == 0:
            ax.set_yticklabels(labels, fontsize=14, fontproperties=font_prop)
        else:
            ax.set_yticklabels(['' for _ in labels], fontsize=14, fontproperties=font_prop)
        ax.invert_yaxis()
        ax.set_xlabel(data['xlabel'], fontsize=14, fontproperties=font_prop)
        ax.set_title(category, fontsize=16, fontproperties=font_prop)
        ax.tick_params(axis='x', labelsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 图例
    handles, legend_labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc='upper center', ncol=2,
        prop=font_prop.copy().set_size(14), frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path+'.pdf', dpi=300, format='pdf')
    plt.show()

def main(baseline: str = "flashmaskv1"):
    plt.rcParams['font.family'] = "Liberation Mono"
    
    root_dir = '.'
    # for dtype in ['bf16', 'fp16']:
    for kernel in ["fwd", "bwd"]:
        for dtype in ['bf16']:
            # for headdim in [128, 64]:
            for headdim in [128]:
                categories = {}
                for seqlen in [8192, 32768, 131072]:
                    method_to_df = {}
                    for method in [baseline, 'flashmaskv3']:
                        filenames = glob.glob(f'{root_dir}/{dtype}/{method}_*{seqlen}_*_{headdim}*.csv')
                        print(filenames)
                        dataframes = []
                        non_numeric_column = 'Operation              '

                        if kernel == "fwd":
                            metric = '  FW TFLOPs/s'
                        elif kernel == "bwd":
                            metric = '  BW TFLOPs/s'
                        else:
                            raise ValueError(f"kernel must be fwd or bwd, but got {kernel}")

                        columns_to_average = [metric, '  Sparsity']

                        for file_path in filenames:
                            df = read_tsv_to_dataframe(file_path)
                            dataframes.append(df)
        
                        aligned_dataframes = [df[columns_to_average] for df in dataframes]
                        combined_data = pd.concat(aligned_dataframes, axis=0, keys=range(len(dataframes)))
                        mean_df = combined_data.groupby(level=1).mean()
                        mean_df[non_numeric_column] = dataframes[0][non_numeric_column]
                        mean_df = mean_df[[non_numeric_column] + columns_to_average] 
                        method_to_df[method] = mean_df
                        print('='*20)
                        print(mean_df)
                    one_item = {}
                    labels = method_to_df['flashmaskv3']['Operation              '].tolist()
                    labels = [label.strip() for label in labels]
                    one_item['labels'] = labels
                    one_item[baseline] = method_to_df[baseline][metric].tolist()
                    one_item['flashmaskv3 improvement'] = (method_to_df['flashmaskv3'][metric] - method_to_df[baseline][metric]).tolist()
                    one_item['flashmaskv3'] = method_to_df['flashmaskv3'][metric].tolist()
                    if kernel == "fwd":
                        one_item['xlabel'] = 'Fwd Speed (TFLOPs/s)'
                    elif kernel == "bwd":
                        one_item['xlabel'] = 'Bwd Speed (TFLOPs/s)'
                    else:
                        raise ValueError(f"kernel must be fwd or bwd, but got {kernel}")

                    categories[f'Sequence length {seqlen//1024}K, head dim {headdim}'] = one_item
                #plot_bar(categories, f'{root_dir}/flashmaskv3_vs_{baseline}_{dtype}_{headdim}_{kernel}', baseline)
                #plot_radar(categories, f'{root_dir}/flashmaskv3_vs_{baseline}_{dtype}_{headdim}_{kernel}', baseline, show_baseline_label=False, show_flashmaskv3_label=False, show_percent_label=True)
                plot_radar(categories, f'{root_dir}/flashmaskv3_vs_{baseline}_{dtype}_{headdim}_{kernel}', baseline)

if __name__ == "__main__":
    from jsonargparse import ArgumentParser
    parser = ArgumentParser(description="Run specific examples or all examples.")

    parser.add_argument(
        "--baseline",
        type=str,
        default="flashmaskv1"
    )

    args = parser.parse_args()
    main(**vars(args))
