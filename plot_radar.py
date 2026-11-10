import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import glob
import os

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

# kernel key -> (CSV column, axis label)
METRICS = {
    "fwd": ('FW TFLOPs/s', 'Fwd Speed (TFLOPs/s)'),
    "bwd": ('BW TFLOPs/s', 'Bwd Speed (TFLOPs/s)'),
    "total": ('TOTAL TFLOPs/s', 'Total Speed (TFLOPs/s)'),
    "fwd_time": ('FW Time (ms)', 'FW Time (ms)'),
    "bwd_time": ('BW Time (ms)', 'BW Time (ms)'),
    "total_time": ('TOTAL Time (ms)', 'TOTAL Time (ms)'),
    "sparsity": ('Sparsity', 'Sparsity'),
}


def method_label(method, labels=None):
    """Display name: --labels first, then METHOD_LABELS, then the raw method name."""
    return (labels or {}).get(method, METHOD_LABELS.get(method, method))


def save_figure(fig, save_path):
    """Save as <dir>/png/<name>.png and <dir>/pdf/<name>.pdf.

    Returns both absolute paths, so callers can print them as-is and the terminal
    / IDE can open them directly.
    """
    out_dir, name = os.path.split(save_path)
    paths = []
    for ext in ('png', 'pdf'):
        sub_dir = os.path.join(out_dir, ext)
        os.makedirs(sub_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(sub_dir, f'{name}.{ext}'))
        fig.savefig(path, dpi=300, format=ext)
        paths.append(path)
    return paths


def plot_radar_grid(rows, save_path, methods,
                    suptitle=None,
                    show_value_labels=True,
                    show_percent_label=True,
                    show_second_label=False,
                    method_labels=None):
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
    legend_labels = [method_label(label, method_labels) for label in legend_labels]

    fig.legend(
        handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.955),
        ncol=min(len(methods), 4),
        prop=font_prop.copy().set_size(12), frameon=False
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontproperties=font_prop, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    paths = save_figure(fig, save_path)
    plt.close(fig)
    return paths


def plot_bar_grid(rows, save_path, methods,
                  suptitle=None,
                  show_value_labels=True,
                  show_percent_label=True,
                  method_labels=None):
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
    legend_labels = [method_label(label, method_labels) for label in legend_labels]
    fig.legend(
        handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
        ncol=min(len(methods), 4),
        prop=font_prop.copy().set_size(12), frameon=False
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontproperties=font_prop, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    paths = save_figure(fig, save_path)
    plt.close(fig)
    return paths



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


def series_label(path):
    """Default legend name of a series directory: '<series> (<run>, <commit>)' from meta.json."""
    import json
    name = os.path.basename(os.path.normpath(path))
    run = os.path.basename(os.path.dirname(os.path.normpath(path)))
    extra = [run] if run else []
    try:
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        commits = [p.get("git_commit") for p in meta.get("packages", []) if p.get("git_commit")]
        if commits:
            extra.append(commits[0])
    except (OSError, ValueError):
        pass
    return f"{name} ({', '.join(extra)})" if extra else name


def load_samples(root_dir, dtype, method, seqlen, headdim, headdim_v):
    """Read every sample CSV of one method / shape into one long DataFrame.

    benchmark_flashmask.py writes {method}_{B}_{S}_{H}_{HKV}_{D}_{DV}_{sample}.csv.
    Returns (DataFrame with columns sample, op and one per METRICS key, files),
    or (None, files) when nothing was readable.
    """
    if os.path.isdir(method):
        # --series: results/<run>/<series>/{B}_{S}_{H}_{HKV}_{D}_{DV}_{sample}.csv
        pattern = f'{method}/*_{seqlen}_*_*_{headdim}_{headdim_v}_*.csv'
    else:
        pattern = f'{root_dir}/{dtype}/{method}_*_{seqlen}_*_*_{headdim}_{headdim_v}_*.csv'
    files = sorted(glob.glob(pattern))
    frames = []
    for path in files:
        df = read_tsv_to_dataframe(path)
        if df is None:
            continue
        frame = pd.DataFrame({
            'sample': os.path.basename(path)[:-len('.csv')].rsplit('_', 1)[1],
            'op': df[get_column_name(df, 'Operation')].astype(str).str.strip(),
        })
        for key, (col, _) in METRICS.items():
            frame[key] = pd.to_numeric(df[get_column_name(df, col)], errors='coerce')
        frames.append(frame)
    if not frames:
        return None, files
    return pd.concat(frames, ignore_index=True), files


def op_means(samples):
    """Mean of every metric per Operation, plus n = number of samples averaged.

    Grouped by Operation NAME, not by row position: --dedup_static_masks makes
    benchmark_flashmask.py emit the S-only masks (Full, Causal, ...) on the first
    sample only, so the files of one seqlen no longer share a row count or a row
    order. A positional mean would blend different masks together.
    """
    grouped = samples.drop(columns='sample').groupby('op', sort=False)
    means = grouped.mean()
    means['n'] = grouped.size()
    return means


def check_sparsity(samples_by_method, tol=1e-4):
    """Check that every method measured the same masks as the baseline.

    Compares each (sample, Operation) with the first method: sparsity must agree
    within tol and the pair must exist on both sides. A speed ratio between two
    different masks means nothing, so callers drop the Operations listed here.
    Returns a list of dicts: method, sample, op, kind ('mismatch' / 'missing'),
    base (baseline sparsity), value (this method's sparsity).
    """
    methods = list(samples_by_method)
    keys = ['sample', 'op', 'sparsity']
    base = samples_by_method[methods[0]][keys]
    issues = []
    for method in methods[1:]:
        merged = base.merge(samples_by_method[method][keys], on=['sample', 'op'],
                            how='outer', suffixes=('_base', '_cmp'), indicator=True)
        for rec in merged.to_dict('records'):
            if rec['_merge'] != 'both':
                kind = 'missing'
            elif abs(rec['sparsity_base'] - rec['sparsity_cmp']) > tol:
                kind = 'mismatch'
            else:
                continue
            issues.append({'method': method, 'sample': rec['sample'], 'op': rec['op'], 'kind': kind,
                           'base': rec['sparsity_base'], 'value': rec['sparsity_cmp']})
    return issues


def collect_shape(root_dir, dtype, methods, headdim, headdim_v, seqlen, sparsity_tol=1e-4):
    """Load every method for one (head dim, seqlen) and align them for comparison.

    Returns None when no method has data, else a dict with
      means:    {method: op_means() DataFrame}
      labels:   Operations present in all methods with matching sparsity (baseline order)
      issues:   check_sparsity() result against the first method that has data
      excluded: Operations dropped because of those issues
      files:    {method: [csv paths]}
    """
    samples, files = {}, {}
    for method in methods:
        df, files[method] = load_samples(root_dir, dtype, method, seqlen, headdim, headdim_v)
        if df is None:
            print(f"Warning: no data for {method}, seqlen {seqlen}, head dim {headdim}/{headdim_v}")
            continue
        samples[method] = df
    if not samples:
        return None
    means = {m: op_means(df) for m, df in samples.items()}
    # samples keeps the methods order minus the absent ones, so the reference is the
    # first method with data here (a series without this head dim must not skip the check).
    issues = check_sparsity(samples, sparsity_tol)
    excluded = sorted({rec['op'] for rec in issues})
    first = next(iter(means))
    labels = [op for op in means[first].index
              if op not in excluded and all(op in means[m].index for m in means)]
    for method, df in means.items():
        dropped = [op for op in df.index if op not in labels and op not in excluded]
        if dropped:
            print(f"Note: {method} seqlen {seqlen}: categories not shared by all methods, dropped: {dropped}")
    return {'means': means, 'labels': labels, 'issues': issues,
            'excluded': excluded, 'files': files}


def report_sparsity(shapes, methods, headdim, headdim_v):
    """Print the sparsity check of one head dim; returns the number of issues."""
    issues = [(seqlen, rec) for seqlen, shape in shapes.items() for rec in shape['issues']]
    checked = sum(len(shape['labels']) + len(shape['excluded']) for shape in shapes.values())
    if not issues:
        print(f"Sparsity check [head dim {headdim}/{headdim_v}]: OK, {checked} (seqlen, Operation) "
              f"pairs identical across {len(methods)} methods")
        return 0
    print(f"Sparsity check [head dim {headdim}/{headdim_v}]: {len(issues)} issue(s); "
          f"these Operations are EXCLUDED from the comparison:")
    for seqlen, rec in issues:
        print(f"  seqlen {seqlen} sample {rec['sample']} {rec['op']:<24} {rec['kind']:<8} "
              f"{methods[0]}={rec['base']} {rec['method']}={rec['value']}")
    return len(issues)


def build_rows(shapes, methods, kernels, headdim):
    """rows structure of plot_radar_grid / plot_bar_grid from collect_shape() results."""
    rows = []
    for kernel in kernels:
        categories = {}
        for seqlen, shape in shapes.items():
            item = {'labels': shape['labels'], 'xlabel': METRICS[kernel][1]}
            for method in methods:
                if method in shape['means']:
                    item[method] = shape['means'][method].loc[shape['labels'], kernel].tolist()
                else:
                    print(f"Warning: {method} has no data at seqlen {seqlen}, head dim {headdim}: "
                          f"plotting zeros for this seqlen")
                    item[method] = [0] * len(shape['labels'])
            categories[f'Sequence length {seqlen // 1024}K, head dim {headdim}'] = item
        if categories:
            rows.append({'row_label': kernel, 'categories': categories})
    return rows


def main(methods: list = None,
         series: list = None,
         head_dims: list = [128, 192, 256],
         seqlens: list = [4096, 8192, 32768, 65536, 131072],
         chart: str = "radar",
         labels: list = None,
         root_dir: str = '.',
         out_dir: str = './fig',
         sparsity_tol: float = 1e-4):
    plt.rcParams['font.family'] = "DejaVu Sans"
    if series:
        if methods:
            raise ValueError("pass either --series (result directories) or --methods (old bf16/ layout)")
        missing = [d for d in series if not os.path.isdir(d)]
        if missing:
            raise ValueError(f"--series: not a directory: {missing}")
        methods = list(series)
        if not labels:
            labels = [series_label(d) for d in series]
    methods = methods or ["flashmaskv1", "flashmaskv3"]
    print("Drawing radar plot with : ", methods)
    if labels and len(labels) != len(methods):
        raise ValueError(f"--labels needs one name per method ({len(methods)}), got {len(labels)}")
    display = dict(zip(methods, labels)) if labels else None
    kernels = ["fwd", "bwd", "total"]
    saved = []

    for dtype in ['bf16']:
        for headdim, headdim_v in parse_head_dims(head_dims):
            shapes = {}
            for seqlen in seqlens:
                shape = collect_shape(root_dir, dtype, methods, headdim, headdim_v, seqlen, sparsity_tol)
                if shape is None:
                    print(f"Error: No data found for sequence length {seqlen}")
                    continue
                shapes[seqlen] = shape
                print(f"[head dim {headdim}/{headdim_v}, seqlen {seqlen}] "
                      + ", ".join(f"{m}: {len(f)} files" for m, f in shape['files'].items()))
            if not shapes:
                print(f"Warning: No data for {dtype}_{headdim}, skip figure")
                continue
            # Series that have no CSV at all for this head dim (e.g. fa4 / cudnn at
            # 576/512, kvshared at 128) are left out of this figure instead of being
            # drawn as zeros; the baseline is the first series that has data.
            present = [m for m in methods if any(m in s['means'] for s in shapes.values())]
            absent = [m for m in methods if m not in present]
            if absent:
                print(f"Note: head dim {headdim}/{headdim_v}: skip "
                      f"{', '.join(method_label(m, display) for m in absent)} (no CSV)")
            n_issues = report_sparsity(shapes, present, headdim, headdim_v)
            n_excluded = sum(len(shape['excluded']) for shape in shapes.values())
            rows = build_rows(shapes, present, kernels, headdim)

            # --- output improvement percentage: total only ---
            records, a, b = compute_improvement_range(rows, present)
            suptitle = None
            total_records = [rec for rec in records if rec['kernel'] == 'total']
            if total_records:
                total_incs = [rec['inc_pct'] for rec in total_records]
                a, b = min(total_incs), max(total_incs)
                print('#' * 40)
                print(f"[headdim={headdim}] TOTAL improvement of '{present[-1]}' over baseline '{present[0]}':")
                for rec in total_records:
                    print(f"  {rec['category']} | {rec['op']:<24} {rec['inc_pct']:+.1f}%")
                print(f"  total improvement range: {a:+.1f}% ~ {b:+.1f}%")
                print('#' * 40)
                suptitle = (f"{method_label(present[0], display)} vs {method_label(present[-1], display)}  |  "
                            f"dtype={dtype}, head dim={headdim}  |  "
                            f"total improvement range: {a:+.1f}% ~ {b:+.1f}%")
                if n_issues:
                    suptitle += f"  |  sparsity mismatch: {n_excluded} op(s) excluded"

            methods_str = "_vs_".join(
                "-".join(os.path.normpath(m).split(os.sep)[-2:]) if os.path.isdir(m) else m
                for m in present)
            kind = 'radar' if chart == 'radar' else 'bar'
            save_path = os.path.join(out_dir, f'{methods_str}_{dtype}_{headdim}_{headdim_v}_fwd_bwd_total_{kind}')
            plot_fn = plot_radar_grid if chart == 'radar' else plot_bar_grid
            paths = plot_fn(rows, save_path, present,
                            suptitle=suptitle,
                            show_value_labels=True, show_percent_label=True,
                            method_labels=display)
            for path in paths:
                print(f"Saved figure: {path}")
            saved.extend(paths)
    return saved


if __name__ == "__main__":
    from jsonargparse import ArgumentParser
    parser = ArgumentParser(description="Run specific examples or all examples.")

    parser.add_argument(
        "--series",
        type=str,
        nargs='+',
        default=None,
        help="Result directories to compare, first one is the baseline, e.g. "
        "results/20261109_0900/fm4 results/20261110_1200/fm4 results/20261110_1200/fa4. "
        "Legend defaults to '<series> (<run>, <commit>)' from each meta.json."
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        default=None,
        help="Old layout: method prefixes under <root_dir>/<dtype>/ "
        "(e.g. flashmaskv1 flashmaskv3 flexattention). Use --series for results/ runs."
    )

    parser.add_argument(
        "--head_dims",
        type=str,
        nargs='+',
        default=["128", "192", "256", "512,512", "576,512"],
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

    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        default=None,
        help="Display names for --methods, same order (e.g. base new). "
        "Defaults to METHOD_LABELS or the raw method name."
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Directory holding the <dtype>/ CSV folders."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./fig",
        help="Figures go to <out_dir>/png/ and <out_dir>/pdf/."
    )

    parser.add_argument(
        "--sparsity_tol",
        type=float,
        default=1e-4,
        help="Max allowed |sparsity difference| per (sample, Operation) between "
        "a method and the baseline; Operations beyond it are excluded."
    )

    args = parser.parse_args()
    main(**vars(args))
