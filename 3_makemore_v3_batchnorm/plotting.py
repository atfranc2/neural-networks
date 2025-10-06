import numpy as np
import matplotlib.pyplot as plt

def plot_hist(data: np.ndarray, title: str, file_name: str, xlim=None) -> str:
    """
    Create and save a clean histogram.

    Parameters:
    - data: 1D numpy array of values to histogram
    - title: figure title
    - file_path: output path, including filename (e.g., '.../figures/hist.png')

    Returns the saved file path.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    data = data.ravel()

    # Determine sensible bins via Freedman–Diaconis rule, fallback to 50
    if data.size > 1:
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        bin_width = 2 * iqr / (data.size ** (1 / 3) + 1e-12)
        if bin_width > 0:
            bins = int(np.clip(np.ceil((data.max() - data.min()) / bin_width), 10, 200))
        else:
            bins = 50
    else:
        bins = 10

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, color='tab:blue', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_title(title)
    if xlim is not None: 
        ax.set_xlim(xlim)
    ax.set_xlabel('value')
    ax.set_ylabel('count')
    ax.grid(alpha=0.2, linestyle=':')
    fig.tight_layout()

    fig.savefig(f"3_makemore_v3_batchnorm/hists/{file_name}.png", dpi=150)
    plt.close(fig)
    return file_name

# Basic

data_rng  = np.random.default_rng(seed=35)
data = np.asarray(data_rng.standard_normal((1, 10000)))

plot_hist(
    data,
    f"Standard Normal Data: Mean {data.mean():.2f}, VAR: {data.var():.2f}",
    "standard_normal",
    (-4,4)
)

scale_down = data * 0.1

plot_hist(
    scale_down,
    f"Standard Normal Data Scaled by 0.1: Mean {scale_down.mean():.2f}, VAR: {scale_down.var():.2f}",
    "scaled_down_standard_normal",
    (-4,4)
)

shifted = data + 1

plot_hist(
    shifted,
    f"Standard Normal Data Shifted By 1: Mean {shifted.mean():.2f}, VAR: {shifted.var():.2f}",
    "shifted_standard_normal",
    (-4,4)
)

shifted_and_scaled = (data * 0.1) + 1

plot_hist(
    shifted_and_scaled,
    f"Standard Normal Data Shifted By 1 and Scaled By 0.1: Mean {shifted_and_scaled.mean():.2f}, VAR: {shifted_and_scaled.var():.2f}",
    "shifted_and_scaled_standard_normal",
    (-4,4)
)


# Activations

weight_rng = np.random.default_rng(seed=399)
weight = np.asarray(weight_rng.standard_normal((200, 10000)))

input_rng = np.random.default_rng(seed=9498)
input = np.asarray(input_rng.standard_normal((1, 200)))

stn_pre_activation = input @ weight

plot_hist(
    stn_pre_activation,
    f"Raw Pre-Activation: Mean {stn_pre_activation.mean():.2f}, VAR: {stn_pre_activation.var():.2f}",
    "raw_pre_activation"
)

stn_activation = np.tanh(input @ weight)

plot_hist(
    stn_activation,
    f"Raw Activation: Mean {stn_activation.mean():.2f}, VAR: {stn_activation.var():.2f}",
    "raw_activation",
    (-1,1)
)


scale = 0.01
raw_scale_pre_activation = input @ (weight * scale)

plot_hist(
    raw_scale_pre_activation,
    f"Raw Scaled Pre-Activation: Mean {raw_scale_pre_activation.mean():.2f}, VAR: {raw_scale_pre_activation.var():.2f}",
    "raw_scaled_pre_activation"
)

raw_scale_stn_activation = np.tanh(input @ (weight * scale))

plot_hist(
    raw_scale_stn_activation,
    f"Raw Scaled Activation: Mean {raw_scale_stn_activation.mean():.2f}, VAR: {raw_scale_stn_activation.var():.2f}",
    "raw_scaled_activation",
    (-1,1)
)

def kaiming(arr:np.ndarray, gain=5/3):
    # return arr * ((5/3)/(arr.shape[0])**0.5)
    return arr * (gain/(arr.shape[0])**0.5)

fan_in = weight.shape[0]
kaiming_pre_activation_no_gain = input @ kaiming(weight, 1)

plot_hist(
    kaiming_pre_activation_no_gain,
    f"Kaiming Pre Activation (Without Gain): Mean {kaiming_pre_activation_no_gain.mean():.2f}, VAR: {kaiming_pre_activation_no_gain.var():.2f}",
    "kaiming_pre_activation_no_gain"
)

kaiming_activation_no_gain = np.tanh(input @ kaiming(weight, 1))

plot_hist(
    kaiming_activation_no_gain,
    f"Kaiming Activation (Without Gain): Mean {kaiming_activation_no_gain.mean():.2f}, VAR: {kaiming_activation_no_gain.var():.2f}",
    "kaiming_activation_no_gain",
    (-1,1)
)


kaiming_pre_activation = input @ kaiming(weight)

plot_hist(
    kaiming_pre_activation,
    f"Kaiming Pre Activation: Mean {kaiming_pre_activation.mean():.2f}, VAR: {kaiming_pre_activation.var():.2f}",
    "kaiming_pre_activation"
)

kaiming_activation = np.tanh(input @ kaiming(weight))

plot_hist(
    kaiming_activation,
    f"Kaiming Activation: Mean {kaiming_activation.mean():.2f}, VAR: {kaiming_activation.var():.2f}",
    "kaiming_activation",
    (-1,1)
)

# Out 1 x 10,0000


def create_weights(layers=24, inputs=10000, outputs=500):
    in_size = outputs
    more_weights_rng  = np.random.default_rng(seed=984984635)
    weight_arrs = [np.asarray(more_weights_rng.standard_normal((inputs, outputs)))]
    out_rng  = np.random.default_rng(seed=4946511)
    for i in range(layers):
        out_size = max(150, int((np.abs(more_weights_rng.standard_normal())) * 2000))
        weight_arrs.append(np.asarray(more_weights_rng.standard_normal((in_size, out_size))))
        in_size =out_size

    return weight_arrs


def tail_prop(arr: np.ndarray, prop=0.98):
    mask = (arr > prop) | (arr < -prop)
    return np.count_nonzero(mask) / arr.size
    
def range_prop(arr: np.ndarray, prop=0.05):
    mask = (arr < prop) & (arr > -prop)
    return np.count_nonzero(mask) / arr.size

def plot_lines(
    data: list[np.ndarray] | list[list[float]],
    title: str,
    file_name: str,
    labels: list[str] | None = None,
    y_label:str|None = None,
    y_lim = None,
    x: np.ndarray | list[float] | None = None,
) -> str:
    """
    Plot multiple arrays as lines with different colors and optional legend labels.

    Parameters:
    - data: list of 1D arrays/lists (same length)
    - title: plot title
    - file_name: output filename without extension (saved under hists/)
    - labels: optional legend labels, same length as data
    - x: optional x-axis values; defaults to 0..N-1

    Returns the saved file path.
    """
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise ValueError("data must be a non-empty list of arrays")
    series = [np.asarray(d).ravel() for d in data]
    n = len(series[0])
    if any(len(s) != n for s in series):
        raise ValueError("all series must have the same length")

    if x is None:
        x_vals = np.arange(n)
    else:
        x_vals = np.asarray(x).ravel()
        if len(x_vals) != n:
            raise ValueError("x must be the same length as each series")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    for i, s in enumerate(series):
        color = colors[i % len(colors)] if colors else None
        label = labels[i] if labels and i < len(labels) else None
        ax.plot(x_vals, s, lw=2, label=label, color=color)

    ax.set_title(title)
    if y_lim:
        ax.get_ylim(y_lim)

    ax.set_xlabel('Layer')
    ax.set_ylabel('value' if y_label is None else y_label)
    ax.grid(alpha=0.2, linestyle=':')
    if labels:
        ax.legend(frameon=False)
    fig.tight_layout()

    out_path = f"3_makemore_v3_batchnorm/lines/{file_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

more_weights = create_weights(layers=18)
prop_tail = 0.95
prop_range = 0.03
raw_output = input @ (weight * scale)
raw_activation = np.tanh(raw_output)
raw_output_var = [raw_output.var()]
raw_activation_var = [raw_activation.var()]
raw_in_tail = [tail_prop(raw_activation, prop_tail)]
raw_in_around_0 = [range_prop(raw_activation, prop_range)]

kaiming_output = input @ kaiming(weight)
kaiming_activation = np.tanh(kaiming_output)
kaiming_output_var = [kaiming_output.var()]
kaiming_activation_var = [kaiming_activation.var()]
kaiming_in_tail = [tail_prop(kaiming_activation, prop_tail)]
kaiming_in_around_0 = [range_prop(kaiming_activation, prop_range)]

plot_hist(raw_output, "Raw Pre-Activations", "1_pre_activation_hist")
plot_hist(raw_activation, "Raw Activations", "1_activation_hist", (-1,1))
plot_hist(kaiming_output, "Kaiming Pre-Activations", "1_kai_pre_activation_hist")
plot_hist(kaiming_activation, "Kaiming Activations", "1_kai_activation_hist", (-1,1))

for index, more_weight in enumerate(more_weights, 2):
    raw_output = raw_activation @ (more_weight * scale)
    raw_activation = np.tanh(raw_output)
    
    raw_output_var.append(raw_output.var())
    raw_activation_var.append(raw_activation.var())
    raw_in_tail.append(tail_prop(raw_activation, prop_tail))
    raw_in_around_0.append(range_prop(raw_activation, prop_range))

    fan_in = more_weight.shape[0]
    kaiming_output = kaiming_activation @ kaiming(more_weight)
    kaiming_activation = np.tanh(kaiming_output)
    kaiming_in_tail.append(tail_prop(kaiming_activation, prop_tail))
    kaiming_in_around_0.append(range_prop(kaiming_activation, prop_range))

    kaiming_output_var.append(kaiming_output.var())
    kaiming_activation_var.append(kaiming_activation.var())

    if index < 5:
        plot_hist(raw_output, "Raw Pre-Activations", f"{index}_pre_activation_hist")
        plot_hist(raw_activation, "Raw Activations", f"{index}_activation_hist", (-1,1))
        plot_hist(kaiming_output, "Kaiming Pre-Activations", f"{index}_kai_pre_activation_hist")
        plot_hist(kaiming_activation, "Kaiming Activations", f"{index}_kai_activation_hist", (-1,1))

plot_lines(
    [
        np.asarray(raw_activation_var) - np.asarray(raw_output_var),
        np.asarray(kaiming_activation_var) - np.asarray(kaiming_output_var)
    ],
    "Variance Gap Between Output and Activations",
    f"variance_gap_scale_{scale}",
    [f"Raw (Scale={scale})", "Kaiming"],
    "Variance Difference"
)

plot_lines(
    [raw_in_tail, raw_in_around_0,],
    "Proportion of Activations",
    f"raw_activation_proportion_{scale}",
    [f"Raw In Tail (Scale={scale})", f"Raw Near 0 (Scale={scale})",],
    "Proportion"
)
plot_lines(
    [kaiming_in_tail, kaiming_in_around_0],
    "Proportion of Activations",
    f"kaiming_activation_proportion",
    ["Kaiming In Tail", "Kaiming Near 0"],
    "Proportion"
)

plot_lines(
    [raw_output_var, raw_activation_var, kaiming_output_var, kaiming_activation_var],
    "Output Variance",
    f"output_variance_{scale}",
    [f"Raw (Scale={scale})", f"Raw Activation (Scale={scale})", "Kaiming", "Kaiming Activation"],
    "Variance"
)