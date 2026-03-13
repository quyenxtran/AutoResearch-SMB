import os
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import value

from .smb_config import SMBConfig, SMBInputs


def extract_profiles(m, inputs: SMBInputs, time_index=None):
    if time_index is None:
        time_index = m.t.at(-1)

    c = len(m.x)
    x = np.zeros(c * inputs.ncols)
    c_num = np.zeros((inputs.ncomp, c * inputs.ncols))


    for j in range(inputs.ncomp):
        count = 0   
        for i in range(inputs.ncols):
            for l in range(c):
                c_num[j, count] = value(m.C[i + 1, j + 1, time_index, m.x.at(l + 1)])
                x[count] = count
                count += 1

    return x, c_num


def _profile_title(config: SMBConfig, inputs: SMBInputs) -> str:
    title_prefix = ''
    if inputs.run_name:
        title_prefix = f"{inputs.run_name} - "
    title_string = f"{title_prefix}SMB "
    for comp in inputs.comps:
        title_string += f"{comp}/"
    tmb_string = ' ('
    for n in inputs.nc:
        tmb_string += f"{n}-"

    title_string = title_string[:-1]
    tmb_string = tmb_string[:-1]

    return title_string + tmb_string


def _plot_profile_axes(ax, m, x, y, config: SMBConfig, inputs: SMBInputs, ylabel: str, ylim, mode: str, title: str):
    colors = ['k', 'b', 'g', 'fuchsia']
    for j in range(inputs.ncomp):
        ax.plot(x, y[j], c=colors[j])

    count = len(x)
    ax.set_xticks(
        np.linspace(0, count, 9),
        [str(int(i)) for i in np.linspace(0, config.L * inputs.ncols, 9)],
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel('SMB Length (cm)')
    ax.set_title(title + f") D/F = {value(m.UD / m.UF):.3f}")
    ax.legend(inputs.comps, loc='upper center', bbox_to_anchor=(0.5, -0.26), ncol=2)
    ax.set_ylim(ylim)

    s = ['D', 'E', 'F', 'R']
    a = ['->', '<-', '->', '<-']
    nc_string = [0, inputs.nc[0], inputs.nc[0] + inputs.nc[1], inputs.nc[0] + inputs.nc[1] + inputs.nc[2]]

    for i in range(len(s)):
        ax.annotate(
            s[i],
            xy=((nc_string[i]) / inputs.ncols * count, 0),
            xytext=((nc_string[i]) / inputs.ncols * count, 20 if mode == 'wt%' else 100),
            horizontalalignment='center',
            arrowprops=dict(arrowstyle=a[i], lw=1),
            fontsize=14,
        )


def _format_sigfigs(value, sigfigs: int) -> str:
    try:
        return f"{float(value):.{sigfigs}g}"
    except (TypeError, ValueError):
        return str(value)


def _flow_info_text(m, config: SMBConfig, inputs: SMBInputs) -> str:
    u_f = value(m.UF) if hasattr(m, "UF") else inputs.u_f
    u_d = value(m.UD) if hasattr(m, "UD") else inputs.u_d
    u_e = value(m.UE) if hasattr(m, "UE") else inputs.u_e
    u_r = value(m.UR) if hasattr(m, "UR") else inputs.u_r
    q_f = u_f * inputs.area * inputs.eb
    q_d = u_d * inputs.area * inputs.eb
    q_e = u_e * inputs.area * inputs.eb
    q_r = u_r * inputs.area * inputs.eb
    return (
        "Flows: "
        f"Fdes={_format_sigfigs(q_d, 3)}, "
        f"Fex={_format_sigfigs(q_e, 3)}, "
        f"Ffeed={_format_sigfigs(q_f, 3)}, "
        f"Fraf={_format_sigfigs(q_r, 3)},"
        f"tstep={_format_sigfigs(inputs.tstep, 2)},"
        
        # f"Frec={_format_sigfigs(q_d + , 2)}"
    )


def _metrics_lines(metrics: dict) -> list:
    ordered_keys = [
        'Frec',
        'productivity_ex_ga_ma',
        'purity_ex_meoh_free',
        'purity_ex_overall',
        'recovery_ex',
        'recovery_ex_GA',
        'recovery_ex_MA',
        'recovery_raff',
        'recovery_raff_GA',
        'recovery_raff_MA',

    ]
    parts = []
    for key in ordered_keys:
        if key in metrics:
            try:
                val = float(metrics[key])
                parts.append(f"{key}={_format_sigfigs(val, 3)}")
            except (TypeError, ValueError):
                parts.append(f"{key}={metrics[key]}")
    if not parts:
        return []
    split_idx = (len(parts) + 1) // 2
    line1 = " ".join(parts[:split_idx])
    line2_parts = parts[split_idx:]
    if not line2_parts:
        return [line1]
    line2 = " ".join(line2_parts)
    return [line1, line2]


def _safe_run_name(run_name: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in run_name.strip())
    return safe or "run"


def plot_profiles(
    m,
    config: SMBConfig,
    inputs: SMBInputs,
    mode: str = 'wt%',
    show_time_endpoints: bool = True,
    save_dir: str = "reports",
    metrics: dict = None,
):
    x, c_num = extract_profiles(m, inputs)

    if mode == 'wt%':
        y = c_num / np.sum(c_num, axis=0) * 100.0
        ylabel = 'wt%'
        ylim = (0, 100)
        # print(y)
        
    elif mode == 'g/L':
        y = c_num * 1000.0
        ylabel = 'g/L'
        ylim = (0, 1000)
    else:
        raise ValueError("mode must be 'wt%' or 'g/L'")

    # Plot final time profile
    title_base = _profile_title(config, inputs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    _plot_profile_axes(ax, m, x, y, config, inputs, ylabel, ylim, mode, title_base)
    if mode == 'wt%':
        ax.axhline(40, color='r', linestyle='--')

    if not show_time_endpoints:
        return
    # Plot time endpoint profiles
    x0, c0 = extract_profiles(m, inputs, time_index=m.t.at(1)) # initial time
    x1, c1 = extract_profiles(m, inputs, time_index=m.t.at(-1)) # final time

    if mode == 'wt%':
        y0 = c0 / np.sum(c0, axis=0) * 100.0
        y1 = c1 / np.sum(c1, axis=0) * 100.0
    else:
        y0 = c0 * 1000.0
        y1 = c1 * 1000.0

    ax = axes[1]
    colors = ['k', 'b', 'g', 'fuchsia']
    for j in range(inputs.ncomp):
        ax.plot(x0, y0[j], c=colors[j], linestyle='-', label=f"{inputs.comps[j]} (t=0)")
        ax.plot(x1, y1[j], c=colors[j], linestyle='--', label=f"{inputs.comps[j]} (t=t_step)")

    count0 = len(x0)
    ax.set_xticks(
        np.linspace(0, count0, 9),
        [str(int(i)) for i in np.linspace(0, config.L * inputs.ncols, 9)],
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel('SMB Length (cm)')
    ax.set_title(title_base + ") t=0 vs t=t_step")
    ax.set_ylim(ylim)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
    if mode == 'wt%':
        ax.axhline(40, color='r', linestyle='--')

    s = ['D', 'E', 'F', 'R']
    a = ['->', '<-', '->', '<-']
    nc_string = [0, inputs.nc[0], inputs.nc[0] + inputs.nc[1], inputs.nc[0] + inputs.nc[1] + inputs.nc[2]]

    for i in range(len(s)):
        ax.annotate(
            s[i],
            xy=((nc_string[i]) / inputs.ncols * count0, 0),
            xytext=((nc_string[i]) / inputs.ncols * count0, 20 if mode == 'wt%' else 100),
            horizontalalignment='center',
            arrowprops=dict(arrowstyle=a[i], lw=1),
            fontsize=14,
        )

    flow_text = _flow_info_text(m, config, inputs)
    fig.text(0.5, 0.15, flow_text, ha='center', va='bottom', fontsize=10)
    if metrics:
        metrics_lines = _metrics_lines(metrics)
        if metrics_lines:
            base_y = 0.03
            line_step = 0.06
            for i, line in enumerate(metrics_lines):
                fig.text(0.5, base_y - i * line_step, line, ha='center', va='bottom', fontsize=9)

    fig.tight_layout(rect=(0, 0.1, 1, 1)) # leave space at bottom for text

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        run_label = _safe_run_name(inputs.run_name) if inputs.run_name else "run"
        filename = f"{run_label}_{mode}.png"
        fig.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()
