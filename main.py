"""

"""
import numpy as np
import scipy
import colorednoise
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"


y_scalar_formatter = ScalarFormatterClass(useMathText=True)
y_scalar_formatter.set_powerlimits((0, 0))
plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["figure.autolayout"] = True


def voltage_trace_from_point_current_trace(current_trace_A, current_xyz_um, voltage_xyz_um):
    current_trace_A = np.array(current_trace_A)
    current_xyz_um = np.array(current_xyz_um).flatten()
    voltage_xyz_um = np.array(voltage_xyz_um).flatten()
    if current_xyz_um.size != 3:
        raise ValueError("current_xyz_um must be an array of size 3.")
    if voltage_xyz_um.size != 3:
        raise ValueError("voltage_xyz_um must be an array of size 3.")
    distance_m = np.linalg.norm(current_xyz_um - voltage_xyz_um) * 1e-6
    return current_trace_A / (4 * np.pi * 0.3333 * distance_m)


def generate_pink_noise(peak_to_peak_magnitude, timesteps):
    np_pink_noise = np.array(colorednoise.powerlaw_psd_gaussian(1, timesteps))
    peak_to_peak_original = np.max(np_pink_noise) - np.min(np_pink_noise)
    np_pink_noise_scaled = np_pink_noise * peak_to_peak_magnitude / peak_to_peak_original
    return np_pink_noise_scaled


def lerp(val_t0, val_t1, t):
    return val_t0 + (val_t1 - val_t0) * t


class TraceRepeater:

    def __init__(self, waveform, period_min_timesteps, period_max_timesteps):
        if len(waveform.shape) != 2:
            raise ValueError("waveform must have a shape of (n_vars, n_timesteps).")
        self._waveform_timesteps = waveform.shape[1]
        self._waveform = waveform
        self._period_min_timesteps = period_min_timesteps
        self._period_max_timesteps = period_max_timesteps
        self._counter_next = None
        self._interpolation_scale_factor = None
        self._rng = np.random.default_rng()
        self._restart_counter()

    def _restart_counter(self):
        period_timesteps = self._rng.integers(self._period_min_timesteps, self._period_max_timesteps, endpoint=True)
        first = self._counter_next is None
        self._counter_next = self._waveform_timesteps - period_timesteps
        self._interpolation_scale_factor = 1 / (self._counter_next - 1)
        if first:
            offset_timesteps = self._rng.integers(period_timesteps, endpoint=False)
            self._counter_next += offset_timesteps

    def get_next(self):
        return_value = (
            self._waveform[:, self._counter_next]
            if self._counter_next >= 0
            else lerp(
                val_t0=self._waveform[:, 0],
                val_t1=self._waveform[:, -1],
                t=(self._counter_next * self._interpolation_scale_factor)))
        if self._counter_next < self._waveform_timesteps - 1:
            self._counter_next += 1
        else:
            self._restart_counter()
        return return_value

    def get_next_n(self, n):
        return_values = []
        for _ in range(n):
            return_values.append(self.get_next())
        return np.stack(return_values, axis=1)


def main(input_path_currents_big, input_path_currents_small, input_path_K_matrix, input_path_K_matrix_dbs, output_dir):

    # =========================================================================
    # load and preprocess data
    # =========================================================================

    timestep_ms = 0.025

    _dict_currents_big = scipy.io.loadmat(input_path_currents_big)
    np_currents_big_xyz_um = _dict_currents_big["XYZ"]
    np_currents_big_trace_A = _dict_currents_big["currents"]
    np_currents_big_time_ms = np.arange(np_currents_big_trace_A.shape[1]) * timestep_ms

    _dict_currents_small = scipy.io.loadmat(input_path_currents_small)
    np_currents_small_xyz_um = _dict_currents_small["XYZ"]
    np_currents_small_trace_A = _dict_currents_small["currents"]
    np_currents_small_time_ms = np.arange(np_currents_small_trace_A.shape[1]) * timestep_ms

    _np_K_matrix = np.loadtxt(input_path_K_matrix, comments="%")
    np_K_matrix_xyz_um = _np_K_matrix[:, 0:3] * 1e6
    np_K_matrix_voltage_V = _np_K_matrix[:, 3]

    _np_K_matrix_dbs = np.loadtxt(input_path_K_matrix_dbs, comments="%")
    np_K_matrix_dbs_xyz_um = _np_K_matrix_dbs[:, 0:3] * 1e6
    np_K_matrix_dbs_voltage_V = _np_K_matrix_dbs[:, 3]

    # =========================================================================
    # plot voltage stuff
    # =========================================================================

    np_interpolation_distances_um = np.linspace(1, 1000, num=100)
    delta_distance_um = np_interpolation_distances_um[1] - np_interpolation_distances_um[0]
    np_interpolation_points_xyz_um = np.linspace((0, 1, 0), (0, 1000, 0), num=100)

    K_matrix_interpolated_values_V = scipy.interpolate.griddata(
        np_K_matrix_xyz_um, np_K_matrix_voltage_V, np_interpolation_points_xyz_um)
    fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
    axes.plot(np_interpolation_distances_um, K_matrix_interpolated_values_V)
    axes.set_title(f"electric potential over distance")
    axes.set_xlabel("distance (um)")
    axes.set_ylabel("voltage (V)")
    axes.yaxis.set_major_formatter(y_scalar_formatter)
    plt.tight_layout()
    plt.savefig(output_dir + f"electric potential decay over distance.png")
    plt.close()

    # K_matrix_dbs_interpolated_values_V = scipy.interpolate.griddata(
    #     np_K_matrix_dbs_xyz_um, np_K_matrix_dbs_voltage_V, np_interpolation_points_xyz_um)
    # first_derivative_V_per_um = (
    #         (K_matrix_dbs_interpolated_values_V[1:] - K_matrix_dbs_interpolated_values_V[:-1]) / delta_distance_um)
    # second_derivative_V_per_um2 = (
    #         (first_derivative_V_per_um[1:] - first_derivative_V_per_um[:-1]) / delta_distance_um)
    # fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
    # axes.plot(
    #     np_interpolation_distances_um[1:-1][second_derivative_V_per_um2 < 1e-5],
    #     second_derivative_V_per_um2[second_derivative_V_per_um2 < 1e-5])
    # axes.set_title(f"second derivative of electric potential over distance")
    # axes.set_xlabel("distance (um)")
    # axes.set_ylabel("d2voltage/dx2 (V/um2)")
    # axes.yaxis.set_major_formatter(y_scalar_formatter)
    # plt.tight_layout()
    # plt.savefig(output_dir + f"second derivative of electric potential over distance.png")
    # plt.close()

    # =========================================================================
    # calculate net voltage traces with and without noise
    # =========================================================================

    for current_type, electrode_dist_um in (
            ("big", 50), ("big", 100), ("big", 200), ("big", 300),
            ("small", 50), ("small", 100), ("small", 200), ("small", 300))[::-1]:
        np_currents_xyz_um = np_currents_big_xyz_um if current_type == "big" else np_currents_small_xyz_um
        np_currents_trace_A = np_currents_big_trace_A if current_type == "big" else np_currents_small_trace_A
        np_currents_time_ms = np_currents_big_time_ms if current_type == "big" else np_currents_small_time_ms
        electrode_xyz_um = (0, electrode_dist_um, 0)

        voltage_traces_to_superimpose_V = []
        for np_current_xyz_um, np_current_trace_A in zip(np_currents_xyz_um, np_currents_trace_A):
            voltage_traces_to_superimpose_V.append(
                voltage_trace_from_point_current_trace(
                    current_trace_A=np_current_trace_A,
                    current_xyz_um=np_current_xyz_um,
                    voltage_xyz_um=electrode_xyz_um))
        np_voltage_trace_clean_V = np.sum(voltage_traces_to_superimpose_V, axis=0)

        np_voltage_trace_pink_noise_V = generate_pink_noise(
            peak_to_peak_magnitude=10e-6,
            timesteps=np_voltage_trace_clean_V.shape[0])
        np_voltage_trace_noisy_V = np_voltage_trace_clean_V + np_voltage_trace_pink_noise_V

        fig, axes = plt.subplots(2, 1, sharex="all", sharey="all")
        axes[0].plot(np_currents_time_ms, np_voltage_trace_clean_V)
        axes[0].set_title(f"clean voltage trace from current_{current_type} at {electrode_dist_um}um")
        axes[1].plot(np_currents_time_ms, np_voltage_trace_noisy_V)
        axes[1].set_title(f"noisy voltage trace from current_{current_type} at {electrode_dist_um}um")
        axes[1].set_xlabel("time (ms)")
        [axes[i].set_ylabel("voltage (V)") for i in (0, 1)]
        [axes[i].yaxis.set_major_formatter(y_scalar_formatter) for i in (0, 1)]
        plt.tight_layout()
        plt.savefig(output_dir + f"voltage from current_{current_type} at {electrode_dist_um}um.png")
        plt.close()

    # =========================================================================
    # simulate multi neuron environment
    # =========================================================================

    n_neurons = 10
    simulation_duration_s = 1

    rng = np.random.default_rng()
    np_neurons_offset_xyz_um = rng.uniform(-50, 50, size=3*n_neurons).reshape((-1, 3))
    np_neurons_freq_Hz = rng.integers(2, 50, endpoint=True, size=n_neurons)
    n_timesteps = int(np.round(simulation_duration_s / (timestep_ms * 1e-3)))
    neurons_currents_trace_A = []
    for np_neuron_freq_Hz in np_neurons_freq_Hz:
        neuron_period_mean_timesteps = 1 / (np_neuron_freq_Hz * timestep_ms * 1e-3)
        trace_repeater = TraceRepeater(
            waveform=np_currents_big_trace_A,
            period_min_timesteps=int(np.round(neuron_period_mean_timesteps * 0.95)),
            period_max_timesteps=int(np.round(neuron_period_mean_timesteps * 1.05)))
        neurons_currents_trace_A.append(trace_repeater.get_next_n(n_timesteps))

    voltage_traces_to_superimpose_incl_deadzone_V = []
    voltage_traces_to_superimpose_excl_deadzone_V = []
    for np_neuron_offset_xyz_um, neuron_currents_trace_A in zip(np_neurons_offset_xyz_um, neurons_currents_trace_A):
        for np_current_xyz_um, np_current_trace_A in zip(
                np_currents_xyz_um + np_neuron_offset_xyz_um[np.newaxis, :],
                neuron_currents_trace_A):
            np_current_voltage_trace_V = voltage_trace_from_point_current_trace(
                current_trace_A=np_current_trace_A,
                current_xyz_um=np_current_xyz_um,
                voltage_xyz_um=[0, 0, 0])
            voltage_traces_to_superimpose_incl_deadzone_V.append(np_current_voltage_trace_V)
            if np.linalg.norm(np_neuron_offset_xyz_um) > 25:
                voltage_traces_to_superimpose_excl_deadzone_V.append(np_current_voltage_trace_V)

    np_voltage_trace_incl_deadzone_clean_V = np.sum(voltage_traces_to_superimpose_incl_deadzone_V, axis=0)
    np_voltage_trace_excl_deadzone_clean_V = np.sum(voltage_traces_to_superimpose_excl_deadzone_V, axis=0)
    np_voltage_trace_pink_noise_V = generate_pink_noise(
        peak_to_peak_magnitude=10e-6,
        timesteps=np_voltage_trace_incl_deadzone_clean_V.shape[0])
    np_voltage_trace_incl_deadzone_noisy_V = np_voltage_trace_incl_deadzone_clean_V + np_voltage_trace_pink_noise_V
    np_voltage_trace_excl_deadzone_noisy_V = np_voltage_trace_excl_deadzone_clean_V + np_voltage_trace_pink_noise_V

    fig, axes = plt.subplots(2, 1, sharex="all", sharey="all")
    axes[0].plot(np.arange(n_timesteps) * timestep_ms, np_voltage_trace_incl_deadzone_noisy_V, linewidth=1)
    axes[0].set_title(f"voltage trace from {n_neurons} neurons including deadzone")
    axes[1].plot(np.arange(n_timesteps) * timestep_ms, np_voltage_trace_excl_deadzone_noisy_V, linewidth=1)
    axes[1].set_title(f"voltage trace from {n_neurons} neurons excluding deadzone")
    axes[1].set_xlabel("time (ms)")
    [axes[i].set_ylabel("voltage (V)") for i in (0, 1)]
    [axes[i].yaxis.set_major_formatter(y_scalar_formatter) for i in (0, 1)]
    plt.tight_layout()
    plt.savefig(output_dir + f"voltage trace from {n_neurons} neurons.png")
    plt.close()

    # =========================================================================
    # simulate variable conductivity environment
    # =========================================================================

    # np_neuron_offset_xyz_um = np.array((0, 50, 0))

    # print("Building interpolator...")
    # K_matrix_interpolator_um_V = scipy.interpolate.LinearNDInterpolator(np_K_matrix_xyz_um, np_K_matrix_voltage_V)
    # print("Interpolator built!")
    #
    # voltage_traces_to_superimpose_part3_V = []
    # for np_current_big_xyz_um, np_current_big_trace_A in zip(np_currents_big_xyz_um, np_currents_big_trace_A):
    #     voltage_traces_to_superimpose_part3_V.append(
    #         K_matrix_interpolator_um_V(np_current_big_xyz_um + np_neuron_offset_xyz_um)[0] * np_current_big_trace_A)
    # np_voltage_trace_clean_part3_V = np.sum(voltage_traces_to_superimpose_part3_V, axis=0)

    # print("Starting interpolation")
    # interpolation_points_xyz_um = []
    # for np_current_big_xyz_um in np_currents_big_xyz_um:
    #     interpolation_points_xyz_um.append(np_current_big_xyz_um + np_neuron_offset_xyz_um)
    # K_matrix_interpolated_values_V = scipy.interpolate.griddata(
    #     np_K_matrix_xyz_um, np_K_matrix_voltage_V, interpolation_points_xyz_um)
    # voltage_traces_to_superimpose_part3_V = []
    # for np_current_big_trace_A, K_matrix_interpolated_value_V in zip(
    #         np_currents_big_trace_A, K_matrix_interpolated_values_V):
    #     voltage_traces_to_superimpose_part3_V.append(np_current_big_trace_A * K_matrix_interpolated_value_V)
    # np_voltage_trace_clean_part3_V = np.sum(voltage_traces_to_superimpose_part3_V, axis=0)
    #
    # fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
    # axes.plot(np_currents_big_time_ms, np_voltage_trace_clean_V, label="analytical, const. cond.")
    # axes.plot(np_currents_big_time_ms, np_voltage_trace_clean_part3_V, label="numerical, var. cond.")
    # axes.set_title(f"comparison of constant and variable conductivity")
    # axes.set_xlabel("time (ms)")
    # axes.set_ylabel("voltage (V)")
    # axes.yaxis.set_major_formatter(y_scalar_formatter)
    # axes.legend()
    # plt.tight_layout()
    # plt.savefig(output_dir + f"comparison of constant and variable conductivity.png")
    # plt.close()


if __name__ == "__main__":

    _input_path_currents_big = r"D:\Documents\Academics\BME517\bme_lab_4\data\currents_big.mat"
    _input_path_currents_small = r"D:\Documents\Academics\BME517\bme_lab_4\data\currents_small.mat"
    _input_path_K_matrix = r"D:\Documents\Academics\BME517\bme_lab_4\data\lab3_output_attempt_7.txt"
    _input_path_K_matrix_dbs = r"D:\Documents\Academics\BME517\bme_lab_4\data\lab3b_output_attempt_8.txt"
    _output_dir = r"D:\Documents\Academics\BME517\bme_lab_4_report\\"
    main(
        _input_path_currents_big, _input_path_currents_small,
        _input_path_K_matrix, _input_path_K_matrix_dbs,
        _output_dir)
