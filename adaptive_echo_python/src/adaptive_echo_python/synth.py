import torch
import torch.nn as nn

from adaptive_echo_python.envelope import EnvelopeUniform, env_uniform
from adaptive_echo_python.interpolation import exp_interp, linear_interp
from adaptive_echo_python.osc import OscillatorModulatedUniform, osc_uniform


class Synth(nn.Module):
    def __init__(self):
        super(Synth, self).__init__()
        self.env_vol_a = EnvelopeUniform()
        self.env_vol_b = EnvelopeUniform()
        self.env_mod = EnvelopeUniform()
        self.osc_a = OscillatorModulatedUniform()
        self.osc_b = OscillatorModulatedUniform()
        self.env_fm = EnvelopeUniform()
        self.fm_range_low = nn.Parameter(torch.tensor(0.0))
        self.fm_range_high = nn.Parameter(torch.tensor(0.0))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # Calculate envelopes
        env_vol_a = self.env_vol_a(time)
        env_vol_b = self.env_vol_b(time)
        env_mod = self.env_mod(time)
        env_fm = self.env_fm(time)

        # Interpolate oscillator settings based on modulation envelope
        # For osc_a: interpolate between osc_a (no modulation) and osc_a_mod (full modulation)
        # Calculate frequency modulation amount
        min_fm = torch.tensor(0.001, dtype=time.dtype, device=time.device)
        max_fm = torch.tensor(0.1, dtype=time.dtype, device=time.device)
        start_fm = exp_interp(min_fm, max_fm, self.fm_range_low)
        end_fm = exp_interp(min_fm, max_fm, self.fm_range_high)
        fm_amount = linear_interp(start_fm, end_fm, env_fm)

        osc_b_output = self.osc_b(time, env_mod)
        osc_a_output = self.osc_a(
            time, env_mod, modulation=osc_b_output, fm_amount=fm_amount
        )

        # Multiply by volume envelopes
        osc_a_output = osc_a_output * env_vol_a
        osc_b_output = osc_b_output * env_vol_b

        # Add them together
        return osc_a_output + osc_b_output

    @torch.jit.export
    def encode_settings(self) -> torch.Tensor:
        # convert all parameters to a single flattened tensor
        params = torch.cat(
            [
                self.env_vol_a.encode_settings(),
                self.env_vol_b.encode_settings(),
                self.env_mod.encode_settings(),
                self.osc_a.encode_settings(),
                self.osc_b.encode_settings(),
                self.env_fm.encode_settings(),
                self.fm_range_low.view((1,)),
                self.fm_range_high.view((1,)),
            ]
        )
        return params

    @torch.jit.export
    def decode_settings(self, settings_input: torch.Tensor):
        num_env_params = 5
        num_osc_params = 12

        index = 0
        self.env_vol_a.decode_settings(settings_input[index : index + num_env_params])
        index += num_env_params
        self.env_vol_b.decode_settings(settings_input[index : index + num_env_params])
        index += num_env_params
        self.env_mod.decode_settings(settings_input[index : index + num_env_params])
        index += num_env_params
        self.osc_a.decode_settings(settings_input[index : index + num_osc_params])
        index += num_osc_params
        self.osc_b.decode_settings(settings_input[index : index + num_osc_params])
        index += num_osc_params
        self.env_fm.decode_settings(settings_input[index : index + num_env_params])
        index += num_env_params
        self.fm_range_low.data.copy_(settings_input[index])
        index += 1
        self.fm_range_high.data.copy_(settings_input[index])
        index += 1

        if index != 46:
            raise ValueError(f"Expected 46 index, got {index}")
        if settings_input.shape[0] != 46:
            raise ValueError(f"Expected 46 settings, got {settings_input.shape[0]}")


def synth(settings: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    env_vol_a = env_uniform(
        times, settings[0], settings[1], settings[2], settings[3], settings[4]
    )
    env_vol_b = env_uniform(
        times, settings[5], settings[6], settings[7], settings[8], settings[9]
    )
    env_mod = env_uniform(
        times, settings[10], settings[11], settings[12], settings[13], settings[14]
    )
    env_fm = env_uniform(
        times, settings[39], settings[40], settings[41], settings[42], settings[43]
    )

    fm_range_low = settings[44]
    fm_range_high = settings[45]
    fm_amount = linear_interp(fm_range_low, fm_range_high, env_fm[None])[0]

    osc_b_low_settings = settings[27:39:2]
    osc_b_high_settings = settings[28:39:2]
    osc_b_settings = linear_interp(
        osc_b_low_settings[..., None], osc_b_high_settings[..., None], env_mod
    )[..., 0]
    osc_b = osc_uniform(
        times,
        osc_b_settings[0],
        osc_b_settings[1],
        osc_b_settings[2],
        osc_b_settings[3],
        osc_b_settings[4],
        osc_b_settings[5],
    )

    osc_a_low_settings = settings[15:27:2]
    osc_a_high_settings = settings[16:27:2]
    osc_a_settings = linear_interp(
        osc_a_low_settings[..., None], osc_a_high_settings[..., None], env_mod
    )[..., 0]
    osc_a = osc_uniform(
        times,
        osc_a_settings[0],
        osc_a_settings[1],
        osc_a_settings[2],
        osc_a_settings[3],
        osc_a_settings[4],
        osc_a_settings[5],
        modulation=osc_b,
        fm_amount=fm_amount,
    )

    return osc_a * env_vol_a + osc_b * env_vol_b
