import torch
import torch.nn as nn

from adaptive_echo_python.envelope import EnvelopeUniform
from adaptive_echo_python.interpolation import exp_interp, linear_interp
from adaptive_echo_python.osc import OscillatorModulatedUniform


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
        min_fm = torch.tensor(0.005, dtype=time.dtype, device=time.device)
        max_fm = torch.tensor(0.5, dtype=time.dtype, device=time.device)
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
