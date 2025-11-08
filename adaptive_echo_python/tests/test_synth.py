import torch

from adaptive_echo_python.synth import Synth


class TestSynth:
    def setup_method(self):
        self.synth = Synth()
        self.sample_rate = 48000
        self.duration = 1.0
        self.time = torch.linspace(
            0, self.duration, int(self.sample_rate * self.duration)
        )

    def test_synth_initialization(self):
        assert isinstance(self.synth, torch.nn.Module)
        assert hasattr(self.synth, "env_vol_a")
        assert hasattr(self.synth, "env_vol_b")
        assert hasattr(self.synth, "env_mod")
        assert hasattr(self.synth, "osc_a")
        assert hasattr(self.synth, "osc_b")
        assert hasattr(self.synth, "env_fm")
        assert hasattr(self.synth, "fm_range_low")
        assert hasattr(self.synth, "fm_range_high")

    def test_synth_forward_basic(self):
        output = self.synth(self.time)
        assert isinstance(output, torch.Tensor)
        assert output.shape == self.time.shape
        assert torch.isfinite(output).all()

    def test_synth_output_range(self):
        output = self.synth(self.time)
        assert output.min() >= -2.0
        assert output.max() <= 2.0

    def test_synth_different_time_inputs(self):
        short_time = torch.linspace(0, 0.1, 100)
        long_time = torch.linspace(0, 2.0, 2000)

        short_output = self.synth(short_time)
        long_output = self.synth(long_time)

        assert short_output.shape == short_time.shape
        assert long_output.shape == long_time.shape
        assert torch.isfinite(short_output).all()
        assert torch.isfinite(long_output).all()

    def test_synth_batch_processing(self):
        batch_time = self.time.unsqueeze(0).repeat(3, 1)
        batch_output = self.synth(batch_time)

        assert batch_output.shape == batch_time.shape
        assert torch.isfinite(batch_output).all()

    def test_synth_output_consistency(self):
        output1 = self.synth(self.time)
        output2 = self.synth(self.time)

        assert torch.allclose(output1, output2)

    def test_synth_parameter_gradients(self):
        self.synth.zero_grad()
        output = self.synth(self.time)
        loss = output.sum()
        loss.backward()

        params_with_grad = 0
        for param in self.synth.parameters():
            if param.grad is not None:
                params_with_grad += 1

        assert params_with_grad > 0

    def test_synth_fm_range_parameters(self):
        self.synth.fm_range_low.data = torch.tensor(0.5)
        self.synth.fm_range_high.data = torch.tensor(0.8)

        output = self.synth(self.time)
        assert torch.isfinite(output).all()

        assert self.synth.fm_range_low.item() == 0.5
        assert abs(self.synth.fm_range_high.item() - 0.8) < 1e-6

    def test_synth_envelope_integration(self):
        with torch.no_grad():
            for env in [
                self.synth.env_vol_a,
                self.synth.env_vol_b,
                self.synth.env_mod,
                self.synth.env_fm,
            ]:
                env.length.data = torch.tensor(0.5)
                env.attack.data = torch.tensor(0.1)
                env.decay.data = torch.tensor(0.1)
                env.sustain.data = torch.tensor(0.7)
                env.release.data = torch.tensor(0.1)

        output = self.synth(self.time)
        assert torch.isfinite(output).all()

    def test_synth_oscillator_integration(self):
        with torch.no_grad():
            for osc in [self.synth.osc_a, self.synth.osc_b]:
                osc.low_freq.data = torch.tensor(0.3)
                osc.high_freq.data = torch.tensor(0.7)
                osc.low_warmth.data = torch.tensor(0.2)
                osc.high_warmth.data = torch.tensor(0.8)
                osc.low_harshness.data = torch.tensor(0.3)
                osc.high_harshness.data = torch.tensor(0.7)
                osc.low_amplitude.data = torch.tensor(0.4)
                osc.high_amplitude.data = torch.tensor(0.9)

        output = self.synth(self.time)
        assert torch.isfinite(output).all()

    def test_synth_extreme_parameters(self):
        with torch.no_grad():
            self.synth.fm_range_low.data = torch.tensor(1.0)
            self.synth.fm_range_high.data = torch.tensor(1.0)

            for env in [
                self.synth.env_vol_a,
                self.synth.env_vol_b,
                self.synth.env_mod,
                self.synth.env_fm,
            ]:
                env.length.data = torch.tensor(1.0)
                env.attack.data = torch.tensor(1.0)
                env.decay.data = torch.tensor(1.0)
                env.sustain.data = torch.tensor(1.0)
                env.release.data = torch.tensor(1.0)

        output = self.synth(self.time)
        assert torch.isfinite(output).all()

    def test_synth_zero_parameters(self):
        with torch.no_grad():
            self.synth.fm_range_low.data = torch.tensor(0.0)
            self.synth.fm_range_high.data = torch.tensor(0.0)

            for env in [
                self.synth.env_vol_a,
                self.synth.env_vol_b,
                self.synth.env_mod,
                self.synth.env_fm,
            ]:
                env.length.data = torch.tensor(0.0)
                env.attack.data = torch.tensor(0.0)
                env.decay.data = torch.tensor(0.0)
                env.sustain.data = torch.tensor(0.0)
                env.release.data = torch.tensor(0.0)

        output = self.synth(self.time)
        assert torch.isfinite(output).all()

    def test_synth_single_time_point(self):
        single_time = torch.tensor([0.5])
        output = self.synth(single_time)
        assert output.shape == (1,)
        assert torch.isfinite(output).all()

    def test_synth_empty_input(self):
        empty_time = torch.tensor([])
        output = self.synth(empty_time)
        assert output.shape == (0,)

    def test_synth_device_compatibility(self):
        if torch.cuda.is_available():
            synth_cuda = self.synth.cuda()
            time_cuda = self.time.cuda()
            output_cuda = synth_cuda(time_cuda)
            assert output_cuda.device.type == "cuda"
            assert torch.isfinite(output_cuda).all()
        if torch.backends.mps.is_available():
            synth_mps = self.synth.to(torch.device("mps"))
            time_mps = self.time.to(torch.device("mps"))
            output_mps = synth_mps(time_mps)
            assert output_mps.device.type == "mps"
            assert torch.isfinite(output_mps).all()
