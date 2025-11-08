import torch

from adaptive_echo_python.two_encoders import TwoEncoders


class TestTwoEncoders:
    def setup_method(self):
        self.audio_input_size = 128
        self.settings_input_size = 16
        self.embedding_size = 64
        self.hidden_size = 256
        self.num_layers = 3
        self.batch_size = 4

        self.model = TwoEncoders(
            audio_input_size=self.audio_input_size,
            settings_input_size=self.settings_input_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

    def test_two_encoders_initialization(self):
        assert isinstance(self.model, torch.nn.Module)
        assert hasattr(self.model, "audio_encoder")
        assert hasattr(self.model, "settings_encoder")
        assert hasattr(self.model, "log_t")
        assert hasattr(self.model, "b")
        assert isinstance(self.model.log_t, torch.nn.Parameter)
        assert isinstance(self.model.b, torch.nn.Parameter)

    def test_two_encoders_forward_basic(self):
        audio_input = torch.randn(self.batch_size, self.audio_input_size)
        settings_input = torch.randn(self.batch_size, self.settings_input_size)

        audio_embedding, settings_embedding = self.model(audio_input, settings_input)

        assert isinstance(audio_embedding, torch.Tensor)
        assert isinstance(settings_embedding, torch.Tensor)
        assert audio_embedding.shape == (self.batch_size, self.embedding_size)
        assert settings_embedding.shape == (self.batch_size, self.embedding_size)
        assert torch.isfinite(audio_embedding).all()
        assert torch.isfinite(settings_embedding).all()

    def test_two_encoders_forward_single_batch(self):
        audio_input = torch.randn(1, self.audio_input_size)
        settings_input = torch.randn(1, self.settings_input_size)

        audio_embedding, settings_embedding = self.model(audio_input, settings_input)

        assert audio_embedding.shape == (1, self.embedding_size)
        assert settings_embedding.shape == (1, self.embedding_size)

    def test_two_encoders_forward_different_batch_sizes(self):
        for batch_size in [1, 2, 8, 16]:
            audio_input = torch.randn(batch_size, self.audio_input_size)
            settings_input = torch.randn(batch_size, self.settings_input_size)

            audio_embedding, settings_embedding = self.model(
                audio_input, settings_input
            )

            assert audio_embedding.shape == (batch_size, self.embedding_size)
            assert settings_embedding.shape == (batch_size, self.embedding_size)

    def test_two_encoders_loss_basic(self):
        audio_input = torch.randn(self.batch_size, self.audio_input_size)
        settings_input = torch.randn(self.batch_size, self.settings_input_size)

        loss = self.model.loss(audio_input, settings_input)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()
        assert loss.item() >= 0  # Loss should be non-negative

    def test_two_encoders_loss_different_batch_sizes(self):
        for batch_size in [1, 2, 8]:
            audio_input = torch.randn(batch_size, self.audio_input_size)
            settings_input = torch.randn(batch_size, self.settings_input_size)

            loss = self.model.loss(audio_input, settings_input)

            assert torch.isfinite(loss).all()
            assert loss.shape == ()

    def test_two_encoders_loss_gradients(self):
        self.model.zero_grad()
        audio_input = torch.randn(self.batch_size, self.audio_input_size)
        settings_input = torch.randn(self.batch_size, self.settings_input_size)

        loss = self.model.loss(audio_input, settings_input)
        loss.backward()

        params_with_grad = 0
        for param in self.model.parameters():
            if param.grad is not None:
                params_with_grad += 1
                assert torch.isfinite(param.grad).all()

        assert params_with_grad > 0

    def test_two_encoders_hyperparameters(self):
        assert self.model.log_t.item() == 3.0
        assert self.model.b.item() == -10.0

        # Test that hyperparameters can be modified
        with torch.no_grad():
            self.model.log_t.data = torch.tensor(2.0)
            self.model.b.data = torch.tensor(-5.0)

        assert abs(self.model.log_t.item() - 2.0) < 1e-6
        assert abs(self.model.b.item() - (-5.0)) < 1e-6

    def test_two_encoders_forward_consistency(self):
        audio_input = torch.randn(self.batch_size, self.audio_input_size)
        settings_input = torch.randn(self.batch_size, self.settings_input_size)

        self.model.eval()
        with torch.no_grad():
            audio_emb1, settings_emb1 = self.model(audio_input, settings_input)
            audio_emb2, settings_emb2 = self.model(audio_input, settings_input)

        assert torch.allclose(audio_emb1, audio_emb2)
        assert torch.allclose(settings_emb1, settings_emb2)

    def test_two_encoders_loss_consistency(self):
        audio_input = torch.randn(self.batch_size, self.audio_input_size)
        settings_input = torch.randn(self.batch_size, self.settings_input_size)

        self.model.eval()
        with torch.no_grad():
            loss1 = self.model.loss(audio_input, settings_input)
            loss2 = self.model.loss(audio_input, settings_input)

        assert torch.allclose(loss1, loss2)

    def test_two_encoders_embedding_normalization(self):
        audio_input = torch.randn(self.batch_size, self.audio_input_size)
        settings_input = torch.randn(self.batch_size, self.settings_input_size)

        audio_embedding, settings_embedding = self.model(audio_input, settings_input)

        # Check that embeddings are normalized in loss computation
        z_audio = audio_embedding / torch.norm(audio_embedding, dim=-1, keepdim=True)
        z_settings = settings_embedding / torch.norm(
            settings_embedding, dim=-1, keepdim=True
        )

        # Check that normalized embeddings have unit norm
        audio_norms = torch.norm(z_audio, dim=-1)
        settings_norms = torch.norm(z_settings, dim=-1)

        assert torch.allclose(audio_norms, torch.ones_like(audio_norms), atol=1e-5)
        assert torch.allclose(
            settings_norms, torch.ones_like(settings_norms), atol=1e-5
        )

    def test_two_encoders_device_compatibility(self):
        audio_input = torch.randn(self.batch_size, self.audio_input_size)
        settings_input = torch.randn(self.batch_size, self.settings_input_size)

        if torch.cuda.is_available():
            model_cuda = self.model.cuda()
            audio_cuda = audio_input.cuda()
            settings_cuda = settings_input.cuda()

            audio_emb, settings_emb = model_cuda(audio_cuda, settings_cuda)
            loss = model_cuda.loss(audio_cuda, settings_cuda)

            assert audio_emb.device.type == "cuda"
            assert settings_emb.device.type == "cuda"
            assert torch.isfinite(audio_emb).all()
            assert torch.isfinite(settings_emb).all()
            assert torch.isfinite(loss).all()

        if torch.backends.mps.is_available():
            model_mps = self.model.to(torch.device("mps"))
            audio_mps = audio_input.to(torch.device("mps"))
            settings_mps = settings_input.to(torch.device("mps"))

            audio_emb, settings_emb = model_mps(audio_mps, settings_mps)
            loss = model_mps.loss(audio_mps, settings_mps)

            assert audio_emb.device.type == "mps"
            assert settings_emb.device.type == "mps"
            assert torch.isfinite(audio_emb).all()
            assert torch.isfinite(settings_emb).all()
            assert torch.isfinite(loss).all()

    def test_two_encoders_different_configurations(self):
        configs = [
            {"embedding_size": 32, "hidden_size": 128, "num_layers": 2},
            {"embedding_size": 128, "hidden_size": 512, "num_layers": 4},
        ]

        for config in configs:
            model = TwoEncoders(
                audio_input_size=self.audio_input_size,
                settings_input_size=self.settings_input_size,
                embedding_size=config["embedding_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
            )

            audio_input = torch.randn(2, self.audio_input_size)
            settings_input = torch.randn(2, self.settings_input_size)

            audio_emb, settings_emb = model(audio_input, settings_input)
            loss = model.loss(audio_input, settings_input)

            assert audio_emb.shape == (2, config["embedding_size"])
            assert settings_emb.shape == (2, config["embedding_size"])
            assert torch.isfinite(loss).all()
