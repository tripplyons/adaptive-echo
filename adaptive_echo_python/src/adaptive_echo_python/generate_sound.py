import soundfile as sf
import torch

from adaptive_echo_python.synth import synth_parallel
from adaptive_echo_python.train import get_device, model_path, num_samples, num_seconds
from adaptive_echo_python.two_encoders import TwoEncoders


def generate_sound():
    two_encoders: TwoEncoders = torch.load(model_path, weights_only=False)

    device = get_device()
    print("Generating sound")
    time = torch.linspace(0, num_seconds, num_samples, device=device)
    target_audio = torch.sin(2 * torch.pi * time * 440).unsqueeze(0)
    print(target_audio.shape, time.shape)
    # settings = two_encoders.reconstruct_settings(target_audio, time) # <- this is for gradient descent
    settings = two_encoders.reconstruct_settings_genetic(target_audio, time) # <- this is for genetic reconstruction
    with torch.inference_mode():
        eval_time = torch.linspace(0, num_seconds, num_seconds * 48000, device=device)
        eval_audio = synth_parallel(settings, eval_time)
        # save eval_audio to wav file
        sf.write("eval_audio.wav", eval_audio.cpu().numpy()[0], 48000)
    with torch.inference_mode():
        settings_predicted = two_encoders.predict_settings(target_audio)
        eval_time = torch.linspace(0, num_seconds, num_seconds * 48000, device=device)
        eval_audio = synth_parallel(settings_predicted, eval_time)
        sf.write("eval_audio_predicted.wav", eval_audio.cpu().numpy()[0], 48000)


if __name__ == "__main__":
    generate_sound()
