"""Abre PickCube-v1 com um ambiente na GPU e grava a garra se mexendo.

Ações aleatórias não tentam pegar o cubo. O vídeo só confirma que a
simulação e a câmera Vulkan estão funcionando nesta máquina.
"""

from pathlib import Path

import gymnasium as gym

from mani_skill.utils.wrappers.record import RecordEpisode

RAIZ = Path(__file__).resolve().parents[1]
VIDEOS = RAIZ / "videos"
PASSOS = 40


def main():
    # Um único ambiente cabe na RTX 4060 Ti de 8 GB junto com a área de trabalho.
    env = gym.make(
        "PickCube-v1",
        obs_mode="none",
        render_mode="rgb_array",
        sim_backend="gpu",
        render_backend="gpu",
        num_envs=1,
    )
    env = RecordEpisode(
        env,
        output_dir=str(VIDEOS),
        save_trajectory=False,
        save_video=True,
        max_steps_per_video=PASSOS,
        video_fps=20,
    )
    print("Espaço de ação:", env.action_space)
    print("Modo de controle:", env.unwrapped.control_mode)
    print("Backend de simulação:", env.unwrapped.device)

    obs, info = env.reset(seed=0, options=dict(reconfigure=True))
    del obs, info
    for passo in range(PASSOS):
        acao = env.action_space.sample()
        obs, recompensa, terminou, truncou, info = env.step(acao)
        if passo % 10 == 0:
            valor = recompensa.detach().float().cpu().reshape(-1)[0].item()
            print(f"passo {passo} recompensa {valor:.3f}")
        acabou = terminou | truncou
        if hasattr(acabou, "any"):
            acabou = bool(acabou.any().item())
        if acabou:
            break
    env.close()
    videos = sorted(VIDEOS.glob("*.mp4"))
    print("Vídeos em", VIDEOS)
    for video in videos:
        print(" ", video.name)


if __name__ == "__main__":
    main()
