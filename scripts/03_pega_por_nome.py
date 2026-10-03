"""Você escreve 'pega a banana' e a garra usa a pose que o simulador conhece.

A visão ainda não entra. Assim dá para ver se o planejador escolheu
o objeto certo antes de colocar o Gemma no meio.
"""

import sys
from pathlib import Path

import gymnasium as gym

import cena_varios_objetos
from cena_varios_objetos import carregar_catalogo, nome_do_comando
from mani_skill.utils.wrappers.record import RecordEpisode
from pegar import pegar_ator

RAIZ = Path(__file__).resolve().parents[1]


def main():
    texto = " ".join(sys.argv[1:]).strip() or input("Comando: ")
    catalogo = carregar_catalogo()
    alvo = nome_do_comando(texto, catalogo)
    if alvo is None:
        nomes = ", ".join(catalogo)
        raise SystemExit(f"Não reconheci o objeto em '{texto}'. Use um destes: {nomes}.")

    print(f"Comando: {texto}")
    print(f"Objeto escolhido pela pose do simulador: {alvo}")
    env = gym.make(
        "PegaPorNome-v1",
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="gpu",
        num_envs=1,
    )
    destino = RAIZ / "videos"
    env = RecordEpisode(
        env,
        output_dir=str(destino),
        save_trajectory=False,
        save_video=True,
        save_on_reset=False,
        video_fps=20,
    )
    env.reset(seed=0, options={"alvo": alvo, "reconfigure": True})
    resultado = pegar_ator(env, env.unwrapped.objetos[alvo])
    sucesso = False
    if resultado != -1:
        sucesso = bool(resultado[-1]["success"].item())
    env.flush_video(name=f"pega_{alvo}")
    env.close()
    print("Sucesso:", sucesso)
    print("Vídeo:", destino / f"pega_{alvo}.mp4")


if __name__ == "__main__":
    # O import registra o ambiente no Gymnasium.
    del cena_varios_objetos
    main()
