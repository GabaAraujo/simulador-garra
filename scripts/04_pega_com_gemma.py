"""O Gemma entende o comando. O Florence marca o objeto. O planejador move.

Cada execução grava um arquivo em logs/ com a mensagem enviada, o tipo
de dado que voltou e como esse dado foi interpretado.
"""

import gc
import os
import sys
from datetime import datetime
from pathlib import Path

# Sem gcc, o PyTorch 2.14 tenta compilar um kernel Triton e a geração quebra.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

import cena_varios_objetos
from camera_3d import nuvem_mundo, ponto_da_caixa
from cena_varios_objetos import carregar_catalogo, nome_do_comando
from mani_skill.utils.wrappers.record import RecordEpisode
from pegar import pegar_ponto

RAIZ = Path(__file__).resolve().parents[1]
LOGS = RAIZ / "logs"
MODELO_GEMMA = "google/gemma-3-4b-it"
MODELO_CAIXA = "florence-community/Florence-2-base"
FRASE = {"cubo": "red cube", "banana": "banana", "lata": "blue can"}


class Diario:
    """Texto corrido, em português, de uma execução."""

    def __init__(self, comando):
        LOGS.mkdir(parents=True, exist_ok=True)
        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "_".join(comando.lower().split()) or "comando"
        self.caminho = LOGS / f"{carimbo}_{slug}.log"
        self.linhas = [
            f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Comando enviado: {comando}",
            "",
        ]

    def etapa(self, titulo):
        self.linhas.append(f"## {titulo}")

    def linha(self, texto):
        self.linhas.append(texto)

    def salvar(self):
        self.caminho.write_text("\n".join(self.linhas) + "\n", encoding="utf-8")
        print("Log:", self.caminho)


def escolher_com_gemma(imagem, comando, catalogo, diario):
    """O Gemma olha a foto e a frase e devolve só o nome do objeto."""
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    nomes = ", ".join(catalogo)
    pedido = (
        "Na mesa há um cubo vermelho, uma banana amarela e uma lata azul. "
        f"O pedido é: {comando}. "
        f"Responda com uma única palavra, entre estas: {nomes}."
    )
    diario.etapa("1. Mensagem enviada ao Gemma")
    diario.linha("Papel: entender a frase e escolher o nome do objeto.")
    diario.linha("Entrada imagem: foto RGB da câmera, " f"{imagem.width} x {imagem.height} pixels, 3 canais de cor.")
    diario.linha("Entrada texto:")
    diario.linha(pedido)
    diario.linha("Saída esperada: uma palavra entre cubo, banana e lata.")
    mensagens = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": imagem},
                {"type": "text", "text": pedido},
            ],
        }
    ]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    print("Carregando Gemma 3 4B em 4 bits para escolher o objeto.")
    modelo = Gemma3ForConditionalGeneration.from_pretrained(
        MODELO_GEMMA,
        quantization_config=quant,
        device_map="auto",
    )
    processador = AutoProcessor.from_pretrained(MODELO_GEMMA)
    entradas = processador.apply_chat_template(
        mensagens,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    entradas = {chave: valor.to(modelo.device) if hasattr(valor, "to") else valor for chave, valor in entradas.items()}
    with torch.inference_mode():
        gerados = modelo.generate(**entradas, max_new_tokens=20)
    texto = processador.batch_decode(gerados, skip_special_tokens=True)[0]
    del modelo
    gc.collect()
    torch.cuda.empty_cache()
    resposta = texto.split("model")[-1].strip()
    escolha = nome_do_comando(resposta, catalogo)
    diario.etapa("2. O que o Gemma devolveu")
    diario.linha("Tipo do dado: texto.")
    diario.linha("Texto completo:")
    diario.linha(texto.strip())
    diario.linha(f"Trecho usado na interpretação (depois de 'model'): {resposta}")
    diario.linha(
        "Interpretação: procura no trecho o apelido mais longo do catálogo "
        "(cubo, banana, lata) e fica com esse nome."
    )
    diario.linha(f"Nome escolhido: {escolha}")
    print("Resposta do Gemma:", texto)
    print(f"Gemma escolheu: {escolha}")
    if escolha is None:
        diario.salvar()
        raise RuntimeError(f"O Gemma não escolheu um objeto conhecido. Texto: {texto}")
    return escolha


def caixa_do_florence(imagem, alvo, diario):
    """O Florence marca na foto o objeto que o Gemma nomeou."""
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    tarefa = "<OPEN_VOCABULARY_DETECTION>"
    frase = FRASE[alvo]
    diario.etapa("3. Mensagem enviada ao Florence")
    diario.linha("Papel: achar o objeto na foto e devolver uma caixa.")
    diario.linha(f"Entrada imagem: a mesma foto, {imagem.width} x {imagem.height} pixels.")
    diario.linha(f"Entrada texto: {tarefa}{frase}")
    diario.linha("Saída esperada: texto com marcas <loc_...> que viram cantos da caixa, em pixels.")
    print(f"Carregando Florence-2 para localizar: {frase}")
    modelo = Florence2ForConditionalGeneration.from_pretrained(MODELO_CAIXA, dtype=torch.float16)
    modelo.to("cuda")
    processador = AutoProcessor.from_pretrained(MODELO_CAIXA)
    entradas = processador(text=f"{tarefa}{frase}", images=imagem, return_tensors="pt")
    entradas = {
        chave: valor.to("cuda").to(torch.float16) if valor.dtype.is_floating_point else valor.to("cuda")
        for chave, valor in entradas.items()
    }
    with torch.inference_mode():
        gerados = modelo.generate(**entradas, max_new_tokens=128)
    texto = processador.batch_decode(gerados, skip_special_tokens=False)[0]
    saida = processador.post_process_generation(text=texto, task=tarefa, image_size=imagem.size)
    caixas = saida[tarefa].get("bboxes") or []
    del modelo
    gc.collect()
    torch.cuda.empty_cache()
    diario.etapa("4. O que o Florence devolveu")
    diario.linha("Tipo do dado: texto com localizações, convertido em lista de caixas.")
    diario.linha(f"Texto bruto: {texto.strip()}")
    if caixas:
        diario.linha(f"Caixas encontradas ({len(caixas)}): {[[round(v, 1) for v in c] for c in caixas]}")
    print("Resposta do Florence:", texto)
    if not caixas:
        diario.salvar()
        raise RuntimeError(f"O Florence não marcou '{frase}'. Texto: {texto}")
    caixa = max(caixas, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    diario.linha(
        "Interpretação: cada caixa é [x1, y1, x2, y2], o canto superior "
        "esquerdo e o inferior direito, em pixels. Fica a maior."
    )
    diario.linha(f"Caixa usada: {[round(v, 1) for v in caixa]}")
    return caixa


def main():
    texto = " ".join(sys.argv[1:]).strip() or input("Comando: ")
    diario = Diario(texto)
    catalogo = carregar_catalogo()
    env = gym.make(
        "PegaPorNome-v1",
        obs_mode="rgb+depth+position",
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
    obs, _ = env.reset(seed=0, options={"reconfigure": True})
    camera = obs["sensor_data"]["base_camera"]
    rgb = camera["rgb"][0].detach().cpu().numpy()
    nuvem = nuvem_mundo(camera["position"], obs["sensor_param"]["base_camera"]["cam2world_gl"])
    imagem = Image.fromarray(rgb)
    diario.etapa("0. Câmera")
    diario.linha(f"Tipo da cor: matriz de inteiros, forma {rgb.shape} (altura, largura, canais).")
    diario.linha("Tipo da profundidade: posição 3D de cada pixel, em milímetros, no sistema da câmera.")
    diario.linha("A pose interna dos objetos não é lida para mover a garra.")

    escolha = escolher_com_gemma(imagem, texto, catalogo, diario)
    env.unwrapped.alvo = escolha
    caixa = caixa_do_florence(imagem, escolha, diario)
    marcado = imagem.copy()
    ImageDraw.Draw(marcado).rectangle([int(v) for v in caixa], outline=(255, 0, 0), width=3)
    foto = destino / f"teste_{escolha}.png"
    marcado.save(foto)
    ponto, fechamento = ponto_da_caixa(nuvem, caixa)
    diario.etapa("5. Da caixa ao ponto 3D")
    diario.linha("Tipo de entrada: a caixa em pixels e o mapa de profundidade.")
    diario.linha("Interpretação: os pixels de dentro da caixa que estão acima da mesa viram um ponto em metros.")
    diario.linha(f"Ponto (x, y, z) em metros: {np.round(ponto, 3).tolist()}")
    diario.linha(f"Direção em que os dedos fecham (x, y, z): {np.round(fechamento, 3).tolist()}")
    print(f"Ponto 3D vindo da câmera: {np.round(ponto, 3).tolist()}")
    resultado = pegar_ponto(env, ponto, fechamento)
    sucesso = False
    if resultado != -1:
        sucesso = bool(resultado[-1]["success"].item())
    env.flush_video(name=f"teste_{escolha}")
    env.close()
    diario.etapa("6. Planejador")
    diario.linha("Tipo de entrada: o ponto 3D e a direção dos dedos.")
    diario.linha("Interpretação: ir até o ponto, fechar a garra e levantar.")
    diario.linha(f"Sucesso: {sucesso}")
    diario.linha(f"Foto com a caixa: {foto}")
    diario.linha(f"Vídeo: {destino / f'teste_{escolha}.mp4'}")
    diario.salvar()
    print("Sucesso:", sucesso)
    print("Vídeo:", destino / f"teste_{escolha}.mp4")


if __name__ == "__main__":
    del cena_varios_objetos
    main()
