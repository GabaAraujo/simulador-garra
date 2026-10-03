"""Cena com cubo, banana e lata. O comando escolhe qual pegar.

A pose de cada objeto vem do simulador. A câmera entra só no script
que usa o Gemma.
"""

from pathlib import Path
import importlib

import numpy as np
import sapien
import torch
import yaml

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
# mani_skill.utils às vezes aponta para envs.utils; o importlib pega o pacote certo.
ycb_mod = importlib.import_module("mani_skill.utils.building.actors.ycb")
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

RAIZ = Path(__file__).resolve().parents[1]
CONFIG = RAIZ / "configs" / "objetos.yaml"


def carregar_catalogo():
    with CONFIG.open(encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)["objetos"]


def nome_do_comando(texto, catalogo):
    """Escolhe o objeto cujo apelido aparece no texto. O apelido mais longo ganha."""
    texto = texto.lower()
    escolhido = None
    tamanho = -1
    for nome, info in catalogo.items():
        for apelido in info["aliases"]:
            if apelido in texto and len(apelido) > tamanho:
                escolhido = nome
                tamanho = len(apelido)
    return escolhido


@register_env("PegaPorNome-v1", max_episode_steps=200)
class PegaPorNomeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.catalogo = carregar_catalogo()
        self.alvo = "cubo"
        self._z_apoio = {}
        self._z_inicial = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # Câmera alta, olhando a mesa: a imagem e a profundidade saem daqui.
        pose = sapien_utils.look_at(eye=[0.35, 0.0, 0.65], target=[-0.05, 0.0, 0.05])
        return [CameraConfig("base_camera", pose, 512, 512, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.objetos = {}
        for nome, info in self.catalogo.items():
            if info["tipo"] == "cubo":
                ator = actors.build_cube(
                    self.scene,
                    half_size=0.02,
                    color=info["cor"],
                    name=nome,
                    initial_pose=sapien.Pose(p=[info["xy"][0], info["xy"][1], 0.02]),
                )
            elif info["tipo"] == "cilindro":
                ator = actors.build_cylinder(
                    self.scene,
                    radius=info["raio"],
                    half_length=info["meio_comprimento"],
                    color=info["cor"],
                    name=nome,
                    initial_pose=sapien.Pose(p=[info["xy"][0], info["xy"][1], 0.0]),
                )
            else:
                builder = self._ycb(info["modelo"], info.get("escala", 1.0))
                builder.initial_pose = sapien.Pose(p=[info["xy"][0], info["xy"][1], 0.0])
                ator = builder.build(name=nome)
            self.objetos[nome] = ator

    def _ycb(self, modelo, escala):
        # A função oficial recarrega o JSON e apaga o dicionário. Por isso
        # a escala é aplicada no módulo, e a chave YCB impede um segundo load.
        ycb_mod._load_ycb_dataset()
        ycb_mod.YCB_DATASET["YCB"] = True
        meta = ycb_mod.YCB_DATASET["model_data"][modelo]
        escalas = list(meta.get("scales", [1.0]))
        meta["scales"] = [escalas[0] * escala]
        try:
            return ycb_mod.get_ycb_builder(self.scene, id=modelo)
        finally:
            meta["scales"] = escalas

    def _after_reconfigure(self, options: dict):
        # Altura para o objeto nascer apoiado na mesa, não enterrado nem flutuando.
        for nome, ator in self.objetos.items():
            malha = ator.get_first_collision_mesh()
            self._z_apoio[nome] = float(-malha.bounding_box.bounds[0, 2])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.alvo = options.get("alvo", self.alvo)
            for nome, ator in self.objetos.items():
                xy = self.catalogo[nome]["xy"]
                xyz = torch.tensor([[xy[0], xy[1], self._z_apoio[nome]]], dtype=torch.float32)
                ator.set_pose(Pose.create_from_pq(xyz))
                self._z_inicial[nome] = float(xyz[0, 2])

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp_pose.raw_pose)

    def evaluate(self):
        alvo = self.objetos[self.alvo]
        levantou = alvo.pose.p[:, 2] > self._z_inicial[self.alvo] + 0.04
        segurou = self.agent.is_grasping(alvo)
        parado = self.agent.is_static(0.2)
        return {
            "success": segurou & levantou & parado,
            "is_grasped": segurou,
            "lifted": levantou,
        }

    def compute_dense_reward(self, obs, action, info):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
