"""Leva a garra até um objeto e o levanta, com o planejador do ManiSkill.

Na etapa da pose, o centro vem da malha do simulador. Na etapa da
câmera, o centro é o ponto 3D medido na imagem.
"""

import numpy as np
import sapien

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def _fechar_e_levantar(env, planner, grasp_pose):
    # Chega um pouco acima, desce, fecha e sobe. O eixo local z aponta
    # para baixo, então um deslocamento negativo sobe no mundo.
    alcance = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(alcance)
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    return planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0, 0, -0.12]))


def pegar_ator(env, ator):
    base = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    caixa = get_actor_obb(ator)
    aproximacao = np.array([0.0, 0.0, -1.0])
    fechamento = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    info = compute_grasp_info_by_obb(
        caixa,
        approaching=aproximacao,
        target_closing=fechamento,
        depth=0.025,
    )
    # O centro da caixa envolve o objeto. A origem do modelo da lata
    # fica na base, e mirar nela faz a garra bater na mesa.
    pose = base.agent.build_grasp_pose(info["approaching"], info["closing"], info["center"])
    resultado = _fechar_e_levantar(env, planner, pose)
    planner.close()
    return resultado


def pegar_ponto(env, ponto, fechamento=None):
    """Ponto em metros no mundo, vindo da câmera, não da pose do objeto."""
    base = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    aproximacao = np.array([0.0, 0.0, -1.0])
    if fechamento is None:
        fechamento = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        fechamento = fechamento - aproximacao * np.dot(aproximacao, fechamento)
    fechamento = np.asarray(fechamento, dtype=np.float64)
    fechamento = fechamento / np.linalg.norm(fechamento)
    pose = base.agent.build_grasp_pose(aproximacao, fechamento, np.asarray(ponto, dtype=np.float64))
    resultado = _fechar_e_levantar(env, planner, pose)
    planner.close()
    return resultado
