# Simulador Garra

Simulação de uma garra Panda, no ManiSkill, que obedece a um comando em português e pega um objeto na mesa.

O teste separa três tarefas:

1. O Gemma 3 olha a foto e a frase e escolhe o nome: cubo, banana ou lata.
2. O Florence-2 marca esse objeto na imagem.
3. A profundidade da câmera vira um ponto 3D, e o planejador fecha a garra nesse ponto.

A pose interna do simulador não é usada para mirar.

## Como rodar

O ambiente usa Python 3.11. O NumPy precisa ficar abaixo da versão 2, porque o mplib 0.1.1 encerra o processo com as versões mais novas.

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

O Gemma 3 é um modelo com acesso controlado. Entre na conta do Hugging Face no terminal antes de rodar:

```bash
hf auth login
```

Depois:

```bash
export PATH="$PWD/.bin:$PATH"
export GIT_PYTHON_REFRESH=quiet
export TORCH_DISABLE_NATIVE_JIT=1
.venv/bin/python scripts/04_pega_com_gemma.py "pega a banana"
```

Cada execução grava um arquivo em `logs/` com a mensagem enviada, o tipo de dado devolvido e como ele foi interpretado. O vídeo fica em `videos/`.
