#!/bin/sh
# O exemplo oficial grava a trajetória com GitPython. Sem um executável
# chamado git, esse import derruba o processo antes da garra se mexer.
# O arquivo .bin/git só responde a versão; este projeto não é um repositório.
raiz=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
export PATH="$raiz/.bin:$PATH"
export GIT_PYTHON_REFRESH=quiet
cd "$raiz"
exec .venv/bin/python -m mani_skill.examples.motionplanning.panda.run \
  -e PickCube-v1 --save-video -n 1 --only-count-success --record-dir demos -b cpu
