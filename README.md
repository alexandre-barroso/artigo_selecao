# Suplemento computacional e formal

`Selection.py` reúne verificações simbólicas, testes determinísticos e simulações. `SelectionStability.lean` e `SelectionCounts.lean` contêm 26 declarações formais.

## Execução

Com Python 3 disponível, execute na pasta dos arquivos:

```bash
pip install numpy==2.4.4 sympy==1.14.0 matplotlib
```
Para garantir o uso das versões que utilizei. E aí rodar...

```bash
Selection.py --output resultados_selecao
```

... para execução dos 120 cenários, 200 réplicas por cenário e semente específica. O gerador utiliza interfaces internas do NumPy via `ctypes`, com estrutura de 136 bytes.

## Saídas

A pasta `resultados_selecao/` recebe `tabela2.csv` (Tabela 2), `figura1.pdf` e `figura1.png` (Figura 1), `resultados.csv` (resultados numéricos, sem cabeçalho) e `resultados.npz` (parâmetros, amostras, riscos e diagnósticos). O programa não imprime relatório, mas o código de saída `0` indica sucesso; `1`, falha de execução/verificação; `2`, argumentos inválidos ou otimização ativada; `3`, NumPy/SymPy ausentes ou em versões incompatíveis.

## Auditoria Lean

Requer Lean/Lake 4.34.0 (e, consequentemente, mathlib4 por Git) disponíveis. Na mesma pasta:

```bash
git clone https://github.com/leanprover-community/mathlib4.git .mathlib
git -C .mathlib checkout 5ed2965256430c3649e86755f9576b54eca72435
(cd .mathlib && lake exe cache get)
```
Isso instala o Lean, caso você não tenha. E aí...

```bash
Selection.py --mathlib .mathlib --output resultados_selecao
```

Esse comando repete a execução e acrescenta a compilação e a auditoria dos axiomas. O resultado fica no campo `A` de `resultados.npz`. Sem `--mathlib`, esse campo fica vazio e nenhuma prova Lean é compilada. A formalização cobre identidades e limites algébricos auxiliares, não a totalidade dos argumentos probabilísticos, adaptativos ou minimax. 
