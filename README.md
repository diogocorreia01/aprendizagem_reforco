# Modelo de Aprendizagem por Reforço (Problema do Labirinto)

## Índice

- [Descrição do Projeto](#descrição-do-projeto)
- [Como Usar](#como-usar)
- [Funcionamento do Modelo](#funcionamento-do-modelo)
- [Como Usar](#como-usar)
- [Possíveis Melhorias](#possíveis-melhorias-na-implementação-do-problema-labirinto)

## Descrição do Projeto

Este projeto implementa um modelo de Aprendizagem por Reforço para resolver um problema de navegação num labirinto. O agente aprende, através de tentativas e erros, a encontrar o caminho ideal a partir de uma entrada (E) até uma saída (S), evitando paredes e penalidades causadas por movimentos inválidos. O algoritmo utilizado combina o Q-Learning com a estratégia de exploração Epsilon-Greedy.

## Tecnologias Utilizadas

- Python 3.12

## Funcionamento do Modelo

### 1. Ambiente

O ambiente é um labirinto representado como uma matriz 2D: 

- **1** representa uma parede (movimento bloqueado).
- **0** representa um caminho livre.
- **'E'** representa a entrada (ponto inicial do agente).
- **'S'** representa a saída (objetivo do agente).

### 2. Agente

O agente aprende ao interagir com o ambiente:

- **Ações:** O agente pode mover-se em quatro direções: cima, baixo, esquerda, e direita.
- **Recompensas**:
  - Quando o agente alcança a saída.
  - -1: Por cada movimento válido.
  - -10: Por bater em uma parede ou tentar um movimento inválido.

### 3. Aprendizagem

O agente utiliza **Q-Learning**:
- **Exploração**: O agente utiliza uma estratégia Epsilon-Greedy para equilibrar exploração (testar novas ações) e exploração (usar o conhecimento atual para maximizar as recompensas).

## Como Usar

### 1. Clonar o repositório:

   ```bash
   git clone https://github.com/diogocorreia01/aprendizagem_reforco.git
   ```

### 2. Estrutura do Labirinto

O labirinto pode ser configurado manualmente no código como uma matriz. Por defeito já está definido um labirinto 10x10:

   ```bash
    # Configuração do labirinto
    labirinto = Labirinto([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'E', 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 'S', 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
   ```
### 3. Executar o Código

```bash
    python aplicacao_problema.py
```

### 4. Visualização

Durante a execução, o labirinto é desenhado no terminal com destaque para:

- **Agente:** Posição atual do agente (vermelho).
- **Caminho Percorrido:** Caminho já percorrido pelo agente (amarelo).
- **Paredes:** Representadas como # (branco).
- **Entrada e Saída:** Destacadas em verde e azul, respetivamente.

Exemplo do caminho ótimo desenhado no terminal:
```bash
    === Caminho Ótimo ===
     #  #  #  #  #  #  #  #  #  # 
     #  O  O  O  #  .  .  .  .  # 
     #  .  #  O  #  .  #  #  .  # 
     #  .  #  O  O  O  #  .  .  # 
     #  .  #  #  #  O  #  .  #  # 
     #  .  .  .  #  O  O  O  O  # 
     #  #  #  .  #  #  #  #  O  # 
     #  .  .  .  .  .  .  #  O  # 
     #  .  #  #  #  #  .  #  O  # 
     #  #  #  #  #  #  #  #  #  # 
    
    
    Caminho ótimo: [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (8, 8)]
    Custo total: -12
```

## Possíveis Melhorias na Implementação do Problema (Labirinto)

**- Visualização Gráfica:** Usar bibliotecas como o pygame para melhorar a interface gráfica.
**- Geração do Labirinto:** Usar bibliotecas para gerar labirintos aleatórios.
**- Treinar o modelo com escalas maiores:** Usar labirintos de maior dimensão (ex. 50x50 ou 100x100).
