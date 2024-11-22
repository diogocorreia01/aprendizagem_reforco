from aprendizagem_reforco import MemoriaEsparsa, EGreedy, QLearning, MecAprendRef
from colorama import Fore, Style

class Labirinto:
    """
    Classe que representa o labirinto
    """

    MOVIMENTOS = {
        'cima': (-1, 0),
        'baixo': (1, 0),
        'esquerda': (0, -1),
        'direita': (0, 1),
    }

    def __init__(self, matriz):
        """
        Inicialização do labirinto com a matriz fornecida
        """
        self.matriz = matriz
        self.pos_inicial = self.encontrar_posicao('E')
        self.pos_final = self.encontrar_posicao('S')
        self.estado_atual = self.pos_inicial

    def encontrar_posicao(self, simbolo):
        """
        Metodo que encontra a posição de um símbolo específico na matriz
        """
        for i, linha in enumerate(self.matriz):
            for j, celula in enumerate(linha):
                if celula == simbolo:
                    return (i, j)

    def reset(self):
        """
        Metodo que faz reset no estado atual para a posição inicial
        """
        self.estado_atual = self.pos_inicial
        return self.estado_atual

    def realizar_acao(self, acao):
        """
        Metodo que realiza uma ação no labirinto e retorna o novo estado e a recompensa (Utiliza as constantes MOVIMENTO defenidas no inicio da classe)
        """
        movimento = self.MOVIMENTOS.get(acao)

        nova_posicao = (
            self.estado_atual[0] + movimento[0],
            self.estado_atual[1] + movimento[1],
        )

        # Verifica se o movimento é válido
        if (0 <= nova_posicao[0] < len(self.matriz) and
            0 <= nova_posicao[1] < len(self.matriz[0]) and
            self.matriz[nova_posicao[0]][nova_posicao[1]] != 1):
            self.estado_atual = nova_posicao
            if self.estado_atual == self.pos_final:
                return self.estado_atual, 1  # Recompensa ao chegar na saída
            return self.estado_atual, -1  # Recompensa por movimento
        return self.estado_atual, -10  # Recompensa negativa se bater na parede


def desenhar_labirinto(matriz, pos_agente, caminho_percorrido, caminho_otimo=None):
    """
    Metodo que desenha o labirinto
    """
    for i, linha in enumerate(matriz):
        linha_str = ''
        for j, celula in enumerate(linha):
            if (i, j) == pos_agente:
                linha_str += Fore.RED + ' A ' + Style.RESET_ALL  # Agente (vermelho)
            elif caminho_otimo and (i, j) in caminho_otimo:
                linha_str += Fore.CYAN + ' O ' + Style.RESET_ALL  # Caminho ótimo (ciano)
            elif (i, j) in caminho_percorrido:
                linha_str += Fore.YELLOW + ' * ' + Style.RESET_ALL  # Caminho percorrido (amarelo)
            elif celula == 1:
                linha_str += Fore.WHITE + ' # ' + Style.RESET_ALL  # Paredes (branco)
            elif celula == 'E':
                linha_str += Fore.GREEN + ' E ' + Style.RESET_ALL  # Entrada (verde)
            elif celula == 'S':
                linha_str += Fore.BLUE + ' S ' + Style.RESET_ALL  # Saída (azul)
            else:
                linha_str += ' . '  # Caminhos livres
        print(linha_str)
    print('\n')


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


# Configuração da aprendizagem por reforço
memoria = MemoriaEsparsa()
acoes = list(Labirinto.MOVIMENTOS.keys())
estrategia = EGreedy(memoria, acoes, epsilon=0.1)
qlearning = QLearning(memoria, estrategia, alfa=0.1, gama=0.9)
agente = MecAprendRef(qlearning, acoes)


# Treino
num_episodios = 200

for episodio in range(num_episodios):
    estado_atual = labirinto.reset()
    custo_total = 0
    caminho_percorrido = []

    print(f"\nEpisódio {episodio + 1}")
    desenhar_labirinto(labirinto.matriz, estado_atual, caminho_percorrido)

    while estado_atual != labirinto.pos_final:
        acao = agente.selecionar_acao(estado_atual)
        novo_estado, recompensa = labirinto.realizar_acao(acao)
        agente.aprender(estado_atual, acao, recompensa, novo_estado)

        # Atualiza o estado, custo e caminho percorrido
        estado_atual = novo_estado
        if estado_atual not in caminho_percorrido:  # Evita duplicações
            caminho_percorrido.append(estado_atual)
        custo_total += recompensa

        # Atualiza o desenho do labirinto
        desenhar_labirinto(labirinto.matriz, estado_atual, caminho_percorrido)

    print(f"Episódio {episodio + 1}: Custo total = {custo_total}")


# Caminho ótimo obtido
print("\n=== Caminho Ótimo ===")
estado_atual = labirinto.reset()
caminho_otimo = [estado_atual]
custo_total = 0

while estado_atual != labirinto.pos_final:
    acao = estrategia.aproveitar(estado_atual)
    estado_atual, recompensa = labirinto.realizar_acao(acao)
    caminho_otimo.append(estado_atual)
    custo_total += recompensa

# Desenha o labirinto com o caminho ótimo
desenhar_labirinto(labirinto.matriz, None, [], caminho_otimo)
print(f"Caminho ótimo: {caminho_otimo}")
print(f"Custo total: {custo_total}")
