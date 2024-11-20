from abc import ABC, abstractmethod
from random import random, shuffle, choice, sample

class MecAprendRef:
    """
    Classe principal que coordena o aprendizado por reforço.
    """
    def __init__(self, aprend_ref, acoes):
        """
        :param aprend_ref: Instância de um algoritmo de aprendizado (AprendRef).
        :param acoes: Lista de ações possíveis.
        """
        self.aprend_ref = aprend_ref
        self.acoes = acoes

    def aprender(self, s, a, r, sn, an=None):
        """
        Encaminha o aprendizado para a instância de AprendRef.
        """
        # Verifica se o algoritmo é SARSA (que precisa de `an`)
        if isinstance(self.aprend_ref, SARSA):
            self.aprend_ref.aprender(s, a, r, sn, an)
        else:
            self.aprend_ref.aprender(s, a, r, sn)

    def selecionar_acao(self, s):
        """
        Encaminha a seleção de ação para a instância de AprendRef.
        """
        return self.aprend_ref.sel_accao.selecionar_acao(s)

# Classe abstrata para Memória de Aprendizado
class MemoriaAprend(ABC):
    @abstractmethod
    def atualizar(self, s, a, q):
        """Atualiza o valor Q para o par estado-ação."""
        pass

    @abstractmethod
    def Q(self, s, a):
        """Retorna o valor Q para o par estado-ação."""
        pass

# Classe abstrata para Seleção de Ações
class SelAcao(ABC):
    def __init__(self, mem_aprend):
        self.mem_aprend = mem_aprend

    @abstractmethod
    def selecionar_acao(self, s):
        pass

    def max_acao(self, s, acoes):
        """
        Retorna a ação com o maior valor Q para um estado dado.
        :param s: Estado atual.
        :param acoes: Lista de ações disponíveis.
        :return: Ação com o maior valor Q.
        """
        shuffle(acoes)  # Embaralha as ações para desempate aleatório.
        return max(acoes, key=lambda a: self.mem_aprend.Q(s, a))

# Classe para Aprendizado por Reforço
class AprendRef(ABC):
    """
    Classe abstrata para mecanismos de aprendizado por reforço.
    """
    def __init__(self, mem_aprend, sel_accao, alfa, gama):
        """
        Inicializa os parâmetros comuns para aprendizado por reforço.
        :param mem_aprend: Instância de memória de aprendizado.
        :param sel_accao: Instância de seleção de ações.
        :param alfa: Taxa de aprendizado (valor entre 0 e 1).
        :param gama: Fator de desconto (valor entre 0 e 1).
        """
        self.mem_aprend = mem_aprend
        self.sel_accao = sel_accao
        self.alfa = alfa
        self.gama = gama

    @abstractmethod
    def aprender(self, s, a, r, sn, an=None):
        """
        Metodo abstrato para implementar o algoritmo de aprendizado.
        :param s: Estado atual.
        :param a: Ação tomada.
        :param r: Recompensa recebida.
        :param sn: Próximo estado.
        :param an: Próxima ação (opcional, usado em algoritmos como SARSA).
        """
        pass

class EGreedy(SelAcao):
    """
    Implementação da estratégia epsilon-greedy para seleção de ações.
    """
    def __init__(self, mem_aprend, acoes, epsilon):
        """
        Estratégia epsilon-greedy para seleção de ações.
        :param mem_aprend: Instância de memória de aprendizado (MemoriaAprend).
        :param acoes: Lista de ações disponíveis.
        :param epsilon: Probabilidade de exploração (valor entre 0 e 1).
        """
        super().__init__(mem_aprend)
        self.acoes = acoes
        self.epsilon = epsilon

    def aproveitar(self, s):
        """
        Seleciona a melhor ação com base no maior valor Q.
        :param s: Estado atual.
        :return: Ação com o maior valor Q.
        """
        return self.max_acao(s, self.acoes)

    def explorar(self):
        """
        Seleciona uma ação aleatória (exploração).
        :return: Ação selecionada aleatoriamente.
        """
        return choice(self.acoes)

    def selecionar_acao(self, s):
        """
        Seleciona uma ação com base na estratégia epsilon-greedy.
        :param s: Estado atual.
        :return: Ação selecionada.
        """
        if random() > self.epsilon:
            acao = self.aproveitar(s)  # Aproveitamento (escolhe a melhor ação).
        else:
            acao = self.explorar()  # Exploração (escolhe uma ação aleatória).
        return acao

class MemoriaEsparsa(MemoriaAprend):
    """
    Implementação de uma memória esparsa que herda de MemoriaAprend.
    """
    def __init__(self, valor_omissao=0.0):
        """
        Inicializa a memória esparsa.
        :param valor_omissao: Valor padrão retornado se o par (estado, ação) não estiver na memória.
        """
        self.valor_omissao = valor_omissao
        self.memoria = {}

    def Q(self, s, a):
        """
        Retorna o valor Q armazenado para o par (estado, ação).
        Se o par não estiver presente, retorna o valor de omissão.
        :param s: Estado.
        :param a: Ação.
        :return: Valor Q associado ao par (estado, ação) ou valor_omissao.
        """
        return self.memoria.get((s, a), self.valor_omissao)

    def atualizar(self, s, a, q):
        """
        Atualiza o valor Q para o par (estado, ação).
        :param s: Estado.
        :param a: Ação.
        :param q: Novo valor Q para o par (estado, ação).
        """
        self.memoria[(s, a)] = q

class SARSA(AprendRef):
    """
    Implementação do algoritmo SARSA (State-Action-Reward-State-Action).
    """
    def aprender(self, s, a, r, sn, an):
        """
        Atualiza os valores Q usando o algoritmo SARSA.
        :param s: Estado atual.
        :param a: Ação tomada no estado atual.
        :param r: Recompensa recebida pela transição.
        :param sn: Próximo estado.
        :param an: Próxima ação no próximo estado.
        """
        # Valor Q atual para o par (s, a)
        qsa = self.mem_aprend.Q(s, a)

        # Valor Q esperado para o próximo estado e próxima ação (SARSA)
        qsn_an = self.mem_aprend.Q(sn, an)

        # Atualiza o valor Q usando a fórmula do SARSA
        q = qsa + self.alfa * (r + self.gama * qsn_an - qsa)

        # Atualiza a memória com o novo valor Q
        self.mem_aprend.atualizar(s, a, q)
        
class QLearning(AprendRef):
    """
    Implementação do algoritmo Q-Learning.
    """
    def aprender(self, s, a, r, sn):
        """
        Atualiza os valores Q usando o algoritmo Q-Learning.
        :param s: Estado atual.
        :param a: Ação tomada no estado atual.
        :param r: Recompensa recebida pela transição.
        :param sn: Próximo estado.
        """
        # Seleciona a melhor ação no próximo estado (sn) usando max_acao
        an = self.sel_accao.max_acao(sn, self.sel_accao.acoes)

        # Obtém o valor Q atual para o par (s, a)
        qsa = self.mem_aprend.Q(s, a)

        # Obtém o valor Q para o próximo estado e a melhor ação (Q-Learning usa a melhor ação)
        qsn_an = self.mem_aprend.Q(sn, an)

        # Calcula o novo valor Q usando a fórmula do Q-Learning
        q = qsa + self.alfa * (r + self.gama * qsn_an - qsa)

        # Atualiza a memória com o novo valor Q
        self.mem_aprend.atualizar(s, a, q)

class DynaQ(QLearning):
    """
    Implementação do algoritmo Dyna-Q, que estende o Q-Learning com simulações.
    """
    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim):
        """
        Inicializa os parâmetros do Dyna-Q.
        :param mem_aprend: Instância de memória de aprendizado.
        :param sel_accao: Instância de seleção de ações.
        :param alfa: Taxa de aprendizado.
        :param gama: Fator de desconto.
        :param num_sim: Número de simulações a serem realizadas.
        """
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim
        self.modelo = ModeloTR()  # Modelo transitório para simulações

    def aprender(self, s, a, r, sn):
        """
        Atualiza os valores Q e o modelo transitório, e realiza simulações.
        :param s: Estado atual.
        :param a: Ação tomada.
        :param r: Recompensa recebida.
        :param sn: Próximo estado.
        """
        # Atualiza Q-Learning para a transição real
        super().aprender(s, a, r, sn)

        # Atualiza o modelo transitório com a transição real
        self.modelo.atualizar(s, a, r, sn)

        # Realiza simulações
        self.simular()

    def simular(self):
        """
        Realiza simulações baseadas no modelo transitório.
        """
        for _ in range(self.num_sim):
            # Amostra uma transição do modelo
            s, a, r, sn = self.modelo.amostrar()

            # Atualiza os valores Q usando a transição simulada
            super().aprender(s, a, r, sn)


class ModeloTR:
    """
    Modelo de Transição e Recompensa (ModeloTR).
    Armazena transições determinísticas para simulações no algoritmo Dyna-Q.
    """

    def __init__(self):
        """
        Inicializa o modelo com dois dicionários:
        - T: Mapeia (estado, ação) para o próximo estado (sn).
        - R: Mapeia (estado, ação) para a recompensa (r).
        """
        self.T = {}  # Transições determinísticas: (s, a) -> sn
        self.R = {}  # Recompensas determinísticas: (s, a) -> r

    def atualizar(self, s, a, r, sn):
        """
        Atualiza o modelo com uma transição.
        :param s: Estado atual.
        :param a: Ação tomada.
        :param r: Recompensa recebida.
        :param sn: Próximo estado.
        """
        self.T[(s, a)] = sn  # Atualiza o próximo estado para (s, a)
        self.R[(s, a)] = r  # Atualiza a recompensa para (s, a)

    def amostrar(self):
        """
        Retorna uma transição aleatória do modelo.
        :return: Um tuple (s, a, r, sn) representando uma transição.
        """
        if not self.T:
            raise ValueError("Nenhuma transição armazenada no modelo.")

        # Seleciona aleatoriamente um par (s, a) armazenado em T
        s, a = choice(list(self.T.keys()))

        # Recupera o próximo estado e a recompensa correspondentes
        sn = self.T[(s, a)]
        r = self.R[(s, a)]

        return s, a, r, sn

class QME(QLearning):
    """
    Implementação do algoritmo QME (Q-Learning com Memória de Experiência).
    """
    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim, dim_max):
        """
        Inicializa os parâmetros do QME.
        :param mem_aprend: Instância de memória de aprendizado.
        :param sel_accao: Instância de seleção de ações.
        :param alfa: Taxa de aprendizado.
        :param gama: Fator de desconto.
        :param num_sim: Número de simulações a serem realizadas.
        :param dim_max: Capacidade máxima da memória de experiência.
        """
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim  # Número de simulações
        self.memoria_experiencia = MemoriaExperiencia(dim_max)  # Memória de experiência

    def aprender(self, s, a, r, sn):
        """
        Atualiza os valores Q e armazena a transição na memória de experiência.
        :param s: Estado atual.
        :param a: Ação tomada.
        :param r: Recompensa recebida.
        :param sn: Próximo estado.
        """
        # Atualiza os valores Q usando Q-Learning
        super().aprender(s, a, r, sn)

        # Armazena a transição (experiência) na memória
        e = (s, a, r, sn)
        self.memoria_experiencia.atualizar(e)

        # Realiza simulações baseadas na memória de experiência
        self.simular()

    def simular(self):
        """
        Realiza simulações baseadas na memória de experiência.
        """
        # Amostra experiências da memória de experiência
        amostras = self.memoria_experiencia.amostrar(self.num_sim)

        # Realiza aprendizado para cada amostra
        for s, a, r, sn in amostras:
            super().aprender(s, a, r, sn)

class MemoriaExperiencia:
    """
    Classe para armazenar experiências (transições) em uma memória limitada.
    """
    def __init__(self, dim_max):
        """
        Inicializa a memória de experiência com capacidade máxima.
        :param dim_max: Capacidade máxima da memória.
        """
        self.dim_max = dim_max  # Capacidade máxima da memória
        self.memoria = []  # Lista para armazenar as transições

    def atualizar(self, e):
        """
        Armazena uma nova experiência na memória. Remove a mais antiga se a capacidade for excedida.
        :param e: Uma experiência representada como um tuple (s, a, r, sn).
        """
        if len(self.memoria) == self.dim_max:
            # Remove a experiência mais antiga (primeira na lista)
            self.memoria.pop(0)
        self.memoria.append(e)  # Adiciona a nova experiência ao final da lista

    def amostrar(self, n):
        """
        Retorna uma lista de amostras aleatórias da memória.
        :param n: Número de amostras a retornar.
        :return: Lista de experiências selecionadas aleatoriamente.
        """
        n_amostras = min(n, len(self.memoria))  # Limita o número de amostras ao tamanho atual da memória
        return sample(self.memoria, n_amostras)  # Retorna amostras aleatórias