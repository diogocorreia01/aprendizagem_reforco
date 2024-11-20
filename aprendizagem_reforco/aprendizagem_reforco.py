from abc import ABC, abstractmethod
from random import random, shuffle, choice, sample


class MecAprendRef:
    """
    Classe principal que vai coordenar a aprendizagem por reforço
    """
    def __init__(self, aprend_ref, acoes):
        """
        Inicialização do mecanismo de aprendizagem por reforço
        """

        self.aprend_ref = aprend_ref
        self.acoes = acoes

    def aprender(self, s, a, r, sn, an=None):
        """
        Executa a aprendizagem no ambiente

        Encaminha os parametros de aprendizagem para o algoritmo de aprendizagem associado,
        Se usarmos o algoritmo SARSA, ele vai precisar da variavel 'an' que é a próxima ação
        no próximo estado, se não usarmos o SARSA passamos apenas os parametros padrão

        Caso esta validação seja removida o algoritmo Q-Learning não vai funcionar como esperado,
        porque ao passar o valor 'an' (mesmo sendo None) vai causar calculos errados.
        """

        # Verifica se o algoritmo é o SARSA
        if isinstance(self.aprend_ref, SARSA):
            self.aprend_ref.aprender(s, a, r, sn, an)
        else:
            self.aprend_ref.aprender(s, a, r, sn)

    def selecionar_acao(self, s):
        """
        Metodo que seleciona a melhor ação para um determinado estado.
        """

        return self.aprend_ref.sel_accao.selecionar_acao(s)


class MemoriaAprend(ABC):
    """
    Classe abstrata que define a interface para implementar a memoria de aprendizagem

    Esta memoria é onde os valores Q (estado-ação) são guardados e atualizados
    durante o processo de aprendizagem por reforço.
    """

    @abstractmethod
    def atualizar(self, s, a, q):
        """
        Atualiza o valor Q associado a um par estado-ação.

        Este metodo vai ser implementado por subclasses para definir como
        o valor Q será guardado.
        """

        pass

    @abstractmethod
    def Q(self, s, a):
        """
        Retorna o valor Q associado a um estado-ação.

        Este metodo vai ser implementado por subclasses para definir como
        o valor Q será recuperado da memória.
        """

        pass


class SelAcao(ABC):
    """
    Classe abstrata que define a interface para as estratégias de seleção de ações.

    A seleção de ações vai definir como o agente escolhe as ações em diferentes estados com base
    nos valores Q guardados.
    """

    def __init__(self, mem_aprend):
        """
        Inicializa a estratégia de seleção de ações.
        """

        self.mem_aprend = mem_aprend

    @abstractmethod
    def selecionar_acao(self, s):
        """
        Seleciona uma ação para um estado dado.

        Este metodo abstrato é implementado pelas subclasses
        que definem estratégias específicas de seleção de ações.
        """

        pass

    def max_acao(self, s, acoes):
        """
        Retorna a ação com o maior valor Q para um estado dado.

        Este metodo escolhe a ação que maximiza o valor Q no estado atual.
        Caso duas ações tenham o mesmo valor Q, as ações são baralhadas para garantir uma escolha
        aleatória entre as melhores.
        """

        shuffle(acoes)
        return max(acoes, key=lambda a: self.mem_aprend.Q(s, a))


class AprendRef(ABC):
    """
    Classe abstrata para os mecanismos de aprendizagem por reforço

    Esta classe serve como base para implementar os diferentes algoritmos de aprendizagem por reforço,
    o SARSA e Q-Learning. Ela define os parametros comuns necessários para a aprendizagem
    e um metodo abstrato que vai ser implementado pelas subclasses.
    """

    def __init__(self, mem_aprend, sel_accao, alfa, gama):
        """
        Inicializa os parametros comuns para a aprendizagem por reforço
        """

        self.mem_aprend = mem_aprend
        self.sel_accao = sel_accao
        self.alfa = alfa
        self.gama = gama

    @abstractmethod
    def aprender(self, s, a, r, sn, an=None):
        """
        Metodo abstrato para implementar o algoritmo de aprendizagem

        Este metodo é implementado pelas subclasses específicas (SARSA e Q-Learning),
        que definem como os valores Q são atualizados.
        """

        pass


class EGreedy(SelAcao):
    """
    Classe da estratégia epsilon-greedy para a seleção de ações
    """

    def __init__(self, mem_aprend, acoes, epsilon):
        """
        Inicialização a estratégia epsilon-greedy
        """

        super().__init__(mem_aprend)
        self.acoes = acoes
        self.epsilon = epsilon

    def aproveitar(self, s):
        """
        Seleciona a melhor ação com base no maior valor Q.

        Este metodo implementa o aproveitamento, retornando a ação que maximiza
        o valor Q para o seu estado atual
        """

        return self.max_acao(s, self.acoes)

    def explorar(self):
        """
        Seleciona uma ação aleatória (exploração)

        Este metodo implementa a exploração, escolhe uma ação aleatória
        da lista de ações disponíveis.
        """

        return choice(self.acoes)

    def selecionar_acao(self, s):
        """
        Seleciona uma ação com base na estratégia epsilon-greedy.

        Este metodo decide entre exploração e aproveitamento com base no valor
        de epsilon. Quanto maior o seu valor, maior a probabilidade de explorar.
        """

        if random() > self.epsilon:
            acao = self.aproveitar(s)  # Aproveitamento (escolhe a melhor ação).
        else:
            acao = self.explorar()  # Exploração (escolhe uma ação aleatória).
        return acao


class MemoriaEsparsa(MemoriaAprend):
    """
    Implementação da classe memória esparsa

    A memória esparsa guarda os valores Q associados a pares estado-ação
    utilizando um dicionário.
    """

    def __init__(self, valor_omissao=0.0):
        """
        Inicialização da memória esparsa.
        """

        self.valor_omissao = valor_omissao
        self.memoria = {}

    def Q(self, s, a):
        """
        Retorna o valor Q armazenado para um par (estado, ação).

        Se o par (s, a) não estiver na memória, retorna o valor_omissao.
        Permitindo ao agente tratar pares não visitados como tendo valor Q zero
        ou outro valor padrão.
        """

        return self.memoria.get((s, a), self.valor_omissao)

    def atualizar(self, s, a, q):
        """
        Atualiza o valor Q para um par (estado, ação).

        Insere o valor Q para o par (s, a) na memória.
        """
        self.memoria[(s, a)] = q


class SARSA(AprendRef):
    """
    Implementação do algoritmo SARSA

    O SARSA é um algoritmo de aprendizagem por reforço que atualiza os valores Q
    considerando tanto o estado e ação atuais quanto o próximo estado e a próxima ação.
    """

    def aprender(self, s, a, r, sn, an):
        """
        Atualiza os valores Q utilizando o algoritmo SARSA.
        """

        qsa = self.mem_aprend.Q(s, a)  # Valor Q atual para o par (s, a)

        qsn_an = self.mem_aprend.Q(sn, an)  # Valor Q esperado para o próximo estado e próxima ação (SARSA)

        q = qsa + self.alfa * (r + self.gama * qsn_an - qsa)  # Atualiza o valor Q utilizando a fórmula do SARSA

        self.mem_aprend.atualizar(s, a, q)  # Atualiza a memória com o novo valor Q


class QLearning(AprendRef):
    """
    Implementação do algoritmo Q-Learning

    O Q-Learning é um algoritmo de aprendizagem por reforço off-policy
    que atualiza os valores Q baseando-se na melhor ação possível para o próximo
    estado, independentemente da política seguida atualmente.
    """

    def aprender(self, s, a, r, sn):
        """
        Atualiza os valores Q usando o algoritmo Q-Learning.
        """

        an = self.sel_accao.max_acao(sn, self.sel_accao.acoes)  # Seleciona a melhor ação no próximo estado (sn) utilizando o max_acao

        qsa = self.mem_aprend.Q(s, a)  # Obtém o valor Q atual para o par (s, a)

        qsn_an = self.mem_aprend.Q(sn, an)  # Obtém o valor Q para o próximo estado e a melhor ação (Q-Learning usa a melhor ação)

        q = qsa + self.alfa * (r + self.gama * qsn_an - qsa)  # Calcula o novo valor Q utilizando a fórmula do Q-Learning

        self.mem_aprend.atualizar(s, a, q)  # Atualiza a memória com o novo valor Q


class DynaQ(QLearning):
    """
    Implementação do algoritmo Dyna-Q, que estende o Q-Learning com simulações.

    O Dyna-Q combina a aprendizagem do Q-Learning com aprendizagem baseada em simulações.
    Ele utiliza um modelo transitório para gerar transições simuladas, permitindo ao agente
    aprender mais rapidamente explorando estados mesmo sem visitá-los diretamente.
    """

    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim):
        """
        Inicialização os parâmetros do Dyna-Q.
        """

        super().__init__(mem_aprend, sel_accao, alfa, gama)  # Inicializa Q-Learning
        self.num_sim = num_sim  # Quantidade de simulações para acelerar a aprendizagem
        self.modelo = ModeloTR()  # Modelo transitório para simulações

    def aprender(self, s, a, r, sn):
        """
        Atualiza os valores Q e o modelo transitório, e realiza simulações
        """

        super().aprender(s, a, r, sn)  # Atualiza os valores Q utilizando o Q-Learning para a transição real

        self.modelo.atualizar(s, a, r, sn)  # Atualiza o modelo transitório com a transição observada

        self.simular()  # Realiza simulações para explorar transições adicionais

    def simular(self):
        """
        Realiza simulações baseadas no modelo transitório

        Durante as simulações, amostras de transições guardados no modelo transitório
        são utilizadas para atualizar os valores Q, acelerando a aprendizagem
        """

        for _ in range(self.num_sim):
            s, a, r, sn = self.modelo.amostrar()  # Amostra de uma transição do modelo transitório

            super().aprender(s, a, r, sn)  # Atualiza os valores Q com base na transição simulada.


class ModeloTR:
    """
    Modelo de Transição (ModeloTR).

    Este modelo é utilizado pelo algoritmo Dyna-Q para guardar as transições observadas no ambiente.
    Ele mantém informações determinísticas sobre, as transições de estado associadas a cada ação e
    as recompensas obtidas ao executar essas transições.

    O modelo permite que o agente realize simulações, utilizando as transições armazenadas para
    atualizar os valores Q sem interagir diretamente com o ambiente.
    """

    def __init__(self):
        """
        Inicialização do modelo de transições
        """

        self.T = {}  # Dicionário para as transições: (s, a) -> sn
        self.R = {}  # Dicionário para as recompensas: (s, a) -> r

    def atualizar(self, s, a, r, sn):
        """
        Atualiza o modelo com uma nova transição observada

        Este metodo guarda o próximo estado (sn) e a recompensa (r)
        associados a um par estado-ação (s, a)
        """

        self.T[(s, a)] = sn  # Atualiza o próximo estado para o par (s, a)
        self.R[(s, a)] = r  # Atualiza a recompensa para o par (s, a)

    def amostrar(self):
        """
        Retorna uma transição aleatória armazenada no modelo

        Este metodo é utilizado para gerar transições simuladas no algoritmo Dyna-Q,
        permitindo ao agente aprender com experiências passadas
        """

        s, a = choice(list(self.T.keys()))  # Seleciona aleatoriamente um par (estado, ação) guardado no dicionário T

        # Seleciona aleatoriamente um par (estado, ação) guardado no dicionário T
        sn = self.T[(s, a)]
        r = self.R[(s, a)]

        return s, a, r, sn  # Retorna a transição simulada


class QME(QLearning):
    """
    Implementação do algoritmo QME (Q-Learning com Memória de Experiência)

    O QME combina a aprendizagem do Q-Learning com o uso de uma memória de experiência.
    Ele guarda as transições observadas numa memória limitada, permitindo que o agente
    reutilize experiências passadas para melhorar a aprendizagem por meio de simulações.
    """

    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim, dim_max):
        """
        Inicialização os parâmetros do QME.
        """

        super().__init__(mem_aprend, sel_accao, alfa, gama)  # Inicialização o Q-Learning padrão
        self.num_sim = num_sim  # Define quantas simulações serão realizadas por iteração
        self.memoria_experiencia = MemoriaExperiencia(dim_max)  # Inicializa a memória de experiência

    def aprender(self, s, a, r, sn):
        """
        Atualiza os valores Q e guarda a transição na memória de experiência.
        """

        super().aprender(s, a, r, sn)  # Atualiza os valores Q para a transição real utilizando o Q-Learning

        # Guarda a transição na memória de experiência
        e = (s, a, r, sn)
        self.memoria_experiencia.atualizar(e)

        self.simular()  # Realiza simulações com base nas experiências guardadas

    def simular(self):
        """
        Realiza simulações baseadas na memória de experiência

        Este metodo utiliza transições guardadas na memória de experiência para
        reforçar a aprendizagem sem interagir diretamente com o ambiente.
        """

        amostras = self.memoria_experiencia.amostrar(self.num_sim)  # Obtém um conjunto de amostras da memória de experiência

        # Para cada transição simulada, realiza a aprendizagem utilizando o Q-Learning
        for s, a, r, sn in amostras:
            super().aprender(s, a, r, sn)


class MemoriaExperiencia:
    """
    Classe para guardar as experiências (transições) numa memória limitada

    A memória de experiência é utilizada para guardar transições observadas pelo agente.
    Essas transições podem ser reutilizadas para simulações, reforçando a aprendizagem
    sem depender de interações diretas com o ambiente.
    """

    def __init__(self, dim_max):
        """
        Inicialização da memória de experiência com capacidade máxima
        """

        self.dim_max = dim_max  # Define o tamanho máximo da memória
        self.memoria = []  # Lista que guarda as transições

    def atualizar(self, e):
        """
        Guarda uma nova experiência na memória e remove a mais antiga se a capacidade for excedida

        Este metodo é chamado sempre que uma nova transição (experiência) é observada.
        Se a memória já estiver cheia, a transição mais antiga é removida antes
        de adicionar a nova.
        """

        # Remove a experiência mais antiga para libertar espaço
        if len(self.memoria) == self.dim_max:
            self.memoria.pop(0)
        self.memoria.append(e)  # Adiciona a nova experiência no final da lista

    def amostrar(self, n):
        """
        Retorna uma lista de amostras aleatórias da memória

        Este metodo seleciona aleatoriamente n transições guardadas na memória.
        Se o número de experiências guardadas for menor que n, retorna todas as
        transições disponíveis.
        """

        n_amostras = min(n, len(self.memoria))  # Garante que o número de amostras não excede o tamanho da memória
        return sample(self.memoria, n_amostras)  # Retorna uma lista de transições selecionadas aleatoriamente