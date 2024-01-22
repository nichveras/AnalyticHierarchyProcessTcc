# coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt


class AHP():
    def __init__(self, metodo, precisao, alternativas, criterios, subCriterios, matrizesPreferencias, log=False):
        self.metodo = metodo
        self.precisao = precisao
        self.alternativas = alternativas
        self.criterios = criterios
        self.subCriterios = subCriterios
        self.matrizesPreferencias = matrizesPreferencias
        self.log = log
        self.prioridadesGlobais = []

    @staticmethod
    def Aproximado(matriz, precisao):
        soma_colunas = matriz.sum(axis=0)
        matriz_norm = np.divide(matriz, soma_colunas)
        media_linhas = matriz_norm.mean(axis=1)
        return np.round(media_linhas, precisao)

    @staticmethod
    def Geometrico(matriz, precisao):
        media_geometrica = [np.prod(linha) ** (1/len(linha)) for linha in matriz]
        media_geometrica_norm = media_geometrica/sum(media_geometrica)
        return media_geometrica_norm.round(precisao)

    @staticmethod
    def AutoValor(matriz, precisao, interacao=100, autovetor_anterior=None):
        matriz_quadrada = np.linalg.matrix_power(matriz, 2)
        soma_linhas = np.sum(matriz_quadrada, axis=1)
        soma_coluna = np.sum(soma_linhas, axis=0)
        autovetor_atual = np.divide(soma_linhas, soma_coluna)

        if autovetor_anterior is None:
            autovetor_anterior = np.zeros(matriz.shape[0])

        diferenca = np.subtract(autovetor_atual, autovetor_anterior).round(precisao)
        if not np.any(diferenca):
            return autovetor_atual.round(precisao)

        interacao -= 1
        if interacao > 0:
            return AHP.AutoValor(matriz_quadrada, precisao, interacao, autovetor_atual)
        else:
            return autovetor_atual.round(precisao)

    @staticmethod
    def Consistencia(matriz):
        if matriz.shape[0] > 2 and matriz.shape[1] > 2:
            lambda_max = np.real(np.linalg.eigvals(matriz).max())
            ic = (lambda_max - len(matriz)) / (len(matriz) - 1)
            ri = {3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45,
                  10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}
            rc = ic / ri[len(matriz)]
            return lambda_max, ic, rc
        else:
            return 0, 0, 0

    def VetorPrioridadesLocais(self):
        vetor_prioridades_locais = {}

        for criterio in self.matrizesPreferencias:
            matriz = np.array(self.matrizesPreferencias[criterio])
            if self.metodo == 'aproximado':
                prioridades_locais = self.Aproximado(matriz, self.precisao)
            elif self.metodo == 'geometrico':
                prioridades_locais = self.Geometrico(matriz, self.precisao)
            else:
                if matriz.shape[0] and matriz.shape[1] >= 2:
                    prioridades_locais = self.AutoValor(matriz, self.precisao)
                else:
                    prioridades_locais = self.Aproximado(matriz, self.precisao)

            vetor_prioridades_locais[criterio] = prioridades_locais

            lambda_max, ic, rc = self.Consistencia(matriz)

            if self.log:
                print('\nPrioridades locais do criterio ' + criterio + ':\n', prioridades_locais)
                print('Soma:', np.round(np.sum(prioridades_locais), self.precisao))
                print('Lambda_max =', lambda_max)
                print('Indice de Consistencia ' + criterio + '=', round(ic, self.precisao))
                print('Razao de Consistencia ' + criterio + '=', round(rc, 2))

        return vetor_prioridades_locais

    def VetorPrioridadesGlobais(self, prioridades, pesos, criterios):
        global_prioridades = np.zeros(len(self.alternativas))

        for criterio in criterios:
            peso = pesos[criterios.index(criterio)]
            prioridades_locais = prioridades.get(criterio, np.zeros(len(self.alternativas)))
            prioridades_global = peso * prioridades_locais

            if criterio in self.subCriterios:
                for subcriterio in self.subCriterios[criterio]:
                    if subcriterio in self.matrizesPreferencias:
                        matriz_subcriterio = np.array(self.matrizesPreferencias[subcriterio])
                        if matriz_subcriterio.size > 0 and matriz_subcriterio.shape[0] == matriz_subcriterio.shape[1]:
                            prioridades_locais_sub = self.AutoValor(matriz_subcriterio, self.precisao, interacao=100)
                            prioridades_global_sub = peso * prioridades_locais_sub
                            global_prioridades += prioridades_global_sub[:len(self.alternativas)]  # Ajuste aqui
                        else:
                            print(f"Erro: A matriz de preferências para o subcriterio {subcriterio} é inválida.")
            else:
                matriz_pref = self.matrizesPreferencias.get(criterio, [])
                if matriz_pref:
                    prioridades_global += peso * self.Aproximado(matriz_pref, self.precisao)[
                                                 :len(self.alternativas)]  # Ajuste aqui
                else:
                    print(f"Erro: A matriz de preferências para o critério {criterio} não foi encontrada.")

                if self.log:
                    print('\nPrioridades globais do criterio ' + criterio + '\n', prioridades_global)
                    print('Soma:', np.sum(prioridades_global).round(self.precisao))

        return global_prioridades

    def Resultado(self):
        prioridades_locais = self.VetorPrioridadesLocais()
        global_prioridades = self.VetorPrioridadesGlobais(prioridades_locais, [1] * len(self.criterios), self.criterios)

        if global_prioridades.size > 0:
            global_prioridades = global_prioridades.round(self.precisao)

            # Obtenha os índices dos carros ordenados por prioridade global
            indices_ordenados = np.argsort(global_prioridades)[::-1]

            # Crie um dicionário com os resultados para todos os carros
            resultados = {carro: prioridade for carro, prioridade in zip(self.alternativas, global_prioridades)}

            # Retorne o dicionário completo de resultados
            return {self.alternativas[i]: resultados[self.alternativas[i]] for i in indices_ordenados}

        else:
            print("Erro: As prioridades globais não puderam ser calculadas corretamente.")
            return {}


if __name__ == '__main__':
    matriz = np.array([
        [1, 6, 2],
        [1 / 6, 1, 1 / 3],
        [1 / 2, 3, 1]
    ])
    precisao = 2
    escolher_melhor_carro = AHP(
        metodo='multicriterio',
        precisao=3,
        alternativas=['Sedan', 'SUV', 'Hatchback','Pickup','Crossover','Compactos'],
        criterios=['Valor', 'Desempenho', 'Segurança', 'Modelo'],
        subCriterios={
            'Valor': ['Valor de Venda', 'Valor de Revenda'],
            'Desempenho': ['Potencia do Motor', 'Eficiencia', 'Velocidade Máxima'],
            'Segurança': ['Freios ABS', 'Airbags', 'Sistema de Estabilidade', 'Confiabilidade'],
            'Modelo': ['Sedan', 'SUV', 'Hatchback', 'Pickup', 'Crossover', 'Compactos', 'Esportivos', 'Eletricos']
        },
        matrizesPreferencias={
            'criterios': [
                [1, 3, 7, 3],
                [1 / 3, 1, 9, 1],
                [1 / 7, 1 / 9, 1, 1 / 7],
                [1 / 3, 1, 7, 1]
            ],
            'valor': [
                [1, 2, 5, 3],
                [1 / 2, 1, 2, 2],
                [1 / 5, 1 / 2, 1, 1 / 2],
                [1 / 3, 1 / 2, 2, 1]
            ],
            'Valor de Venda': [
                [1, 9, 9, 1, 1 / 2, 5],
                [1 / 9, 1, 1, 1 / 9, 1 / 9, 1 / 7],
                [1 / 9, 1, 1, 1 / 9, 1 / 9, 1 / 7],
                [1, 9, 9, 1, 1 / 2, 5],
                [2, 9, 9, 2, 1, 6],
                [1 / 5, 7, 7, 1 / 5, 1 / 6, 1]
            ],
            'Valor de Revenda': [
                [1, 1 / 1.13, 1.41, 1.15, 1.24, 1.19],
                [1.13, 1, 1.59, 1.3, 1.4, 1.35],
                [1 / 1.41, 1 / 1.59, 1, 1 / 1.23, 1 / 1.14, 1 / 1.18],
                [1 / 1.15, 1 / 1.3, 1.23, 1, 1.08, 1.04],
                [1 / 1.24, 1 / 4, 1.14, 1 / 1.08, 1, 1 / 1.04],
                [1 / 1.19, 1 / 1.35, 1.18, 1 / 1.04, 1.04, 1]
            ],
            'desempenho': [
                [1, 1.5, 4, 4, 4, 5],
                [1 / 1.5, 1, 4, 4, 4, 5],
                [1 / 4, 1 / 4, 1, 1, 1.2, 1],
                [1 / 4, 1 / 4, 1, 1, 1, 3],
                [1 / 4, 1 / 4, 1 / 1.2, 1, 1, 2],
                [1 / 5, 1 / 5, 1, 1 / 3, 1 / 2, 1]
            ],
            'Potencia do Motor': [
                [1, 3, 4, 1 / 2, 2, 2],
                [1 / 3, 1, 2, 1 / 5, 1, 1],
                [1 / 4, 1 / 2, 1, 1 / 6, 1 / 2, 1 / 2],
                [2, 5, 6, 1, 4, 4],
                [1 / 2, 1, 2, 1 / 4, 1, 1],
                [1 / 2, 1, 2, 1 / 4, 1, 1]
            ],
            'Eficiencia': [
                [1, 1, 5, 7, 9, 1 / 3],
                [1, 1, 5, 7, 9, 1 / 3],
                [1 / 5, 1 / 5, 1, 2, 9, 1 / 8],
                [1 / 7, 1 / 7, 1 / 2, 1, 2, 1 / 8],
                [1 / 9, 1 / 9, 1 / 9, 1 / 2, 1, 1 / 9],
                [3, 3, 8, 8, 9, 1]
            ],
            'Velocidade Máxima': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'segurança': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Freios ABS': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Airbags': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Sistema de Estabilidade': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Confiabilidade': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'modelo': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Sedan': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'SUV': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Hatchback': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Pickup': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Crossover': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Compactos': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Esportivos': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]

            ],
            'Eletricos': [
                [1, 1, 7, 5, 9, 6],
                [1, 1, 7, 5, 9, 6],
                [1 / 7, 1 / 7, 1, 1 / 6, 3, 1 / 3],
                [1 / 5, 1 / 5, 6, 1, 7, 5],
                [1 / 9, 1 / 9, 1 / 3, 1 / 7, 1, 1 / 5],
                [1 / 6, 1 / 6, 3, 1 / 5, 5, 1]
            ]
        },
        log=True
    )
    resultado = escolher_melhor_carro.Resultado()
    print("Resultado final:", resultado)

    plt.bar(resultado.keys(), resultado.values())
    plt.ylabel("Prioridade")
    plt.show()

