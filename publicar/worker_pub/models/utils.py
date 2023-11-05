# ----------------------------------------------------------------------------------------------------
# Funções úteis para a implementação das funcionalidades para publicação do modelo via API
# ----------------------------------------------------------------------------------------------------
import re
import numpy as np
from unicodedata import normalize


def concatenar_registros(x, algumas_stop_words, remover_stop_words=False):
    """
    Trata e concatena o conteúdo das colunas do dataset e transforma numa string única.
        :param x: Dataset a ser concatenado.
        :param algumas_stop_words: Palavras que serão retiradas do dataset por não serem relevantes para o treino.
        :param remover_stop_words: Se esta opção for True, todas as palavras da lista 'algumas_stop_words' serão
                                   removidas no pré-processamento do texto.
        :return: Dataset tratado e concatenado.
    """
    registros_tratados = []

    for i in range(len(x)):
        # Retira espaços em branco no final e no início do campo. Também retira acentuação e caracteres especiais
        aux = ' '.join([normalize('NFKD', re.sub(r'[^\w\s]', ' ', t.strip())).encode('ASCII', 'ignore').decode('ASCII')
                        for t in list(x[i])])

        # Se for o caso, remove as stopwords
        if remover_stop_words:
            # Faz join e depois split novamente porque no join anterior são juntadas palavras e frases,
            # aqui é para separar só em palavras
            split_aux = aux.split()
            aux = [w for w in split_aux if w.lower() not in algumas_stop_words]
            aux = ' '.join(aux)

        registros_tratados.append(aux)

    return np.array(registros_tratados)
