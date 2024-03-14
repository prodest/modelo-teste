# ----------------------------------------------------------------------------------------------------
# Funções úteis para a implementação das funcionalidades para o treino automatizado para criar uma
# nova versão do modelo que está em produção.
# ----------------------------------------------------------------------------------------------------
import re
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize
from tensorflow.keras.utils import to_categorical


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


def tratar_texto_tfidf(t, le, x_train, y_train, x_test, y_test, x_val, y_val, logger):
    """
    Trata o dataset utilizando TF-IDF (Term Frequency–Inverse Document Frequency) e One-Hot encoding.
        :param t: Tokenizer contendo o vocabulário de palavras.
        :param le: Label Encoder com os labels de y codificados.
        :param x_train: Features de treino que serão tratadas utilizando TFIDF.
        :param y_train: Labels de treino que serão tratados utilizando One-Hot encoding.
        :param x_test: Features de teste que serão tratadas utilizando TFIDF.
        :param y_test: Labels de teste que serão tratados utilizando One-Hot encoding.
        :param x_val: Features de validação que serão tratadas utilizando TFIDF.
        :param y_val: Labels de validação que serão tratados utilizando One-Hot encoding.
        :param logger: Logger para gravação de logs.
        :return: Datasets tratados.
    """
    # Obs.: Está considerando o y completo
    qtd_classes_y = len(le.classes_)

    # Codifica o texto com base no TF-IDF (Term Frequency - Inverse Document Frequency) que leva em conta a relevância
    # da palavra no texto
    try:
        cod_x_train = t.texts_to_matrix(x_train, mode='tfidf')
        cod_x_test = t.texts_to_matrix(x_test, mode='tfidf')
        cod_x_val = t.texts_to_matrix(x_val, mode='tfidf')
    except MemoryError as e:
        msg = f"Erro na alocação de memória. Mensagem do Tokenizer: '{e}'. Programa abortado!"
        logger.error(msg)
        print(f"\n\n{msg} Consulte o log de execução para mais detalhes!\n")
        exit(1)

    # Codifica as classes com o One-Hot Encoding
    y_train_one_hot = to_categorical(le.transform(y_train), num_classes=qtd_classes_y)
    y_test_one_hot = to_categorical(le.transform(y_test), num_classes=qtd_classes_y)
    y_val_one_hot = to_categorical(le.transform(y_val), num_classes=qtd_classes_y)

    return qtd_classes_y, cod_x_train, y_train_one_hot, cod_x_test, y_test_one_hot, cod_x_val, y_val_one_hot


def predizer_tfidf(t, le, modelo, x, logger):
    """
    Prediz labels em um dataset que foi vetorizado utilizado TF-IDF.
        :param t: Tokenizer contendo o vocabulário de palavras.
        :param le: Label Encoder com os labels de y codificados.
        :param modelo: Modelo treinado que será utilizado para a realização da predição.
        :param x: Dataset que será utilizado para realizar a predição.
        :param logger: Logger para gravação de logs.
        :return: Lista com os labels preditos.
    """
    # Vetoriza e faz a predição
    try:
        x = t.texts_to_matrix(x, mode='tfidf')
    except MemoryError as e:
        msg = f"Não foi possível alocar memória. Mensagem do Tokenizer: '{e}'."
        logger.error(msg)
        print(f"\n\n{msg}")
        exit(1)

    ypred = modelo.predict(x)

    # Obtém a classe que foi inferida
    resultado_argmax = np.argmax(ypred, axis=1)

    # Recupera as classes no formato string
    ypred_inverse = list(le.inverse_transform(resultado_argmax))

    return ypred_inverse


def construir_modelo_rn(input_shape, output_dim, neuronios=2, aplicar_dropout=False, perc_dropout=0.1):
    """
    Constroi um modelo baseado em rede neural.
        :param input_shape: Shape da entrada.
        :param output_dim: Dimensão da saida.
        :param neuronios: Quantidade de neurônios que serão utilizados na rede neural.
        :param aplicar_dropout: Indica se será aplicado Dropout.
        :param perc_dropout: Se for aplicar dropout, qual o percentual.
        :return: Modelo baseado em rede neural instanciado com os parâmetros passados.
    """
    modelo = tf.keras.models.Sequential()
    modelo.add(tf.keras.layers.Input(shape=input_shape))
    modelo.add(tf.keras.layers.Dense(units=neuronios, activation='relu'))

    if aplicar_dropout:
        modelo.add(tf.keras.layers.Dropout(perc_dropout))

    modelo.add(tf.keras.layers.Dense(units=output_dim, activation='sigmoid'))
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo


def gerar_grafico_historico_treino(hist, caminho):
    """
    Gera um gráfico com dois subplots contendo dados de histórico do treinamento.
        :param hist: Objeto contendo os dados históricos do treinamento.
        :param caminho: Caminho onde o gráfico será salvo.
    """
    figure, axis = plt.subplots(1, 2, figsize=(15, 4))

    # Plota o gráfico de acurácia
    axis[0].plot(hist.history['accuracy'])
    axis[0].plot(hist.history['val_accuracy'])
    axis[0].set_title('Acurácia do modelo')
    axis[0].set_ylabel('Acurácia')
    axis[0].set_xlabel('Epoch')
    axis[0].legend(['Treino', 'Validação'], loc='upper left')

    # Plota o gráfico de perda
    axis[1].plot(hist.history['loss'])
    axis[1].plot(hist.history['val_loss'])
    axis[1].set_title('Perda do modelo')
    axis[1].set_ylabel('Perda')
    axis[1].set_xlabel('Epoch')
    axis[1].legend(['Treino', 'Validação'], loc='upper left')

    caminho_arq_grafico = os.path.join(caminho, "TrainingChart.jpeg")
    plt.savefig(caminho_arq_grafico)
