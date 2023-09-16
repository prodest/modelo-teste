# ----------------------------------------------------------------------------------------------------------
# Funções para parametrização de modelo baseado em redes neurais
# ----------------------------------------------------------------------------------------------------------
import tensorflow as tf
import logging
import numpy as np
from utils import tratar_texto_tfidf
import configs as c


def construir_modelo_rn(input_shape, output_dim, neuronios=2, aplicar_dropout=False, perc_dropout=0.1):
    """
    Constroi um modelo baseado em rede neural.
        :param input_shape: Shape da entrada.
        :param output_dim: Dimensão da saida.
        :param neuronios: Quantidade de neurônios que serão utilizados na rede neural.
        :param aplicar_dropout: Indica se será aplicado Dropout.
        :param perc_dropout: Se for aplicar dropout, qual o percentual.
        :return: modelo baseado em rede neural instanciado com os parâmetros passados.
    """
    modelo = tf.keras.models.Sequential()
    modelo.add(tf.keras.layers.Dense(units=neuronios, input_shape=input_shape, activation='relu'))

    if aplicar_dropout:
        modelo.add(tf.keras.layers.Dropout(perc_dropout, input_shape=input_shape))

    modelo.add(tf.keras.layers.Dense(units=output_dim, activation='sigmoid'))
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo


def clf_rn_tfidf(t, le, x_train, y_train, x_test, y_test, x_val, y_val):
    """
    Treina um modelo que utiliza Rede Neural e TFIDF.
        :param t: Tokenizer contendo o vocabulário de palavras.
        :param le: Label Encoder com os labels de y codificados.
        :param x_train: Features de treino.
        :param y_train: Labels de treino.
        :param x_test: Features de teste.
        :param y_test: Labels de teste.
        :param x_val: Features de validação.
        :param y_val: Labels de validação.
        :return: Modelo treinado, histórico do treinamento e resultado do teste com o x_test e y_test.
    """
    # Vetoriza o dataset utilizando TF-IDF. Junto ao texto vetorizado retorna a quantidade de classes encontradas no y
    qtd_classes_y, x_train, y_train, x_test, y_test, x_val, y_val = \
        tratar_texto_tfidf(t, le, x_train, y_train, x_test, y_test, x_val, y_val)

    print("=> Treinando o modelo 'clf_rn_tfidf'...\n")

    # Prepara as dimensões iniciais para a rede
    input_shape = x_train[0].shape

    # Tem que ser do y porque o le.fit() foi feito com o y inteiro. Se colocar a do y_train dará erro de shape
    # lá no treinamento
    output_dim = qtd_classes_y

    modelo = construir_modelo_rn(input_shape, output_dim, neuronios=c.qtd_neuronios, aplicar_dropout=c.aplicar_dropout,
                                 perc_dropout=c.perc_dropout)

    print("\n  # Arquitetura do modelo:\n")
    print(modelo.summary())
    print("\n")

    # Treina o modelo e guarda o histórico do treino
    hist = modelo.fit(x_train, y_train, batch_size=c.batch_size, epochs=c.epochs, validation_data=(x_val, y_val))

    resultado_teste = modelo.evaluate(x_test, y_test)

    msg_log = f"\n\n*************** RESULTADOS DO TREINO ***************\n"
    msg_log += f"\n    Acurácia...: {resultado_teste[1]}\n"
    msg_log += f"\n    Perda......: {resultado_teste[0]}\n\n"
    logging.info(msg_log)
    print(msg_log)

    return modelo, hist, resultado_teste


def predizer_tfidf(t, le, modelo, x):
    """
    Prediz labels num dataset que foi vetorizado utilizado TF-IDF.
        :param t: Tokenizer contendo o vocabulário de palavras.
        :param le: Label Encoder com os labels de y codificados.
        :param modelo: Modelo treinado que será utilizado para a realização da predição.
        :param x: Dataset que será utilizado para realizar a predição.
        :return: Targets preditos.
    """
    # Vetoriza e faz a predição
    try:
        x = t.texts_to_matrix(x, mode='tfidf')
    except MemoryError as e:
        msg = f"Erro na alocação de memória. Mensagem do Tokenizer: '{e}'. Programa abortado!"
        logging.error(msg)
        print(f"\n\n{msg} Consulte o log de execução ({c.arquivo_log}) para mais detalhes!\n")
        exit(1)

    ypred = modelo.predict(x)

    # Obtém a classe que foi inferida
    resultado_argmax = np.argmax(ypred, axis=1)

    # Recupera as classes no formato string
    ypred_inverse = list(le.inverse_transform(resultado_argmax))

    return ypred_inverse
