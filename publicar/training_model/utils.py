# ----------------------------------------------------------------------------------------------------------
# Funções diversas que podem ser utilizadas em qualquer parte do programa
# ----------------------------------------------------------------------------------------------------------
import pickle
import matplotlib.pyplot as plt
import logging
import re
import pandas as pd
import numpy as np
from tensorflow.saved_model import save
from unicodedata import normalize
from keras.utils import to_categorical
from configs import arquivo_log
from os import makedirs
from logging.handlers import RotatingFileHandler
from pathlib import Path


def make_log(filename: str, log_output: str = "file") -> logging.Logger:
    """
    Cria um logger para gerar logs na console ou gravar em um arquivo. Se o arquivo já existir, inicia a gravação a
    partir do final dele.
        :param filename: Nome do arquivo de logs (caso o log seja gravado em arquivo).
        :param log_output: Destino para geração dos logs.
        :return: Um logger para geração dos logs.
    """
    # Configurações básicas
    logger_name = filename.split(".")[0]
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s | %(funcName)s: %(message)s')
    logger.propagate = False

    if log_output == "console":
        consolehandler = logging.StreamHandler()
        consolehandler.setLevel(logging.INFO)
        consolehandler.setFormatter(formatter)
        logger.addHandler(consolehandler)
    elif log_output == "file":
        # Se a pasta de logs não existir, cria
        try:
            makedirs("logs", exist_ok=True)
        except PermissionError:
            msg = "Erro ao criar a pasta 'logs'. Permissão de escrita negada!"
            print(f"\n{msg}\n")
            exit(1)

        log_file_path = str(Path("logs") / filename)

        # Configuração de parâmetros para gravação de logs
        try:
            rotatehandler = RotatingFileHandler(log_file_path, mode='a', maxBytes=10485760, backupCount=5)
            rotatehandler.setLevel(logging.INFO)
            rotatehandler.setFormatter(formatter)
            logger.addHandler(rotatehandler)
        except FileNotFoundError:
            msg = f"Não foi possível encontrar/criar o arquivo de log no caminho '{log_file_path}'."
            print(f"\n{msg}\n")
            exit(1)
        except PermissionError:
            msg = f"Não foi possível criar/acessar o arquivo de log no caminho '{log_file_path}'. Permissão de " \
                  f"escrita/leitura negada."
            print(f"\n{msg}\n")
            exit(1)
    else:
        msg = f"O parâmetro 'log_output' contém um tipo de saída do log incorreto ('{log_output}'). Os possíveis " \
              f"valores são: 'console' ou 'file'."
        print(f"\n{msg}\n")
        exit(1)

    return logger


# Para facilitar, define um logger único para todas as funções
LOGGER = make_log(filename="train.log", log_output="file")


def imprimir_parametros_treino_rn_tfidf(param):
    """
    Imprime os parâmetros utilizados no treino.
        :param param: Parâmetros carregados através do arquivo 'configs.py' do módulo 'conf'.
        :return: String com os parâmetros.
    """
    parametros = f"""
    # >> Parâmetros que foram utilizados para configuração do modelo:

        aplicar_dropout = {param.aplicar_dropout}
        perc_dropout = {param.perc_dropout}
        qtd_neuronios = {param.qtd_neuronios}
        batch_size = {param.batch_size}
        epochs = {param.epochs}
        trava_ramdomica = {param.trava_randomica}
    """

    LOGGER.info(parametros)
    return parametros


def imprimir_alguns_parametros_utilizados(param):
    """
    Imprime alguns parâmetros utilizados.
        :param param: Parâmetros carregados através do arquivo 'configs.py' do módulo 'conf'.
        :return: String com os parâmetros.
    """
    parametros = f"""
    # >> Parâmetros que foram utilizados para pré-processamento da base de dados e testes:

        caminho_arquivo = {param.caminho_arquivo}
        nome_arquivo = {param.nome_arquivo}
        separador = {param.separador}
        encoding_arquivo = {param.encoding_arquivo}
        percentual_exemplos_teste = {param.percentual_exemplos_teste}
        qtd_exemplos = {param.qtd_exemplos}
        qtd_palavras_considerar = {param.qtd_palavras_considerar}
        colunas_selecionadas_dataset = {param.colunas_selecionadas_dataset}
        colunas_selecionadas_x = {param.colunas_selecionadas_x}
        colunas_selecionadas_y = {param.colunas_selecionadas_y}
        descartar_registros_vazios = {param.descartar_registros_vazios}
        remover_stop_words = {param.remover_stop_words}
        algumas_stop_words = {param.algumas_stop_words}
        
    # >> Parâmetros que foram utilizados para definição do tipo e nome do modelo:
    
        dados_modelo = {param.dados_modelo}
    """

    LOGGER.info(parametros)
    return parametros


def imprimir_shapes(msg_titulo, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Imprime os shapes dos datasets.
        :param msg_titulo: Mensagem que será mostrada no título da impressão.
        :param x_train: Features do dataset de treino.
        :param y_train: Targets do dataset de treino.
        :param x_val:  Features do dataset de validação.
        :param y_val: Targets do dataset de validação.
        :param x_test:  Features do dataset de teste.
        :param y_test: Targets do dataset de teste.
        :return: String com os shapes.

    """
    msg = f"\n  # {msg_titulo}:\n"
    msg += f"\n    x Treino.....: {x_train.shape}"
    msg += f"\n    y Treino.....: {y_train.shape}\n"
    msg += f"\n    x Validação..: {x_val.shape}"
    msg += f"\n    y Validação..: {y_val.shape}\n"
    msg += f"\n    x Teste......: {x_test.shape}"
    msg += f"\n    y Teste......: {y_test.shape}\n"
    msg += f"\n    - Total de exemplos: {len(x_train) + len(x_val) + len(x_test)}\n"

    LOGGER.info(msg)
    print(msg)
    return msg


def gerar_grafico_historico_treino(hist, caminho_modelo):
    """
    Gera um gráfico com dois subplots contendo dados de histórico do treinamento.
        :param hist: Objeto contendo os dados históricos do treinamento.
        :param caminho_modelo: Caminho do modelo onde o gráfico será salvo.
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

    caminho_arq_grafico = caminho_modelo + "grafico_treinamento.jpeg"
    LOGGER.info(f">>> Um gráfico com o histórico do treinamento foi gerado no arquivo '{caminho_arq_grafico}'")
    plt.savefig(caminho_arq_grafico)


def salvar_objeto(caminho, objeto, nome_objeto):
    """
    Salva um objeto.
        :param caminho: Caminho onde o objeto será salvo.
        :param objeto: Objeto que será salvo.
        :param nome_objeto: Nome do objeto que será salvo.
    """
    caminho_arq_objeto = f"{caminho}{nome_objeto}.pkl"

    try:
        arq = open(caminho_arq_objeto, 'wb')
    except FileNotFoundError:
        msg = f"Erro ao salvar o objeto '{nome_objeto}'. O caminho '{caminho_arq_objeto}' está incorreto."
        LOGGER.error(msg)
        print(f"\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)
    except PermissionError:
        msg = f"Erro ao salvar o objeto '{nome_objeto}' no caminho '{caminho}'. Permissão de escrita negada."
        LOGGER.error(msg)
        print(f"\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    try:
        pickle.dump(objeto, arq)
    except TypeError as e:
        msg = f"Erro ao persistir o objeto '{nome_objeto}' com o Pickle (mensagem Pickle: {e})."
        LOGGER.error(msg)
        print(f"\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    arq.close()
    LOGGER.info(f"O objeto '{nome_objeto}' foi salvo no arquivo '{caminho_arq_objeto}'")


def carregar_dataset(caminho_arquivo, separador, encoding_arquivo, colunas_selecionadas_dataset,
                     descartar_registros_vazios, colunas_selecionadas_x, colunas_selecionadas_y, qtd_exemplos):
    """
    Carrega os datasets para realização do treinamento e testes.
        :param caminho_arquivo: Caminho do arquivo para carregar os datsets.
        :param separador: Separador dos campos do arquivo de dados CSV.
        :param encoding_arquivo: Encode do arquivo CSV.
        :param colunas_selecionadas_dataset: Colunas (features) que serão usadas no treino e teste.
        :param descartar_registros_vazios: Define se os registros que possuem algum valor vazio devem ser descartados.
        :param colunas_selecionadas_x: Features (características utilizadas para inferir o target).
        :param colunas_selecionadas_y: Target (alvo pretendido para cada linha do arquivo de dados CSV).
        :param qtd_exemplos: Quantidade de exemplos extraidos do dataset para rodar o modelo.
        :return: Datasets carregados e tamanho do dataset.
    """
    print(f"=> Carregando dataset ('{caminho_arquivo}')...", end='', flush=True)

    try:
        dados = pd.read_csv(caminho_arquivo, dtype=str, sep=separador, encoding=encoding_arquivo)
    except FileNotFoundError:
        msg = f"O arquivo '{caminho_arquivo}' não foi encontrado."
        LOGGER.error(msg)
        print(f"\n\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    print(" OK!")

    tamanho_dataset = len(dados)

    # Deixa somente as colunas de interesse para o treino e teste
    try:
        dados = dados[colunas_selecionadas_dataset]
    except KeyError as e:
        msg = f"Coluna em 'colunas_selecionadas_dataset' não encontrada no dataset (erro Pandas: {e})."
        LOGGER.error(msg)
        print(f"\n\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    print(f"=> Verificando se existem valores vazios (descartar_registros_vazios = {descartar_registros_vazios})...",
          end='', flush=True)

    LOGGER.info(f"Opção: descartar_registros_vazios = {descartar_registros_vazios}")

    # Verifica se tem valores vazios e trata
    msg = f"\n\n  # Quantidade de valores vazios por coluna:\n{dados.isna().sum()}"

    if descartar_registros_vazios:
        # Retira os registros com valores vazios
        dados = dados.dropna()

        tamanho_dataset_limpo = len(dados)

        if tamanho_dataset != tamanho_dataset_limpo:
            qtd_registros_excluidos = tamanho_dataset - tamanho_dataset_limpo
            msg += f"\n\n  - Quantidade de registros antes da limpeza: {tamanho_dataset}"
            msg += f"\n  - Quantidade de registros removidos: {qtd_registros_excluidos}"
            msg += f"\n  - Quantidade de registros depois da limpeza: {tamanho_dataset_limpo}"
            msg += f"\n  - {(qtd_registros_excluidos / tamanho_dataset) * 100:.2f}% dos registros foram excluidos " \
                   f"porque possuiam alguma coluna com valor vazio!\n"
    else:
        # Preenche os valores vazios com o valor padrão 'nanvazio'
        dados = dados.fillna('nanvazio')
        msg += f"\n  - Confirmando se os valores foram preenchidos:\n{dados.isna().sum()}\n"

    print(" OK!")

    LOGGER.info(msg)

    #  Limita a quantidade de exemplos para não estourar a memória do computador!
    if tamanho_dataset > qtd_exemplos:
        dados = dados.sample(frac=1, random_state=0)  # Mistura os exemplos para tentar pegar todas as classes
        dados = dados[:qtd_exemplos]

    # Separa em features e targets
    try:
        x = dados[colunas_selecionadas_x]
    except KeyError as e:
        msg = f"Coluna em 'colunas_selecionadas_x' não encontrada no dataset (erro Pandas: {e})."
        LOGGER.error(msg)
        print(f"\n\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    # Extrai a coluna com os targets
    try:
        y = dados[colunas_selecionadas_y]
    except KeyError as e:
        msg = f"Coluna em 'colunas_selecionadas_y' não encontrada no dataset (erro Pandas: {e})."
        LOGGER.error(msg)
        print(f"\n\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    del dados

    # Transforma o dataframe num array
    x = np.array(x)
    y = np.array(y).reshape(-1)

    print(msg)

    return x, y


def concatenar_registros(x, algumas_stop_words, remover_stop_words=False, nome_dataset=''):
    """
    Trata e concatena o conteúdo das colunas do dataset e transforma numa string única.
        :param x: Dataset a ser concatenado.
        :param algumas_stop_words: Stopwords que poderão ser removidas do dataset.
        :param remover_stop_words: Se esta opção for True, todas as palavras da lista 'algumas_stop_words' serão
                                   removidas no pré-processamento do texto.
        :param nome_dataset: Nome do Dataset que está sendo concatenado. Se não passar algum nome de dataset as
                             mensagens na tela serão inibidas.
        :return: Dataset tratado e concatenado.
    """
    if nome_dataset != '':
        print(f"=> Concatenando as colunas dos registros (dataset: {nome_dataset})...", end='', flush=True)

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

    if nome_dataset != '':
        print(" OK!")

    return np.array(registros_tratados)


def tratar_texto_tfidf(t, le, x_train, y_train, x_test, y_test, x_val, y_val):
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
        :return: Datasets tratados.
    """
    print("=> Vetorizando o texto com base no TF-IDF...", end='', flush=True)

    # Obs.: Está considerando o y completo
    qtd_classes_y = len(le.classes_)

    # Obs.: Considerando somente o y_train para auxiliar na verificação dos resultados
    qtd_classes_y_train = len(set(y_train))

    # Algumas estatísticas sobre os registros
    # (Fonte: https://machinelearningknowledge.ai/
    # keras-tokenizer-tutorial-with-examples-for-fit_on_texts-texts_to_sequences-texts_to_matrix-sequences_to_matrix)

    # Ordena as palavras pela maior ocorrência no texto
    palavras = list(t.word_counts.items())
    palavras.sort(reverse=True, key=lambda p: p[1])

    msg_log = f"\n\n        ************ Estatísticas dos documentos (Obs.: Somente o X e Y para treino) ************\n"
    msg_log += f"\n==> Quantidade de documentos no treino: {t.document_count}"
    msg_log += f"\n==> Quantidade de Classes identificadas (y completo): {qtd_classes_y}"
    msg_log += f"\n==> Quantidade de Classes identificadas (y_train): {qtd_classes_y_train}"
    msg_log += f"\n==> Quantidade de palavras únicas no treino: {len(t.word_index.items())}"
    msg_log += f"\n\n==> Quantidade de ocorrência das palavras nos documentos no treino (20 palavras mais " \
               f"frequentes): \n{palavras[:20]}"
    msg_log += f"\n\n==> Indice das palavras do treino (primeiras 20 palavras): \n{list(t.word_index.items())[:20]}\n"

    LOGGER.info(msg_log)
    print(msg_log)

    # Codifica o texto com base no TF-IDF (Term Frequency - Inverse Document Frequency) que leva em conta a relevância
    # da palavra no texto
    try:
        cod_x_train = t.texts_to_matrix(x_train, mode='tfidf')
        cod_x_test = t.texts_to_matrix(x_test, mode='tfidf')
        cod_x_val = t.texts_to_matrix(x_val, mode='tfidf')
    except MemoryError as e:
        msg = f"Erro na alocação de memória. Mensagem do Tokenizer: '{e}'."
        LOGGER.error(msg)
        print(f"\n\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    # Codifica as classes com o One-Hot Encoding
    y_train_one_hot = to_categorical(le.transform(y_train), num_classes=qtd_classes_y)
    y_test_one_hot = to_categorical(le.transform(y_test), num_classes=qtd_classes_y)
    y_val_one_hot = to_categorical(le.transform(y_val), num_classes=qtd_classes_y)

    imprimir_shapes("Dimensões dos datasets depois do texto vetorizado", cod_x_train, y_train_one_hot,
                    cod_x_test, y_test_one_hot, cod_x_val, y_val_one_hot)
    LOGGER.info(f"Exemplo do dataset com o texto processado (10 primeiros registros):\n\n{cod_x_train[:10]}\n")

    return qtd_classes_y, cod_x_train, y_train_one_hot, cod_x_test, y_test_one_hot, cod_x_val, y_val_one_hot


def salvar_modelo(t, le, modelo, hist, resultado_teste, tipo_modelo, nome_modelo, caminho, id_exec, nome_experimento,
                  run_id, labels_dataset, info_modelo, decremento_acuracia, parametros_treino, nomes_datasets):
    """
    Salva um modelo no hard drive.
        :param t: Objeto contendo o Tokenizer.
        :param le: Objeto contendo o Label Encoder.
        :param modelo: Objeto com o modelo que será salvo.
        :param hist: Histórico da acurácia no treinamento.
        :param resultado_teste: Resultado do teste com o x_test e y_test.
        :param tipo_modelo: Tipo do modelo que será utilizado para efetuar os testes.
        :param nome_modelo: Nome do modelo. O arquivo criado terá o nome do modelo.
        :param caminho: Caminho onde o modelo será salvo.
        :param nome_experimento: Nome do experimento no MLFlow.
        :param run_id: ID de execução do experimento no MLFlow.
        :param id_exec: ID de execução do treinamento.
        :param labels_dataset: Labels do dataset.
        :param info_modelo: Algumas informações do modelo no treinamento.
        :param decremento_acuracia: Decremento da acurácia que será utilizado como base para o decidir o início do
                                    treino automatizado.
        :param parametros_treino: Parâmetros do treino que serão utilizados no treino automatizado.
        :param nomes_datasets: Nomes dos datasets que serão utilizados no treino automatizado.
    """
    print(f"\n=> Salvando o modelo '{nome_modelo}' na pasta '{caminho}'...\n")
    caminho_modelo = f"{caminho}{nome_modelo}/"

    try:
        makedirs(caminho_modelo, exist_ok=True)
    except PermissionError:
        msg = f"Erro ao criar a pasta '{caminho_modelo}'. Permissão de escrita negada."
        LOGGER.error(msg)
        print(f"\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    LOGGER.info(f">>> Salvando o modelo '{nome_modelo}' em '{caminho_modelo}'")

    if tipo_modelo == "clf_rn_tfidf":
        save(modelo, caminho_modelo)
    else:
        salvar_objeto(caminho_modelo, modelo, nome_modelo + '.pkl')

    # Salva o gráfico da acurácia no treinamento
    gerar_grafico_historico_treino(hist, caminho_modelo)

    # Salva algumas informações do treinamento do modelo num arquivo de texto
    with open(caminho_modelo + "algumas_informacoes_modelo.txt", 'w') as arq:
        arq.write(info_modelo)

    # Gera um CSV com a contabilização das palavras utilizadas no treino
    with open(caminho_modelo + "palavras_utilizadas_contabilizadas.csv", 'w') as arq:
        palavras = list(t.word_counts.items())
        palavras.sort(reverse=True, key=lambda p: p[1])
        total_palavras = sum(list(t.word_counts.values()))

        arq.write("palavra;quantidade;percentual\n")

        for p in palavras[:t.num_words]:
            arq.write(f"{p[0]};{p[1]};{(p[1] / total_palavras) * 100}\n")

    LOGGER.info(f">>> Foram gerados alguns arquivos txt e CSV com informações sobre o modelo na pasta "
                f"'{caminho_modelo}'")

    # Salva os parâmetros do treino que serão utilizados no treino automatizado
    salvar_objeto(caminho_modelo, parametros_treino, "TrainingParams")

    # Salva os parâmetros do treino num arquivo de texto
    with open(caminho_modelo + "TrainingParams.txt", 'w') as arq:
        arq.write(f"{parametros_treino}")

    # Salva os nomes dos datasets que serão utilizados no treino automatizado
    salvar_objeto(caminho_modelo, nomes_datasets, "TrainingDatasetsNames")

    # Salva os nomes dos datasets num arquivo de texto
    with open(caminho_modelo + "TrainingDatasetsNames.txt", 'w') as arq:
        arq.write(f"{nomes_datasets}")

    # Salva algumas informações do modelo treinado
    acuracia_treino = f"{np.mean(hist.history['accuracy']) * 100:.2f}%"
    acuracia_validacao = f"{np.mean(hist.history['val_accuracy']) * 100:.2f}%"
    acuracia_teste = f"{resultado_teste[1] * 100:.2f}%"

    informacoes_modelo = {'experiment_name': nome_experimento, 'run_name': id_exec, 'run_id': run_id,
                          'train_accuracy': acuracia_treino, 'val_accuracy': acuracia_validacao,
                          'test_accuracy': acuracia_teste, 'features': parametros_treino['colunas_selecionadas_x'],
                          'targets': parametros_treino['colunas_selecionadas_y'],
                          'predictable_labels': labels_dataset}
    salvar_objeto(caminho_modelo, informacoes_modelo, "ModelInfo")

    # Salva algumas informações do modelo treinado num arquivo txt
    with open(caminho_modelo + "ModelInfo.txt", 'w') as arq:
        arq.write(f"{informacoes_modelo}")

    # Salva o Tokenizer e o Label Encoder para utilizar nas predições futuras
    salvar_objeto(caminho_modelo, t, "Tokenizer")
    salvar_objeto(caminho_modelo, le, "LabelEncoder")

    # Salva um baseline para comparação com treinos futuros
    baseline = {'acuracia_media_treino': np.mean(hist.history['accuracy']),
                'acuracia_media_validacao': np.mean(hist.history['val_accuracy']),
                'acuracia_teste': resultado_teste[1],
                'decremento_acuracia': decremento_acuracia,
                'labels_dataset': labels_dataset}
    salvar_objeto(caminho_modelo, baseline, "BaselineMetrics")

    # Salva o baseline num arquivo de texto
    with open(caminho_modelo + "BaselineMetrics.txt", 'w') as arq:
        arq.write(f"{baseline}")


def obter_parametros_modelo(cf):
    """
    Obtém os parâmetros do modelo definidos no arquivo de configuração.
        :param cf: Arquivo de configuração importado.
        :return: Tipo e nome do modelo.
    """
    if 'Tipo modelo' in cf.dados_modelo and 'Nome modelo' in cf.dados_modelo:
        tipo_modelo = cf.dados_modelo['Tipo modelo']
        nome_modelo = cf.dados_modelo['Nome modelo']
    else:
        msg = f"Os parâmetros 'Tipo modelo' e/ou 'Nome modelo' não foram encontrados no arquivo de configuaração " \
              f"'{cf.caminho_arquivo_conf}'."
        LOGGER.error(msg)
        print(f"\n\n{msg} Consulte o log de execução ({arquivo_log}) para mais detalhes!\n")
        exit(1)

    return tipo_modelo, nome_modelo
