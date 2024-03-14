# ----------------------------------------------------------------------------------------------------------
# Script para treinamento do modelo
# ----------------------------------------------------------------------------------------------------------
import os
import configs as cf
from datetime import datetime
from time import time, sleep
from utils import LOGGER, obter_parametros_modelo


def treinar(c):
    """
    Rotina de treinamento do modelo.
        :param c: Arquivo de configuração importado.
    """
    # Gera um id de execução para rastreamento do modelo gerado no arquivo de log
    id_exec = datetime.fromtimestamp(int(time())).strftime('%Y-%m-%d_%H%M%Shs')
    msg = f"# ID da execução: {id_exec}"
    print(f"{msg}\n")

    print(f"=> Importando bibliotecas...", end='', flush=True)

    from sklearn.model_selection import train_test_split
    from utils import LOGGER, salvar_modelo, carregar_dataset, concatenar_registros, imprimir_shapes, \
        imprimir_alguns_parametros_utilizados, imprimir_parametros_treino_rn_tfidf
    from rede_neural_tfidf import clf_rn_tfidf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from sklearn import preprocessing
    import mlflow.tensorflow

    print(" OK!")

    LOGGER.info("---------------- APLICAÇÃO INICIADA (Treinamento) ----------------")
    LOGGER.info(msg)

    info_modelo = ""  # Guardará algumas informações da rotina de treinamento para salvar com o modelo
    ts_inicio_treinamento = time()

    # Monta o caminho completo do arquivo
    caminho_arquivo = c.caminho_arquivo + c.nome_arquivo

    # imprimir parâmetros utilizados
    info_modelo += imprimir_alguns_parametros_utilizados(c)

    # Carrega o dataset
    x, y = carregar_dataset(caminho_arquivo, c.separador, c.encoding_arquivo, c.colunas_selecionadas_dataset,
                            c.descartar_registros_vazios, c.colunas_selecionadas_x, c.colunas_selecionadas_y,
                            c.qtd_exemplos)

    # Divide os dados para o treinamento, validação e teste
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=c.percentual_exemplos_teste, shuffle=True,
                                                            random_state=c.trava_randomica)
    except ValueError as e:
        msg = f"O parâmetro 'percentual_exemplos_teste' está inconsistente. Mensagem do 'train_test_split': '{e}'. " \
              f"Programa abortado!"
        LOGGER.error(msg)
        print(f"\n\n{msg} Consulte o log de execução ({c.arquivo_log}) para mais detalhes!\n")
        exit(1)

    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=True,
                                                    random_state=c.trava_randomica)

    del x

    info_modelo += imprimir_shapes("Dimensões dos datasets depois do split", x_train, y_train, x_val, y_val,
                                   x_test, y_test)
    msg = f"Exemplo do dataset carregado (10 primeiros registros):\n\n{x_train[:10]}\n"
    LOGGER.info(msg)
    info_modelo += "\n" + msg
    print(msg)

    # Guarda os labels do dataset
    labels_dataset = list(set(y))

    # Transforma os campos dos datasets numa string única
    x_train = concatenar_registros(x_train, c.algumas_stop_words, remover_stop_words=c.remover_stop_words,
                                   nome_dataset="x_train")
    x_test = concatenar_registros(x_test, c.algumas_stop_words, remover_stop_words=c.remover_stop_words,
                                  nome_dataset="x_test")
    x_val = concatenar_registros(x_val, c.algumas_stop_words, remover_stop_words=c.remover_stop_words,
                                 nome_dataset="x_val")

    msg = f"Exemplo do dataset limpo (caracteres especiais) e concatenado (10 primeiros registros " \
          f"remover_stop_words = {c.remover_stop_words}):\n\n{x_train[:10]}\n"
    LOGGER.info(msg)
    info_modelo += "\n" + msg
    print(f"\n{msg}")

    print("=> Instanciando e dando fit() no Tokenizer e no Label Encoder...", end='', flush=True)

    # Tokenização dos documentos (Será utilizado somente o X_train para que a fase do treinamento não seja contaminada
    # com dados das fases de validação e teste)
    t = Tokenizer(oov_token='***UNK***', num_words=c.qtd_palavras_considerar)
    t.fit_on_texts(x_train)

    # Transforma os rótulos em números
    le = preprocessing.LabelEncoder()

    # Obs.: Tem que fazer o fit com o y inteiro, para que o transform retorne os mesmos números para os labels y_train,
    # y_test, y_val, y_teste_adicional, correspondentes
    le.fit(y)

    print(" OK!")

    # imprimir parâmetros utilizados
    info_modelo += imprimir_parametros_treino_rn_tfidf(c)

    tipo_modelo, nome_modelo = obter_parametros_modelo(c)

    print("=> Configurando o experimento no MLflow...", end='', flush=True)

    # Configura onde o MLflow vai salvar os experimentos.
    if os.environ.get('MLFLOW_TRACKING_URI') is None:
        mlflow.set_tracking_uri('sqlite:///teste_mlflow.db')

    # Configura o experimento no MLFlow
    mlflow.set_experiment(experiment_name=c.nome_experimento)
    desc = "Experimentos para construção de um modelo para classificar se um tweet é bullying. E se for, qual o tipo."
    run_id = mlflow.start_run(run_name=id_exec, description=desc).info.run_id  # Para usar na verificação de resultados
    mlflow.tensorflow.autolog(every_n_iter=1)
    tags = {"Projeto": "Classificador de Cyberbullying Tweets", "team": "TESTES"}
    mlflow.set_tags(tags)

    # Define os parâmetros do treino que serão utilizados no treino automatizado
    parametros_treino = {'algumas_stop_words': c.algumas_stop_words, 'remover_stop_words': c.remover_stop_words,
                         'qtd_palavras_considerar': c.qtd_palavras_considerar, 'qtd_neuronios': c.qtd_neuronios,
                         'qtd_exemplos': c.qtd_exemplos, 'aplicar_dropout': c.aplicar_dropout,
                         'perc_dropout': c.perc_dropout, 'batch_size': c.batch_size, 'epochs': c.epochs,
                         'decremento_acuracia': c.decremento_acuracia,
                         'colunas_selecionadas_x': c.colunas_selecionadas_x,
                         'colunas_selecionadas_y': c.colunas_selecionadas_y,
                         'percentual_exemplos_teste': c.percentual_exemplos_teste, 'separador': c.separador,
                         'encoding_arquivo': c.encoding_arquivo}

    print(" OK!")

    # Treina o modelo
    modelo, hist, resultado_teste = clf_rn_tfidf(t, le, x_train, y_train, x_test, y_test, x_val, y_val)
    msg = f"Histórico do treinamento: {hist.history}"
    LOGGER.info(msg)
    info_modelo += "\n" + msg

    del y, x_train, y_train, x_test, y_test, x_val, y_val

    # Calcula o tempo de duração do treinamento em horas, minutos e segundos
    ts_fim_treinamento = time()
    tempo_duracao = ts_fim_treinamento - ts_inicio_treinamento
    horas = int(tempo_duracao // 3600)
    tempo_duracao = tempo_duracao - horas * 3600
    minutos = int(tempo_duracao // 60)
    tempo_duracao = tempo_duracao - minutos * 60
    segundos = int(tempo_duracao)

    msg = f"=> Duração do treinamento: {horas} hora(s), {minutos} minuto(s) e {segundos} segundo(s)"
    LOGGER.info(msg)
    print(msg)

    salvar_modelo(t, le, modelo, hist, resultado_teste, tipo_modelo, nome_modelo, c.caminho_modelos_salvos, id_exec,
                  c.nome_experimento, run_id, labels_dataset, info_modelo, c.decremento_acuracia, parametros_treino,
                  c.nomes_datasets)

    # Salva alguns artefatos no MLFlow (Obs.: foram criados na rotina 'salvar_modelo')
    caminho_modelo = f"{c.caminho_modelos_salvos}{nome_modelo}/"
    mlflow.log_artifact(caminho_modelo + "LabelEncoder.pkl")
    mlflow.log_artifact(caminho_modelo + "Tokenizer.pkl")
    mlflow.log_artifact(caminho_modelo + "BaselineMetrics.pkl")
    mlflow.log_artifact(caminho_modelo + "BaselineMetrics.txt")
    mlflow.log_artifact(caminho_modelo + "ModelInfo.pkl")
    mlflow.log_artifact(caminho_modelo + "ModelInfo.txt")
    mlflow.log_artifact(caminho_modelo + "TrainingParams.pkl")
    mlflow.log_artifact(caminho_modelo + "TrainingParams.txt")
    mlflow.log_artifact(caminho_modelo + "TrainingDatasetsNames.pkl")
    mlflow.log_artifact(caminho_modelo + "TrainingDatasetsNames.txt")
    mlflow.log_artifact(caminho_modelo + "algumas_informacoes_modelo.txt")
    mlflow.log_artifact(caminho_modelo + "grafico_treinamento.jpeg")
    mlflow.log_artifact(caminho_modelo + "palavras_utilizadas_contabilizadas.csv")

    # Finaliza o experimento no MLFlow
    mlflow.end_run()


if __name__ == "__main__":
    print("\n\n\033[1m\033[92m")
    print("                    ======================================================================================")
    print("                    |               C L A S S I F I C A D O R   C Y B E R B U L L Y I N G                |")
    print("                    ======================================================================================")
    print("\033[0m")

    print(f"\n   ATENÇÃO: Este é o backend do 'CLASSIFICADOR' responsável pelo treinamento do modelo. Já existem "
          f"vários \n   parâmetros pré-configurados no arquivo \033[96m'{cf.caminho_arquivo_conf}'\033[0m.\n"
          f"   Caso queira ou necessite personalizar: Aborte a execução; edite o arquivo; salve as alterações e "
          f"execute o \n   programa novamente!")
    print(f"\n\n\033[1m   *** INFO: Consulte os detalhes da execução em '{cf.arquivo_log}'\n\n\033[0m")

    # Somente para chamar atenção para a mensagem inicial do classificador
    print(f"=> Iniciando o classificador", end='', flush=True)

    for _ in range(5):
        sleep(1.1)
        print(".", end='', flush=True)

    print("\n\n")

    # Treina os modelos
    treinar(cf)

    print("\n\033[1m\033[92m=> Rotina de treinamento finalizada com sucesso!\033[0m\n")
    LOGGER.info("---------------- APLICAÇÃO FINALIZADA ----------------")
    exit(0)
