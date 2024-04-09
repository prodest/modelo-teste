# ----------------------------------------------------------------------------------------------------
# Implementação das funcionalidades para o treino automatizado para criar uma nova versão do
# modelo que está em produção.
#
# ATENÇÃO: O modelo em produção não será atualizado automaticamente. Somente será avaliado e informado
#          que existe a possibilidade de melhorá-lo. O cientista de dados será responsável por fazer
#          testes e validar se o modelo, de fato, tem condições de evoluir para uma nova versão.
# ----------------------------------------------------------------------------------------------------
import os
from logging import getLogger, FATAL
import mlflow
import numpy as np
import pandas as pd
from mllibprodest.interfaces import ModelPublicationInterfaceRETRAIN
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .utils import concatenar_registros, predizer_tfidf, construir_modelo_rn, gerar_grafico_historico_treino, \
    tratar_texto_tfidf

""" Desativa as mensagens de warning e info do TensorFlow """
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
getLogger('tensorflow').setLevel(FATAL)

""" Desativa o uso de GPU """
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ModeloRETRAIN(ModelPublicationInterfaceRETRAIN):
    def __init__(self, model_name: str, model_provider_name: str, experiment_name: str, dataset_provider_name: str):
        self.__logger_name = model_name + "_retrain.log"
        self.__logger = self.make_log(self.__logger_name)

        # Definição dos atributos necessários para a implementação dos métodos get
        self.__model_name = model_name
        self.__model_provider_name = model_provider_name
        self.__experiment_name = experiment_name
        self.__dataset_provider_name = dataset_provider_name

    def get_model_name(self) -> str:
        return self.__model_name

    def get_model_provider_name(self) -> str:
        return self.__model_provider_name

    def get_experiment_name(self) -> str:
        return self.__experiment_name

    def get_dataset_provider_name(self) -> str:
        return self.__dataset_provider_name

    def evaluate(self, model, datasets, baseline_metrics, training_params, artifacts_path="temp_area",
                 batch_size=100000) -> (bool, dict):
        try:
            # Obtém os datasets
            x = datasets['features']
            y = datasets['targets']
        except KeyError as e:
            msg = f"O dataset {e} não foi encontrado no dicionário de datasets."
            self.__logger.error(msg)
            print(f"\n\n{msg}")
            exit(1)

        try:
            # Obtém os parâmetros necessários
            separador = training_params['separador']
            encoding_arquivo = training_params['encoding_arquivo']
            algumas_stop_words = training_params['algumas_stop_words']
            remover_stop_words = training_params['remover_stop_words']
            colunas_selecionadas_x = training_params['colunas_selecionadas_x']
            colunas_selecionadas_y = training_params['colunas_selecionadas_y']
        except KeyError as e:
            msg = f"O parâmetro {e} não foi fornecido no dicionário de parâmetros."
            self.__logger.error(msg)
            print(f"\n\n{msg}")
            exit(1)

        # Lê os datasets e depois transforma em um numpy array
        x = pd.read_csv(x, dtype=str, sep=separador, encoding=encoding_arquivo)
        y = pd.read_csv(y, dtype=str, sep=separador, encoding=encoding_arquivo)

        # Garante que as colunas de features e target do dataset sejam as mesmas utilizadas no treino
        try:
            x = x[colunas_selecionadas_x]
            y = y[colunas_selecionadas_y]
        except KeyError as e:
            msg = f"Coluna(s) em 'colunas_selecionadas_x' ou 'colunas_selecionadas_y' não encontrada(s) no dataset " \
                  f"(erro Pandas: {e})."
            self.__logger.error(msg)
            print(f"\n\n{msg}")
            exit(1)

        x = np.array(x)
        y = np.array(y).reshape(-1)

        # Obtém os artefatos necessários
        t = self.convert_artifact_to_object(self.__model_name, "Tokenizer.pkl", artifacts_path)
        le = self.convert_artifact_to_object(self.__model_name, "LabelEncoder.pkl", artifacts_path)

        x = concatenar_registros(x, algumas_stop_words=algumas_stop_words, remover_stop_words=remover_stop_words)

        tamanho_dataset = len(x)
        lotes = self.generate_batch_indices(tamanho_dataset, batch_size=batch_size)
        ypred = []
        msg = f"=> Fazendo a avaliação das métricas (tamanho do dataset: {tamanho_dataset}) lote(s)..."
        self.__logger.info(f"{msg}")
        print(f"\n{msg}")

        for inicio, fim in lotes:
            msg = f"   {inicio+1} a {fim}"
            self.__logger.info(msg)
            print(f"{msg}; ", end="", flush=True)

            # Prediz cada parte separadamente e concatena com as anteriores
            ypred += predizer_tfidf(t, le, model, x[inicio:fim], self.__logger_name)

        # Apura as métricas e informações adicionais
        qtd_labels = len(y)
        qtd_acertos = accuracy_score(y, ypred, normalize=False)
        qtd_erros = qtd_labels - qtd_acertos
        acuracia = accuracy_score(y, ypred)
        acuracia_percentual = f"{acuracia * 100:.4f}%"

        # Apura se existem labels do dataset y que não estavam no dataset utilizado no processo de treinamento.
        # Caso exista algum, é mandatório que o modelo seja retreinado para conseguir predizer tal label.
        labels_eval = set(y)
        labels_ausentes = [label for label in labels_eval if label not in baseline_metrics['labels_dataset']]

        necessita_retreinar = False
        info = {'evaluator_model_run_id': model.metadata.run_id}
        limiar = baseline_metrics['acuracia_media_validacao'] - baseline_metrics['decremento_acuracia']
        limiar_percentual = f"{limiar * 100:.4f}%"

        if acuracia <= limiar:
            necessita_retreinar = True
            info['acuracia_avaliada'] = acuracia_percentual
            info['info_acerto'] = f"O modelo avaliado acertou {qtd_acertos} de {qtd_labels} labels (errou " \
                                  f"{qtd_erros})."
            info['info_adicional_acuracia'] = f"A acurácia avaliada foi menor do que a acurácia do modelo em " \
                                              f"produção em {baseline_metrics['decremento_acuracia'] * 100:.4f}% " \
                                              f"(Limiar: acurácia <= {limiar_percentual})"

        if labels_ausentes:
            y = list(y)
            labels_ausentes_cont = []
            exemplos_count = 0

            # Contabiliza a quantidade de elementos para cada label ausente e apura total de elementos
            for label in labels_ausentes:
                qtd = y.count(label)
                exemplos_count += qtd
                labels_ausentes_cont.append((label, qtd))

            # Ordena por quantidade de elementos por label em ordem decrescente
            labels_ausentes_cont.sort(reverse=True, key=lambda p: p[1])

            necessita_retreinar = True
            info['labels_ausentes'] = labels_ausentes_cont
            info['info_adicional_labels'] = f"Existem um ou mais labels no dataset avaliado que não fizeram parte do " \
                                            f"treinamento do modelo que está em produção. Quantidade de labels fora " \
                                            f"do treino: {len(labels_ausentes)}. Quantidade de elementos do dataset " \
                                            f"que foram impactados pela falta dos labels: {exemplos_count}."

        if not necessita_retreinar:
            msg = f"O modelo com o 'run_id' {model.metadata.run_id} não precisa ser retreinado. Ele acertou " \
                  f"{qtd_acertos} de {qtd_labels} labels (errou {qtd_erros}), obtendo uma acurácia " \
                  f"de {acuracia_percentual} (Limiar: acurácia <= {limiar_percentual})"
            self.__logger.info(f"{msg}")
            print(f"\n\n{msg}")
        else:
            msg = f"O modelo precisa ser retreinado. Detalhes: {info}"
            self.__logger.info(f"{msg}")
            print(f"\n\n{msg}")

        return necessita_retreinar, info

    def retrain(self, production_model_name, production_params, experiment_name, datasets, reasons):
        msg = f"=> Um novo modelo está sendo treinado {production_model_name}..."
        self.__logger.info(f"{msg}")
        print(f"\n\n{msg}\n")
        caminho_artefatos = "temp_area/" + production_model_name + "/"

        datasets_names = self.load_production_datasets_names(production_model_name)

        try:
            algumas_stop_words = production_params['algumas_stop_words']
            remover_stop_words = production_params['remover_stop_words']
            qtd_palavras_considerar = production_params['qtd_palavras_considerar']
            qtd_exemplos = production_params['qtd_exemplos']
            qtd_neuronios = production_params['qtd_neuronios']
            aplicar_dropout = production_params['aplicar_dropout']
            perc_dropout = production_params['perc_dropout']
            batch_size = production_params['batch_size']
            epochs = production_params['epochs']
            decremento_acuracia = production_params['decremento_acuracia']
            percentual_exemplos_teste = production_params['percentual_exemplos_teste']
            separador = production_params['separador']
            encoding_arquivo = production_params['encoding_arquivo']
            colunas_selecionadas_x = production_params['colunas_selecionadas_x']
            colunas_selecionadas_y = production_params['colunas_selecionadas_y']
        except KeyError as e:
            msg = f"O parâmetro {e} não foi fornecido no dicionário de parâmetros."
            self.__logger.error(msg)
            print(f"\n\n{msg}")
            exit(1)

        # Lê os datasets e depois transforma em um numpy array
        x = pd.read_csv(datasets['features'], dtype=str, sep=separador, encoding=encoding_arquivo)
        y = pd.read_csv(datasets['targets'], dtype=str, sep=separador, encoding=encoding_arquivo)

        # Utiliza a mesma quantidade de exemplos usada no treino
        x = x[:qtd_exemplos]
        y = y[:qtd_exemplos]

        # Garante que as colunas de features e target do dataset sejam as mesmas utilizadas no treino
        try:
            x = x[colunas_selecionadas_x]
            y = y[colunas_selecionadas_y]
        except KeyError as e:
            msg = f"Coluna(s) em 'colunas_selecionadas_x' ou 'colunas_selecionadas_y' não encontrada(s) no dataset " \
                  f"(erro Pandas: {e})."
            self.__logger.error(msg)
            print(f"\n\n{msg}")
            exit(1)

        x = np.array(x)
        y = np.array(y).reshape(-1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percentual_exemplos_teste, shuffle=True,
                                                            random_state=1)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=1)

        del x

        # Transforma os campos dos datasets numa string única
        x_train = concatenar_registros(x_train, algumas_stop_words, remover_stop_words)
        x_test = concatenar_registros(x_test, algumas_stop_words, remover_stop_words)
        x_val = concatenar_registros(x_val, algumas_stop_words, remover_stop_words)

        # Tokenização dos documentos (Será utilizado somente o X_train para que a fase do treinamento não seja
        # contaminada com dados das fases de validação e teste)
        t = Tokenizer(oov_token='***UNK***', num_words=qtd_palavras_considerar)
        t.fit_on_texts(x_train)

        # Transforma os rótulos em números
        le = preprocessing.LabelEncoder()

        # Obs.: Tem que fazer o fit com o y inteiro, para que o transform retorne os mesmos números para os labels
        # y_train, y_test, y_val, correspondentes
        le.fit(y)

        labels_dataset = list(set(y))
        del y

        # Vetoriza o dataset utilizando TF-IDF e One-Hot Encoder
        qtd_classes_y, x_train, y_train, x_test, y_test, x_val, y_val = tratar_texto_tfidf(t, le, x_train, y_train,
                                                                                           x_test, y_test, x_val, y_val,
                                                                                           self.__logger_name)
        # Prepara as dimensões iniciais para a rede
        input_shape = x_train[0].shape

        # Tem que ser do y porque o le.fit() foi feito com o y inteiro. Se colocar o shape do y_train dará erro de shape
        # no treinamento
        output_dim = qtd_classes_y

        # Configura onde o MLflow vai salvar os experimentos.
        if os.environ.get('MLFLOW_TRACKING_URI') is None:
            mlflow.set_tracking_uri('sqlite:///teste_mlflow.db')

        # Configura o experimento no MLFlow
        mlflow.set_experiment(experiment_name=experiment_name)
        run_name = "self_training"
        desc = "Experimentos automatizados do modelo para classificar se um tweet é bullying. E se for, qual o tipo."
        run_id = mlflow.start_run(run_name=run_name, description=desc).info.run_id
        mlflow.tensorflow.autolog(every_n_iter=1)
        tags = {"Projeto": "Classificador de Cyberbullying Tweets", "team": "TESTES"}
        mlflow.set_tags(tags)

        # Constrói o modelo
        modelo = construir_modelo_rn(input_shape, output_dim, neuronios=qtd_neuronios, aplicar_dropout=aplicar_dropout,
                                     perc_dropout=perc_dropout)

        # Treina o modelo e guarda o histórico do treino
        hist = modelo.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

        resultado_teste = modelo.evaluate(x_test, y_test)

        # Salva o motivo do treino num arquivo txt
        with open(caminho_artefatos + "ReasonsForTraining.txt", 'w') as arq:
            arq.write(f"{reasons}")

        # Salva algumas informações do modelo treinado
        acuracia_treino = f"{np.mean(hist.history['accuracy']) * 100:.2f}%"
        acuracia_validacao = f"{np.mean(hist.history['val_accuracy']) * 100:.2f}%"
        acuracia_teste = f"{resultado_teste[1] * 100:.2f}%"

        informacoes_modelo = {'experiment_name': experiment_name, 'run_name': run_name, 'run_id': run_id,
                              'train_accuracy': acuracia_treino, 'val_accuracy': acuracia_validacao,
                              'test_accuracy': acuracia_teste, 'features': colunas_selecionadas_x,
                              'targets': colunas_selecionadas_y, 'predictable_labels': labels_dataset}
        self.convert_artifact_to_pickle(self.__model_name, informacoes_modelo, "ModelInfo.pkl", "temp_area")

        # Salva algumas informações do modelo treinado num arquivo txt
        with open(caminho_artefatos + "ModelInfo.txt", 'w') as arq:
            arq.write(f"{informacoes_modelo}")

        # cria um baseline para comparação com treinos futuros
        baseline = {'acuracia_media_treino': np.mean(hist.history['accuracy']),
                    'acuracia_media_validacao': np.mean(hist.history['val_accuracy']),
                    'acuracia_teste': resultado_teste[1],
                    'decremento_acuracia': decremento_acuracia,
                    'labels_dataset': labels_dataset}

        # Salva o baseline num arquivo de texto
        with open(caminho_artefatos + "BaselineMetrics.txt", 'w') as arq:
            arq.write(f"{baseline}")

        # Salva os parâmetros de produção num arquivo de texto
        with open(caminho_artefatos + "TrainingParams.txt", 'w') as arq:
            arq.write(f"{production_params}")

        # Salva os nomes dos datasets num arquivo de texto
        with open(caminho_artefatos + "TrainingDatasetsNames.txt", 'w') as arq:
            arq.write(f"{datasets_names}")

        # Salva o histórico do treinamento num arquivo de texto
        with open(caminho_artefatos + "TrainingHistory.txt", 'w') as arq:
            arq.write(f"{hist.history}")

        # Gera um gráfico do treinamento
        gerar_grafico_historico_treino(hist, caminho_artefatos)

        # Salva alguns artefatos para mandar para o MLflow
        self.convert_artifact_to_pickle(self.__model_name, baseline, "BaselineMetrics.pkl", "temp_area")
        self.convert_artifact_to_pickle(self.__model_name, le, "LabelEncoder.pkl", "temp_area")
        self.convert_artifact_to_pickle(self.__model_name, t, "Tokenizer.pkl", "temp_area")
        self.convert_artifact_to_pickle(self.__model_name, production_params, "TrainingParams.pkl", "temp_area")
        self.convert_artifact_to_pickle(self.__model_name, datasets_names, "TrainingDatasetsNames.pkl", "temp_area")

        # Envia os artefatos gerados para o MLFlow
        mlflow.log_artifact(caminho_artefatos + "BaselineMetrics.pkl")
        mlflow.log_artifact(caminho_artefatos + "BaselineMetrics.txt")
        mlflow.log_artifact(caminho_artefatos + "LabelEncoder.pkl")
        mlflow.log_artifact(caminho_artefatos + "Tokenizer.pkl")
        mlflow.log_artifact(caminho_artefatos + "TrainingParams.pkl")
        mlflow.log_artifact(caminho_artefatos + "TrainingParams.txt")
        mlflow.log_artifact(caminho_artefatos + "TrainingDatasetsNames.pkl")
        mlflow.log_artifact(caminho_artefatos + "TrainingDatasetsNames.txt")
        mlflow.log_artifact(caminho_artefatos + "ModelInfo.pkl")
        mlflow.log_artifact(caminho_artefatos + "ModelInfo.txt")
        mlflow.log_artifact(caminho_artefatos + "TrainingChart.jpeg")
        mlflow.log_artifact(caminho_artefatos + "TrainingHistory.txt")
        mlflow.log_artifact(caminho_artefatos + "ReasonsForTraining.txt")

        # Finaliza o experimento no MLFlow
        mlflow.end_run()

        msg = f"Modelo treinado com sucesso, entretanto precisa ser validado e registrado manualmente! Dados do " \
              f"modelo novo: {informacoes_modelo}"
        self.__logger.info(f"{msg}")
        print(f"\n\n{msg}")
