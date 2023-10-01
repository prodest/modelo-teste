# ----------------------------------------------------------------------------------------------------
# Implementação das funcionalidades para a publicação do modelo via API
# ----------------------------------------------------------------------------------------------------
import logging
import os
import numpy as np
from mllibprodest.interfaces import ModelPublicationInterfaceCLF
from sklearn.metrics import accuracy_score
from .utils import *


""" Desativa as mensagens de warning e info do TensorFlow """
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

""" Desativa o uso de GPU """
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ModeloCLF(ModelPublicationInterfaceCLF):
    def __init__(self, model_name: str, model_provider_name: str):
        """
        Classe para publicação dos modelos de classificação.
        """
        self.__logger = self.make_log(model_name + "_pub.log")
        self.__model_name = model_name
        self.__artifacts_destination_path = "temp_area"
        self.__model_provider_name = model_provider_name

        # Obtém modelo, info, configurações e os artefatos necessários
        self.__modelo = self.load_model(self.__model_name, self.__model_provider_name, self.__artifacts_destination_path)
        self.__info_modelo = self.convert_artifact_to_object(self.__model_name, "ModelInfo.pkl", self.__artifacts_destination_path)
        self.__parametros_treino = self.convert_artifact_to_object(self.__model_name, 
                                                                   "TrainingParams.pkl",
                                                                   self.__artifacts_destination_path)
        self.__t = self.convert_artifact_to_object(self.__model_name, "Tokenizer.pkl", self.__artifacts_destination_path)
        self.__le = self.convert_artifact_to_object(self.__model_name, "LabelEncoder.pkl", self.__artifacts_destination_path)

    def get_model_name(self) -> str:
        return self.__model_name

    def get_model_provider_name(self) -> str:
        return self.__model_provider_name

    def get_model_info(self) -> dict:
        if type(self.__info_modelo) is dict:
            return self.__info_modelo
        else:
            return {'Model name': self.__model_name}

    def predict(self, dataset) -> list | str:
        # Separa os campos por ponto-e-virgula, se for o caso, e transforma num array
        # Obs.: Este array tem que ser nos mesmos moldes de quando é gerado via arquivo (2 dimensões)
        if type(dataset) is list:
            if len(dataset) > 0:
                try:
                    dados_lista = [d.split(";") for d in dataset]
                except AttributeError as e:
                    msg = f"Não foi possível realizar o split dos dados. Mensagem: {e}."
                    self.__logger.error(msg)
                    return msg

                # Verifica se todas as listas dentro da lista de dados possuem o mesmo tamanho, para evitar erros
                # na criação do array
                tamanho_padrao = len(dados_lista[0])

                for e in dados_lista:
                    if len(e) != tamanho_padrao:
                        msg = f"A Lista possui dados com tamanho inválido. Obs.: Se estiver passando uma lista de " \
                              f"strings que possuem o separador ';', verifique se a quantidade de campos que serão " \
                              f"gerados ao realizar o split será igual para todas as strings."
                        self.__logger.error(msg)
                        return msg

                x = np.array(dados_lista)
            else:
                msg = f"Foi passada uma lista vazia."
                self.__logger.error(msg)
                return msg
        else:
            msg = f"Esta função espera uma lista, mas foi passado um objeto do tipo '{type(dataset)}'."
            self.__logger.error(msg)
            return msg

        x = concatenar_registros(x, algumas_stop_words=self.__parametros_treino['algumas_stop_words'],
                                 remover_stop_words=self.__parametros_treino['remover_stop_words'])

        # Vetoriza e faz a predição
        try:
            x = self.__t.texts_to_matrix(x, mode='tfidf')
        except MemoryError as e:
            msg = f"Não foi possível alocar memória. Mensagem do Tokenizer: '{e}'."
            self.__logger.error(msg)
            return msg

        # faz a predição usando o modelo
        ypred = self.__modelo.predict(x)

        # Obtém a classe que foi inferida
        resultado_argmax = np.argmax(ypred, axis=1)

        # Recupera as classes no formato string
        ypred_inverse = list(self.__le.inverse_transform(resultado_argmax))

        return ypred_inverse

    def evaluate(self, data_features: list, data_targets: list) -> dict | str:
        ypred = self.predict(data_features)

        try:
            metricas = {'Accuracy': f"{accuracy_score(data_targets, ypred) * 100:.2f}%"}
        except ValueError as e:
            msg = f"Não foi possível avaliar as métricas. Mensagem do accuracy_score: '{e}'."
            self.__logger.error(msg)
            return msg

        return metricas

    def get_feedback(self, y_pred: list, y_true: list) -> dict | str:
        metricas = {}

        try:
            metricas['Acuracia'] = f"{accuracy_score(y_true, y_pred) * 100:.2f}%"
            metricas['qtd_acertos'] = int(accuracy_score(y_true, y_pred, normalize=False))
        except ValueError as e:
            msg = f"Não foi possível avaliar as métricas. Mensagem do accuracy_score: '{e}'."
            self.__logger.error(msg)
            return msg

        metricas['qtd_labels_avaliados'] = len(y_true)
        metricas['qtd_erros'] = metricas['qtd_labels_avaliados'] - metricas['qtd_acertos']

        # Verifica se tem algum label, informado pelos usuários, que não participou do treinamento e contabiliza a
        # ocorrência dos mesmos
        if 'predictable_labels' in self.__info_modelo:
            tipos_label_y_true = set(y_true)
            labels_ausentes = []

            for label in tipos_label_y_true:
                if label not in self.__info_modelo['predictable_labels']:
                    labels_ausentes.append(label)

            if labels_ausentes:
                metricas['labels_ausentes'] = labels_ausentes
                ocor_labels_ausentes = []
                total_ocorrencia_labels_ausentes = 0

                for la in labels_ausentes:
                    qtd = y_true.count(la)
                    ocor_labels_ausentes.append((la, qtd))
                    total_ocorrencia_labels_ausentes += qtd

                metricas['ocorrencia_labels_ausentes'] = sorted(ocor_labels_ausentes, key=lambda x: x[1], reverse=True)
                metricas['total_ocorrencia_labels_ausentes'] = total_ocorrencia_labels_ausentes
                perc = f"{(total_ocorrencia_labels_ausentes / metricas['qtd_erros']) * 100:.2f}%"
                metricas['percentual_erros_labels_ausentes'] = perc

        return metricas
