# ----------------------------------------------------------------------------------------------------------
# Contém as configurações do modelo
#
# ATENÇÃO: Tenha cuidado ao editar este arquivo. Parâmetros incorretos causarão erros na execução
#          do programa! Todos os parâmetros estão comentados com o intuito de ajudar na escolha.
# ----------------------------------------------------------------------------------------------------------
""" Diminua este valor caso esteja extrapolando o limite de memória/CPU disponíveis no seu computador """
# Quantidade de exemplos (linhas do dataset) para rodar o modelo.
qtd_exemplos = 50000


""" Caminhos utilizados """

# Caminho base para composição dos outros caminhos
caminho_base = "./"

# Caminho padrão para este arquivo de configuração
caminho_arquivo_conf = caminho_base + "configs.py"

# Gravação de logs do programa
arquivo_log = caminho_base + "logs/train.log"

# Caminho para persistência dos modelos treinados
caminho_modelos_salvos = caminho_base + "saidas/modelos_salvos/"

# Caminho onde se encontram os datasets
caminho_arquivo = caminho_base + "datasets/"


""" Parâmetros para pré-processamento da base de dados e testes"""

# Informe o nome do arquivo de dados CSV que será utilizado para treino e testes
nome_arquivo = "cyberbullying_tweets_some_cleaning.csv"

# Guarda os nomes dos datasets para utilizar no treino automatizado
nomes_datasets = {'features': 'datasets/features_cyberbullying_tweets.csv',
                  'targets': 'datasets/targets_cyberbullying_tweets.csv'}

# Separador dos campos do arquivo de dados CSV
separador = ','

# Encode do arquivo CSV (evita a apresentação de caracteres estranhos no arquivo!)
encoding_arquivo = 'utf-8'

# Percentual de exemplos que serão utilizados para teste na fase de treinamento. Obs.: 0.002 = 0.2%, 0.10 = 10%, etc.
percentual_exemplos_teste = 0.30

# Quantidade de palavras (mais frequentes) que serão consideradas para construir o vocabulário utilizando o TF-IDF
# Se quiser que todas as palavras sejam consideradas, informe como "None" (sem aspas)
qtd_palavras_considerar = None

# Informe as colunas que serão utilizadas no treino e teste
colunas_selecionadas_dataset = ['tweet_text', 'cyberbullying_type']

# Features (características utilizadas para inferir o target)
colunas_selecionadas_x = ['tweet_text']

# Target (alvo pretendido para cada linha do arquivo de dados CSV)
colunas_selecionadas_y = ['cyberbullying_type']

# Define se os registros que possuem algum valor vazio devem ser descartados ou não. Caso seja definido para False,
# os registros vazios serão preenchidos com o valor padrão 'nanvazio'. Pense bem nas consequências ao escolher False!
descartar_registros_vazios = False

# Se esta opção for True, todas as palavras da lista 'algumas_stop_words' serão removidas no pré-processamento do texto
remover_stop_words = True

# Obs.: Coloque somente palavras em minúsculo (Fonte: https://gist.github.com/sebleier/554280, com algumas adaptações)
algumas_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                      'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'm',
                      'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                      'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
                      'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                      'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                      'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                      'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                      'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'http', 'https',
                      'amp']

# Decremento da acurácia que será utilizado como base para o decidir o início do treino automatizado. Caso a acurácia
# do treino automatizado menos o 'decremento_acuracia' seja inferior à acurácia do modelo registrado no MLflow, o treino
# automatizado será iniciado.
decremento_acuracia = 0.02

""" Parâmetros para definição do tipo e nome do modelo """

# Este parâmetro serve para, caso deseje, salvar diferentes modelos (sem sobrescrevê-los) e depois utilizá-los nas
# rotinas do programa
dados_modelo = {'Tipo modelo': "clf_rn_tfidf",
                'Nome modelo': "CLF_CYBER_BULLYING_TWEETS"
                }

nome_experimento = 'CLF-Cyberbullying_Tweets'

""" Parâmetros para configuração do modelo 'clf_rn_tfidf' """

# Altere para True caso o modelo esteja dando overfitting, ou seja, a acurácia esteja boa no teinamento mas ruim no
# teste
aplicar_dropout = False

# Percentual de neurônios que serão mortos quando o parâmetro 'aplicar_dropout' for True. Obs.: 0.4=40%, 0.1=10%, etc.
perc_dropout = 0.4

# Quanto maior a quantidade de neurônios utilizados no modelo mais recursos computacionais serão necessários, por outro
# lado, se este valor for muito baixo a capacidade de aprendizagem do modelo diminui
qtd_neuronios = 5

# Tamanho dos lotes que serão treinados simultâneamente. Quanto maior o tamanho do lote, mais recursos computacionais
# serão necessários
batch_size = 32

# Quantidade de iterações utilizadas para cada treinamento. Quanto maior este valor, mais demorado será o treinamento e
# maior a chance de dar overfitting, porém, se for um valor muito baixo, pode ser que o modelo não tenha tempo de
# aprender
epochs = 10

# Seed da função randômica de embaralhamento da base de dados para escolha dos exemplos para treino, teste e validação.
# Este valor é utilizado para manter a reproducibilidade dos testes, pois assim, sempre os mesmos exemplos serão
# escolhidos a cada rodada. Se quiser que o conjunto de exemplos escolhidos mude, informe um número inteiro no
# parâmetro 'trava_randomica', porém não é aconselhável ficar mudando de valor entre as mesmas rodadas de testes
trava_randomica = 1


""" Parâmetros para testes do modelo """

# Tamanho do lote que será utilizado para dividir o arquivo CSV para evitar estouro de memória
tamanho_lote = 50000

# ***************************** NÃO ALTERAR OS PARÂMETROS ABAIXO!!! *****************************

""" Desativa as mensagens de warning e info do TensorFlow """
if True:  # If somente para tirar o aviso de importação fora do topo do arquivo
    import os
    import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

""" Desativa o uso de GPU """
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
