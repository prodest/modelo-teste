# ----------------------------------------------------------------------------------------------------
# Este teste verifica se a implementação das interfaces está atendendo à algumas premissas
# necessárias para publicação do modelo.
# ----------------------------------------------------------------------------------------------------
import sys
from logging import getLogger
from mllibprodest.utils import validate_params
from mllibprodest.validators.test import Test


if __name__ == "__main__":
    """
    Se você estiver rodando o MLflow localmente, execute o script e informe o caminho completo da pasta 'mlruns'
    através do parâmetro '--mlruns_path=<caminho completo para a pasta mlruns>'. A pasta 'mlruns' é criada na pasta 
    local onde o servidor do MLflow, utilizado para registrar o modelo, foi iniciado.

    Uso: python test_retrain.py --mlruns_path="caminho completo para a pasta mlruns"
    """
    # Evita a propagação dos logs na tela ao realizar os testes
    logger = getLogger("LOG_TESTS")
    logger.propagate = False

    params = sys.argv
    mlruns_path = ""

    # Se recebeu algum parâmetro, verifica se é o esperado
    if len(params) > 1:
        parametros_esperados = {'--mlruns_path': str}
        resultado, retorno = validate_params(params, parametros_esperados)

        if resultado:
            mlruns_path = retorno['--mlruns_path']
        else:
            print(f"\nERRO: {retorno}")
            exit(1)

    validador = Test()
    validador.validate(mlruns_path=mlruns_path)
