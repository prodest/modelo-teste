# ----------------------------------------------------------------------------------------------------
# Script que contém a função personalizada pelo usuário para realização de testes
# ----------------------------------------------------------------------------------------------------

# Coloque aqui os imports


def test(modelos: dict):
    """
    Implemente nesta função os testes adicionais que você julgar necessários para validar o código do worker.
    Esta função recebe um dicionário com os nomes dos modelos como chave e um modelo instanciado como valor.
    Exemplo: {'NOME_DO_MODELO': <models.retrain1.ModeloRETRAIN object at 0x7f70045bf1f0>}

    REGRAS:
    1. Não altere o nome do script 'mytest_retrain.py' nem o nome da função 'test()';
    2. Não altere os nomes nem os tipos de parâmetros recebidos pela função 'test()';
    3. Esta função não deve retornar nada;
    4. A responsabilidade de criar os testes personalizados e verificar se passaram é do desenvolvedor do modelo;
    5. Imprima na tela ou salve em logs as informações que você achar que são úteis.
    """
    # TODO: Implementar os testes personalizados.

    for nome_modelo, modelo in modelos.items():
        # TODO: Implementar as verificações para cada modelo recebido.
        pass
