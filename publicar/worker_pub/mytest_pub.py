# ----------------------------------------------------------------------------------------------------
# Script que contém a função personalizada pelo usuário para realização de testes
# ----------------------------------------------------------------------------------------------------

def test(modelos: dict):
    """
    Implemente nesta função os testes adicionais que você julgar necessários para validar o código do worker.
    Esta função recebe um dicionário com os nomes dos modelos como chave e um modelo instanciado como valor.
    Exemplo: {'NOME_DO_MODELO': <models.pub1.ModeloCLF object at 0x6f61145cf1f8>}

    REGRAS:
    1. Não altere o nome do script 'mytest_pub.py' nem o nome da função 'test()';
    2. Não altere os nomes nem os tipos de parâmetros recebidos pela função 'test()';
    3. Esta função não deve ter retorno;
    4. A responsabilidade de criar os testes personalizados e verificar se passaram é do desenvolvedor do modelo;
    5. Imprima na tela ou salve em logs as informações que você achar que são úteis.
    """

    for nome_modelo, modelo in modelos.items():
        print(f"\n>> ModeloCLF - {nome_modelo}:\n")
        print("Predict:", modelo.predict(["yes exactly the police can murder black people and we can be okay with it because it’s in the past and they’re dead now."]))
        print("Evaluate:", modelo.evaluate(["Today’s society so sensitive it’s sad they joke about everything but they take out the gay jokes before race, rape, and other 'sensitive' jokes",
                                            "aposto que vou sofrer bullying depois do meu próximo tweet"],
                                           ['gender', 'not_cyberbullying']))
        print("Get Info:", modelo.get_model_info())
