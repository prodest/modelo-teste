# Modelo de teste

Modelo classificador para testes de *deploy* da **Stack de ML do Prodest**.

---


### Este modelo foi baseado no dataset "Cyberbullying Classification".
O dataset foi obtido através do endereço (acessado em 02/09/2023): https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification?resource=download

- **Citação:**

    J. Wang, K. Fu, C.T. Lu, "SOSNet: A Graph Convolutional Network Approach to Fine-Grained Cyberbullying Detection", Proceedings of the 2020 IEEE International Conference on Big Data (IEEE BigData 2020), December 10-13, 2020.

---
### AVISO IMPORTANTE: 
Considerando que este modelo servirá apenas para realização de testes, **NÃO** houve preocupação com:

- Escolha do classificador (foi utilizada uma Rede Neural de uso genérico);
- Tratamento e análise mais apurada do dataset;
- *Benchmarking* para escolha de ferramentas e abordagens diferentes;
- Validação cruzada;
- Otimizações do código;
- etc.

Enfim, o desenvolvimento deste modelo teve como objetivo principal ilustrar a implementação de um modelo 
utilizando a [Biblioteca de Machine Learning (ML) do Prodest](https://github.com/prodest/mllibprodest), que servisse como
base para *deploys* de teste da [Stack de ML do Prodest](https://github.com/prodest/prodest-ml-stack) (versão standalone).