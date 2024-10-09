# UFV
 repositório para codigos das aulas da pós graduação

## relatório
Relatório de atividade prática: 
Titanic Machine Learning from Disaster

Jefferson Silva dos Anjos - 118852
Pós-graduação em inteligência artificial e computacional
Jefferson.ti@hotmail.com.br
---

Este relatório e o comparativo que aplica modelos de aprendizado de máquina e deep learnin usando a pytorch, como o objetivo de prever quem sobreviveria no naufrágio do Titanic.
# INTRODUÇÃO
Este relatório apresenta os resultados de um teste de sobrevivência com base no conjunto de dados do Titanic. Diversos modelos de Machine Learning foram aplicados para prever a probabilidade de sobrevivência dos passageiros foram escolhidos sete modelos.
# OBJETIVOS
Desenvolver modelos e aplicá-los para previsão de sobrevivência do titanic com base em características disponíveis no conjunto de dados. Essa tarefa é uma clássica de classificação binária.
# METODOLOGIA
A metodologia aplicada ao projeto seguiu um fluxo estruturado de coleta e pré-processamento de dados, modelagem, ajuste de hiperparâmetros, avaliação e visualização. As técnicas de balanceamento e ensemble learning foram essenciais para melhorar o desempenho do modelo, resultando em previsões precisas sobre a sobrevivência dos passageiros do Titanic.
##	Coleta de Dados
Os dados foram coletados do conjunto de dados Titanic, disponível no Kaggle. O conjunto contém informações sobre os passageiros, incluindo características como idade, sexo, classe, tarifa e local de embarque. Os dados foram divididos em um conjunto de treinamento e um conjunto de teste.
##	Pré-processamento dos Dados
A etapa de pré-processamento é crucial para garantir que os dados estejam prontos para o modelo. As seguintes etapas foram realizadas:
-	Tratamento de Valores Nulos: Valores ausentes nas colunas Age e Fare foram preenchidos com a mediana das respectivas colunas. A coluna Embarked foi preenchida com o valor mais frequente (S).
-	Feature Engineering: Novas características foram criadas para melhorar a capacidade preditiva do modelo:
-	familia: Soma de SibSp (irmãos/irmãs) e Parch (pais/filhos) para representar o tamanho da família.
-	crianca: Uma variável binária indicando se o passageiro é uma criança (idade < 12).
-	Title: Extraído dos nomes dos passageiros para representar diferentes grupos sociais.
-	Mapeamento de Variáveis Categóricas: As variáveis categóricas (Sex, Embarked, Title) foram convertidas em variáveis numéricas para serem utilizadas no modelo.
-	Remoção de Colunas Irrelevantes: Colunas que não contribuíam para a previsão, como Name, Ticket e Cabin, foram removidas.
##	Divisão dos Dados
Os dados foram divididos em conjuntos de treino e validação utilizando a função train_test_split do scikit-learn, garantindo que a distribuição das classes fosse preservada (stratify=y).
##	Balanceamento de Classes
Dado o desbalanceamento entre as classes (sobreviventes vs. não sobreviventes), foi aplicado o SMOTE (Synthetic Minority Over-sampling Technique) ao conjunto de treinamento. Essa técnica aumentou o número de instâncias da classe minoritária (sobreviventes) para balancear a distribuição das classes.
##	Modelo e Hiperparâmetros
Os seguintes modelos foram utilizados para a classificação:
-	REGRESSÃO LOGÍSTICA
-	NAIVE BAYES
-	K-NEAREST NEIGHBORS (KNN)
-	SUPPORT VECTOR CLASSIFIER (SVC)
-	ÁRVORE DE DECISÃO
-	RANDOM FOREST
-	REDE NEURAL (PYTORCH)
Para cada modelo, os hiperparâmetros foram ajustados usando RandomizedSearchCV, permitindo encontrar a combinação ideal de parâmetros para melhorar o desempenho do modelo.
##	Ensemble Learning
Um Voting Classifier foi utilizado para combinar as previsões dos diferentes modelos, permitindo que a decisão final fosse baseada na média ponderada das previsões. Isso geralmente resulta em uma melhoria no desempenho geral do modelo.
##	Avaliação do Modelo
O desempenho do modelo foi avaliado utilizando as seguintes métricas:
-	Acurácia: Proporção de previsões corretas sobre o total de previsões.
-	ROC AUC: Medida da capacidade do modelo em distinguir entre as classes.
-	Matriz de Confusão: Para visualizar o desempenho do modelo em cada classe.
-	Relatório de Classificação: Incluindo precisão, recall e F1-score.
Além disso, a curva de precisão-recall foi plotada para avaliar a relação entre precisão e recall em diferentes thresholds.
##	Predições e Submissão
Após a avaliação e ajustes do modelo, foram feitas previsões no conjunto de teste, e os resultados foram organizados em um DataFrame para submissão.
## ANÁLISE DOS RESULTADOS
- Desempenho da Rede Neural: A Rede Neural obteve a maior acurácia (83.84%), superando todos os outros modelos, indicando sua eficácia em capturar as complexidades dos dados.
- Comparação com Outros Modelos: O SVC e o Random Forest apresentaram desempenhos robustos, seguidos por KNN e Árvores de Decisão. Modelos mais simples, como a Regressão Logística e Naive Bayes, mostraram desempenhos mais modestos.
- Validade e Robustez: É fundamental realizar uma validação adicional para assegurar que o desempenho do modelo não seja afetado por overfitting.
A tabela abaixo resume a acurácia de cada modelo avaliado:

| Modelo                          | Acurácia (%)  |
|---------------------------------|---------------|
| Regressão Logística             | 80.47         |
| Naive Bayes                     | 80.36         |
| K-Nearest Neighbors (KNN)       | 81.15         |
| Support Vector Classifier (SVC) | 83.16         |
| Árvore de Decisão               | 81.71         |
| Random Forest                   | 82.61         |
| Rede Neural                     | 83.84         |

Apesar dos modelos de Machine Learning aplicados apresentarem um desempenho promissor em termos de acurácia, com uma média de 81.78%, os resultados obtidos na competição do Kaggle, onde a submissão final alcançou uma acurácia de apenas 77.51%, ficaram aquém das expectativas. Isso sugere que, embora a modelagem tenha sido eficaz em um cenário controlado, fatores como a complexidade dos dados, a presença de ruídos ou a seleção de características podem ter impactado negativamente o desempenho na competição. Portanto, é fundamental revisar as estratégias de pré-processamento, explorar novas variáveis e considerar abordagens adicionais para melhorar a performance do modelo em ambientes de competição.

![Figura 1. Public Score kaggle Titanic - Machine Learning from Disaster](D:\github\UFV\UFV\img_2.png)
# CONCLUSÃO
Os resultados demonstram que modelos de Machine Learning, especialmente a Rede Neural, são eficazes para prever a sobrevivência dos passageiros do Titanic. A análise e a modelagem podem ser aprimoradas através do ajuste de hiperparâmetros, validação cruzada e exploração de novas características.
