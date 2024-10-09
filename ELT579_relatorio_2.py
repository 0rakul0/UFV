import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

#%% importando a base
df = pd.read_csv('./data/dataset_problema2.csv')

X = df.drop(['id', 'Severidade'], axis=1)
y = df['Severidade']

#%% separando os dados
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% padronizando os dados
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns
X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_test.columns

#%% modelo padronizado regressão linear múltipla RLM
model_lin_m = LinearRegression()
model_lin_m.fit(X_train_sc, y_train)
coef = model_lin_m.coef_
feat = pd.DataFrame(coef)
feat['features'] = X_train.columns  # mostra as colunas mais importantes
print(feat)

#%% seleção das features menos impactantes RFE
lista_score = []
for i in range(1, X.shape[1]+1):
    modelo_linear = LinearRegression()
    selecto = RFE(modelo_linear, n_features_to_select=i, step=1)
    selecto = selecto.fit(X_train_sc, y_train)
    mask = selecto.support_
    features = X_train_sc.columns[mask]
    X_sel = X_train_sc[features]
    scores = cross_val_score(modelo_linear, X_sel, y_train, cv=10, scoring='r2')
    lista_score.append(np.mean(scores))

#%% grafico
plt.plot(lista_score)
plt.savefig('Score.png')
plt.show()

#%% seleção final
modelo_linear = LinearRegression()
selecto = RFE(modelo_linear, n_features_to_select=10, step=1)
selecto = selecto.fit(X_train_sc, y_train)
mask = selecto.support_
features = X_train_sc.columns[mask]
X_sel = X_train_sc[features]
scores = cross_val_score(modelo_linear, X_sel, y_train, cv=10, scoring='r2')
print(np.mean(scores))
print(features)
modelo_linear.fit(X_sel, y_train)

#%% teste
from sklearn.metrics import mean_squared_error, mean_absolute_error
y_pred = modelo_linear.predict(X_test_sc[features])
r2 = modelo_linear.score(X_test_sc[features], y_test)
rmse = (mean_squared_error(y_test, y_pred)**0.5)
mae = (mean_absolute_error(y_test, y_pred))
print('r2_score: ', r2)
print('mae: ', mae)
print('rmse: ', rmse)

#%% outros modelos Ridge
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_sc, y_train)
y_pred_ridge = ridge_model.predict(X_test_sc)

#%% modelo de Lasso
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_sc, y_train)
y_pred_lasso = lasso_model.predict(X_test_sc)

#%% modelo ElasticNet
from sklearn.linear_model import ElasticNet

elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticnet_model.fit(X_train_sc, y_train)
y_pred_elasticnet = elasticnet_model.predict(X_test_sc)

#%% arvore de desição
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

#%% floresta randomica
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

#%% gradiente
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

#%% xbg
import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)


#%%SVR
from sklearn.svm import SVR

svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_sc, y_train)
y_pred_svr = svr_model.predict(X_test_sc)


#%% implementação de todos os modelos
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_model(model, X_train, X_test, y_train, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Armazenar os resultados
    results.append({
        'Model': model.__class__.__name__,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    })

models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=0.1),
    ElasticNet(alpha=0.1, l1_ratio=0.5),
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(n_estimators=100, random_state=0),
    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0),
    xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1),
    SVR(kernel='rbf', C=1.0, epsilon=0.1)
]

results = []

for model in models:
    evaluate_model(model, X_train_sc, X_test_sc, y_train, y_test, results)

results_df = pd.DataFrame(results)

# Plotar o gráfico
fig, ax = plt.subplots(figsize=(10, 6))
results_df.plot(x='Model', y=['R2', 'RMSE', 'MAE'], kind='bar', ax=ax)
plt.title('Comparação de Modelos - Métricas R², RMSE, MAE')
plt.ylabel('Valores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('todos_os_modelos.png')
plt.show()

# R²
plt.figure(figsize=(8, 6))
results_df.plot(x='Model', y='R2', kind='bar', color='blue', legend=False)
plt.title('R²')
plt.ylabel('Valores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('R2_score.png')
plt.show()

# RMSE
plt.figure(figsize=(8, 6))
results_df.plot(x='Model', y='RMSE', kind='bar', color='orange', legend=False)
plt.title('RMSE')
plt.ylabel('Valores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('RMSE_score.png')  # Salva o gráfico RMSE
plt.show()

# MAE
plt.figure(figsize=(8, 6))
results_df.plot(x='Model', y='MAE', kind='bar', color='green', legend=False)
plt.title('MAE')
plt.ylabel('Valores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('MAE_score.png')  # Salva o gráfico MAE
plt.show()

"""
as métricas de desempenho: R², RMSE e MAE:

### 1. **R² (Coeficiente de Determinação)**
   - **O que é**: O coeficiente de determinação, ou \( R^2 \), mede a proporção da variância dos dados que é explicada pelo modelo. Em outras palavras, ele indica o quanto o modelo consegue "explicar" a variabilidade dos dados de saída (a variável dependente) a partir dos dados de entrada (as variáveis independentes).
   
   - **Intervalo de Valores**: O valor do \( R^2 \) varia entre 0 e 1, embora possa ser negativo se o modelo for muito ruim. Um valor de 1 indica que o modelo ajusta perfeitamente os dados (explica 100% da variação dos dados), enquanto um valor de 0 significa que o modelo não explica nada da variação.
   
   - **Como interpretar**:
     - \( R^2 \) perto de 1: O modelo tem um bom ajuste e explica a maior parte da variabilidade.
     - \( R^2 \) perto de 0: O modelo não explica bem a variabilidade dos dados.
     - \( R^2 \) negativo: O modelo ajusta pior do que um modelo simples, como a média das respostas.

   - **Exemplo**: Se um modelo tem \( R^2 = 0.90 \), isso significa que ele explica 90% da variabilidade dos dados. Isso sugere que o modelo é bastante bom.

---

### 2. **RMSE (Root Mean Squared Error - Erro Quadrático Médio)**
   - **O que é**: O RMSE é uma métrica que calcula o erro médio ao quadrado entre os valores preditos e os valores reais, e depois tira a raiz quadrada desse valor. Ele mede a magnitude dos erros de previsão, dando mais peso a erros grandes, já que os erros são elevados ao quadrado antes da média ser tirada.

   - **Fórmula**:
     \[
     RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2}
     \]
     Onde \( y_i \) são os valores reais e \( \hat{y_i} \) são os valores preditos pelo modelo.

   - **Intervalo de Valores**: Quanto menor o RMSE, melhor. O RMSE nunca será negativo e pode variar de 0 ao infinito. Um valor de 0 indicaria que o modelo previu perfeitamente todos os valores.

   - **Como interpretar**:
     - RMSE baixo: O modelo tem bons ajustes e está fazendo previsões próximas dos valores reais.
     - RMSE alto: As previsões do modelo estão mais distantes dos valores reais, e há um erro maior nos ajustes.

   - **Exemplo**: Se o RMSE for 6.93, significa que, em média, a diferença entre os valores preditos e os valores reais é de cerca de 6.93 unidades. Esse erro considera grandes desvios mais gravemente devido ao uso do quadrado.

---

### 3. **MAE (Mean Absolute Error - Erro Médio Absoluto)**
   - **O que é**: O MAE é a média das diferenças absolutas entre os valores preditos e os valores reais. Ele mede a magnitude média do erro de previsão sem considerar a direção do erro (ou seja, não diferencia se o erro é positivo ou negativo).

   - **Fórmula**:
     \[
     MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|
     \]
     Onde \( y_i \) são os valores reais e \( \hat{y_i} \) são os valores preditos.

   - **Intervalo de Valores**: O MAE também nunca será negativo e pode variar de 0 ao infinito. Um valor de 0 indicaria que o modelo previu perfeitamente todos os valores.

   - **Como interpretar**:
     - MAE baixo: Indica que o modelo tem uma boa precisão e que as previsões estão muito próximas dos valores reais.
     - MAE alto: O modelo tem um erro médio significativo, com previsões distantes dos valores reais.

   - **Diferença com RMSE**: Diferente do RMSE, o MAE não dá mais peso a erros maiores, pois não eleva os erros ao quadrado. Isso faz com que o RMSE penalize erros grandes mais do que o MAE. Como resultado, o RMSE é mais sensível a outliers do que o MAE.

   - **Exemplo**: Se o MAE for 5.33, significa que, em média, o modelo erra as previsões em cerca de 5.33 unidades.

---

### Comparação entre as métricas:
- **R²** é útil para entender a capacidade do modelo em capturar a variação dos dados. Um \( R^2 \) alto sugere que o modelo explica bem a variabilidade da variável dependente.
- **RMSE** e **MAE** medem o erro médio, mas o RMSE penaliza mais os erros grandes devido ao uso do quadrado na fórmula, enquanto o MAE considera todos os erros da mesma forma. Portanto, se você se preocupa mais com erros grandes, prefira o RMSE; se deseja uma métrica mais simples e menos sensível a grandes erros, use o MAE.

"""