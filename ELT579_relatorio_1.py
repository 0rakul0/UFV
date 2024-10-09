import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import torch
import torch.nn as nn
import torch.optim as optim

# Carregar dados
train = pd.read_csv(r'.\data\train.csv')
test = pd.read_csv(r'.\data\test.csv')


# Função para criar features
def criar_features(X):
    subs = {'female': 1, 'male': 0}
    X['mulher'] = X['Sex'].replace(subs)
    X['Fare'] = X['Fare'].fillna(X['Fare'].mean())
    X['Age'] = X['Age'].fillna(X['Age'].mean())
    X['Embarked'] = X['Embarked'].fillna('S')
    subs = {'S': 1, 'C': 2, 'Q': 3}
    X['porto'] = X['Embarked'].replace(subs)
    X['crianca'] = np.where(X['Age'] < 12, 1, 0)
    return X


# Preparar dados
X_train = criar_features(train.drop(['PassengerId', 'Survived'], axis=1))
X_test = criar_features(test.drop(['PassengerId'], axis=1))

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'mulher', 'porto', 'crianca']
X_train = X_train[features]
X_test = X_test[features]

y_train = train['Survived']

# Normalização
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Dicionário para armazenar os scores
scores = {}

# Modelos de Machine Learning
models = {
    'Logistic Regression': LogisticRegression(random_state=0),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVC': SVC(C=3, kernel='rbf', degree=2, gamma=0.1),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0),
    'Random Forest': RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=5, random_state=0)
}

# Avaliar modelos
for name, model in models.items():
    score = cross_val_score(model, X_train_sc, y_train, cv=10)
    scores[name] = np.mean(score)
    print(f"{name} Accuracy: {np.mean(score):.4f}")


# Modelo de RNA com PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Treinar modelo de RNA
X_train_tensor = torch.FloatTensor(X_train_sc)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

model_nn = NeuralNetwork(input_size=X_train_sc.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

# Treinamento
for epoch in range(100):  # Número de épocas
    model_nn.train()
    optimizer.zero_grad()
    outputs = model_nn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Avaliação
model_nn.eval()
with torch.no_grad():
    predictions = model_nn(X_train_tensor)
    predicted_classes = (predictions > 0.5).float()
    nn_accuracy = (predicted_classes.view(-1) == y_train_tensor.view(-1)).float().mean().item()
    scores['Neural Network'] = nn_accuracy

# Imprimir a acurácia do modelo de RNA
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

# Gráfico de comparação
plt.figure(figsize=(10, 10))
plt.bar(scores.keys(), scores.values(), color='blue')
plt.xlabel('Modelos')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia dos Modelos')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Previsões e submissão
# Para prever no conjunto de teste, siga o mesmo procedimento de transformação
X_test_tensor = torch.FloatTensor(X_test_sc)
with torch.no_grad():
    nn_predictions = model_nn(X_test_tensor)
    nn_predicted_classes = (nn_predictions > 0.5).float()

submission = pd.DataFrame(test['PassengerId'])
submission['Survived'] = nn_predicted_classes.numpy().astype(int)
submission.to_csv('../data/submission_nn.csv', index=False)
