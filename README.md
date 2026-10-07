# 🎯 Perceptron Multiclasse com Estratégia One-vs-Rest (OvR)

O trabalho a seguir foi um projeto simples para a disciplina de Inteligência Artificial do semestre 2025/2.
Este repositório contém uma demonstração prática do algoritmo **Perceptron** aplicado ao problema de classificação multiclasse (4 classes em um espaço bidimensional 2D). Para estender o Perceptron — que é nativamente um classificador binário — para múltiplas classes, utiliza-se a estratégia **One-vs-Rest (OvR)** (ou *One-vs-All*).

## 📌 Sumário

* [Visão Geral](#-visão-geral)

* [Como Funciona](#-como-funciona)

  * [O Algoritmo Perceptron](#o-algoritmo-perceptron)

  * [Estratégia One-vs-Rest (OvR)](#estratégia-one-vs-rest-ovr)

* [Estrutura do Conjunto de Dados](#-estrutura-do-conjunto-de-dados)

* [Estrutura do Projeto](#-estrutura-do-projeto)

* [Instalação e Execução](#-instalação-e-execução)

* [Resultados e Visualização](#-resultados-e-visualização)

* [Licença](#-licença)

## 📖 Visão Geral

O objetivo deste projeto é demonstrar a leitura de dados pontuais em duas dimensões ($x, y$), o treinamento de classificadores lineares Perceptron e a renderização gráfica das **fronteiras de decisão** geradas no plano cartesiano.

O conjunto de dados é composto por 4 classes distintas (`C1`, `C2`, `C3` e `C4`), agrupadas em diferentes regiões do plano.

## 🧠 Como Funciona

### O Algoritmo Perceptron

O Perceptron é o modelo de rede neural artificial mais fundamental. Ele aprende uma fronteira de decisão linear (hiperplano) para separar duas classes calculando a combinação linear das entradas mais um termo de viés (*bias*):

$$
f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

Para problemas binários, a predição é dada por $\text{sign}(f(\mathbf{x}))$.

### Estratégia One-vs-Rest (OvR)

Como o Perceptron resolve apenas problemas binários diretamente, a estratégia **One-vs-Rest** treina $K$ classificadores independentes, onde $K$ é o número de classes (no nosso caso, $K = 4$):

1. **Classificador 1 (**$C_1$**):** Treinado para separar $C_1$ das classes $\{C_2, C_3, C_4\}$.

2. **Classificador 2 (**$C_2$**):** Treinado para separar $C_2$ das classes $\{C_1, C_3, C_4\}$.

3. **Classificador 3 (**$C_3$**):** Treinado para separar $C_3$ das classes $\{C_1, C_2, C_4\}$.

4. **Classificador 4 (**$C_4$**):** Treinado para separar $C_4$ das classes $\{C_1, C_2, C_3\}$.

Para classificar um novo ponto $\mathbf{x}$, é calculada a função de decisão (`decision_function`) de cada um dos 4 modelos. O ponto é atribuído à classe correspondente ao maior score:

$$
\hat{y} = \arg\max_{c \in \{C_1, C_2, C_3, C_4\}} \text{score}_c(\mathbf{x})
$$

## 📊 Estrutura do Conjunto de Dados

O arquivo `perceptron.txt` contém os dados no formato CSV com três colunas:

| Coluna | Tipo | Descrição | 
| ----- | ----- | ----- | 
| `x` | `float` | Coordenada no eixo das abscissas | 
| `y` | `float` | Coordenada no eixo das ordenadas | 
| `classe` | `string` | Rótulo da classe (`C1`, `C2`, `C3`, `C4`) | 

Os pontos das 4 classes estão distribuídos aproximadamente nas seguintes regiões do plano:

* **`C1`**: Próximo à origem $(0, 0)$

* **`C2`**: Deslocado no eixo X $(5, 0)$

* **`C3`**: Deslocado no eixo Y $(0, 5)$

* **`C4`**: Deslocado em ambos os eixos $(5, 5)$

## 📁 Estrutura do Projeto

```
.
├── perceptron.py       # Script principal com leitura, treinamento e plots
├── perceptron.txt      # Conjunto de dados sintético (4 classes)
└── README.md           # Documentação do projeto

```

## 🚀 Instalação e Execução

### Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado.

### 1. Clonar o Repositório

```
git clone https://github.com/alisson-xavier/Perceptron_IA_2025_2.git
cd Perceptron_IA_2025.2

```

### 2. Instalar Dependências

Instale as bibliotecas necessárias utilizando o `pip`:

```
pip install pandas numpy matplotlib scikit-learn

```

### 3. Executar o Script

```
python perceptron.py

```

## 🖼️ Resultados e Visualização

O fluxo de execução do script realiza duas etapas de visualização:

1. **Plot do Dataset Original:** Exibe a distribuição espacial dos pontos das 4 classes com cores distintas.

2. **Mapeamento das Regiões de Decisão:** Utiliza uma malha de pontos (`np.meshgrid`) no intervalo $[-1, 6.5] \times [-1, 6.5]$ para avaliar as regiões de influência de cada Perceptron via `plt.contourf()`.

## 📜 Sobre a utilização ou modificação por terceiros

Este projeto não tem fins lucrativos e foi usado apenas de forma educacional e demonstrativa. Qualquer contribuição para melhora será bem-vinda.
