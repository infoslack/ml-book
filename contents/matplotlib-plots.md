# Tipos de plots mais utilizados

As visualizações do Matplotlib são construídas a partir de matrizes NumPy, neste capítulo veremos alguns dos tipos de gráficos mais utilizados. Para solucionar maior parte do problema em descobrir que tipo de plotagem usar, primeiro é ter uma ideia geral dos dados para em seguida ver qual plotagem melhor se adapta. Faremos alguns experimentos!

Antes de continuar, vamos precisar do NumPy para gerar alguns dados para nossas plotagens:

```python
import numpy as np

x = np.linspace(0, 10, 100)
x[:10]

array([0.        , 0.1010101 , 0.2020202 , 0.3030303 , 0.4040404 ,
       0.50505051, 0.60606061, 0.70707071, 0.80808081, 0.90909091])
```

### Line

Gráfico de linha é o tipo padrão de visualização no Matplotlib, geralmente, a menos que seja configurado de outra forma, os gráficos começarão como linhas.

```python
fig, ax = plt.subplots()
ax.plot(x, x**2);
```

![plot line](images/plot-line.png)

### Scatter

Outro tipo de gráfico bastante usado é o `scatter` ou gráfico de dispersão. Lembra um pouco o gráfico de linhas, mas é representado por pontos.

```python
fig, ax = plt.subplots()
ax.scatter(x, np.sin(x));
```

![plot scatter](images/plot-scatter.png)

### Bar

Um gráfico de barras apresenta barras retangulares com comprimentos e alturas proporcionais aos valores que representam. Pode ser plotado horizontalmente ou verticalmente. Bastante utilizado em comparações entre dados.

```python
produtos = {"Pão": 10,
            "Leite": 8,
            "Sorvete": 12}

fig, ax = plt.subplots()
ax.bar(produtos.keys(), produtos.values())
ax.set(title="Lista de produtos", ylabel="Preço");
```

![plot bar](images/plot-bar-h.png)

```python
fig, ax = plt.subplots()
ax.barh(list(produtos.keys()), list(produtos.values()));
```

![plot bar horizontal](images/plot-bar.png)

### Hist

Histograma pode ser um ótimo passo para entender um conjunto de dados.
A função `randn` do NumPy gera valores aleatórios com uma [distribuição normal](https://pt.wikipedia.org/wiki/Distribui%C3%A7%C3%A3o_normal). Vamos gerar dados com essa função e plotar em um gráfico de histograma:

```python
x = np.random.randn(1000)

fig, ax = plt.subplots()
ax.hist(x);
```

![plot hist](images/plot-hist.png)

### Subplots

Já vimos uma breve apresentação de subplots, agora veremos um pouco mais de recursos dessa função. Vamos criar vários plots em uma só figura. Na primeira opção vamos plotar os dados em cada `axis` da figura:

```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,
                                             ncols=2,
                                             figsize=(10, 5))
ax1.plot(x, x/2); #line
ax2.scatter(np.random.random(10), np.random.random(10)) #scatter
ax3.bar(produtos.keys(), produtos.values()) #bar
ax4.hist(np.random.randn(1000)); #hist
```

![plot subplots opt 1](images/plot-subplots-1.png)

Na segunda opção usaremos um índice para plotar os dados, particularmente prefiro essa alternativa:

```python
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

ax[0, 0].plot(x, x/2) #line
ax[0, 1].scatter(np.random.random(10), np.random.random(10)) #scatter
ax[1, 0].bar(produtos.keys(), produtos.values()) #bar
ax[1, 1].hist(np.random.randn(1000)); #hist
```
![plot subplots opt 2](images/plot-subplots-1.png)

## Plotando dados com Pandas

---

WIP
