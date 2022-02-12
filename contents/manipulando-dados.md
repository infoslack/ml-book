# Manipulando dados com Pandas

Como vimos anteriormente, criar Séries e DataFrame do zero é legal, mas normalmente faremos a importação dos dados no formato de um arquivo `.csv` ou planilha.

O Pandas permite a importação de dados de maneira fácil por meio de funções como `pd.read_csv()` e `pd.read_excel()`.

Por exemplo, vamos obter as informações deste documento do Google Sheet:

![Google Sheet](images/excel.png "Dados no google sheet")

Depois de exportá-lo como um arquivo `.csv`, podemos agora importá-lo no Pandas com `pd.read_csv()` e criar o nosso DataFrame:

![Importando dados](images/carros-df.png "Importando CSV")

Agora temos os mesmos dados da planilha disponíveis em um DataFrame Pandas.
Isso permite que você aproveite todas as funcionalidades do Pandas para explorar os dados.

## Anatomia de um DataFrame

Abaixo vemos um resumo dos principais componentes de um DataFrame:

![Anatomia DataFrame](images/anatomia-dataframe.png "Anatomia de um DataFrame")

## Explorando os dados

Uma das primeiras tarefas que fazemos logo após importar os dados para um DataFrame Pandas é começar a explorá-lo. O Pandas possui muitas funções internas que nos permitem obter informações descritivas dos dados, `.dtypes` por exemplo nos mostra que tipo de dados cada coluna contém:

```python
df.dtypes

Fabricante       object
Cor              object
Quilometragem     int64
Portas            int64
Preco            object
dtype: object
```

Note que a coluna `Preco` não é um número inteiro como `Quilometragem` e `Portas`. Não se preocupe isso é fácil de consertar e faremos isso em outro momento.

A função `.describe()` nos mostra uma visão estatística de todas as colunas numéricas:

```python
df.describe()
```

|       | Quilometragem |  Portas   |
| :---: | :-----------: | :-------: |
| count |   10.000000   | 10.000000 |
| mean  | 78601.400000  | 4.000000  |
|  std  | 61983.471735  | 0.471405  |
|  min  | 11179.000000  | 3.000000  |
|  25%  | 35836.250000  | 4.000000  |
|  50%  | 57369.000000  | 4.000000  |
|  75%  | 96384.500000  | 4.000000  |
|  max  | 213095.000000 | 5.000000  |

Outra função muito utilizada é a `.info()`, que mostra quantas linhas existem, se há valores ausentes e os tipos de dados de cada coluna:

```
df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   Fabricante     10 non-null     object
 1   Cor            10 non-null     object
 2   Quilometragem  10 non-null     int64 
 3   Portas         10 non-null     int64 
 4   Preco          10 non-null     object
dtypes: int64(2), object(3)
memory usage: 528.0+ bytes
```

Pandas também disponibiliza várias funções estatísticas e matemáticas como `.mean()` e `.sum()` que podem ser aplicadas diretamente em um DataFrame ou Séries.

```python
valores = pd.Series([3000, 3500, 11250])
valores.mean()

5916.666666666667
```

```python
valores.sum()

17750
```

Usar essas funções em um DataFrame inteiro pode não fazer muito sentido, nesse caso podemos direcionar a função individualmente para uma coluna.
O método `.columns` mostrará todas as colunas de um DataFrame:

```python
df.columns

Index(['Fabricante', 'Cor', 'Quilometragem', 'Portas', 'Preco'], dtype='object')
```

Selecionando uma coluna com dados numéricos podemos utilizar a função estatística `.mean()` novamente:

```python
df["Quilometragem"].mean()

78601.4
```

## Visualizando e selecionando dados

Na prática, realizar alterações nos dados e visualizá-los constantemente é uma tarefa de rotina. Nem sempre precisamos alterar todos os dados de um DataFrame, portanto veremos maneiras diferentes de selecionar.

Para visualizar as primeiras 5 linhas do seu DataFrame existe a função `.head()` que é muito utilizada:

```python
df.head()
```

|     | Fabricante |   Cor    | Quilometragem | Portas |     Preco     |
| :-: | :--------: | :------: | :-----------: | :----: | :-----------: |
|  0  |   Toyota   |  Branco  |    150043     |   4    | R$ 24,000.00  |
|  1  |   Honda    | Vermelho |     87899     |   4    | R$ 25,000.00  |
|  2  |   Toyota   |   Azul   |     32549     |   3    | R$ 27,000.00  |
|  3  |    BMW     |  Preto   |     11179     |   5    | R$ 122,000.00 |
|  4  |   Nissan   |  Branco  |    213095     |   4    | R$ 13,500.00  |

Para selecionar mais de 5, você pode passar o valor desejado como argumento na função, por exemplo: `.head(7)`.

Outro método muito utilizado é o `.tail()` que seleciona as últimas 5 linhas do seu DataFrame:

```python
df.tail()
```

|     | Fabricante |  Cor   | Quilometragem | Portas |    Preco     |
| :-: | :--------: | :----: | :-----------: | :----: | :----------: |
|  5  |   Toyota   | Verde  |     99213     |   4    | R$ 14,500.00 |
|  6  |   Honda    |  Azul  |     45698     |   4    | R$ 17,500.00 |
|  7  |   Honda    |  Azul  |     54738     |   4    | R$ 27,000.00 |
|  8  |   Toyota   | Branco |     60000     |   4    | R$ 26,250.00 |
|  9  |   Nissan   | Branco |     31600     |   4    | R$ 19,700.00 |

Por meio das instruções `.loc[]` e `.iloc[]` podemos selecionar dados de Séries e DataFrames, essas opções são muito utilizadas. Para ilustrar o uso em uma Série, vamos criar uma:

```
animais = pd.Series(["gato", "ave", "cachorro", "cobra", "leão", "cavalo"], 
                   index=[0, 3, 8, 9, 6, 3])
animais

0        gato
3         ave
8    cachorro
9       cobra
6        leão
3      cavalo
dtype: object
```

Agora vamos utilizar o `.loc[]` informando como valor de entrada um número inteiro que corresponde ao índice da nossa Série:

```
animais.loc[3]

3       ave
3    cavalo
dtype: object
```

Temos como retorno os valores que tem como índice o número 3.

Já com o `.iloc[]`:

```
animais.iloc[3]

cobra
```

O valor retornado é bem diferente `cobra` que tem como índice o número 9. Isso acontece pois o `.iloc[]` faz uma seleção pela posição dos dados na Série ou DataFrame. Vamos testar esses recursos agora no nosso DataFrame:

```
df.loc[3]

Fabricante                 BMW
Cor                      Preto
Quilometragem            11179
Portas                       5
Preco            R$ 122,000.00
Name: 3, dtype: object

df.iloc[3]

Fabricante                 BMW
Cor                      Preto
Quilometragem            11179
Portas                       5
Preco            R$ 122,000.00
Name: 3, dtype: object
```

Ambos `.loc[]` e `.iloc[]` retornaram o mesmo valor pois as informações no DataFrame exibidas estão em ordem tanto na posição quanto no índice.

Ainda podemos utilizar o `slicing` com `.loc[]` e `.iloc[]`, selecionando dados em um range:

```
animais.iloc[:3]

0        gato
3         ave
8    cachorro
dtype: object
```

> Lembre-se, utilize `.loc[]` quando estiver selecionando dados pelo índice e `.iloc[]` quando estiver referindo-se a posições no DataFrame.

Se quiser selecionar uma coluna em particular use `['NOME_DA_COLUNA']`:

```
df["Fabricante"]

0    Toyota
1     Honda
2    Toyota
3       BMW
4    Nissan
5    Toyota
6     Honda
7     Honda
8    Toyota
9    Nissan
Name: Fabricante, dtype: object
```

WIP
