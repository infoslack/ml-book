# Arrays, matrizes e operações

NumPy possui algumas funções para criar arrays, veremos agoras as mais utilizadas:

```python
array_simples = np.array([1, 2, 3])
array_simples

array([1, 2, 3])
```

Como visto anteriormente `.array()` cria um array simples. Mas e se eu precisar de um array repleto de números `1` ? NumPy fornece a função `.ones()`:

```python
ones = np.ones((10, 2))
ones

array([[1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.]])
```

A função `.ones()` recebe como parâmetro o shape do array que desejamos criar, nesse exemplo criamos um array de `10x2`.
Existe outra função para criar um array apenas de zeros. Exatamente, ela se chama `.zeros()` e também recebe parâmetros para definir o tamanho do array:

```python
zeros = np.zeros((4, 4, 3))
zeros

array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]])
```

Mas, se precisarmos criar um array informando apenas um range de valores, podemos usar o `.arange()`:

```python
range_a1 = np.arange(0, 10, 2)
range_a1

array([0, 2, 4, 6, 8])
```

A função `.arange()` assim como as outras também recebe parâmetros configuráveis, nesse caso temos 3 valores: *Start*, *Stop* e *Step*. No nosso exemplo o array começa em 0, conta até 10 com saltos de 2 em 2, ou seja, um array de números pares.

Outro recurso interessante é o de poder criar arrays com valores aleatórios:

```python
random_1 = np.random.randint(10, size=(5, 3))
random_1

array([[2, 3, 8],
       [6, 0, 2],
       [4, 0, 0],
       [2, 7, 1],
       [9, 6, 7]])
```

A método `.random` cria valores aleatórios para preencher o array, nesse caso estamos usando a função `.randint()` para gerar valores inteiros, como parâmetro informamos que queremos valores aleatórios entre 0 e 10 inseridos em um array de tamanho `5x3`. Outra função muito utilizada é a própria `.random()` que gera valores do tipo *float*, nesse caso a sua utilização fica `.random.random()`:

```python
random_2 = np.random.random((5, 3))
random_2

array([[0.55826932, 0.81632871, 0.90207164],
       [0.78776093, 0.21937942, 0.88146314],
       [0.01304884, 0.72247319, 0.15347852],
       [0.0923776 , 0.32395553, 0.11888521],
       [0.40217155, 0.04779815, 0.98891801]])
```

Como resultado temos outro array mas dessa vez de floats **entre 0 e 1**, também de tamanho `5x3`. NumPy utiliza números pseudo-aleatórios, significa que os números apenas parecem aleatórios, mas na verdade são predeterminados.

Para consistência em um experimento de Machine Learning, convém manter os números aleatórios gerados semelhantes ao longo dos experimentos. Fazer isso é simples e depende do uso da função `.seed()`:

```python
np.random.seed(42)
np.random.randint(10, size=(5, 3))

array([[6, 3, 7],
       [4, 6, 9],
       [2, 6, 7],
       [4, 3, 7],
       [7, 2, 5]])
```

> Ao reproduzir o exemplo acima no seu ambiente, você deve obter o mesmo resultado.

## Selecionando e visualizando elementos

Lembra dos nossos primeiros arrays ? Veremos agora como selecionar os elementos, mas primeiro vamos visualizar todos os elementos apenas para recordar:

```python
a1

array([1, 2, 3])

a2

array([[4, 5, 6],
       [7, 8, 9]])

a3

array([[[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9]],

       [[10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]]])
```

Bem, agora queremos selecionar apenas o primeiro elemento do `a1` que por ser um vetor, tem apenas uma única linha:

```python
a1[0]

1
```

Se fizermos o mesmo com `a2` o resultado é diferente:

```python
a2[0]

array([4, 5, 6])
```

Isso ocorre pois `a2` é um array de duas dimensões, logo o que estamos recebendo como responta é a primeira linha do array. Para selecionar apenas o primeiro elemento é preciso informar a coluna:

```python
a2[0,0]

4
```

Outra forma de selecionar os elementos é utilizando `slice`, vejamos um exemplo no array `a3`:

```python
a3[:2, :2, :2]

array([[[ 1,  2],
        [ 4,  5]],

       [[10, 11],
        [13, 14]]])
```

> Acessar elementos em arrays requer um pouco de prática, especialmente quando as dimensões aumentam, como no caso do exemplo anterior.

## Manipulando e comparando arrays

Veremos agora um pouco de aritmética com arrays. É possível realizar todas as operações (soma, subtração, multiplicação, potência, etc) e mais, para exemplificar vamos utilizar o nosso primeiro array criado, `a1` e vamos criar um novo utilizando `.ones()`:

```python
a1

array([1, 2, 3])

ones = np.ones(3)
ones

array([1., 1., 1.])
```

> Uma adição simples

```python
a1 + ones

array([2., 3., 4.])
```

> Subtração

```python
a1 - ones

array([0., 1., 2.])
```

> Multiplicação

```python
a1 * ones

array([1., 2., 3.])
```

> Divisão

```python
a1 / ones

array([1., 2., 3.])
```

> Divisão obtendo a parte inteira

```python
a2 // a1

array([[4, 2, 2],
       [7, 4, 3]])
```

> Exponenciação

```python
a1 ** 2

array([1, 4, 9])
```

> Exponenciação utilizando a função `.square()`

```python
np.square(a1)

array([1, 4, 9])
```

> Módulo

```python
a1 % 2

array([1, 0, 1])
```

> Logaritmo

```python
np.log(a1)

array([0.   , 0.69314718, 1.09861229])
```

> Exponencial

```python
np.exp(a1)

array([ 2.71828183,  7.3890561 , 20.08553692])
```

## Agregação

NumPy também disponibiliza várias funções para agregação, `sum()` por exemplo, pode somar todos os elementos de um array:

```python
sum(a1)

6
```

A mesma função disponibilizada pelo NumPy seria `np.sum()`:

```python
np.sum(a1)

6
```

O resultado é o mesmo, porém o desempenho é completamente diferente. Para o nosso teste, vamos criar um array com `1.000.000` de valores aleatório e comparar as duas funções de soma ( do Python e do NumPy ), medindo o tempo de processamento de cada uma:

```python
grande_array = np.random.random(1000000)
grande_array.size

1000000
```

Ok, já temos o nosso array com 1 milhão de valores aleatórios, para medir o tempo de processamento das operações de soma, podemos utilizar a função mágica chamada `%timeit` que vai exibir o tempo total de processamento de cada função, essa operação pode demorar um pouco:

```python
%timeit sum(grande_array)       # Python sum()
%timeit np.sum(grande_array)  # NumPy np.sum()

10 loops, best of 5: 165 ms per loop
1000 loops, best of 5: 365 µs per loop
```

A função `.sum()` do Python levou **165 milissegundos**, enquanto a função `np.sum()` do NumPy fez tudo em **365 microssegundos**, um pouco mais de **450x** mais rápido. Em outras palavras, se estiver trabalhando com NumPy, por questões de desempenho é melhor escolher as suas funções embutidas para grandes volumes de dados, do que utilizar as operações nativas do Python.

Vamos utilizar o array `a2` e experimentar outras funções, por exemplo calcular a média dos valores:

```python
a2

array([[4, 5, 6],
       [7, 8, 9]])

np.mean(a2)

6.5
```

> Encontrar o maior valor `.max()`

```python
np.max(a2)

9
```

> Encontrar o menor valor `.min()`

```python
np.min(a2)

4
```

> Calcular o desvio padrão `.std()` (é uma média de como os valores estão espalhados)

```python
np.std(a2)

1.707825127659933
```

> Calcular a variância `.var()` (é a média das diferenças quadradas da média)

```python
np.var(a2)

2.9166666666666665
```

## Reshaping e Transpose

Imagine que precisamos somar os arrays `a2` e `a3`:

```python
a2 + a3

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-60-7e0cfe75ff3f> in <module>()
----> 1 a2 + a3

ValueError: operands could not be broadcast together with shapes (2,3) (2,3,3) 
```

E para nossa surpresa recebemos esse erro informando que os shapes são diferentes e que por esse motivo a operação não pode ser realizada. Para resolver o problema podemos reordenar um dos arrays ou fazer o `reshape`:

```python
a2

array([[4, 5, 6],
       [7, 8, 9]])

a2.reshape(2, 3, 1)

array([[[4],
        [5],
        [6]],

       [[7],
        [8],
        [9]]])
```

O que fizemos com o reshape foi alterar o formato do nosso array para uma estrutura de uma só coluna, agora é possível somar `a2` e `a3`:

```python
a2.reshape(2, 3, 1) + a3

array([[[ 5,  6,  7],
        [ 9, 10, 11],
        [13, 14, 15]],

       [[17, 18, 19],
        [21, 22, 23],
        [25, 26, 27]]])
```

Já o `transpose` inverte a estrutura do array, por exemplo:

```python
a2

array([[4, 5, 6],
       [7, 8, 9]])

a2.T

array([[4, 7],
       [5, 8],
       [6, 9]])

a2.shape

(2, 3)

a2.T.shape

(3, 2)
```

> Assim, temos `a2` um array de `2x3` quando invertido pelo método transpose vira um array de `3x2`.

## Operações de comparação e ordenação de arrays

Assim como podemos utilizar operadores aritméticos com arrays, também é possível utilizar operadores de comparação:

```python
# Temos os arrays a1 e a2
a1

array([1, 2, 3])

a2

array([[4, 5, 6],
       [7, 8, 9]])
      
# a1 é maior que a2 ?
a1 > a2

array([[False, False, False],
       [False, False, False]])

# a1 é menor ou igual à a2 ?
a1 <= a2

array([[ True,  True,  True],
       [ True,  True,  True]])

# a1 é maior que 5 ?
a1 > 5

array([False, False, False])

# a1 é igual à a2 ?
a1 == a2

array([[False, False, False],
       [False, False, False]])
```

Agora um pouco de ordenação, lembra do nosso array aleatório `random_1` ?
Podemos utilizar a função `.sort()` e ordenar os elementos:

```python
random_1

array([[6, 6, 1],
       [6, 4, 1],
       [1, 1, 6],
       [1, 6, 4],
       [9, 0, 4]])

np.sort(random_1)

array([[1, 6, 6],
       [1, 4, 6],
       [1, 1, 6],
       [1, 4, 6],
       [0, 4, 9]])
```

---
WIP
