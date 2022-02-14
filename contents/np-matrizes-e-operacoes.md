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

---

WIP
