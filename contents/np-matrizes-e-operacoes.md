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

---

WIP

- random.randint
- random.random
- random.rand
- random.seed
- indexing
- select
- operações