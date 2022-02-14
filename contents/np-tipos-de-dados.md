# Tipos de dados e atributos

> O principal tipo de dado usado no NumPy é o `ndarray`, mesmo tipos diferentes de arrays continuam sendo `ndarray`.

```python
# vetor
a1 = np.array([1, 2, 3])

# matrix de duas dimensões
a2 = np.array([[4, 5, 6],
               [7, 8, 9]])

# matrix
a3 = np.array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
                [[10, 11, 12],
                 [13, 14, 15],
                 [16, 17, 18]]])
```

Vamos olhar algumas informações sobre os exemplos criados, usando os métodos `shape`, `ndim`, `size` e a função `type`:

```python
a1.shape

(3,)
```

Usando o método `.shape`, podemos ver que o array NumPy `a1` é um vetor de 3 posições.

```python
a1.ndim, a2.ndim

(1, 2)
```

Aqui, usando `.ndim` vemos que `a1` é um array unidimensional, enquanto `a2` possui duas dimensões.

```python
a3.size

18
```

O método `.size` mostra que o array `a3` possui 18 elementos.

```python
type(a1), type(a2), type(a3)

(numpy.ndarray, numpy.ndarray, numpy.ndarray)
```

Todos os arrays são do tipo `ndarray`.

Assim como no Pandas, você pode exibir os elementos dos arrays apenas digitando o seu nome em uma célula do notebook:

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

## Anatomia de um array NumPy

WIP