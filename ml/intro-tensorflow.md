# TensorFlow

TensorFlow é uma biblioteca de Machine Learning *open-source* para pré-processamento de dados, modelagem de dados e criação de modelos. Em vez de criar modelos de `ML` do zero, podemos utilizar TensorFlow que contém muitas funções de `ML` (*principalmente as mais utilizadas*). Ok, TensorFlow é vasto, mas o foco principal é simples: transformar dados em números que são chamados de (`tensors`) e construir algoritmos de `ML` para encontrar padrões neles.

## Tensors

Os `tensors` são como matrizes NumPy. Daqui pra frente pense em um tensor como uma representação numérica multidimensional (n-dimensional, onde `n` pode ser qualquer número) de algo. Esse algo pode ser quase qualquer coisa que possamos imaginar:

- Pode ser os próprios números (tensors representando o preço de carros).
- Pode ser uma imagem (tensors representando os pixels de uma foto).
- Pode ser texto (tensors representando palavras).
- Ou pode ser alguma outra forma de informação (dados) que você deseja representar como números.

A principal diferença entre tensors e matrizes NumPy é que os tensors podem ser utilizados em `GPUs` (unidades de processamento gráfico, placas de vídeo) e `TPUs` (unidades de processamento de tensor). O benefício disso é poder executar tarefas computacionais mais rápidas, ou seja, para encontrar padrões em representações numéricas nos dados de forma mais rápida.

Vamos começar nossa jornada, a primeira coisa que faremos é importar o TensorFlow, o alias mais comum utilizado é o `tf`:

```python
import tensorflow as tf
print(tf.__version__)
```

> 2.8.0

## Criando Tensors com tf.constant()

No geral, normalmente não criaremos `tensors` por conta própria, pois o TensorFlow possui módulos integrados capazes de ler nossas fontes de dados e convertê-las automaticamente em tensors. Apenas para exemplificar nesse momento em que estamos nos familiarizando, criaremos tensors e veremos como manipulá-los. Começando por `tf.constant()`:

```python
scalar = tf.constant(5)
scalar

<tf.Tensor: shape=(), dtype=int32, numpy=5>
```

> Um `scalar` é conhecido como um tensor de `rank` 0 por não ter dimensões (é apenas um número). No momento não precisamos saber muito sobre os diferentes `ranks` de tensors, veremos mais detalhes sobre isso em outro momento. O importante agora é saber que os tensors podem ter um intervalo ilimitado de dimensões (a quantidade exata, vai depender dos dados que vamos representar).

```python
scalar.ndim

0
```

```python
vetor = tf.constant([10, 10])
vetor

<tf.Tensor: shape=(2,), dtype=int32, numpy=array([10, 10], dtype=int32)>

vetor.ndim

1
```

```python
matriz = tf.constant([[10, 5],
                      [5, 10]])
matriz

<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[10,  5],
       [ 5, 10]], dtype=int32)>

matriz.ndim

2
```

Por padrão o TensorFlow cria tensors utilizando `int32` como tipo de dados ou `float32`. Isso também é conhecido como precisão de `32 bits` (quanto maior o número, mais preciso ele é).

```python
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
tensor


<tf.Tensor: shape=(3, 2, 3), dtype=int32, numpy=
array([[[ 1,  2,  3],
        [ 4,  5,  6]],

       [[ 7,  8,  9],
        [10, 11, 12]],

       [[13, 14, 15],
        [16, 17, 18]]], dtype=int32)>

tensor.ndim

3
```

Esse é um exemplo de um tensor `rank` 3 (possui 3 dimensões), como dito antes, um tensor pode ter uma quantidade ilimitada de dimensões. Imagine que você quer transformar uma série de imagens em tensors, no formato (223,223, 3, 32), onde:

**223, 223** são as primeiras 2 dimensões, altura e largura das imagens em pixels. **3** é o número de canais de cores da imagem (vermelho, verde e azul) e **32** é o tamanho do lote (número de imagens que uma rede neural vê). Todas as variáveis criadas acima, são na verdade tensors, na literatura podemos encontrar referências com nomes diferentes:

- **scalar**: um único número
- **vetor**: um número com direção (ex: velocidade de um carro)
- **matriz**: uma matriz bidimensional numérica
- **tensor**: uma matriz n-dimensional numérica (onde `n` pode ser qualquer número, logo um tensor com dimensão `0` é um scalar, um tensor com 1 dimensão é um vetor).

> MEMO: inserir uma imagem aqui com ex. de álgebra visual comparando as referências para um tensor, algo parecido com: https://www.mathsisfun.com/algebra/scalar-vector-matrix.html

## Criando Tensors com tf.Variable()

---

WIP
