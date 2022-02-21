# Transfer Learning com TensorFlow 2.0 - resolvendo um problema de classificação

Neste capítulo, usaremos Machine Learning para resolver um problema de classificação, identificando diferentes raças de cães. Utilizaremos uma base de dados de identificação de cães [disponível no Kaggle](https://www.kaggle.com/c/dog-breed-identification/overview). A base consiste em uma coleção de mais de 10.000 imagens rotuladas em 120 raças de cães diferentes. Este é um problema de classificação de imagens *multiclasse*, pois estamos tentando classificar várias raças diferentes de cães. Se o problema fosse classificar imagens para identificar se é cão ou gato, isso seria um problema de classificação binária (*como vimos em Scikit-Learn*).

Classificação multiclasse de imagens é um problema interessante, pois envolve o mesmo tipo de tecnologia que a Tesla utiliza em seus carros autônomos. A etapa mais importante em um problema de `ML` é preparar os dados (*transforma-los em números*) e é justamente por onde iremos começar.

> Como utilizaremos TensorFlow para pré-processar os dados, ou seja vamos inserir nossos dados em `Tensors` (*matrizes de números que podem ser executados em GPUs*) permitindo que o modelo encontre padrões nos dados. Veremos como configurar o Google Colab para utilizar GPU.

Para o nosso modelo de `ML`, usaremos um modelo pré-treinado do `TensorFlow Hub`. Esse processo de utilizar um modelo pré-treinado e dadaptá-lo para o nosso problema é chamado de `Transfer Learning` (*aprendizado por transferência*). Faremos isso pois queremos aproveitar os padrões de um modelo que foi treinado exaustivamente para classificar imagens. ***Veremos em outros capítulos como criar modelos do zero para resolver diferentes problemas com TensorFlow.***

## Preparando o ambiente

- ### WIP
  - configurar o colab para usar gpu
  - importar o tensorflow
  - verificar a disponibilidade da gpu

## Preparando os dados

- ### WIP
  - upload dos dados do kaggle para google drive
  - explicar o processo de montagem do drive no colab
> Os dados: https://www.kaggle.com/c/dog-breed-identification/data

### Acessando os dados

Agora que temos os dados que iremos trabalhar disponíveis no Google Drive, podemos começar a verificação dos mesmos. Começando com o arquivo `labels.csv` que contém todos os `IDs` de imagem associada com as raças dos cães, ou seja (dados e rótulos).

```python
import pandas as pd
labels = pd.read_csv("drive/MyDrive/Dog Vision/labels.csv")
print(labels.describe())
print(labels.head())
```

```
count                              10222               10222
unique                             10222                 120
top     000bec180eb18c7604dcecc8fe0dba07  scottish_deerhound
freq                                   1                 126
                                 id             breed
0  000bec180eb18c7604dcecc8fe0dba07       boston_bull
1  001513dfcb2ffafc82cccf4d8bbaba97             dingo
2  001cdf01b096e06d78e9e5112d419397          pekinese
3  00214f311d5d2247d5dfe4fe24b2303d          bluetick
4  0021f9ceb3235effd7fcde7f7538ed62  golden_retriever
```

Existem 10.222 IDs diferentes (ou seja imagens diferentes) e 120 raças diferentes. Como temos os IDs das imagens e os rótulos (labels) em um DataFrame, precisamos criar:

- Uma lista com os caminhos dos arquivos para imagens de treino
- Uma matriz com todos os rótulos únicos
- Uma matriz geral de todos os rótulos

```python
filenames = ["drive/MyDrive/Dog Vision/train/" + fname + ".jpg" for fname in labels["id"]]

# listando os 10 primeiros
filenames[:10]
```

```
['drive/MyDrive/Dog Vision/train/000bec180eb18c7604dcecc8fe0dba07.jpg',
 'drive/MyDrive/Dog Vision/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg',
 'drive/MyDrive/Dog Vision/train/001cdf01b096e06d78e9e5112d419397.jpg',
 'drive/MyDrive/Dog Vision/train/00214f311d5d2247d5dfe4fe24b2303d.jpg',
 'drive/MyDrive/Dog Vision/train/0021f9ceb3235effd7fcde7f7538ed62.jpg',
 'drive/MyDrive/Dog Vision/train/002211c81b498ef88e1b40b9abf84e1d.jpg',
 'drive/MyDrive/Dog Vision/train/00290d3e1fdd27226ba27a8ce248ce85.jpg',
 'drive/MyDrive/Dog Vision/train/002a283a315af96eaea0e28e7163b21b.jpg',
 'drive/MyDrive/Dog Vision/train/003df8b8a8b05244b1d920bb6cf451f9.jpg',
 'drive/MyDrive/Dog Vision/train/0042188c895a2f14ef64a918ed9c7b64.jpg']
```

Criamos uma lista de caminhos de arquivo para as imagens em vez de importá-las para começar. Trabalhar com os caminhos dos arquivos (*nesse caso strings*) é bem mais eficiente do que trabalhar diretamente com as imagens.

Bem, agora que temos os caminhos de arquivo em uma lista, vamos obter os rótulos. Vamos pegar os rótulos de `labels` e transformá-los em uma array NumPy:

```python
import numpy as np
labels_array = labels["breed"].to_numpy()
labels_array[:10]
```

```
array(['boston_bull', 'dingo', 'pekinese', 'bluetick', 'golden_retriever',
       'bedlington_terrier', 'bedlington_terrier', 'borzoi', 'basenji',
       'scottish_deerhound'], dtype=object)
```

Agora é possível verificar se a quantidade de rótulos é igual a quantidade de arquivos:

```python
if len(labels_array) == len(filenames):
  print("Número de labels é igual ao número de arquivos!")
else:
  print("Número de labels não bate com o número de arquivos, verifique os dados!")
```

> Número de labels é igual ao número de arquivos!

Tudo parece em ordem, agora podemos começar, primeiro encontraremos todos os nomes únicos de raças de cães:

```python
racas = np.unique(labels_array)
len(racas)

120
```

Estamos trabalhando com imagens de 120 raças diferentes de cães. Agora podemos utilizar `racas` para transformar o array de rótulos em um array de booleans, assim criaremos uma lista indicando qual é o rótulo real (True) e qual não é (False).

```python
boolean_labels = [label == np.array(racas) for label in labels_array]
boolean_labels[:2]


[array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False]),
 array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False])]
```

Mas pra que fazer isso? Lembre-se que em `ML` é importante converter os dados em números antes de entregá-los para um modelo. Nesse caso estamos transformando os nomes das 120 raças de cães em um array. Vamos ver um exemplo com uma única raça para entender os que houve com essa transformação:

```python
# Ex: transformando um array de boolenas de uma raça em inteiros
print(labels_array[0]) # label de uma raça
print(np.where(racas == labels_array[0])[0][0]) # index referente ao label
print(boolean_labels[0].argmax()) # index onde o label aparece na matriz booleana
print(boolean_labels[0].astype(int)) # convertendo os booleans em inteiros
```

```
boston_bull
19
19
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0]
```

Perceba que agora temos os rótulos em formato numérico, hora de dividir os dados.

## Criando um conjunto de validação

A base de dados do Kaggle não separa um conjunto de validação (dados que podemos utilizar para testar o modelo antes de fazer previsões no conjunto de teste). Poderíamos utilizar a função `train_test_split` do `Scikit-Learn` ou apenas fazer divisões manuais nos dados. Vamos começar definindo nossas variáveis `X` e `y` como `dados` e `rótulos` respectivamente:

```python
X = filenames
y = boolean_labels
```

Como estamos trabalhando com uma base de dados com mais de 10.000 imagens, uma boa abordagem seria trabalhar com apenas uma parte delas para garantir que tudo esteja funcionando antes de treinar o modelo com toda a base. Isso porque os cálculos envolvendo mais de 10.000 imagens pode levar muito tempo. E um dos nossos objetivos em projetos de `ML` é reduzir o tempo entre os experimentos. Vamos começar um experimento com apenas 1000 imagens e aumentar esse valor conforme necessário.

```python
NUMERO_IMAGENS = 1000
NUMERO_IMAGENS

1000
```

Agora faremos a divisão dos dados em conjuntos de treino e validação. Usaremos o mesmo padrão de divisão visto anteriormente no capítulo de `Scikit-Learn`, uma divisão 80/20 (80% dos dados para treinamento e 20% para validação).

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X[:NUMERO_IMAGENS],
                                                  y[:NUMERO_IMAGENS],
                                                  test_size=0.2,
                                                  random_state=13)

len(X_train), len(y_train), len(X_val), len(y_val)

(800, 800, 200, 200)
```

Reservamos 800 imagens para treino e 200 para validação, vamos verificar uma amostra dos dados:

```python
X_train[:3], y_train[:2]


(['drive/MyDrive/Dog Vision/train/032620ae0f847d957d94d1fd76cb17e8.jpg',
  'drive/MyDrive/Dog Vision/train/056b535b441278e83839984f1b1da0a6.jpg',
  'drive/MyDrive/Dog Vision/train/0fbacbbdb4ad588598757b6d4bd111f1.jpg'],
 [array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
          True, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False]),
  array([False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False,  True, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False])])
```

## Pré-processamento de imagens

Até aqui apenas os rótulos estão no formato numérico, as imagens ainda são apenas caminhos de arquivo. Precisamos transformar as imagens em `tensors`.
Lembre-se um `tensor` é uma forma de representar informações em números. Pense em um Tensor como uma combinação de matrizes NumPy, que pode ser processado em uma GPU. Vamos converter uma imagem em um array NumPy para entender o restante do processo:

```python
from matplotlib.pyplot import imread
image = imread(filenames[20]) # lendo uma imagen, no caso de ID 20
image.shape

(375, 500, 3)
```

Perceba o shape (forma) da imagem, que é (375, 500, 3), ou seja altura, largura e cor (RGB). Podemos converter facilmente em um tensor utilizando `tf.constant()`:

```python
tf.constant(image)[:2]

<tf.Tensor: shape=(2, 500, 3), dtype=uint8, numpy=
array([[[127, 128, 120],
        [123, 124, 116],
        [117, 120, 111],
        ...,
        [ 98, 123,  65],
        [ 97, 122,  64],
        [102, 127,  69]],

       [[117, 120, 111],
        [114, 117, 108],
        [112, 115, 106],
        ...,
        [ 97, 122,  64],
        [ 94, 119,  61],
        [ 98, 123,  65]]], dtype=uint8)>
```

> Por padrão vamos redimensionar as imagens para um shape (`224,224`), veremos mais detalhes dessa escolha adiante.

Para converter todas as imagens em tensors criaremos uma função:

```python
# definindo o tamanho padrão das imagens
IMG_SIZE = 224

def processa_imagem(image_path):
  
  # lendo o arquivo de imagem
  image = tf.io.read_file(image_path)
  # transforma a imagem em um tensor com 3 canais de cores (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # converte os valores do canal de cor de 0-255 para valores entre 0-1
  image = tf.image.convert_image_dtype(image, tf.float32)
  # redimensiona a imagem para o novo shape (224,224)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image
```

Agora temos uma função que converte as imagens em `Tensors`, precisamos construir agora uma outra função para transformar os dados em lotes (TensorFlow BatchDataset). Por exemplo, estamos trabalhando com uma base de 10.000 imagens, somados esses arquivos podem ocupar toda a memória da GPU, tentar calcular todo o conjunto de dados ao mesmo tempo poderia resultar em erro. Em vez disso é mais eficiente criar lotes menores dos dados e computar um lote por vez. Antes, precisamos de uma função simples para transformar os nomes dos caminhos dos arquivos de imagem e os rótulos em tuplas:

```python
def image_label(image_path, label):
  image = processa_imagem(image_path)
  return image, label
```

Agora criaremos uma função para gerar os lotes de dados. Como estamos trabalhando com 3 conjuntos diferentes de dados (treino, validação e teste), vamos garantir que a função possa trabalhar com cada conjunto. Por padrão vamos definir um tamanho de lote de `32`.

```python
# tamanho do lote
BATCH_SIZE = 32

# função que retorna os dados em lotes
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  
  # se os dados forem de teste, não temos rótulos (provavelmente)
  if test_data:
    print("Criando lotes de dados de teste...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch = data.map(processa_imagem).batch(BATCH_SIZE)
    return data_batch
  
  # se os dados forem de um conjunto válido, não precisamos embaralhar
  elif valid_data:
    print("Criando lotes de dados de validação...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                               tf.constant(y)))
    data_batch = data.map(image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    # se os dados forem de treino, embaralhe
    print("Criando lotes de dados de treino...")
    # transformando caminhos de arquivos e rótulos em tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                              tf.constant(y)))
    
    # embaralha os caminhos de arquivos e rótulos
    data = data.shuffle(buffer_size=len(x))

    # cria tuplas (imagem, rótulo)
    data = data.map(image_label)

    # transforma os dados em lotes
    data_batch = data.batch(BATCH_SIZE)
  return data_batch
```

```python
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)

Criando lotes de dados de treino...
Criando lotes de dados de validação...


train_data.element_spec, val_data.element_spec

((TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None),
  TensorSpec(shape=(None, 120), dtype=tf.bool, name=None)),
 (TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None),
  TensorSpec(shape=(None, 120), dtype=tf.bool, name=None)))
```

Agora temos os dados em lotes, em pares de tensors (imagens e rótulos) prontos para serem utilizados em um GPU. Esse conceito pode ser um pouco difícil de entender, vamos tentar visualizar o que está acontecendo por baixo dos panos.

## Visualizando os lotes de dados

Vamos criar uma função para "plotar" 25 imagens:

```python
import matplotlib.pyplot as plt

def show_images(images, labels):
  plt.figure(figsize=(10, 10))
  # loop para exibir 25 imagens
  for i in range(25):
    # cria subplots 5 x 5
    ax = plt.subplot(5, 5, i+1)
    # mostra imagem
    plt.imshow(images[i])
    # adiciona label da imagem como título
    plt.title(racas[labels[i].argmax()])
    plt.axis("off")
```

Já sabemos que um lote (aqui no nosso caso) é uma coleção de tensors. Ou seja, para visualizar os dados em um lote é preciso desfazer o lote primeiro. Isso pode ser feito por meio da função `as_numpy_iterator()` em um lote de dados. Dessa forma vamos transformar um lote em algo que possa ser iterado. Esse processo pode demorar um pouco:

```python
train_images, train_labels = next(train_data.as_numpy_iterator())
show_images(train_images, train_labels)
```

![lote imagens tensors](images/lote-tensor.png)

Podemos fazer o mesmo para dados de validação:

```python
val_images, val_labels = next(val_data.as_numpy_iterator())
show_images(val_images, val_labels)
```

![lote imagens validação tensors](images/lote-tensor-val.png)

## Criando e treinando um modelo




---

## WIP
