# Redes neurais convolucionais e visão computacional com TensorFlow

Vamos nos aprofundar em algo específico, veremos um tipo especial de rede neural, as redes neurais convolucionais (`CNNs`) que podem ser utilizadas para visão computacional (*detecção de padrões em dados visuais*).

> Em `Deep Learning`, vários tipos diferentes de arquiteturas de modelo podem ser utilizados para diferentes problemas. Podemos utilizar uma `CNNs` para realizar previsões em dados de imagem ou texto por exemplo. Na prática, algumas arquiteturas funcionam melhor que outras.

Imagine poder classificar uma foto de comida se é pizza ou carne (*no capítulo passado fizemos algo parecido classificando raças de cães*). Detectar se um objeto está presente ou não em uma imagem ou ainda, se uma pessoa específica foi gravada por uma câmera de segurança. Neste capítulo, seguiremos com o workflow do TensorFlow que já vimos, ao mesmo tempo em que aprendemos sobre como construir e utilizar `CNNs`.

As redes neurais convolucionais funcionam muito bem com imagens, para aprender sobre elas, vamos resolver um problema de classificação utilizando uma base de dados de imagens. Usaremos o [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), uma coleção composta por 101 categorias diferentes de 101.000 imagens reais de pratos de comida.
Utilizaremos apenas duas categorias, pizza e carne para construir um classificado binário.

No Google Colab, vamos baixar o arquivo `.zip` com as imagens e descompactá-lo.

```python
import zipfile

# download dos arquivos de imagens zipados
!wget https://infoslack.pro/pizza_steak.zip

# descompactando o zip
zip_ref = zipfile.ZipFile("pizza_steak.zip", "r")
zip_ref.extractall()
zip_ref.close()
```

```
--2022-02-22 23:32:44--  https://infoslack.pro/pizza_steak.zip
Resolving infoslack.pro (infoslack.pro)... 35.202.40.163
Connecting to infoslack.pro (infoslack.pro)|35.202.40.163|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 109497985 (104M) [application/zip]
Saving to: ‘pizza_steak.zip’

pizza_steak.zip     100%[===================>] 104.42M  25.6MB/s    in 4.3s    

2022-02-22 23:32:49 (24.5 MB/s) - ‘pizza_steak.zip’ saved [109497985/109497985]
```

Vamos dar uma olhada nos dados, queremos verificar quantas imagens temos para treinamento:

```python
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))
num_steak_images_train

750
```

Uma boa ideia agora seria visualizar uma das imagens (*sempre que estiver trabalhando com dados é importante visualizá-los o máximo possível*):

```python
# Visualiza uma imagem
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

def view_random_image(target_dir, target_class):
  # diretório de destino
  target_folder = target_dir+target_class

  # Pega um caminho de imagem aleatório
  random_image = random.sample(os.listdir(target_folder), 1)

  # lendo a imagem e plotando com matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # mostra o formato da imagem (tamanho)

  return img
```

Criamos uma função para visualizar uma imagem aleatório, agora vamos utilizá-la:

```python
img = view_random_image(target_dir="pizza_steak/train/", target_class="steak")
```

```
Image shape: (512, 508, 3)
```

![cnn visualizando uma imagem aleatório do dataset](images/cnn/steak-1.png)

Agora que temos ideia do tipo de imagem que vamos trabalhar, todo o conjunto é composto por imagens semelhantes, no nosso caso em 2 classes.

```python
img.shape

(512, 512, 3)
```

Observe a forma (shape) da imagem, temos a largura, altura e os canais de cores (RGB). No nosso caso, a largura e altura variam entre as imagens da base de dados, já o valor do `RGB` é sempre igual a 3. Note os valores da matriz `img` abaixo:

```
img

array([[[182, 150, 103],
        [171, 138,  93],
        [175, 141,  96],
        ...,
        [155, 122,  89],
        [162, 129,  98],
        [169, 136, 105]],

       [[174, 142,  95],
        [166, 132,  87],
        [170, 136,  91],

       ...,

       [[118, 106,  68],
        [117, 105,  67],
        [131, 117,  80],
        ...,
        [ 85,  59,  46],
        [ 91,  68,  54],
        [122,  98,  86]],
        ...,
        [ 95,  67,  55],
        [101,  72,  64],
        [124,  95,  87]]], dtype=uint8)
```

Os valores da matriz estão entre 0 e 255. Isso acontece porque esse é o intervalo possível para os valores de vermelho, verde e azul (`RGB`). Imagine um pixel com um valor vermelho=0, verde=0, azul=255, esse pixel será muito azul! Quando desenvolvermos um modelo para diferenciar entre as imagens de pizza e carne, encontraremos padrões nesses diferentes valores de pixel, que determinam a aparência de cada classe.

Como vimos no capítulo passado, os modelos de `ML`, preferem trabalhar com valores entre 0 e 1. Ou seja, uma das etapas de pré-processamento mais comuns para trabalhar com imagens é normalizar os valores de pixel, dividindo as matrizes de imagens por `255`. Por exemplo:

```
img/255.

array([[[0.71372549, 0.58823529, 0.40392157],
        [0.67058824, 0.54117647, 0.36470588],
        [0.68627451, 0.55294118, 0.37647059],
        ...,
        [0.60784314, 0.47843137, 0.34901961],
        [0.63529412, 0.50588235, 0.38431373],
        [0.6627451 , 0.53333333, 0.41176471]],

       [[0.68235294, 0.55686275, 0.37254902],
        [0.65098039, 0.51764706, 0.34117647],
        [0.66666667, 0.53333333, 0.35686275],

       ...,

       [[0.4627451 , 0.41568627, 0.26666667],
        [0.45882353, 0.41176471, 0.2627451 ],
        [0.51372549, 0.45882353, 0.31372549],
        ...,
        [0.33333333, 0.23137255, 0.18039216],
        [0.35686275, 0.26666667, 0.21176471],
        [0.47843137, 0.38431373, 0.3372549 ]],

       [[0.47058824, 0.42352941, 0.2745098 ],
        [0.49019608, 0.43921569, 0.30196078],
        [0.54509804, 0.49019608, 0.35294118],
```

## Arquitetura de uma rede neural convolucional

---
WIP
