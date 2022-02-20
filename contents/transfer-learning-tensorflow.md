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

---

## WIP
