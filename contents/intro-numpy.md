# Introdução ao NumPy

NumPy (Numerical Python), em outras palavras é o cérebro da computação científica e numérica em Python. Machine Learning aborda a transformação de dados em números para encontrar padrões, nesse momento o NumPy entra em ação.

Nada impede que você consiga fazer cálculos numéricos usando Python puro, no começo você vai achar o Python rápido, mas conforme o volume de dados para trabalhar aumenta, você percebe lentidão ao realizar esses cálculos. E esse é um dos principais motivos pelos quais usamos NumPy, por ser rápido.

Nos bastidores, o código do NumPy foi otimizado para ser executado usando **C**, que é outra linguagem de programação, que pode fazer as coisas muito mais rápidas que Python. Ou seja, podemos escrever os nossos códigos para fazer cálculos numéricos em Python usando NumPy e obter os benefícios adicionais de velocidade.

## Importando a biblioteca NumPy

Para começar a usar NumPy, podemos importar a biblioteca da mesma forma como fizemos com Pandas, no caso a maneira mais comum é importar com a abreviação `np`:

```python
import numpy as np
```