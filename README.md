# Algoritmo Genético para o Problema das 8 Rainhas

Este projeto implementa um algoritmo genético para resolver o problema das 8 rainhas, seguindo as especificações fornecidas.

## Especificações Implementadas

- **Representação**: Permutação de string de bits
- **Recombinação**: "cut-and-crossfill" crossover
- **Probabilidade de Recombinação**: 90%
- **Mutação**: Troca de genes
- **Probabilidade de Mutação**: 40%
- **Seleção de pais**: Ranking - Melhor de 2 de 5 escolhidos aleatoriamente
- **Seleção de sobreviventes**: Substituição do pior
- **Tamanho da população**: 100
- **Número de filhos gerados**: 2
- **Inicialização**: Aleatória
- **Condição de término**: Encontrar a solução ou 10.000 avaliações de fitness

## Requisitos

- Python 3.7+
- NumPy
- Matplotlib

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

Para executar o algoritmo:

```bash
python eight_queens_ga.py
```

## Análise de Resultados

O programa executa 30 experimentos e fornece as seguintes análises:

1. Número de execuções que convergiram
2. Média e desvio padrão das iterações para convergência
3. Fitness médio da população em cada execução
4. Gráficos de convergência mostrando a média e o melhor indivíduo por iteração
5. Fitness médio alcançado nas 30 execuções (média e desvio padrão) 