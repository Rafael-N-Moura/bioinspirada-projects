# Algoritmo Genético para o Problema das 8 Rainhas

Este projeto implementa um algoritmo genético para resolver o problema das 8 rainhas, seguindo as especificações fornecidas.

## Especificações Implementadas

- **Representação**: Permutação de string de bits
- **Recombinação**: "cut-and-crossfill" crossover
- **Probabilidade de Recombinação**: 90%
- **Mutação**: Múltiplas estratégias implementadas:
  - Troca de genes (SWAP)
  - Inserção (INSERTION)
  - Inversão (INVERSION)
  - Perturbação (PERTURBATION)
- **Probabilidade de Mutação**: 40%
- **Seleção de pais**: Ranking - Melhor de 2 de 5 escolhidos aleatoriamente
- **Seleção de sobreviventes**: Substituição do pior
- **Tamanho da população**: 10
- **Número de filhos gerados**: 2
- **Inicialização**: Aleatória
- **Condição de término**: Encontrar a solução ou 10.000 avaliações de fitness

## Estratégias de Mutação

O projeto implementa quatro diferentes estratégias de mutação para comparação:

1. **Mutação por Troca (SWAP)**
   - Dois genes são escolhidos aleatoriamente e suas posições são trocadas
   - Preserva a maior parte da adjacência da informação
   - Útil para ajustes finos em soluções já boas

2. **Mutação por Inserção (INSERTION)**
   - Dois genes são escolhidos aleatoriamente
   - O segundo gene é movido para próximo do primeiro
   - Preserva a maior parte da ordem da informação adjacente
   - Útil quando a ordem relativa dos genes é importante

3. **Mutação por Inversão (INVERSION)**
   - Dois genes são escolhidos aleatoriamente
   - A sub-string entre eles é invertida
   - Preserva a adjacência da informação, mas modifica a ordem
   - Útil para quebrar padrões rígidos mantendo conexões locais

4. **Mutação por Perturbação (PERTURBATION)**
   - Um subconjunto de genes é escolhido aleatoriamente
   - O subconjunto é rearranjado aleatoriamente
   - Permite mudanças mais drásticas na configuração
   - Útil para exploração mais ampla do espaço de busca

## Requisitos

- Python 3.7+
- NumPy
- Matplotlib

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

Para executar o algoritmo com todas as estratégias de mutação:

```bash
python eight_queens_ga.py
```

## Análise de Resultados

O programa executa 30 experimentos para cada estratégia de mutação e fornece as seguintes análises:

1. Número de execuções que convergiram
2. Média e desvio padrão das iterações para convergência
3. Fitness médio da população em cada execução
4. Gráficos de convergência mostrando a média e o melhor indivíduo por iteração
5. Fitness médio alcançado nas 30 execuções (média e desvio padrão) 