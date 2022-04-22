from scipy.optimize import linprog
import numpy as np

# Criação da função para imprimir o relatório
def relatorio_dual_simplex(c, A, b, nomes_variaveis_decisao, nomes_restricoes):
    '''Parâmetros
       c: é um vetor de uma dimensão contendo os coeficientes da função objetivo
       A: é um vetor de duas dimensões contendo os coeficientes das inequações de restrições
       b: é um vetor de uma dimensão contendo os valores correspondentes às inequações de restrição
       nomes_variaveis_decisao: é uma lista contendo os nomes das variáveis de decisão
       nomes_restricoes: é uma lista contendo os nomes das variáveis de restrição'''

    # O parâmetro 'method' deve ser preenchido com o método escolhido pra resolver
    # Nesse caso, 'highs-ds' é o método 'dual simplex'
    res = linprog(c, A, b, method='highs-ds')

    print('=' * 35)

    # Valor ótimo
    print(f'Valor ótimo: {res.fun * (-1):.4f}'.center(35))
    
    print('-' * 35)

    # Quantidades ótimas das variáveis de decisão
    print(f'Quantidade ótima de cada variável:'.center(35))
    for i in range(len(nomes_variaveis_decisao)):
        print(f'  - {nomes_variaveis_decisao[i]}: {np.round(res.x[i], decimals=4)}')

    print('-' * 35)

    # Folga em cada uma das restrições
    print('Folga:'.center(35), end='\n\n')
    for r in range(len(nomes_restricoes)):
        print(f'{nomes_restricoes[r]}: {np.round(res.slack[r], decimals=4)}')
    
    print('-' * 35)

    # Preços sombra
    print('Preço Sombra:'.center(35), end='\n\n')
    for k in range(len(nomes_restricoes)):
        print(f'{nomes_restricoes[k]}: {np.round(res.ineqlin.marginals[k], decimals=4) * -1}')

    print('-' * 35)

    # Utilização das restrições
    print('Utilização:'.center(35), end='\n\n')
    for j in range(len(nomes_restricoes)):
        print(f'{nomes_restricoes[j]}: {np.round(b[j] - res.slack[j], decimals=4)}')

    print('=' * 35)