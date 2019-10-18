import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

""" Este código lê as bases de dados simulados e reais, roda os modelos descritos nesta monografia e plota o gráfico 
dos resíduos. Para rodar o modelo, utilize um terminal ou console Python 3.6.4 e os módulos importados acima. Com as
bases de dados e o código no mesmo diretório, digite: <python main.py> Os resultados são impressos no Terminal e 
gravados em arquivo TXT. """


def reg(col, data, optional_col=""):
    """ Função que roda as regressões """
    res = smf.ols("{} ~  {}".format(col, optional_col), data=data).fit()
    sns.distplot(res.resid)
    plt.show()
    return res


def print_reg3(m1, m2, m3, m4, m5):
    """ Função que organiza os resultados em tabela padrão """
    info_dict={'Log-likelihood': lambda x: f"{x.llf:.2f}",
               'R-squared Adj': lambda x: f"{x.rsquared_adj:.2f}",
               'AIC': lambda x: f"{x.aic:.2f}",
               'BIC': lambda x: f"{x.bic:.2f}",
               'No. observations': lambda x: f"{int(x.nobs):d}"}

    results_table = summary_col(results=[m1, m2, m3, m4, m5],
                                float_format='%0.2f',
                                stars=True,
                                model_names=['Simul 1',
                                             'Simul 2',
                                             'Simul 3',
                                             'Real  4',
                                             'Real  5'],
                                info_dict=info_dict)

    results_table.add_title('Table - OLS Regressions')
    print(results_table)
    return results_table


if __name__ == "__main__":
    # Leitura das bases
    base = pd.read_csv('simulado.csv', sep=';')
    real = pd.read_csv('real.csv', sep=';')

    # Regressões econométricas
    lm1 = reg('average_qli', base, ' + ALTERNATIVE0 + FPM_DISTRIBUTION + siglas')
    lm2 = reg('average_qli', base, ' + ALTERNATIVE0 + FPM_DISTRIBUTION + siglas + gdp_index + unemployment + '
                                   'num_mun_na_acp + firms_profit + average_workers + inflation')
    lm3 = reg('average_qli', base, ' + ALTERNATIVE0 + FPM_DISTRIBUTION  + gdp_index + unemployment + '
                                   'num_mun_na_acp + firms_profit + average_workers + inflation')
    lm4 = reg('average_qli', real, ' + ALTERNATIVE0 + FPM_DISTRIBUTION + siglas')
    lm5 = reg('average_qli', real, ' + ALTERNATIVE0 + FPM_DISTRIBUTION + siglas + ln_populacao_acp + area_acp + '
                                   ' pct_superior_completo + num_mun_na_acp + hhi')

    # Geração de tabelas
    res = print_reg3(lm1, lm2, lm3, lm4, lm5)

    # Gravação de resultados
    with open('table.txt', 'w') as f:
        f.write(res.as_text())
