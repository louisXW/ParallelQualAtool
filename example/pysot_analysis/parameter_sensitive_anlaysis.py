import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def read_pysot_result(filename):
    """
    Read the parameter vector X of each simulation
    :param filename: The pysot result logging file pysot_result.dat
    :return: The parameter vector list in order of simulation id and iterations id
    """
    file = filename
    with open(file) as f:
        reader = f.readlines()
        data = []
        for i, line in enumerate(reader):
            if i >= 1:
                str1 = line.split('\t')
                str2 = str1[3].replace('@[', '').replace('\n', '').replace(']', '').split(', ')
                str3 = str1[0:3] + str2
                if 'None' in str3:
                    pass
                else:
                    str3 = [float(item) for item in str3]
                    data.append(str3)
        data = np.asarray(data)
        indx = np.lexsort((data[:, 1], data[:, 0]))
        data = data[indx]
        return data

def read_each_obj(filename):
    """
    read each sub objective function value of each simulation
    :param filename: The result logging file inclduing the value of each su objective function
    :return: The array of each objective function value in order of simulation id and iterations id
    """
    file = filename
    objs = np.loadtxt(file, skiprows=0)
    indx = np.lexsort((objs[:, 1], objs[:, 0]))
    objs = objs[indx]
    return objs

def main():
    filename_result = "pysot_result.dat"
    filename_eachobj = "pysot_result_eachobj.dat"
    parameter = read_pysot_result(filename_result)
    eachobj = read_each_obj(filename_eachobj)
    parameter = parameter[:, 3:]
    eachobj = eachobj[:, 2:]
    corf = generate_correlation_map(eachobj.T, parameter.T)

    obj_index = ['Chlfa UPR E1', 'Chlfa UPR B1', 'TN UPR E1', 'TN UPR B1', 'NH4 UPR E1', 'NH4 UPR B1',
                 'NO3 UPR E1', 'NO3 UPR B1', 'TP UPR B1','TP UPR E1', 'DO UPR E1', 'DO UPR B1',
                 'TSS UPR E1', 'TSS UPR B1', 'Biomass UPR B1']
    # obj_index = ['chlfa_e1', 'chlfa_b1', 'tn_e1', 'tn_b1', 'nh4_e1', 'nh4_b1', 'no3_e1', 'no3_b1','biomass']
    par_index = ['KsAmNit', 'a_dNpr', 'RcDenWat', 'RcDenSed', 'RcNit20_w', 'RcNit20_Sed']
    par_index = ['kl_f_w', 'kl_f_s', 'b_poc1_2_w', 'b_poc1_2_s', 'kl_m_w', 'kl_m_s', 'b_poc2_3_w', 'b_poc2_3_s','adnpr_w', 'adnpr_s', 'adppr_w', 'adppr_s']
    sns.heatmap(corf, xticklabels=par_index, yticklabels=obj_index, cmap='bwr')
    plt.show()

if __name__ == '__main__':
    main()

