#Extracting relevant data from proxy run
import csv
from scipy import stats

def extract_data_from_csv(file_path):
    #horizon, key, proxy, epsilon, mean_epsilon, min_epsilon, median_epsilon, q1_epsilon, q3_epsilon, proxy_time, simulate_time
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def extract_data_by_horizon(rows, h):
    proxy, eps, mean_eps = [], [], []
    filtered_rows = [row for row in rows[1:] if int(row[0]) == h]
    for row in filtered_rows:
        proxy.append(float(row[2]))
        eps.append(float(row[3]))
        mean_eps.append(float(row[4]))
    return proxy, eps, mean_eps

def calculate_pearson_spearman(eps, prox):
    pearson_corr, p_value_p = stats.pearsonr(eps, prox)
    # print(f"Pearson r: {pearson_corr:.3f}, p-value: {p_value_p:.3f}")

    spearman_corr, p_value_s = stats.spearmanr(eps, prox)
    # print(f"Spearman r: {spearman_corr:.3f}, p-value: {p_value_s:.3f}")

    return pearson_corr, spearman_corr


if __name__ == "__main__":
    #iterate thru file paths.
    paths = [r"unicycle-taylor\proxy_analysis_results_1-5.csv", r"unicycle-taylor\proxy_analysis_results_6-10.csv"]
    for file in paths:
        rows = extract_data_from_csv(file) # returns rows
        pearson_corrs, spearman_corrs = [], []
        h_start = int(rows[1][0]) # get 1st horizon
        h_end = int(rows[-1][0]) # get last horizon
        for horizon in range(h_start, h_end+1):
            proxy, eps, mean_eps = extract_data_by_horizon(rows, horizon)
            pearson_corr, spearman_corr = calculate_pearson_spearman(mean_eps, proxy)
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)

            print(f"Horizon {horizon}: Pearson r: {pearson_corr:.3f}, Spearman r: {spearman_corr:.3f}")
            # print(f"Horizon {horizon}:")
            # print(proxy, "\n", eps, "\n", mean_eps)

