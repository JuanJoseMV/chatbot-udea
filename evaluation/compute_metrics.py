import json
import csv
import numpy as np
import argparse

def compute_accuracy_at_k(rankings, k):
    return sum(1 for rank in rankings if rank <= k) / len(rankings)

def main(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    results = {}
    total_acc_at_1 = []
    total_acc_at_5 = []
    total_acc_at_10 = []

    for article, questions in data.items():
        if article not in results:
            results[article] = []
        for question, ranking in questions:
            acc_at_1 = compute_accuracy_at_k([ranking], 1)
            acc_at_5 = compute_accuracy_at_k([ranking], 5)
            acc_at_10 = compute_accuracy_at_k([ranking], 10)
            results[article].append([question, acc_at_1, acc_at_5, acc_at_10])
            total_acc_at_1.append(acc_at_1)
            total_acc_at_5.append(acc_at_5)
            total_acc_at_10.append(acc_at_10)

    csv_path = data_path.replace('.json', '.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Article', 'Mean Accuracy@1', 'Mean Accuracy@5', 'Mean Accuracy@10'])
        for article, article_results in results.items():
            mean_acc_at_1 = np.mean([result[1] for result in article_results])
            mean_acc_at_5 = np.mean([result[2] for result in article_results])
            mean_acc_at_10 = np.mean([result[3] for result in article_results])
            writer.writerow([article, mean_acc_at_1, mean_acc_at_5, mean_acc_at_10])

    avg_acc_at_1 = np.mean(total_acc_at_1)
    avg_acc_at_5 = np.mean(total_acc_at_5)
    avg_acc_at_10 = np.mean(total_acc_at_10)

    print(f'Average Accuracy@1: {avg_acc_at_1:.4f}')
    print(f'Average Accuracy@5: {avg_acc_at_5:.4f}')
    print(f'Average Accuracy@10: {avg_acc_at_10:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute accuracy metrics from rankings.')
    parser.add_argument('--data_path', type=str, help='Path to the input JSON file')
    args = parser.parse_args()

    data_path = args.data_path
    main(data_path)