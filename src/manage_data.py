import pandas as pd
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def compute_average_percentages(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Define the columns to be considered for average computation
    columns_to_average = [
        'avg_parking_time_B_heatmap',
        'avg_search_parking_time_heatmap',
        'distance_parking_point_B_heatmap',
        'avg_parking_time_B',
        'avg_search_parking_time',
        'distance_parking_point_B'
    ]

    # Round usage percentage to target values (0, 25, 50, 75, 100)
    target_percentages = [0, 25, 50, 75, 100]
    df['usage_percentage_rounded'] = df['usage_percentage_heatmap'].str.rstrip('%').astype(float)
    df['usage_percentage_rounded'] = df['usage_percentage_rounded'].apply(
        lambda x: min(target_percentages, key=lambda p: abs(p - x)))

    # Group data by rounded percentage and compute the mean for specified columns, ignoring NaN values
    df_avg = df.groupby(['usage_percentage_rounded', 'alpha', 'vehicles'])[columns_to_average].mean().reset_index()

    # Save the result in a new CSV file
    df_avg.to_csv(output_csv, index=False)


def extract_vehicle_number(vehicle_id):
    match = re.search(r'(\d+)', vehicle_id)
    return int(match.group(1)) if match else 0


def sort_data(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        data = list(reader)

    header, rows = data[0], data[1:]
    valid_rows = []

    for row in rows:
        try:
            perc_use_heatmap = float(row[0])
            test_percentage_number = float(row[1])
            valid_rows.append(row)
        except ValueError:
            print(f"Row ignored due to conversion error: {row}")

    rows_sorted = sorted(valid_rows, key=lambda x: (
        float(x[0]),
        float(x[1]),
        extract_vehicle_number(x[2])
    ))

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows_sorted)


def create_bar_plot(file_csv):
    df = pd.read_csv(file_csv)

    metrics_heatmap = ['avg_parking_time_B_heatmap', 'avg_search_parking_time_heatmap',
                       'distance_parking_point_B_heatmap']
    metrics_no_heatmap = ['avg_parking_time_B', 'avg_search_parking_time',
                          'distance_parking_point_B']

    metrics_ylabel = {'avg_parking_time_B': 'Time (s)', 'avg_search_parking_time': 'Time (s)',
                      'distance_parking_point_B': 'Distance (m)'}

    grouped_by_percentage_vehicles = df.groupby(['usage_percentage_rounded', 'vehicles'])

    for (percentage, vehicles), group in grouped_by_percentage_vehicles:
        plt.figure(figsize=(12, 6))

        for metric_heatmap, metric_no_heatmap in zip(metrics_heatmap, metrics_no_heatmap):
            group[metric_heatmap] = group[metric_heatmap].fillna(0)
            group[metric_no_heatmap] = group[metric_no_heatmap].fillna(0)

            if group[metric_heatmap].notna().any() and group[metric_no_heatmap].notna().any():
                data = pd.DataFrame({
                    'alpha': group['alpha'].astype(str),
                    'heatmap': group[metric_heatmap],
                    'no_heatmap': group[metric_no_heatmap]
                })

                data_melted = data.melt(id_vars='alpha', value_vars=['heatmap', 'no_heatmap'],
                                        var_name='type', value_name='value')
                data_melted["type"] = data_melted["type"].apply(
                    lambda x: "w Heatmap" if x == "heatmap" else "wo Heatmap")

                sns.barplot(x='alpha', y='value', hue='type', data=data_melted,
                            palette={'w Heatmap': 'lightblue', 'wo Heatmap': 'blue'},
                            dodge=True)

                plt.title(f"Adoption percentage: {percentage}% - Number of vehicles: {vehicles}", fontsize=16)
                plt.xlabel(r'$\alpha$', fontsize=16)
                plt.ylabel(f'{metrics_ylabel[metric_heatmap.split("_heatmap")[0]]}', fontsize=16)

                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.grid(True)
                plt.legend(title='w/wo Heatmap', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=16)
                plt.tight_layout()
                plt.savefig(
                    f"barplot_percentage_{percentage}_vehicles_{vehicles}_{metric_heatmap.split('_heatmap')[0]}.png")
                plt.show()


def create_percentage_graphs(file_csv):
    df = pd.read_csv(file_csv)

    metrics_heatmap = ['avg_parking_time_B_heatmap', 'avg_search_parking_time_heatmap',
                       'distance_parking_point_B_heatmap']
    metrics_no_heatmap = ['avg_parking_time_B', 'avg_search_parking_time',
                          'distance_parking_point_B']

    metrics_ylabel = {'avg_parking_time_B': 'Time (s)', 'avg_search_parking_time': 'Time (s)',
                      'distance_parking_point_B': 'Distance (m)'}
    metrics_title = {'avg_parking_time_B': 'Avg parking time', 'avg_search_parking_time': 'Avg parking search time',
                     'distance_parking_point_B': 'Avg distance from parking to target'}

    for metric_heatmap, metric_no_heatmap in zip(metrics_heatmap, metrics_no_heatmap):
        plt.figure(figsize=(12, 6))

        sns.lineplot(data=df, x='usage_percentage_rounded', y=metric_heatmap, marker='o',
                     label=f"w Heatmap", color='blue')
        sns.lineplot(data=df, x='usage_percentage_rounded', y=metric_no_heatmap, marker='o',
                     label=f"wo Heatmap", color='orange')

        plt.title(f"{metrics_title[metric_heatmap.split('_heatmap')[0]]} varying the adoption percentage", fontsize=16)
        plt.xlabel('Adoption percentage (%)', fontsize=16)
        plt.ylabel(f'{metrics_ylabel[metric_no_heatmap]}', fontsize=16)
        plt.xticks([0, 25, 50, 75, 100], fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=16)
        filename = f"graph_{metric_heatmap.split('_heatmap')[0]}.png"
        plt.savefig(filename)
        print(f"Graph saved as {filename}")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate graphs based on an input CSV")
    parser.add_argument('--input_csv', type=str, default='results_data.csv',
                        help="Path to the input CSV file")
    parser.add_argument('--output_csv', type=str, default=False,
                        help="Path to the output CSV file")
    parser.add_argument('--skip_average', action='store_true',
                        help="Skip average computation and use the existing CSV file")
    args = parser.parse_args()

    if args.skip_average:
        create_bar_plot(args.output_csv)
        create_percentage_graphs(args.output_csv)
    else:
        compute_average_percentages(args.input_csv, args.output_csv)
        create_bar_plot(args.output_csv)
        create_percentage_graphs(args.output_csv)
