import os
import csv
from rule_based4 import RuleBased
from run_rule_based import process_file_with_rule

# Daftar file aturan morfologi
affix_rule_files = [
    "dataset/AM1.json",
    "dataset/AM2.json",
    "dataset/AM3.json",
    "dataset/AM4.json",
    "dataset/AM5.json",
    "dataset/AM6.json",
    "dataset/AM7.json",
    "dataset/Progressive1.json",
    "dataset/Progressive2.json",
    "dataset/Progressive3.json",
    "dataset/Progressive4.json",
    "dataset/Progressive5.json",
    "dataset/Comprehensive.json"
]

input_file = 'data/data_input55.txt'
validation_file = 'data/korpus_validasi_OOD_5K.txt'
metrics_file = 'hasil/rule_based/hasil_metrik_rule_based_data5_5k_baru.csv'

os.makedirs('hasil/rule_based/data5_5k_baru', exist_ok=True)

with open(metrics_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Nama_Aturan', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'F0.5-Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for rule_path in affix_rule_files:
        rule_name = os.path.basename(rule_path).replace(".json", "")
        output_file = f'hasil/rule_based/data5_5k_baru/output_{rule_name}.txt'
        print(f"\nEvaluasi aturan: {rule_name}")

        rb = RuleBased(rule_path)
        metrics = process_file_with_rule(rb, input_file, output_file, validation_file)

        if metrics:
            row = {"Nama_Aturan": rule_name}
            row.update(metrics)
            writer.writerow(row)
            print(f"✔ Selesai: {rule_name} — Akurasi: {metrics['Accuracy']:.4f}")
        else:
            print(f"✖ Gagal evaluasi untuk {rule_name}")
