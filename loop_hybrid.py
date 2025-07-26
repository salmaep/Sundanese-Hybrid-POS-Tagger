import os
import csv
import sys
from Hybrid_POS_Tag import HybridPOSTagger, TeeStringIO

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


# File input dan output
input_file = 'data/data_input22.txt'
validation_file = 'data/korpus_validasi_ID_5K.txt'
training_file = 'dataset/Korpus_Train.txt'
base_output_dir = 'hasil/hybrid/data_2_baru/5k_baru'

# Pastikan folder hasil ada
os.makedirs(base_output_dir, exist_ok=True)

# File output utama
metrics_file = os.path.join(base_output_dir, 'hasil_metrik_hybrid.csv')
rekap_file = os.path.join(base_output_dir, 'rekap_oov_statistics.csv')
tagged_results_file = os.path.join(base_output_dir, 'all_tagged_results.txt')

# Baca input text sekali saja
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        input_text = f.read().strip()
except FileNotFoundError:
    print(f"File input tidak ditemukan: {input_file}")
    sys.exit(1)

# Buka file CSV saja (hapus tagged_results_file dari sini)
with open(metrics_file, 'w', newline='', encoding='utf-8') as csvfile, \
     open(rekap_file, 'w', newline='', encoding='utf-8') as rekap_csv:

    # Setup CSV writers
    metrics_writer = csv.DictWriter(csvfile, fieldnames=[
        'Nama_Aturan', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'F0.5-Score'
    ])
    metrics_writer.writeheader()

    rekap_writer = csv.DictWriter(rekap_csv, fieldnames=[
        'Nama_Aturan', 'Total_Words', 'Total_OOV', 'HMM_Handled',
        'Numeric_Handled', 'Lexicon_Handled', 'Affix_Rules_Handled',
        'Nasal_Rules_Handled', 'Syntax_Rules_Handled', 'Default_NN_Handled'
    ])
    rekap_writer.writeheader()

    # Loop untuk tiap aturan morfologi
    for rule_path in affix_rule_files:
        rule_name = os.path.basename(rule_path).replace(".json", "")
        output_dir = os.path.join(base_output_dir, rule_name)
        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, f"log_{rule_name}.txt")
        log_capture = TeeStringIO(log_file)
        sys.stdout = log_capture

        # Lokasi hasil tagging individual
        tagged_result_path = os.path.join(output_dir, 'tagged_result.txt')

        try:
            print(f"[INFO] Evaluasi aturan: {rule_name}")

            # Inisialisasi hybrid tagger
            hybrid_tagger = HybridPOSTagger()

            # Muat aturan morfologi
            if os.path.exists(rule_path):
                hybrid_tagger.rule_based.affix_rules = hybrid_tagger.rule_based.load_affix_rules(rule_path)
            else:
                print(f"[ERROR] File aturan tidak ditemukan: {rule_path}")
                continue

            # Training HMM
            hybrid_tagger.train_hmm(training_file)

            # Tag kalimat input
            results = hybrid_tagger.tag_sentence(input_text)

            # Evaluasi dan simpan hasil lengkap
            hybrid_tagger.evaluate_and_save_results(results, validation_file, output_dir)

            # Simpan hasil tagging ke file per aturan
            with open(tagged_result_path, 'w', encoding='utf-8') as tagged_file:
                tagged_file.write(f"[{rule_name}] ")
                tagged_output = " ".join([f"{word}|{tag}" for word, tag in zip(results['words'], results['final_tags'])])
                tagged_file.write(tagged_output + "\n")

            # Hitung metrik dan simpan ke CSV
            metrics = hybrid_tagger.evaluator.calculate_metrics_macro(
                hybrid_tagger.evaluator.read_and_extract_labels(validation_file),
                results['final_tags'],
                beta=0.5
            )
            if metrics:
                metrics_writer.writerow({
                    'Nama_Aturan': rule_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'F0.5-Score': metrics['f0.5_score']
                })

                oov_stats = results['oov_stats']
                rekap_writer.writerow({
                    'Nama_Aturan': rule_name,
                    'Total_Words': oov_stats['total_words'],
                    'Total_OOV': oov_stats['total_oov'],
                    'HMM_Handled': oov_stats['hmm_handled'],
                    'Numeric_Handled': oov_stats['numeric_handled'],
                    'Lexicon_Handled': oov_stats['lexicon_handled'],
                    'Affix_Rules_Handled': oov_stats['affix_rules_handled'],
                    'Nasal_Rules_Handled': oov_stats['nasal_rules_handled'],
                    'Syntax_Rules_Handled': oov_stats['syntax_rules_handled'],
                    'Default_NN_Handled': oov_stats['unknown_handled']
                })

        except Exception as e:
            print(f"[ERROR] Gagal memproses {rule_name}: {e}")
            import traceback
            traceback.print_exc()
            with open(tagged_result_path, 'w', encoding='utf-8') as tagged_file:
                tagged_file.write(f"[{rule_name}] ERROR: {e}\n")

        finally:
            sys.stdout = log_capture.original_stdout
            log_capture.close()
            csvfile.flush()
            rekap_csv.flush()

print(f"\n=== EVALUASI SELESAI ===")
print(f"Hasil tagging per aturan disimpan dalam masing-masing folder.")